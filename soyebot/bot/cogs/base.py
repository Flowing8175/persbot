"""Base Cog for Chat-based interactions."""

import logging
import asyncio
from abc import abstractmethod
from typing import Optional, List

import discord
from discord.ext import commands

from soyebot.bot.buffer import MessageBuffer
from soyebot.bot.session import SessionManager
from soyebot.config import AppConfig
from soyebot.services.llm_service import LLMService
from soyebot.bot.chat_handler import ChatReply, create_chat_reply, resolve_session_for_message, send_split_response
from soyebot.utils import GENERIC_ERROR_MESSAGE, extract_message_content

logger = logging.getLogger(__name__)

class BaseChatCog(commands.Cog):
    """Abstract base cog containing shared logic for message buffering and processing."""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        session_manager: SessionManager,
    ):
        self.bot = bot
        self.config = config
        self.llm_service = llm_service
        self.session_manager = session_manager

        # Buffer and Task Management
        self.message_buffer = MessageBuffer(delay=config.message_buffer_delay)
        self.sending_tasks: dict[int, asyncio.Task] = {}
        self.processing_tasks: dict[int, asyncio.Task] = {}
        self.active_batches: dict[int, list[discord.Message]] = {}

    @abstractmethod
    async def _send_response(self, message: discord.Message, reply: ChatReply):
        """Send the generated reply to the channel. Must be implemented by subclasses."""
        pass

    async def _process_batch(self, messages: list[discord.Message]):
        """Standard batch processing logic."""
        if not messages:
            return

        primary_message = messages[0]
        channel_id = primary_message.channel.id

        # Register task
        self.active_batches[channel_id] = messages
        self.processing_tasks[channel_id] = asyncio.current_task()

        try:
            full_text = await self._prepare_batch_context(messages)

            if not full_text:
                return

            logger.info("Processing batch of %d messages from %s: %s", len(messages), primary_message.author.name, full_text[:100])

            async with primary_message.channel.typing():
                resolution = await resolve_session_for_message(
                    primary_message,
                    full_text,
                    session_manager=self.session_manager,
                )

                if not resolution:
                    return

                reply = await create_chat_reply(
                    messages if isinstance(messages, list) else primary_message,
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                )

                if reply:
                    # Use the last message as the anchor for reply/context
                    anchor_message = messages[-1]
                    await self._send_response(anchor_message, reply)

        except asyncio.CancelledError:
            logger.info("Batch processing cancelled for channel %s.", primary_message.channel.name)
            raise

        except Exception as e:
            logger.error("메시지 처리 중 예상치 못한 오류 발생: %s", e, exc_info=True)
            await self._handle_error(primary_message, e)

        finally:
            if self.processing_tasks.get(channel_id) == asyncio.current_task():
                self.processing_tasks.pop(channel_id, None)
                self.active_batches.pop(channel_id, None)

    async def _prepare_batch_context(self, messages: list[discord.Message]) -> str:
        """Prepare the text content for the LLM, including context if needed. Can be overridden."""
        # Default implementation: just combine messages
        combined_content = []
        for msg in messages:
            content = extract_message_content(msg)
            if content:
                 combined_content.append(content)
        return "\n".join(combined_content)

    async def _handle_error(self, message: discord.Message, error: Exception):
        """Handle errors during processing. Can be overridden."""
        await message.channel.send(GENERIC_ERROR_MESSAGE)

    async def _handle_break_cut_sending(self, channel_id: int, channel, reply: ChatReply):
        """Helper for sending split responses."""
        if channel_id in self.sending_tasks and not self.sending_tasks[channel_id].done():
            self.sending_tasks[channel_id].cancel()

        task = asyncio.create_task(
            send_split_response(channel, reply, self.session_manager)
        )
        self.sending_tasks[channel_id] = task

        def _cleanup(t):
            if self.sending_tasks.get(channel_id) == t:
                self.sending_tasks.pop(channel_id, None)

        task.add_done_callback(_cleanup)

    def _cancel_active_tasks(self, channel_id: int, author_name: str, message_type: str = "new message"):
        """Cancel sending and processing tasks for a channel."""
        if self.config.break_cut_mode and channel_id in self.sending_tasks:
            task = self.sending_tasks[channel_id]
            if not task.done():
                logger.info(f"{message_type} from {author_name} interrupted sending in channel {channel_id}")
                task.cancel()

        messages_to_prepend = []
        if channel_id in self.processing_tasks:
            task = self.processing_tasks[channel_id]
            if not task.done():
                logger.info(f"{message_type} from {author_name} interrupted processing in channel {channel_id}. Merging messages.")
                messages_to_prepend = self.active_batches.get(channel_id, [])
                task.cancel()

        return messages_to_prepend

    @commands.Cog.listener()
    async def on_typing(self, channel: discord.abc.Messageable, user: discord.abc.User, when: float):
        if user.bot:
            return

        if hasattr(channel, 'id'):
            # Check filtering in subclass if needed, but BaseCog assumes if we are listening, we care.
            # But we should probably check if it's an interesting channel in the subclass listener
            # or rely on the buffer check (buffer handles non-existent keys gracefully).

            # Subclasses should likely override or wrap this if they have specific ignore logic
            # For now, we put shared logic here:
            if not self.config.break_cut_mode:
                self.message_buffer.handle_typing(channel.id, self._process_batch)

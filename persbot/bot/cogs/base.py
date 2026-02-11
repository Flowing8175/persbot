"""Base Cog for Chat-based interactions."""

import asyncio
import logging
from abc import abstractmethod
from typing import List, Optional

import discord
from persbot.tools.manager import ToolManager
from discord.ext import commands

from persbot.bot.buffer import MessageBuffer
from persbot.bot.chat_handler import (
    ChatReply,
    create_chat_reply,
    create_chat_reply_stream,
    resolve_session_for_message,
    send_split_response,
    send_streaming_response,
)
from persbot.bot.session import SessionManager
from persbot.config import AppConfig
from persbot.services.llm_service import LLMService
from persbot.utils import GENERIC_ERROR_MESSAGE, extract_message_content, send_discord_message

logger = logging.getLogger(__name__)


class ActiveAPICall:
    """Tracks an active API call for cancellation.

    This class ensures that when a new message arrives during batch processing:
    1. The ongoing LLM API-side generation is cancelled (like pressing STOP in chatgpt.com)
    2. The stacked messages are included in new request
    """

    def __init__(self, task: asyncio.Task, cancel_event: asyncio.Event) -> None:
        self.task = task
        self.cancel_event = cancel_event

    def cancel(self) -> None:
        """Cancel both task and set cancel event immediately."""
        if self.cancel_event:
            self.cancel_event.set()
        if self.task and not self.task.done():
            self.task.cancel()


class BaseChatCog(commands.Cog):
    """Abstract base cog containing shared logic for message buffering and processing."""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        session_manager: SessionManager,
        tool_manager: Optional[ToolManager] = None,
    ) -> None:
        self.bot = bot
        self.config = config
        self.llm_service = llm_service
        self.session_manager = session_manager
        self.tool_manager = tool_manager

        # Buffer and Task Management
        self.message_buffer = MessageBuffer(delay=config.message_buffer_delay)
        self.sending_tasks: dict[int, asyncio.Task] = {}
        self.processing_tasks: dict[int, asyncio.Task] = {}
        self.active_batches: dict[int, list[discord.Message]] = {}

        # Cancellation signal tracking (per-channel abort events)
        self.cancellation_signals: dict[int, asyncio.Event] = {}
        # Track active API calls for proper cancellation (per-channel)
        self.active_api_calls: dict[int, ActiveAPICall] = {}

    @abstractmethod
    async def _send_response(self, message: discord.Message, reply: ChatReply):
        """Send the generated reply to the channel. Must be implemented by subclasses."""
        pass

    def _should_use_streaming(self) -> bool:
        """Check if streaming mode should be used based on break_cut_mode config."""
        return self.config.break_cut_mode

    async def _process_with_streaming(
        self,
        messages: list[discord.Message],
        primary_message: discord.Message,
        channel_id: int,
        cancel_event: asyncio.Event,
    ):
        """Process using streaming API with fallback to non-streaming on failure."""
        anchor_message = messages[-1]

        # Create streaming task with fallback
        async def _make_streaming_call():
            try:
                async with primary_message.channel.typing():
                    resolution = await resolve_session_for_message(
                        primary_message,
                        await self._prepare_batch_context(messages),
                        session_manager=self.session_manager,
                        cancel_event=cancel_event,
                    )

                    if not resolution:
                        return None

                    # Use streaming response
                    stream = create_chat_reply_stream(
                        messages if isinstance(messages, list) else primary_message,
                        resolution=resolution,
                        llm_service=self.llm_service,
                        session_manager=self.session_manager,
                        tool_manager=self.tool_manager,
                        cancel_event=cancel_event,
                    )
                    return stream
            except Exception as e:
                logger.warning("Streaming failed, falling back to non-streaming: %s", e)
                # Fall back to non-streaming
                resolution = await resolve_session_for_message(
                    primary_message,
                    await self._prepare_batch_context(messages),
                    session_manager=self.session_manager,
                    cancel_event=cancel_event,
                )

                if not resolution:
                    return None

                reply = await create_chat_reply(
                    messages if isinstance(messages, list) else primary_message,
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                    tool_manager=self.tool_manager,
                    cancel_event=cancel_event,
                )
                return reply

        # Create and track streaming task
        streaming_task = asyncio.create_task(_make_streaming_call())
        self.active_api_calls[channel_id] = ActiveAPICall(streaming_task, cancel_event)

        try:
            result = await streaming_task

            # Check if result is a stream (async iterator) or ChatReply (fallback)
            if hasattr(result, "__aiter__"):
                # It's a stream - use send_streaming_response
                await self._send_streaming_response(channel_id, primary_message.channel, result)
            elif result and isinstance(result, ChatReply):
                # Fallback to non-streaming - send normally
                await self._send_response(anchor_message, result)
        finally:
            # Clean up API call tracking
            self.active_api_calls.pop(channel_id, None)

    async def _process_with_reply(
        self,
        messages: list[discord.Message],
        primary_message: discord.Message,
        channel_id: int,
        cancel_event: asyncio.Event,
    ):
        """Process using non-streaming API (original behavior)."""
        anchor_message = messages[-1]

        async def _make_api_call():
            async with primary_message.channel.typing():
                resolution = await resolve_session_for_message(
                    primary_message,
                    await self._prepare_batch_context(messages),
                    session_manager=self.session_manager,
                    cancel_event=cancel_event,
                )

                if not resolution:
                    return None

                reply = await create_chat_reply(
                    messages if isinstance(messages, list) else primary_message,
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                    tool_manager=self.tool_manager,
                    cancel_event=cancel_event,
                )
                return reply

        # Create and track API call task for immediate cancellation
        api_call_task = asyncio.create_task(_make_api_call())
        self.active_api_calls[channel_id] = ActiveAPICall(api_call_task, cancel_event)

        try:
            reply = await api_call_task
            if reply:
                await self._send_response(anchor_message, reply)
        finally:
            # Clean up API call tracking
            self.active_api_calls.pop(channel_id, None)

    async def _send_streaming_response(self, channel_id: int, channel, stream):
        """Send streaming response to Discord.

        This method handles sending chunks as they arrive from LLM.
        """
        if channel_id in self.sending_tasks and not self.sending_tasks[channel_id].done():
            self.sending_tasks[channel_id].cancel()

        async def _send_task():
            # The stream is an async iterator - pass it directly
            try:
                sent_messages = await send_streaming_response(
                    channel=channel,
                    stream=stream,
                    session_key=f"channel:{channel_id}",
                    session_manager=self.session_manager,
                )
                logger.info(
                    "Streaming response sent: %d messages to channel %s",
                    len(sent_messages),
                    channel_id,
                )
            except Exception as e:
                logger.error("Error sending streaming response: %s", e, exc_info=True)

        task = asyncio.create_task(_send_task())
        self.sending_tasks[channel_id] = task

        def _cleanup(t) -> None:
            if self.sending_tasks.get(channel_id) == t:
                self.sending_tasks.pop(channel_id, None)

        task.add_done_callback(_cleanup)

    async def _process_batch(self, messages: list[discord.Message]):
        """Standard batch processing logic."""
        if not messages:
            return

        primary_message = messages[0]
        channel_id = primary_message.channel.id

        # Register task
        self.active_batches[channel_id] = messages
        current_task = asyncio.current_task()
        self.processing_tasks[channel_id] = current_task

        # Create cancellation event for this channel
        cancel_event = asyncio.Event()
        self.cancellation_signals[channel_id] = cancel_event

        # Track this API call for proper cancellation
        api_call_task = None

        try:
            full_text = await self._prepare_batch_context(messages)

            if not full_text:
                return

            logger.info(
                "Processing batch of %d messages from %s: %s",
                len(messages),
                primary_message.author.name,
                full_text[:100],
            )

            # Choose streaming or non-streaming path based on break_cut_mode
            use_streaming = self._should_use_streaming()

            if use_streaming:
                # Streaming path with fallback to non-streaming on failure
                await self._process_with_streaming(
                    messages, primary_message, channel_id, cancel_event
                )
            else:
                # Non-streaming path (original behavior)
                await self._process_with_reply(messages, primary_message, channel_id, cancel_event)

        except asyncio.CancelledError:
            logger.info("Batch processing cancelled for channel %s.", primary_message.channel.name)
            raise

        except Exception as e:
            logger.error("메시지 처리 중 예상치 못한 오류 발생: %s", e, exc_info=True)
            await self._handle_error(primary_message, e)

        finally:
            if self.processing_tasks.get(channel_id) == current_task:
                self.processing_tasks.pop(channel_id, None)
                self.active_batches.pop(channel_id, None)
                self.cancellation_signals.pop(channel_id, None)
            # Ensure API call is cleaned up if still present
            self.active_api_calls.pop(channel_id, None)

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
        await send_discord_message(message.channel, GENERIC_ERROR_MESSAGE)

    async def _handle_break_cut_sending(self, channel_id: int, channel, reply: ChatReply):
        """Helper for sending split responses.

        DEPRECATED: This method is kept for backward compatibility but streaming
        is now handled by _process_with_streaming.
        """
        # Use the new streaming-based approach
        if reply.text:
            # Create a simple async iterator from the reply text
            async def _stream_from_text():
                if reply.text:
                    yield reply.text

            # Use send_streaming_response for consistent behavior
            asyncio.create_task(
                send_streaming_response(
                    channel=channel,
                    stream=_stream_from_text(),
                    session_key=reply.session_key,
                    session_manager=self.session_manager,
                )
            )
        else:
            # Fallback to split response if streaming fails
            if channel_id in self.sending_tasks and not self.sending_tasks[channel_id].done():
                self.sending_tasks[channel_id].cancel()

            task = asyncio.create_task(send_split_response(channel, reply, self.session_manager))
            self.sending_tasks[channel_id] = task

            def _cleanup(t) -> None:
                if self.sending_tasks.get(channel_id) == t:
                    self.sending_tasks.pop(channel_id, None)

            task.add_done_callback(_cleanup)

    def _cancel_active_tasks(
        self, channel_id: int, author_name: str, message_type: str = "new message"
    ):
        """Cancel sending and processing tasks for a channel.

        This method ensures that when a new message arrives during batch processing:
        1. Any ongoing LLM API-side generation is cancelled immediately (like STOP button)
        2. The messages from cancelled batch are returned for stacking
        3. A new request will be made with all stacked messages
        """
        messages_to_prepend = []

        # Step 1: Cancel ongoing API call FIRST (critical for server-side cancellation)
        # This is the most important step - it cancels the actual HTTP request
        if channel_id in self.active_api_calls:
            active_api = self.active_api_calls[channel_id]
            logger.info(
                f"{message_type} from {author_name} cancelling ongoing API call for channel {channel_id}"
            )
            active_api.cancel()
            # Get messages from the cancelled batch to prepend to new request
            messages_to_prepend = self.active_batches.get(channel_id, [])

        # Step 2: Also trigger cancellation signal (redundant but ensures coverage)
        if channel_id in self.cancellation_signals:
            logger.info(
                f"{message_type} from {author_name} triggered abort signal for channel {channel_id}"
            )
            self.cancellation_signals[channel_id].set()

        # Step 3: Cancel sending task if in break-cut mode
        if self.config.break_cut_mode and channel_id in self.sending_tasks:
            task = self.sending_tasks[channel_id]
            if not task.done():
                logger.info(
                    f"{message_type} from {author_name} interrupted sending in channel {channel_id}"
                )
                task.cancel()

        # Step 4: Cancel the processing task itself
        if channel_id in self.processing_tasks:
            task = self.processing_tasks[channel_id]
            if not task.done():
                logger.info(
                    f"{message_type} from {author_name} interrupted processing in channel {channel_id}. Merging messages."
                )
                # If we didn't get messages from active_api_calls, get from active_batches
                if not messages_to_prepend:
                    messages_to_prepend = self.active_batches.get(channel_id, [])
                task.cancel()

        return messages_to_prepend

    @commands.Cog.listener()
    async def on_typing(
        self, channel: discord.abc.Messageable, user: discord.abc.User, when: float
    ):
        if user.bot:
            return

        if hasattr(channel, "id"):
            # Check filtering in subclass if needed, but BaseCog assumes if we are listening, we care.
            # But we should probably check if it's an interesting channel in the subclass listener
            # or rely on the buffer check (buffer handles non-existent keys gracefully).

            # Subclasses should likely override or wrap this if they have specific ignore logic
            # For now, we put shared logic here:
            if not self.config.break_cut_mode:
                self.message_buffer.handle_typing(channel.id, self._process_batch)

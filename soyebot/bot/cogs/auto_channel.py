"""Auto-reply Cog for channels configured via environment variables."""

import logging
import time
from typing import Optional

import asyncio

import discord
from discord.ext import commands

from soyebot.bot.chat_handler import ChatReply, create_chat_reply, resolve_session_for_message, send_split_response
from soyebot.bot.session import SessionManager
from soyebot.bot.buffer import MessageBuffer
from soyebot.config import AppConfig
from soyebot.metrics import get_metrics
from soyebot.services.llm_service import LLMService
from soyebot.utils import GENERIC_ERROR_MESSAGE, extract_message_content

logger = logging.getLogger(__name__)


class AutoChannelCog(commands.Cog):
    """Automatically responds to messages in configured channels."""

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
        self.sending_tasks: dict[int, asyncio.Task] = {}
        self.message_buffer = MessageBuffer(delay=config.message_buffer_delay)

        # Track active processing tasks (LLM generation) to allow debouncing/merging
        self.processing_tasks: dict[int, asyncio.Task] = {}
        self.active_batches: dict[int, list[discord.Message]] = {}

    async def _send_auto_reply(self, message: discord.Message, reply: ChatReply) -> None:
        if not reply.text:
            logger.debug("LLM returned no text response for the auto-reply message.")
            return

        # If Break-Cut Mode is OFF, send normally
        if not self.config.break_cut_mode:
            sent_message = await message.channel.send(reply.text)
            if sent_message:
                self.session_manager.link_message_to_session(str(sent_message.id), reply.session_key)
            return

        # If Break-Cut Mode is ON, use shared helper
        channel_id = message.channel.id
        if channel_id in self.sending_tasks and not self.sending_tasks[channel_id].done():
            self.sending_tasks[channel_id].cancel()

        task = asyncio.create_task(
            send_split_response(message.channel, reply, self.session_manager)
        )
        self.sending_tasks[channel_id] = task

        def _cleanup(t):
            if self.sending_tasks.get(channel_id) == t:
                self.sending_tasks.pop(channel_id, None)
        
        task.add_done_callback(_cleanup)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        # After attempting to run commands, check if we should auto-reply.
        # We ignore anything that looks like a command, and non-auto-reply channels.
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return

        if message.channel.id not in self.config.auto_reply_channel_ids:
            return
        if message.content.startswith(self.config.command_prefix):
            return
        if message.content and message.content.lstrip().startswith("\\"):
            return

        # Interrupt current sending if any (Break-Cut Mode)
        if self.config.break_cut_mode and message.channel.id in self.sending_tasks:
            task = self.sending_tasks[message.channel.id]
            if not task.done():
                logger.info(f"New message from {message.author} interrupted auto-reply in channel {message.channel.id}")
                task.cancel()

        # Interrupt current processing (Processing/Debounce Mode)
        messages_to_prepend = []
        if message.channel.id in self.processing_tasks:
            task = self.processing_tasks[message.channel.id]
            if not task.done():
                logger.info(f"New message from {message.author} interrupted processing in channel {message.channel.id}. Merging messages.")
                
                # Retrieve messages being processed
                messages_to_prepend = self.active_batches.get(message.channel.id, [])
                
                # Cancel the processing task
                task.cancel()
                
                # We expect the task to handle CancelledError and clean up, 
                # but we've already grabbed the messages we need to retry.

        await self.message_buffer.add_message(message.channel.id, message, self._process_batch)
        
        # If we merged messages from a cancelled task, inject them at the front of the buffer
        # add_message guarantees buffers[channel_id] exists and has at least [message]
        if messages_to_prepend:
             self.message_buffer.buffers[message.channel.id][0:0] = messages_to_prepend

    @commands.Cog.listener()
    async def on_typing(self, channel: discord.abc.Messageable, user: discord.abc.User, when: float):
        """Interrupt auto-reply if user starts typing."""
        if user.bot:
            return
        if not hasattr(channel, 'id'):
            return
        
        # Only care if this is an auto-reply channel
        if channel.id not in self.config.auto_reply_channel_ids:
            return

        # If Break-Cut Mode is OFF, use handle_typing to extend buffer
        if not self.config.break_cut_mode:
            self.message_buffer.handle_typing(channel.id, self._process_batch)

    @commands.command(name="@", aliases=["undo"])
    async def undo_command(self, ctx: commands.Context, num_to_undo_str: Optional[str] = "1"):
        """Deletes the last N user/assistant message pairs from the chat history."""
        # This command should only work in auto-reply channels
        if ctx.channel.id not in self.config.auto_reply_channel_ids:
            return

        # Argument validation
        try:
            num_to_undo = int(num_to_undo_str)
            if num_to_undo < 1:
                await ctx.message.add_reaction("❌")
                return
        except ValueError:
            await ctx.message.add_reaction("❌")
            return

        # Permission check
        session_key = f"channel:{ctx.channel.id}"
        session = self.session_manager.sessions.get(session_key)
        user_message_count = 0
        if session and hasattr(session.chat, 'history'):
            user_role = self.llm_service.get_user_role_name()
            for msg in session.chat.history:
                if msg.role == user_role and msg.author_id == ctx.author.id:
                    user_message_count += 1

        is_admin = isinstance(ctx.author, discord.Member) and ctx.author.guild_permissions.manage_guild
        has_permission = is_admin or user_message_count >= 5

        if not has_permission:
            await ctx.message.add_reaction("❌")
            logger.warning(
                "User %s (admin=%s, messages=%d) tried to use undo command without permission in #%s.",
                ctx.author.name, is_admin, user_message_count, ctx.channel.name
            )
            return

        # Execute undo, respecting the max limit
        num_to_actually_undo = min(num_to_undo, 10)
        removed_messages = self.session_manager.undo_last_exchanges(session_key, num_to_actually_undo)

        if removed_messages:
            try:
                await ctx.message.delete()
            except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                await ctx.message.add_reaction("✅")
                logger.warning(
                    "Could not delete undo command message from %s in #%s; left reaction instead.",
                    ctx.author.name,
                    ctx.channel.name,
                )

            assistant_role = self.llm_service.get_assistant_role_name()
            user_role = self.llm_service.get_user_role_name()

            for msg in removed_messages:
                if hasattr(msg, 'message_ids') and msg.message_ids:
                    for mid in msg.message_ids:
                        if msg.role == assistant_role:
                            try:
                                message_to_delete = await ctx.channel.fetch_message(int(mid))
                                await message_to_delete.delete()
                            except discord.NotFound:
                                logger.warning("Could not find message %s to delete in #%s.", mid, ctx.channel.name)
                            except discord.Forbidden:
                                logger.warning("Could not delete message %s in #%s, probably missing permissions.", mid, ctx.channel.name)
                            except Exception as e:
                                logger.warning("Error deleting message %s: %s", mid, e)

                        elif msg.role == user_role:
                            try:
                                message_to_delete = await ctx.channel.fetch_message(int(mid))
                                await message_to_delete.delete()
                            except discord.NotFound:
                                logger.warning("Could not find user message %s to delete in #%s.", mid, ctx.channel.name)
                            except discord.Forbidden:
                                logger.warning("Could not delete user message %s in #%s, probably missing permissions.", mid, ctx.channel.name)
                            except Exception as e:
                                logger.warning("Error deleting user message %s: %s", mid, e)
        else:
            await ctx.message.add_reaction("❌")

    async def _process_batch(self, messages: list[discord.Message]):
        if not messages:
            return

        primary_message = messages[0]
        channel_id = primary_message.channel.id
        
        # Register this task as the active processing task for the channel
        self.active_batches[channel_id] = messages
        self.processing_tasks[channel_id] = asyncio.current_task()

        start_time = time.perf_counter()
        metrics = get_metrics()

        try:
             # Combine contents
            combined_content = []
            for msg in messages:
                content = extract_message_content(msg)
                if content:
                    if len(messages) > 1 and msg.author.name:
                         combined_content.append(f"{msg.author.name}: {content}")
                    else:
                         combined_content.append(content)

            full_text = "\n".join(combined_content)

            if not full_text:
                return

            logger.info(
                "Processing batch of %d auto-reply messages in #%s: %s",
                len(messages),
                primary_message.channel.name,
                full_text[:100],
            )

            async with primary_message.channel.typing():
                resolution = await resolve_session_for_message(
                    primary_message,
                    full_text,
                    session_manager=self.session_manager,
                )

                if not resolution:
                    return

                reply = await create_chat_reply(
                    primary_message,
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                )

                if reply:
                    await self._send_auto_reply(primary_message, reply)

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
            metrics.increment_counter('messages_processed')

        except asyncio.CancelledError:
            logger.info("Batch processing cancelled for channel #%s (likely due to new message).", primary_message.channel.name)
            raise

        except Exception as exc:
            logger.error("자동 응답 메시지 처리 중 오류 발생: %s", exc, exc_info=True)
            await primary_message.channel.send(GENERIC_ERROR_MESSAGE)
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
        
        finally:
            # Cleanup only if we are the current task (handling race conditions slightly safer)
            if self.processing_tasks.get(channel_id) == asyncio.current_task():
                self.processing_tasks.pop(channel_id, None)
                self.active_batches.pop(channel_id, None)

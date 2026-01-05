"""Auto-reply Cog for channels configured via environment variables."""

import logging
import time
import json
from pathlib import Path
from typing import Optional

import asyncio

import discord
from discord.ext import commands

from soyebot.bot.chat_handler import ChatReply, create_chat_reply, resolve_session_for_message, send_split_response
from soyebot.bot.session import SessionManager
from soyebot.bot.buffer import MessageBuffer
from soyebot.config import AppConfig
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

        # Load dynamic channels
        self.json_file_path = Path("data/auto_channels.json")
        # Initialize dynamic set and preserve env-based config
        self.dynamic_channel_ids: set[int] = set()
        self.env_channel_ids: set[int] = set(self.config.auto_reply_channel_ids)

        self._load_dynamic_channels()

    def _load_dynamic_channels(self):
        """Loads auto-channels from JSON and updates config."""
        self.dynamic_channel_ids = set()
        if self.json_file_path.exists():
            try:
                with open(self.json_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.dynamic_channel_ids = set(data)
            except Exception as e:
                logger.error(f"Failed to load auto channels from {self.json_file_path}: {e}")

        # Merge environment config with dynamic config
        combined = self.env_channel_ids | self.dynamic_channel_ids
        self.config.auto_reply_channel_ids = tuple(combined)
        logger.info(f"Loaded auto-channels. Env: {len(self.env_channel_ids)}, Dynamic: {len(self.dynamic_channel_ids)}, Total: {len(self.config.auto_reply_channel_ids)}")

    def _save_dynamic_channels(self):
        """Saves dynamic auto-channels to JSON and updates config."""
        try:
            # Create data dir if not exists (though it should exist)
            self.json_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.json_file_path, "w", encoding="utf-8") as f:
                json.dump(list(self.dynamic_channel_ids), f)

            # Update config immediately
            combined = self.env_channel_ids | self.dynamic_channel_ids
            self.config.auto_reply_channel_ids = tuple(combined)
        except Exception as e:
            logger.error(f"Failed to save auto channels to {self.json_file_path}: {e}")

    @commands.group(name="자동채널", aliases=["auto"], invoke_without_command=True)
    @commands.has_permissions(manage_guild=True)
    async def auto_channel_group(self, ctx: commands.Context):
        """자동 응답 채널 설정 관리 명령어"""
        await ctx.send_help(ctx.command)

    @auto_channel_group.command(name="등록", aliases=["register", "add"])
    @commands.has_permissions(manage_guild=True)
    async def register_channel(self, ctx: commands.Context):
        """현재 채널을 자동 응답 채널로 등록합니다."""
        channel_id = ctx.channel.id

        if channel_id in self.dynamic_channel_ids:
             await ctx.message.add_reaction("✅")
             return

        self.dynamic_channel_ids.add(channel_id)
        self._save_dynamic_channels()
        await ctx.message.add_reaction("✅")

    @auto_channel_group.command(name="해제", aliases=["unregister", "remove"])
    @commands.has_permissions(manage_guild=True)
    async def unregister_channel(self, ctx: commands.Context):
        """현재 채널을 자동 응답 채널에서 해제합니다."""
        channel_id = ctx.channel.id

        if channel_id in self.env_channel_ids:
             await ctx.reply("⚠️ 이 채널은 시스템 설정(환경 변수)으로 등록되어 있어 명령어로 해제할 수 없습니다.", mention_author=False)
             return

        if channel_id not in self.dynamic_channel_ids:
             await ctx.reply("⚠️ 이 채널은 자동 응답 채널이 아닙니다.", mention_author=False)
             return

        self.dynamic_channel_ids.remove(channel_id)
        self._save_dynamic_channels()
        await ctx.message.add_reaction("✅")

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

        channel_id = ctx.channel.id
        undo_performed_on_pending = False

        # 1. Check and Undo active Processing Task (Thinking)
        # If the bot is currently thinking, we cancel it and delete the triggering user message.
        if channel_id in self.processing_tasks:
            task = self.processing_tasks[channel_id]
            if not task.done():
                logger.info("Undo command interrupted active processing in channel #%s", ctx.channel.name)
                task.cancel()
                
                # Delete the pending messages that were being processed
                pending_messages = self.active_batches.get(channel_id, [])
                for msg in pending_messages:
                    try:
                        await msg.delete()
                    except (discord.NotFound, discord.Forbidden):
                        pass
                
                # We consider this as 1 "undo" action (the current pending turn)
                undo_performed_on_pending = True
                num_to_undo -= 1
        
        # 2. Check and Undo active Sending Task (Break-Cut Mode)
        # If the bot is currently sending a split response, we cancel it.
        # We generally don't decrement num_to_undo here because the partial response 
        # is likely already in history (or parts of it), so we rely on the standard
        # undo logic below to clean up the partial exchange.
        if channel_id in self.sending_tasks:
            task = self.sending_tasks[channel_id]
            if not task.done():
                logger.info("Undo command interrupted active sending in channel #%s", ctx.channel.name)
                task.cancel()
                # We do NOT return or decrement here, as we want to proceed to delete 
                # whatever partial messages made it to the history.

        # If we only wanted to undo 1 item and we already undid the pending one, we are done
        # (unless the user wanted to undo more).
        if num_to_undo <= 0:
            try:
                await ctx.message.delete()
            except (discord.Forbidden, discord.HTTPException):
                pass
            return

        # Permission check for historical undo
        session_key = f"channel:{channel_id}"
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
            # If we successfully cancelled a pending task, we still consider the command "successful" enough to not show X
            # but we won't undo history.
            if not undo_performed_on_pending:
                await ctx.message.add_reaction("❌")
                logger.warning(
                    "User %s (admin=%s, messages=%d) tried to use undo command without permission in #%s.",
                    ctx.author.name, is_admin, user_message_count, ctx.channel.name
                )
            else:
                 try:
                    await ctx.message.delete()
                 except: 
                    pass
            return

        # Execute undo, respecting the max limit
        num_to_actually_undo = min(num_to_undo, 10)
        removed_messages = self.session_manager.undo_last_exchanges(session_key, num_to_actually_undo)

        if removed_messages or undo_performed_on_pending:
            try:
                await ctx.message.delete()
            except (discord.Forbidden, discord.HTTPException):
                pass

            assistant_role = self.llm_service.get_assistant_role_name()
            user_role = self.llm_service.get_user_role_name()

            for msg in removed_messages:
                if hasattr(msg, 'message_ids') and msg.message_ids:
                    for mid in msg.message_ids:
                        if msg.role == assistant_role:
                            try:
                                message_to_delete = await ctx.channel.fetch_message(int(mid))
                                await message_to_delete.delete()
                            except asyncio.CancelledError:
                                pass
                            except (discord.NotFound, discord.Forbidden):
                                pass
                            except Exception as e:
                                logger.warning(f"Error deleting message {mid} during undo: {e}")

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

        try:
             # Combine contents
            combined_content = []
            for msg in messages:
                content = extract_message_content(msg)
                if content:
                    if len(messages) > 1 and msg.author.id:
                         combined_content.append(f"{msg.author.id}: {content}")
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
                    messages, # Pass list of messages
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                )

                if reply:
                    await self._send_auto_reply(primary_message, reply)

        except asyncio.CancelledError:
            logger.info("Batch processing cancelled for channel #%s (likely due to new message).", primary_message.channel.name)
            raise

        except Exception as exc:
            logger.error("자동 응답 메시지 처리 중 오류 발생: %s", exc, exc_info=True)
            await primary_message.channel.send(GENERIC_ERROR_MESSAGE)
        
        finally:
            # Cleanup only if we are the current task (handling race conditions slightly safer)
            if self.processing_tasks.get(channel_id) == asyncio.current_task():
                self.processing_tasks.pop(channel_id, None)
                self.active_batches.pop(channel_id, None)

    async def cog_command_error(self, ctx: commands.Context, error: Exception):
        """Cog 내 명령어 에러 핸들러"""
        if isinstance(error, commands.MissingPermissions):
            await ctx.reply(f"❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: {', '.join(error.missing_permissions)})", mention_author=False)
        elif isinstance(error, commands.BadArgument):
            await ctx.reply("❌ 잘못된 인자가 전달되었습니다. 명령어를 다시 확인해 주세요.", mention_author=False)
        else:
             logger.error(f"Command error in {ctx.command}: {error}", exc_info=True)


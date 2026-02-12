"""Auto-reply Cog for channels configured via environment variables."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import aiofiles
import discord
from discord.ext import commands

from persbot.bot.chat_handler import ChatReply
from persbot.bot.cogs.base import BaseChatCog
from persbot.bot.session import SessionManager
from persbot.config import AppConfig
from persbot.services.llm_service import LLMService
from persbot.utils import (
    GENERIC_ERROR_MESSAGE,
)

logger = logging.getLogger(__name__)


class AutoChannelCog(BaseChatCog):
    """Automatically responds to messages in configured channels."""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        session_manager: SessionManager,
        tool_manager=None,
    ):
        super().__init__(bot, config, llm_service, session_manager, tool_manager)

        # Load dynamic channels
        self.json_file_path = Path("data/auto_channels.json")
        # Initialize dynamic set and preserve env-based config
        self.dynamic_channel_ids: set[int] = set()
        self.env_channel_ids: set[int] = set(self.config.auto_reply_channel_ids)

        # Load dynamic channels (async)
        asyncio.create_task(self._load_dynamic_channels())

    async def _load_dynamic_channels(self) -> None:
        """Loads auto-channels from JSON and updates config."""
        # Using async load for consistency with write operations
        self.dynamic_channel_ids = set()
        if self.json_file_path.exists():
            try:
                async with aiofiles.open(self.json_file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    if isinstance(data, list):
                        self.dynamic_channel_ids = set(data)
            except Exception as e:
                logger.error(f"Failed to load auto channels from {self.json_file_path}: {e}")

        # Merge environment config with dynamic config
        combined = self.env_channel_ids | self.dynamic_channel_ids
        self.config.auto_reply_channel_ids = tuple(combined)
        logger.info(
            f"Loaded auto-channels. Env: {len(self.env_channel_ids)}, Dynamic: {len(self.dynamic_channel_ids)}, Total: {len(self.config.auto_reply_channel_ids)}"
        )

    async def _save_dynamic_channels(self) -> None:
        """Saves dynamic auto-channels to JSON and updates config."""
        try:
            # Create data dir if not exists (though it should exist)
            self.json_file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(self.json_file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(list(self.dynamic_channel_ids)))

            # Update config immediately
            combined = self.env_channel_ids | self.dynamic_channel_ids
            self.config.auto_reply_channel_ids = tuple(combined)
        except Exception as e:
            logger.error(f"Failed to save auto channels to {self.json_file_path}: {e}")

    @commands.group(name="자동채널", aliases=["auto"], invoke_without_command=True)
    async def auto_channel_group(self, ctx: commands.Context) -> None:
        """자동 응답 채널 설정 관리 명령어"""
        # Check permissions unless NO_CHECK_PERMISSION is set
        if not self.config.no_check_permission:
            if (
                not isinstance(ctx.author, discord.Member)
                or not ctx.author.guild_permissions.manage_guild
            ):
                await ctx.reply(
                    "❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: manage_guild)",
                    mention_author=False,
                )
                return
        await ctx.send_help(ctx.command)

    @auto_channel_group.command(name="등록", aliases=["register", "add"])
    async def register_channel(self, ctx: commands.Context) -> None:
        """현재 채널을 자동 응답 채널로 등록합니다."""
        channel_id = ctx.channel.id

        if channel_id in self.dynamic_channel_ids:
            await ctx.message.add_reaction("✅")
            return

        self.dynamic_channel_ids.add(channel_id)
        await self._save_dynamic_channels()
        await ctx.message.add_reaction("✅")

    @auto_channel_group.command(name="해제", aliases=["unregister", "remove"])
    async def unregister_channel(self, ctx: commands.Context) -> None:
        """현재 채널을 자동 응답 채널에서 해제합니다."""
        channel_id = ctx.channel.id

        if channel_id in self.env_channel_ids:
            await ctx.reply(
                "⚠️ 이 채널은 시스템 설정(환경 변수)으로 등록되어 있어 명령어로 해제할 수 없습니다.",
                mention_author=False,
            )
            return

        if channel_id not in self.dynamic_channel_ids:
            await ctx.reply("⚠️ 이 채널은 자동 응답 채널이 아닙니다.", mention_author=False)
            return

        self.dynamic_channel_ids.remove(channel_id)
        await self._save_dynamic_channels()
        await ctx.message.add_reaction("✅")

    async def _send_response(self, message: discord.Message, reply: ChatReply) -> None:
        if not reply.text:
            logger.debug("LLM returned no text response for the auto-reply message.")
            return

        # Use base class method (handles streaming/non-streaming based on break_cut_mode)
        await self._handle_break_cut_sending(message.channel.id, message.channel, reply)

    async def _handle_error(self, message: discord.Message, error: Exception) -> None:
        await message.channel.send(GENERIC_ERROR_MESSAGE)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
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

        messages_to_prepend = self._cancel_active_tasks(message.channel.id, message.author.name)

        await self.message_buffer.add_message(message.channel.id, message, self._process_batch)

        if messages_to_prepend:
            # Ensure the list exists before prepending
            if message.channel.id in self.message_buffer.buffers:
                self.message_buffer.buffers[message.channel.id][0:0] = messages_to_prepend

    @commands.Cog.listener()
    async def on_typing(
        self, channel: discord.abc.Messageable, user: discord.abc.User, when: float
    ) -> None:
        """Interrupt auto-reply if user starts typing."""
        if user.bot:
            return
        if not hasattr(channel, "id"):
            return

        # Only care if this is an auto-reply channel
        if channel.id not in self.config.auto_reply_channel_ids:
            return

        # Use shared logic
        if not self.config.break_cut_mode:
            self.message_buffer.handle_typing(channel.id, self._process_batch)

    @commands.command(name="@", aliases=["undo"])
    async def undo_command(
        self, ctx: commands.Context, num_to_undo_str: Optional[str] = "1"
    ) -> None:
        """Deletes the last N user/assistant message pairs from the chat history."""
        if ctx.channel.id not in self.config.auto_reply_channel_ids:
            return

        # Validate argument
        num_to_undo = self._validate_undo_arg(num_to_undo_str)
        if num_to_undo is None:
            await ctx.message.add_reaction("❌")
            return

        channel_id = ctx.channel.id
        session_key = f"channel:{channel_id}"

        # Cancel pending tasks
        undo_performed_on_pending, num_to_undo = await self._cancel_pending_tasks(
            ctx, channel_id, num_to_undo
        )

        # Check if we're done
        if num_to_undo <= 0:
            await self._try_delete_message(ctx.message)
            return

        # Check permission for historical undo
        if not self._check_undo_permission(ctx, session_key):
            if not undo_performed_on_pending:
                await ctx.message.add_reaction("❌")
            else:
                await self._try_delete_message(ctx.message)
            return

        # Execute undo
        num_to_actually_undo = min(num_to_undo, 10)
        removed_messages = self.session_manager.undo_last_exchanges(
            session_key, num_to_actually_undo
        )

        if removed_messages or undo_performed_on_pending:
            await self._try_delete_message(ctx.message)
            await self._delete_removed_messages(ctx.channel, removed_messages)
        else:
            await ctx.message.add_reaction("❌")

    def _validate_undo_arg(self, num_to_undo_str: Optional[str]) -> Optional[int]:
        """Validate undo count argument. Returns None if invalid."""
        try:
            num = int(num_to_undo_str)
            return num if num >= 1 else None
        except ValueError:
            return None

    async def _cancel_pending_tasks(self, ctx, channel_id: int, num_to_undo: int) -> tuple:
        """Cancel pending tasks and return (was_cancelled, remaining_undo_count)."""
        undo_performed = False

        # Cancel processing task
        if channel_id in self.processing_tasks:
            task = self.processing_tasks[channel_id]
            if not task.done():
                logger.info("Undo interrupted active processing in #%s", ctx.channel.name)
                task.cancel()

                # Delete pending messages
                for msg in self.active_batches.get(channel_id, []):
                    try:
                        await msg.delete()
                    except (discord.NotFound, discord.Forbidden):
                        pass

                undo_performed = True
                num_to_undo -= 1

        # Cancel sending task
        if channel_id in self.sending_tasks:
            task = self.sending_tasks[channel_id]
            if not task.done():
                logger.info("Undo interrupted active sending in #%s", ctx.channel.name)
                task.cancel()

        return undo_performed, num_to_undo

    def _check_undo_permission(self, ctx, session_key: str) -> bool:
        """Check if user has permission for historical undo."""
        session = self.session_manager.sessions.get(session_key)
        user_message_count = 0

        if session and hasattr(session.chat, "history"):
            user_role = self.llm_service.get_user_role_name()
            user_message_count = sum(
                1
                for msg in session.chat.history
                if msg.role == user_role and msg.author_id == ctx.author.id
            )

        is_admin = (
            isinstance(ctx.author, discord.Member) and ctx.author.guild_permissions.manage_guild
        )
        # Bypass permission check if NO_CHECK_PERMISSION is set
        if self.config.no_check_permission:
            is_admin = True
        return is_admin or user_message_count >= 5

    async def _delete_removed_messages(self, channel, removed_messages: list) -> None:
        """Delete Discord messages from removed history entries."""
        assistant_role = self.llm_service.get_assistant_role_name()

        # Collect all message IDs to delete in batches
        message_ids_to_delete: list[tuple[str, str]] = []  # (message_id, role)

        for msg in removed_messages:
            if not hasattr(msg, "message_ids") or not msg.message_ids:
                continue
            for mid in msg.message_ids:
                message_ids_to_delete.append((mid, msg.role))

        # Batch delete operations for better performance
        if message_ids_to_delete:
            tasks = [
                self._try_delete_channel_message(channel, mid, role)
                for mid, role in message_ids_to_delete
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _try_delete_channel_message(self, channel, message_id: str, role: str) -> None:
        """Try to delete a message by ID, logging any errors."""
        try:
            msg = await channel.fetch_message(int(message_id))
            await msg.delete()
        except asyncio.CancelledError:
            pass
        except discord.NotFound:
            logger.debug("Message %s not found for deletion", message_id)
        except discord.Forbidden:
            logger.warning("No permission to delete message %s", message_id)
        except Exception as e:
            logger.warning("Error deleting message %s: %s", message_id, e)

    async def _try_delete_message(self, message) -> None:
        """Try to delete a message, ignoring errors."""
        try:
            await message.delete()
        except (discord.Forbidden, discord.HTTPException, discord.NotFound):
            pass

    async def cog_command_error(self, ctx: commands.Context, error: Exception) -> None:
        """Cog 내 명령어 에러 핸들러"""
        if isinstance(error, commands.MissingPermissions):
            await ctx.reply(
                f"❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: {', '.join(error.missing_permissions)})",
                mention_author=False,
            )
        elif isinstance(error, commands.BadArgument):
            await ctx.reply(
                "❌ 잘못된 인자가 전달되었습니다. 명령어를 다시 확인해 주세요.",
                mention_author=False,
            )
        else:
            logger.error(f"Command error in {ctx.command}: {error}", exc_info=True)

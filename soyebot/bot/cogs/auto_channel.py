"""Auto-reply Cog for channels configured via environment variables."""

import asyncio
import logging
import time
from typing import Optional

import discord
from discord.ext import commands

from soyebot.bot.chat_handler import create_chat_reply, resolve_session_for_message
from soyebot.bot.session import SessionManager
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
        # MessageBuffer removed as per new "immediate" response requirements
        # self.message_buffer = MessageBuffer(delay=config.message_buffer_delay)

    @commands.command(name="끊어치기", aliases=["split"])
    async def toggle_segmented_messaging(self, ctx: commands.Context, mode: Optional[str] = None):
        """Toggles the segmented messaging mode (끊어치기 모드)."""
        if ctx.channel.id not in self.config.auto_reply_channel_ids:
            return

        if not (ctx.author.guild_permissions.manage_guild):
            await ctx.message.add_reaction("❌")
            return

        current = self.config.segmented_messaging

        if mode:
            mode = mode.lower()
            if mode == "on":
                new_state = True
            elif mode == "off":
                new_state = False
            else:
                await ctx.send("사용법: !끊어치기 [on|off]")
                return
        else:
            new_state = not current

        self.config.segmented_messaging = new_state
        status = "ON" if new_state else "OFF"
        await ctx.send(f"끊어치기 모드: **{status}**")
        await ctx.message.add_reaction("✅")

    async def _send_auto_reply(self, message: discord.Message, reply_text: str, session_key: str) -> None:
        if not reply_text:
            logger.debug("LLM returned no text response for the auto-reply message.")
            return

        if not self.config.segmented_messaging:
            # Traditional mode: Send all at once
            sent_message = await message.channel.send(reply_text)
            if sent_message:
                self.session_manager.link_message_to_session(str(sent_message.id), session_key)
        else:
            # Segmented mode: Split by newlines and send sequentially with delay
            lines = [line.strip() for line in reply_text.split('\n') if line.strip()]
            if not lines:
                return

            sent_message_ids = []

            for line in lines:
                # Calculate delay: 0.5s to 1.7s proportional to length
                # Heuristic: 0.05s per character
                delay = min(1.7, max(0.5, len(line) * 0.05))

                async with message.channel.typing():
                    await asyncio.sleep(delay)
                    sent_msg = await message.channel.send(line)
                    if sent_msg:
                        sent_message_ids.append(str(sent_msg.id))

            # Link all sent message IDs to the session (comma-separated)
            if sent_message_ids:
                combined_ids = ",".join(sent_message_ids)
                self.session_manager.link_message_to_session(combined_ids, session_key)

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

        # Direct processing without buffer delay
        await self._process_batch([message])

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
                if msg.message_id:
                    # Handle comma-separated IDs for segmented messages
                    msg_ids = msg.message_id.split(',')

                    for msg_id in msg_ids:
                        msg_id = msg_id.strip()
                        if not msg_id:
                            continue

                        if msg.role == assistant_role:
                            try:
                                message_to_delete = await ctx.channel.fetch_message(msg_id)
                                await message_to_delete.delete()
                            except discord.NotFound:
                                logger.warning("Could not find message %s to delete in #%s.", msg_id, ctx.channel.name)
                            except discord.Forbidden:
                                logger.warning("Could not delete message %s in #%s, probably missing permissions.", msg_id, ctx.channel.name)

                        elif msg.role == user_role:
                            try:
                                message_to_delete = await ctx.channel.fetch_message(msg_id)
                                await message_to_delete.delete()
                            except discord.NotFound:
                                logger.warning("Could not find user message %s to delete in #%s.", msg_id, ctx.channel.name)
                            except discord.Forbidden:
                                logger.warning("Could not delete user message %s in #%s, probably missing permissions.", msg_id, ctx.channel.name)
        else:
            await ctx.message.add_reaction("❌")

    async def _process_batch(self, messages: list[discord.Message]):
        if not messages:
            return

        primary_message = messages[0]
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
                await primary_message.channel.send("❌ 메시지 내용이 없는데요.")
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
                    await primary_message.channel.send("❌ 메시지 내용이 없는데요.")
                    return

                reply = await create_chat_reply(
                    primary_message,
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                )

                if reply:
                    await self._send_auto_reply(primary_message, reply.text, reply.session_key)

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
            metrics.increment_counter('messages_processed')

        except Exception as exc:
            logger.error("자동 응답 메시지 처리 중 오류 발생: %s", exc, exc_info=True)
            await primary_message.channel.send(GENERIC_ERROR_MESSAGE)
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)

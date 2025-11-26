"""Auto-reply Cog for channels configured via environment variables."""

import logging
import time

import discord
from discord.ext import commands

from soyebot.bot.chat_handler import create_chat_reply, resolve_session_for_message
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
        self.message_buffer = MessageBuffer(delay=config.message_buffer_delay)

    def _should_ignore_message(self, message: discord.Message) -> bool:
        if message.author.bot:
            return True
        if message.channel.id not in self.config.auto_reply_channel_ids:
            return True
        if message.content.startswith(self.config.command_prefix):
            return True
        if message.content and message.content.lstrip().startswith("\\"):
            return True
        return False

    async def _send_auto_reply(self, message: discord.Message, reply_text: str, session_key: str) -> None:
        if not reply_text:
            logger.debug("LLM returned no text response for the auto-reply message.")
            return

        sent_message = await message.channel.send(reply_text)
        if sent_message:
            self.session_manager.link_message_to_session(str(sent_message.id), session_key)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # First, check for the undo command
        if await self._handle_undo_command(message):
            return  # If it was an undo command, we're done.

        if self._should_ignore_message(message):
            return

        await self.message_buffer.add_message(message.channel.id, message, self._process_batch)

    async def _handle_undo_command(self, message: discord.Message) -> bool:
        """Check for and handle the !@ or !undo command."""
        # Only operate in auto-reply channels
        if message.channel.id not in self.config.auto_reply_channel_ids:
            return False

        content = message.content.strip()
        is_undo_command = content.startswith("!@") or content.startswith("!undo")

        if not is_undo_command:
            return False

        # Permission check: admin (manage_guild) or user with >= 5 messages in session
        session_key = f"channel:{message.channel.id}"
        session = self.session_manager.sessions.get(session_key)
        user_message_count = 0

        if session and hasattr(session.chat, 'history'):
            user_role = self.llm_service.get_user_role_name()
            for msg in session.chat.history:
                if msg.role == user_role and msg.author_id == message.author.id:
                    user_message_count += 1

        is_admin = isinstance(message.author, discord.Member) and message.author.guild_permissions.manage_guild
        has_permission = is_admin or user_message_count >= 5

        if not has_permission:
            await message.add_reaction("❌")
            logger.warning(
                "User %s (admin=%s, messages=%d) tried to use undo command without permission in #%s.",
                message.author.name, is_admin, user_message_count, message.channel.name
            )
            return True # Command was handled (by denying it)

        # Parse number of exchanges to undo
        num_to_undo = 1
        parts = content.split()
        if len(parts) > 1:
            try:
                num = int(parts[1])
                if num < 1:
                    await message.add_reaction("❌")
                    return True
                num_to_undo = min(num, 10)  # Max 10 at a time
            except ValueError:
                await message.add_reaction("❌")
                return True

        # Execute undo
        success = self.session_manager.undo_last_exchanges(session_key, num_to_undo)

        if success:
            await message.add_reaction("✅")
            try:
                # Edit the original message to reflect the action
                new_content = f"> -# ~~{message.content}~~"
                await message.edit(content=new_content)
            except discord.Forbidden:
                logger.warning("Could not edit undo message in #%s, probably missing permissions.", message.channel.name)
        else:
            await message.add_reaction("❌")

        return True # Command was handled

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

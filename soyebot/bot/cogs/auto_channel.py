"""Auto-reply Cog for channels configured via environment variables."""

import logging
import time

import discord
from discord.ext import commands

from bot.chat_handler import create_chat_reply, resolve_session_for_message
from bot.session import SessionManager
from config import AppConfig
from metrics import get_metrics
from services.llm_service import LLMService
from utils import GENERIC_ERROR_MESSAGE, extract_message_content

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
        if self._should_ignore_message(message):
            return

        start_time = time.perf_counter()
        metrics = get_metrics()

        try:
            user_message = extract_message_content(message)
            if not user_message:
                await message.channel.send("❌ 메시지 내용이 없는데요.")
                return

            logger.info(
                "Processing auto-reply message from %s in #%s: %s",
                message.author.name,
                message.channel.name,
                user_message[:100],
            )

            async with message.channel.typing():
                resolution = await resolve_session_for_message(
                    message,
                    user_message,
                    session_manager=self.session_manager,
                )

                if not resolution:
                    await message.channel.send("❌ 메시지 내용이 없는데요.")
                    return

                reply = await create_chat_reply(
                    message,
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                )

                if reply:
                    await self._send_auto_reply(message, reply.text, reply.session_key)

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
            metrics.increment_counter('messages_processed')

        except Exception as exc:
            logger.error("자동 응답 메시지 처리 중 오류 발생: %s", exc, exc_info=True)
            await message.channel.send(GENERIC_ERROR_MESSAGE)
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)

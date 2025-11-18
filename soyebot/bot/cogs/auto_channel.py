"""Auto-reply Cog for channels configured via environment variables."""

import logging
import time

import discord
from discord.ext import commands

from config import AppConfig
from services.gemini_service import GeminiService
from bot.session import SessionManager
from utils import extract_message_content
from metrics import get_metrics

logger = logging.getLogger(__name__)


class AutoChannelCog(commands.Cog):
    """Automatically responds to messages in configured channels."""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        gemini_service: GeminiService,
        session_manager: SessionManager,
    ):
        self.bot = bot
        self.config = config
        self.gemini_service = gemini_service
        self.session_manager = session_manager

    def _determine_session_id(self, message: discord.Message) -> str:
        """Resolve a stable session key for reply chains."""
        if message.reference and message.reference.message_id:
            referenced_id = str(message.reference.message_id)
            existing_session = self.session_manager.get_session_for_message(referenced_id)
            if existing_session:
                return existing_session
            return referenced_id
        return str(message.id)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Skip bot messages or messages outside configured channels
        if message.author.bot:
            return
        if message.channel.id not in self.config.auto_reply_channel_ids:
            return

        # Skip command messages to avoid interfering with prefix commands
        if message.content.startswith(self.config.command_prefix):
            return

        start_time = time.perf_counter()
        metrics = get_metrics()

        try:
            user_message = extract_message_content(message)
            if not user_message:
                await message.reply("❌ 메시지 내용이 없는데요.", mention_author=False)
                return

            logger.info(
                "Processing auto-reply message from %s in #%s: %s",
                message.author.name,
                message.channel.name,
                user_message[:100],
            )

            async with message.channel.typing():
                session_id = self._determine_session_id(message)

                chat_session, session_key = await self.session_manager.get_or_create(
                    user_id=message.author.id,
                    username=message.author.name,
                    message_id=str(session_id),
                )
                self.session_manager.link_message_to_session(str(message.id), session_key)

                response_result = await self.gemini_service.generate_chat_response(
                    chat_session,
                    user_message,
                    message,
                )

                if response_result:
                    response_text, _response_obj = response_result
                    if response_text:
                        reply_message = await message.reply(response_text, mention_author=False)
                        if reply_message:
                            self.session_manager.link_message_to_session(
                                str(reply_message.id),
                                session_key,
                            )
                    else:
                        logger.debug("Gemini returned no text response for the auto-reply message.")

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
            metrics.increment_counter('messages_processed')

        except Exception as exc:
            logger.error("자동 응답 메시지 처리 중 오류 발생: %s", exc, exc_info=True)
            await message.reply(
                "❌ 봇 내부에서 예상치 못한 오류가 발생했어요. 개발자에게 문의해주세요.",
                mention_author=False,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)

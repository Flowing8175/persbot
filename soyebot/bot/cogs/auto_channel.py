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
                await message.channel.send("❌ 메시지 내용이 없는데요.")
                return

            logger.info(
                "Processing auto-reply message from %s in #%s: %s",
                message.author.name,
                message.channel.name,
                user_message[:100],
            )

            async with message.channel.typing():
                reference_message_id = (
                    str(message.reference.message_id)
                    if message.reference and message.reference.message_id
                    else None
                )

                resolution = self.session_manager.resolve_session(
                    channel_id=message.channel.id,
                    author_id=message.author.id,
                    username=message.author.name,
                    message_id=str(message.id),
                    message_content=user_message,
                    reference_message_id=reference_message_id,
                    created_at=message.created_at,
                )

                if not resolution.cleaned_message:
                    await message.channel.send("❌ 메시지 내용이 없는데요.")
                    return

                chat_session, session_key = await self.session_manager.get_or_create(
                    user_id=message.author.id,
                    username=message.author.name,
                    session_key=resolution.session_key,
                    channel_id=message.channel.id,
                    message_content=resolution.cleaned_message,
                    message_ts=message.created_at,
                    message_id=str(message.id),
                )
                self.session_manager.link_message_to_session(str(message.id), session_key)

                response_result = await self.gemini_service.generate_chat_response(
                    chat_session,
                    resolution.cleaned_message,
                    message,
                )

                if response_result:
                    response_text, _response_obj = response_result
                    if response_text:
                        sent_message = await message.channel.send(response_text)
                        if sent_message:
                            self.session_manager.link_message_to_session(
                                str(sent_message.id),
                                session_key,
                            )
                    else:
                        logger.debug("Gemini returned no text response for the auto-reply message.")

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
            metrics.increment_counter('messages_processed')

        except Exception as exc:
            logger.error("자동 응답 메시지 처리 중 오류 발생: %s", exc, exc_info=True)
            await message.channel.send(
                "❌ 봇 내부에서 예상치 못한 오류가 발생했어요. 개발자에게 문의해주세요."
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)

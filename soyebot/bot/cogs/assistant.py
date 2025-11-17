"""Assistant Cog for SoyeBot."""

import discord
from discord.ext import commands
import logging
import time

from config import AppConfig
from services.gemini_service import GeminiService
from bot.session import SessionManager
from utils import extract_message_content
from metrics import get_metrics

logger = logging.getLogger(__name__)

class AssistantCog(commands.Cog):
    """@mention을 통한 AI 어시스턴트 기능을 처리하는 Cog"""

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
        # Ignore bot messages, messages without bot mention, or @everyone/@here mentions
        if message.author.bot or not self.bot.user.mentioned_in(message):
            return
        
        # Ignore @everyone and @here mentions
        if message.mention_everyone:
            return


        # Track message processing latency
        start_time = time.perf_counter()
        metrics = get_metrics()

        try:
            user_message = extract_message_content(message)
            if not user_message:
                await message.reply("❌ 메시지 내용이 없는데요.", mention_author=False)
                return

            logger.info(f"Processing message from {message.author.name}: {user_message[:100]}")

            async with message.channel.typing():
                # 세션 ID 결정: 리플라이 대상이 있으면 그 메시지 ID, 없으면 현재 메시지 ID
                session_id = message.reference.message_id if message.reference else message.id

                # Get or create user session (async to prevent blocking)
                chat_session, user_id = await self.session_manager.get_or_create(
                    user_id=message.author.id,
                    username=message.author.name,
                    message_id=str(session_id),
                )

                response_result = await self.gemini_service.generate_chat_response(
                    chat_session,
                    user_message,
                    message,
                )

                if response_result:
                    response_text, response_obj = response_result

                    # Only reply if there's text content
                    # (Some responses may be non-text, so skip replying)
                    if response_text:
                        await message.reply(response_text, mention_author=False)
                    else:
                        logger.debug("Gemini returned no text response for the mention.")
                # else: The error message is now handled by gemini_service._api_request_with_retry

            # Track successful message processing
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
            metrics.increment_counter('messages_processed')

        except Exception as e:
            logger.error(f"메시지 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
            await message.reply("❌ 봇 내부에서 예상치 못한 오류가 발생했어요. 개발자에게 문의해주세요.", mention_author=False)

            # Track processing time even on error
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)

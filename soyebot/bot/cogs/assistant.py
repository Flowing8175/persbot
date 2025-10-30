"""Assistant Cog for SoyeBot."""

import discord
from discord.ext import commands
import logging

from config import AppConfig
from services.gemini_service import GeminiService
from bot.session import SessionManager
from utils import extract_message_content, is_bot_mentioned

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
        if message.author.bot or not is_bot_mentioned(message, self.bot):
            return

        logger.debug(f"Message from {message.author.name} ({message.author.id}): {len(message.content)} chars")
        self.session_manager.cleanup_expired()
        logger.debug("Session cleanup completed")

        try:
            user_message = extract_message_content(message)
            if not user_message:
                logger.debug("No message content extracted")
                await message.reply("❌ 메시지 내용이 없는데요.", mention_author=False)
                return

            logger.info(f"Processing message from {message.author.name}: {user_message[:100]}")

            async with message.channel.typing():
                # 세션 ID 결정: 리플라이 대상이 있으면 그 메시지 ID, 없으면 현재 메시지 ID
                session_id = message.reference.message_id if message.reference else message.id
                logger.debug(f"Session ID determined: {session_id}")

                # Get or create user session with memory context
                chat_session, user_id = self.session_manager.get_or_create(
                    user_id=message.author.id,
                    username=message.author.name,
                    message_id=str(session_id),
                )
                logger.debug(f"Session created/retrieved for user {user_id}")

                logger.debug("Sending request to Gemini API")
                response_result = await self.gemini_service.generate_chat_response(
                    chat_session,
                    user_message,
                    message,
                )
                logger.debug(f"Received response: {response_result is not None}")

                if response_result:
                    response_text, response_obj = response_result
                    logger.debug(f"Response text length: {len(response_text)}, Has response object: {response_obj is not None}")

                    # Only reply if there's text content
                    # (Gemini might return only function calls without text)
                    if response_text:
                        logger.debug(f"Sending reply with {len(response_text)} characters")
                        await message.reply(response_text, mention_author=False)
                    else:
                        # If only function calls were returned, log but don't reply
                        logger.debug("Response contained only function calls, no text to reply with")
                # else: The error message is now handled by gemini_service._api_request_with_retry

        except Exception as e:
            logger.error(f"메시지 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
            await message.reply("❌ 봇 내부에서 예상치 못한 오류가 발생했어요. 개발자에게 문의해주세요.", mention_author=False)

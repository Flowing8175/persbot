"""Assistant Cog for SoyeBot."""

import discord
from discord.ext import commands
import logging
import time

from config import AppConfig
from services.llm_service import LLMService
from bot.session import SessionManager
from utils import GENERIC_ERROR_MESSAGE, extract_message_content
from metrics import get_metrics

logger = logging.getLogger(__name__)

class AssistantCog(commands.Cog):
    """@mention을 통한 AI 어시스턴트 기능을 처리하는 Cog"""

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
                reference_message_id = (
                    str(message.reference.message_id)
                    if message.reference and message.reference.message_id
                    else None
                )

                resolution = await self.session_manager.resolve_session(
                    channel_id=message.channel.id,
                    author_id=message.author.id,
                    username=message.author.name,
                    message_id=str(message.id),
                    message_content=user_message,
                    reference_message_id=reference_message_id,
                    created_at=message.created_at,
                )

                if not resolution.cleaned_message:
                    await message.reply("❌ 메시지 내용이 없는데요.", mention_author=False)
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

                response_result = await self.llm_service.generate_chat_response(
                    chat_session,
                    resolution.cleaned_message,
                    message,
                )

                if response_result:
                    response_text, response_obj = response_result

                    # Only reply if there's text content
                    # (Some responses may be non-text, so skip replying)
                    if response_text:
                        reply_message = await message.reply(response_text, mention_author=False)
                        if reply_message:
                            self.session_manager.link_message_to_session(
                                str(reply_message.id),
                                session_key,
                            )
                    else:
                        logger.debug("LLM returned no text response for the mention.")
                # else: The error message is now handled by llm_service._api_request_with_retry

            # Track successful message processing
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
            metrics.increment_counter('messages_processed')

        except Exception as e:
            logger.error(f"메시지 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
            await message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

            # Track processing time even on error
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)

    @commands.command(name='초기화', aliases=['reset'])
    async def reset_session(self, ctx: commands.Context):
        """현재 채널의 대화 세션을 수동으로 초기화합니다."""

        try:
            self.session_manager.reset_session_by_channel(ctx.channel.id)
            await ctx.message.add_reaction("✅")
        except Exception as exc:
            logger.error("세션 초기화 실패: %s", exc, exc_info=True)
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

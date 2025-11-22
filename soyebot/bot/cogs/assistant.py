"""Assistant Cog for SoyeBot."""

import logging
import time
import asyncio

import discord
from discord.ext import commands

from bot.chat_handler import ChatReply, create_chat_reply, resolve_session_for_message
from bot.session import SessionManager
from bot.buffer import MessageBuffer
from config import AppConfig
from metrics import get_metrics
from services.llm_service import LLMService
from utils import GENERIC_ERROR_MESSAGE, extract_message_content

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
        self.message_buffer = MessageBuffer(delay=config.message_buffer_delay)

    def _should_ignore_message(self, message: discord.Message) -> bool:
        """Return True when the bot should not process the message."""

        if message.author.bot:
            return True
        if not self.bot.user or not self.bot.user.mentioned_in(message):
            return True
        return message.mention_everyone

    async def _send_llm_reply(self, message: discord.Message, reply: ChatReply) -> None:
        if not reply.text:
            logger.debug("LLM returned no text response for the mention.")
            return

        reply_message = await message.reply(reply.text, mention_author=False)
        if reply_message:
            self.session_manager.link_message_to_session(str(reply_message.id), reply.session_key)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if self._should_ignore_message(message):
            return

        await self.message_buffer.add_message(message.channel.id, message, self._process_batch)

    async def _process_batch(self, messages: list[discord.Message]):
        if not messages:
            return

        # Use the first message for context/reply target, but could be improved
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
                await primary_message.reply("❌ 메시지 내용이 없는데요.", mention_author=False)
                return

            logger.info("Processing batch of %d messages from %s: %s", len(messages), primary_message.author.name, full_text[:100])

            async with primary_message.channel.typing():
                resolution = await resolve_session_for_message(
                    primary_message,
                    full_text,
                    session_manager=self.session_manager,
                )

                if not resolution:
                    await primary_message.reply("❌ 메시지 내용이 없는데요.", mention_author=False)
                    return

                reply = await create_chat_reply(
                    primary_message,
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                )

                if reply:
                    # We reply to the last message in the batch so the user sees it at the bottom
                    last_message = messages[-1]
                    await self._send_llm_reply(last_message, reply)

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)
            metrics.increment_counter('messages_processed')

        except Exception as e:
            logger.error("메시지 처리 중 예상치 못한 오류 발생: %s", e, exc_info=True)
            await primary_message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_latency('message_processing', duration_ms)

    @commands.command(name='초기화', aliases=['reset'])
    async def reset_session(self, ctx: commands.Context):
        """현재 채널의 대화 세션을 수동으로 초기화합니다."""

        try:
            self.session_manager.reset_session_by_channel(ctx.channel.id)
            await ctx.message.add_reaction("✅")
            await asyncio.sleep(5)
            await ctx.message.remove_reaction("✅", ctx.bot.user)
        except Exception as exc:
            logger.error("세션 초기화 실패: %s", exc, exc_info=True)
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

"""Assistant Cog for SoyeBot."""

import logging
import time
import asyncio
from typing import Optional

import discord
from discord.ext import commands

from bot.chat_handler import ChatReply, create_chat_reply, resolve_session_for_message
from bot.session import SessionManager, ResolvedSession
from bot.buffer import MessageBuffer
from config import AppConfig
from metrics import get_metrics
from services.llm_service import LLMService
from services.base import ChatMessage
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
        # If this channel is handled by AutoChannelCog, let it handle the response
        # to avoid duplicate replies (one plain, one reply).
        if message.channel.id in self.config.auto_reply_channel_ids:
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

    @commands.Cog.listener()
    async def on_typing(self, channel: discord.abc.Messageable, user: discord.abc.User, when: float):
        """
        Listener for typing events.
        Extends the processing delay if a user is typing in a channel where we have pending messages.
        """
        if user.bot:
            return

        # We need the channel ID. 'channel' can be TextChannel, DMChannel, etc.
        if hasattr(channel, 'id'):
            self.message_buffer.handle_typing(channel.id, self._process_batch)

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

    @commands.command(name='retry', aliases=['재생성', '다시'])
    async def retry_command(self, ctx: commands.Context):
        """Re-generate the last assistant response."""
        session_key = f"channel:{ctx.channel.id}"

        # Undo the last exchange (assistant + user message)
        removed_messages: list[ChatMessage] = self.session_manager.undo_last_exchanges(session_key, 1)

        if not removed_messages:
            await ctx.reply("❌ 되돌릴 대화가 없습니다.", mention_author=False)
            return

        # Identify the user message content and the previous assistant message ID
        user_role = self.llm_service.get_user_role_name()
        assistant_role = self.llm_service.get_assistant_role_name()

        user_content = ""
        assistant_message_id = None

        # Process removed messages to find content and ID
        # Removed messages are chronological. Expect [User, Assistant] usually.
        for msg in removed_messages:
            if msg.role == user_role:
                user_content = msg.content
            elif msg.role == assistant_role:
                if msg.message_id:
                    assistant_message_id = msg.message_id

        if not user_content:
            await ctx.send("❌ 재시도할 사용자 메시지를 찾을 수 없습니다.")
            return

        # Re-generate response
        async with ctx.channel.typing():
            resolution = ResolvedSession(session_key, user_content)

            # Create a reply using the original context author (ctx.author) which is acceptable for retry
            reply = await create_chat_reply(
                ctx.message,
                resolution=resolution,
                llm_service=self.llm_service,
                session_manager=self.session_manager,
            )

            if reply and reply.text:
                # Try to edit the old message if it exists
                edited = False
                if assistant_message_id:
                    try:
                        old_message = await ctx.channel.fetch_message(int(assistant_message_id))
                        await old_message.edit(content=reply.text)
                        # Link the old message ID to the new session state
                        self.session_manager.link_message_to_session(str(old_message.id), session_key)
                        edited = True
                    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                        logger.warning("Could not edit message %s during retry.", assistant_message_id)

                # If editing failed or wasn't possible, send a new message.
                if not edited:
                    await self._send_llm_reply(ctx.message, reply)
            else:
                 await ctx.send(GENERIC_ERROR_MESSAGE)

        # Attempt to delete the retry command message itself for cleanliness
        try:
            await ctx.message.delete()
        except (discord.Forbidden, discord.HTTPException):
            pass

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

    @commands.command(name='temp')
    @commands.has_permissions(manage_guild=True)
    async def set_temperature(self, ctx: commands.Context, value: Optional[float] = None):
        """Set the temperature parameter for the LLM (0.0 - 2.0)."""
        if value is None:
            current_temp = getattr(self.config, 'temperature', 1.0)
            await ctx.reply(f"🌡️ 현재 Temperature: {current_temp}", mention_author=False)
            return

        if not (0.0 <= value <= 2.0):
            await ctx.reply("❌ Temperature는 0.0에서 2.0 사이여야 합니다.", mention_author=False)
            return

        try:
            self.llm_service.update_parameters(temperature=value)
            await ctx.message.add_reaction("✅")
            await asyncio.sleep(5)
            await ctx.message.remove_reaction("✅", ctx.bot.user)
        except Exception as e:
            logger.error("Temperature 설정 실패: %s", e, exc_info=True)
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

    @commands.command(name='topp')
    @commands.has_permissions(manage_guild=True)
    async def set_top_p(self, ctx: commands.Context, value: Optional[float] = None):
        """Set the top_p parameter for the LLM (0.0 - 1.0)."""
        if value is None:
            current_top_p = getattr(self.config, 'top_p', 1.0)
            await ctx.reply(f"📊 현재 Top-p: {current_top_p}", mention_author=False)
            return

        if not (0.0 <= value <= 1.0):
            await ctx.reply("❌ Top-p는 0.0에서 1.0 사이여야 합니다.", mention_author=False)
            return

        try:
            self.llm_service.update_parameters(top_p=value)
            await ctx.message.add_reaction("✅")
            await asyncio.sleep(5)
            await ctx.message.remove_reaction("✅", ctx.bot.user)
        except Exception as e:
            logger.error("Top-p 설정 실패: %s", e, exc_info=True)
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

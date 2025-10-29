"Summarizer Cog for SoyeBot."

import discord
from discord.ext import commands
from datetime import datetime, timedelta, timezone
import time
import logging
from typing import Optional, Literal

from config import AppConfig
from services.gemini_service import GeminiService
from services.memory_service import MemoryService
from utils import DiscordUI, parse_korean_time

logger = logging.getLogger(__name__)

class SummarizerCog(commands.Cog):
    """'!요약' 관련 명령어를 처리하는 Cog"""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        gemini_service: GeminiService,
        memory_service: Optional[MemoryService] = None,
    ):
        self.bot = bot
        self.config = config
        self.gemini_service = gemini_service
        self.memory_service = memory_service

    async def _fetch_messages(
        self,
        channel: discord.TextChannel,
        **kwargs
    ) -> tuple[str, int]:
        """지정된 조건으로 메시지를 가져와 텍스트로 합칩니다."""
        messages = []
        message_count = 0

        async for message in channel.history(**kwargs):
            if not message.author.bot:
                messages.append(f"{message.author.display_name}: {message.content}")
                message_count += 1

        return "\n".join(messages), message_count

    async def _extract_and_save_summary_memory(
        self,
        user_id: int,
        summary: str,
        channel_name: str,
    ) -> None:
        """요약에서 핵심 정보를 추출하여 기억으로 저장합니다.

        Args:
            user_id: 명령어를 실행한 사용자 ID
            summary: 생성된 요약 텍스트
            channel_name: 요약한 채널 이름
        """
        if not self.memory_service or not self.config.enable_memory_system:
            return

        try:
            # 요약의 첫 100자를 기억으로 저장
            summary_preview = summary[:150]
            if len(summary) > 150:
                summary_preview += "..."

            memory_content = f"[{channel_name} 채널 요약] {summary_preview}"

            self.memory_service.save_memory(
                user_id=str(user_id),
                memory_type='key_memory',
                content=memory_content,
                importance_score=0.6,  # 중간 정도의 중요도
            )

            logger.info(f"Saved summary memory for user {user_id}: {memory_content[:50]}")

        except Exception as e:
            logger.warning(f"Failed to save summary memory: {e}")

    @commands.group(name="요약", invoke_without_command=True)
    async def summarize(self, ctx: commands.Context, *args):
        """
        최근 메시지를 요약합니다.
        - `!요약`: 최근 30분 요약
        - `!요약 <시간>`: 지정된 시간만큼 요약 (예: 20분, 1시간)
        - `!요약 id <ID>`: 특정 메시지 ID 이후 요약 (자동 감지)
        - `!요약 range <ID> <이후/이전> <시간>`: 범위 요약 (자동 감지)
        """
        if ctx.invoked_subcommand is None:
            # 첫 번째 인자 확인
            if len(args) == 0:
                # 기본값: 최근 30분 요약
                time_str = None
            elif args[0].lower() == 'id' and len(args) >= 2:
                # !요약 id <ID> 형식 - id 서브커맨드로 자동 라우팅
                try:
                    message_id = int(args[1])
                    await self.summarize_by_id(ctx, message_id)
                except (ValueError, IndexError):
                    await DiscordUI.safe_send(ctx.channel, f"❌ 올바른 메시지 ID를 입력해주세요. 예: `!요약 id 1234567890`")
                return
            elif args[0].lower() == 'range' and len(args) >= 4:
                # !요약 range <ID> <이후/이전> <시간> 형식 - range 서브커맨드로 자동 라우팅
                try:
                    message_id = int(args[1])
                    direction = args[2]
                    time_str = args[3]

                    if direction not in ["이후", "이전"]:
                        await DiscordUI.safe_send(ctx.channel, f"❌ 방향은 '이후' 또는 '이전'이어야 합니다. 예: `!요약 range 1234567890 이후 2시간`")
                        return

                    await self.summarize_by_range(ctx, message_id, direction, time_str)
                except (ValueError, IndexError):
                    await DiscordUI.safe_send(ctx.channel, f"❌ 올바른 형식을 사용해주세요. 예: `!요약 range 1234567890 이후 2시간`")
                return
            else:
                # 일반 시간 인자 처리
                time_str = args[0] if len(args) > 0 else None

            # 기본 요약 로직 (시간 기반)
            minutes = parse_korean_time(time_str) if time_str else 30
            if minutes is None:
                await DiscordUI.safe_send(ctx.channel, f"❌ 시간 형식이 올바르지 않아요. (예: '20분', '1시간')")
                return

            async with ctx.channel.typing():
                time_limit = datetime.now(timezone.utc) - timedelta(minutes=minutes)
                full_text, count = await self._fetch_messages(
                    ctx.channel, limit=self.config.max_messages_per_fetch, after=time_limit, oldest_first=True
                )

                if count == 0:
                    await DiscordUI.safe_send(ctx.channel, f"ℹ️ 최근 {minutes}분 동안 메시지가 없어요.")
                    return

                summary = await self.gemini_service.summarize_text(full_text)
                if summary:
                    # 요약을 기억으로 저장
                    await self._extract_and_save_summary_memory(
                        user_id=ctx.author.id,
                        summary=summary,
                        channel_name=ctx.channel.name,
                    )
                    await DiscordUI.safe_send(ctx.channel, f"**최근 {minutes}분 {count}개 메시지 요약:**\n>>> {summary}")
                else:
                    await DiscordUI.safe_send(ctx.channel, "❌ 요약 생성 중 오류가 발생했어요. 다시 시도해주세요.")

    @summarize.command(name="id")
    async def summarize_by_id(self, ctx: commands.Context, message_id: int):
        """특정 메시지 ID 이후의 메시지를 요약합니다."""
        try:
            start_message = await ctx.channel.fetch_message(message_id)
        except discord.NotFound:
            await DiscordUI.safe_send(ctx.channel, f"❌ 메시지 ID `{message_id}`를 찾을 수 없어요.")
            return

        async with ctx.channel.typing():
            full_text, count = await self._fetch_messages(
                ctx.channel, limit=self.config.max_messages_per_fetch, after=start_message, oldest_first=True
            )
            # 시작 메시지도 포함
            if not start_message.author.bot:
                full_text = f"{start_message.author.display_name}: {start_message.content}\n" + full_text
                count += 1

            if count == 0:
                await DiscordUI.safe_send(ctx.channel, f"ℹ️ 메시지 ID `{message_id}` 이후 메시지가 없어요.")
                return

            summary = await self.gemini_service.summarize_text(full_text)
            if summary:
                # 요약을 기억으로 저장
                await self._extract_and_save_summary_memory(
                    user_id=ctx.author.id,
                    summary=summary,
                    channel_name=ctx.channel.name,
                )
                await DiscordUI.safe_send(ctx.channel, f"**메시지 ID `{message_id}` 이후 {count}개 메시지 요약:**\n>>> {summary}")
            else:
                await DiscordUI.safe_send(ctx.channel, "❌ 요약 생성 중 오류가 발생했어요. 다시 시도해주세요.")

    @summarize.command(name="range")
    async def summarize_by_range(self, ctx: commands.Context, message_id: int, direction: Literal["이후", "이전"], time_str: str):
        """메시지 ID 기준 특정 시간 범위의 메시지를 요약합니다."""
        minutes = parse_korean_time(time_str)
        if minutes is None:
            await DiscordUI.safe_send(ctx.channel, f"❌ 시간 형식이 올바르지 않아요. (예: '20분', '1시간')")
            return

        try:
            start_message = await ctx.channel.fetch_message(message_id)
        except discord.NotFound:
            await DiscordUI.safe_send(ctx.channel, f"❌ 메시지 ID `{message_id}`를 찾을 수 없어요.")
            return

        async with ctx.channel.typing():
            start_time = start_message.created_at
            history_args = {"limit": self.config.max_messages_per_fetch, "oldest_first": True}
            if direction == "이후":
                history_args["after"] = start_message
                history_args["before"] = start_time + timedelta(minutes=minutes)
            else: # 이전
                history_args["after"] = start_time - timedelta(minutes=minutes)
                history_args["before"] = start_message

            full_text, count = await self._fetch_messages(ctx.channel, **history_args)

            if count == 0:
                await DiscordUI.safe_send(ctx.channel, f"ℹ️ 해당 범위에 메시지가 없어요.")
                return

            summary = await self.gemini_service.summarize_text(full_text)
            if summary:
                # 요약을 기억으로 저장
                await self._extract_and_save_summary_memory(
                    user_id=ctx.author.id,
                    summary=summary,
                    channel_name=ctx.channel.name,
                )
                await DiscordUI.safe_send(ctx.channel, f"**메시지 ID `{message_id}` {direction} {minutes}분 {count}개 메시지 요약:**\n>>> {summary}")
            else:
                await DiscordUI.safe_send(ctx.channel, "❌ 요약 생성 중 오류가 발생했어요. 다시 시도해주세요.")

    @summarize.error
    @summarize_by_id.error
    @summarize_by_range.error
    async def summarize_error(self, ctx, error):
        if isinstance(error, commands.BadArgument):
            await ctx.send("❌ 인수가 잘못되었어요. 숫자를 입력해야 하는 곳에 문자를 넣지 않았는지 확인해주세요.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"❌ 명령어를 완성해주세요! `!도움말 {ctx.command.name}`으로 사용법을 볼 수 있어요.")
        else:
            logger.error(f"요약 명령어 에러: {error}", exc_info=True)
            await ctx.send(f"❌ 예상치 못한 에러가 발생했어요: {error}")

"""Summarizer Cog for SoyeBot."""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import discord
from discord import app_commands
from discord.ext import commands

from persbot.config import AppConfig
from persbot.services.llm_service import LLMService
from persbot.utils import GENERIC_ERROR_MESSAGE, DiscordUI, parse_korean_time, send_discord_message

logger = logging.getLogger(__name__)


class SummarizerCog(commands.Cog):
    """'!요약' 관련 명령어를 처리하는 Cog"""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
    ):
        self.bot = bot
        self.config = config
        self.llm_service = llm_service

    async def _fetch_messages(self, channel: discord.TextChannel, **kwargs) -> tuple[str, int]:
        """지정된 조건으로 메시지를 가져와 텍스트로 합칩니다.

        Memory-optimized: Single-pass collection to reduce intermediate objects.

        CRITICAL: All callers pass oldest_first=True in kwargs, ensuring messages
        are in CHRONOLOGICAL ORDER (oldest → newest). This is essential for accurate
        LLM summarization. DO NOT reverse the order.
        """
        # Build formatted message list in single pass (memory optimization for 1GB RAM)
        # Eliminates intermediate message_list to reduce peak memory by ~20%
        text_parts = [
            f"{msg.author.id}: {msg.content}"
            async for msg in channel.history(**kwargs)
            if not msg.author.bot
        ]

        # Messages are in chronological order (oldest first) - critical for LLM context
        return "\n".join(text_parts), len(text_parts)

    @commands.hybrid_command(name="요약", description="채널의 대화 내용을 요약합니다.")
    @app_commands.describe(
        대상="시간 (예: '20분') 또는 메시지 ID",
        방향="방향 ('이후' 또는 '이전', ID 사용 시)",
        범위="범위 시간 (예: '30분', ID/방향 사용 시)",
    )
    async def summarize(
        self,
        ctx: commands.Context,
        대상: Optional[str] = None,
        방향: Optional[Literal["이후", "이전"]] = None,
        범위: Optional[str] = None,
    ) -> None:
        """
        채널의 대화 내용을 요약합니다.

        사용법:
        - `!요약`: 최근 30분 요약
        - `!요약 20분`: 최근 20분 요약
        - `!요약 <ID> 이후`: 해당 메시지 이후 요약
        - `!요약 <ID> <이후|이전> 30분`: 특정 시점 기준 범위 요약
        """
        # Collect non-None arguments into a tuple to mimic existing logic
        args = []
        if 대상:
            args.append(대상)
        if 방향:
            args.append(방향)
        if 범위:
            args.append(범위)

        # Defer immediately to prevent timeout errors
        await ctx.defer()

        await self._handle_summarize_args(ctx, tuple(args))

    async def _handle_summarize_args(self, ctx: commands.Context, args: tuple) -> None:
        """인자를 분석하여 적절한 요약 메서드를 호출합니다."""
        if len(args) == 0:
            # !요약 - 기본값: 최근 30분
            await self._summarize_by_time(ctx, 30)
        elif len(args) == 1:
            # !요약 <시간> - 지정된 시간만큼 요약
            # 또는 !요약 <ID> (단독 ID는 지원 안함, 기존 로직 따름 -> 기존엔 <ID>만 오면 시간 파싱 시도 후 실패함)
            # 하지만 사용성을 위해 arg1이 ID처럼 보이면 안내를 할 수도 있음.
            # 일단 기존 로직 유지: 시간 파싱 시도
            arg = args[0]
            minutes = parse_korean_time(arg)
            if minutes is not None:
                await self._summarize_by_time(ctx, minutes)
            else:
                # ID일 수도 있음, 하지만 기존 로직은 ID 단독 처리를 안함.
                await send_discord_message(
                    ctx, f"❌ 시간 형식이 올바르지 않아요. (예: '20분', '1시간')"
                )

        elif len(args) == 2:
            # !요약 <메시지ID> <이후|이전>
            대상, 방향 = args[0], args[1]
            if not self._is_message_id(대상):
                await send_discord_message(ctx, f"❌ 첫 번째 인자는 메시지 ID여야 해요.")
                return
            if 방향 not in ["이후", "이전"]:
                await send_discord_message(
                    ctx, f"❌ 두 번째 인자는 '이후' 또는 '이전'이어야 합니다."
                )
                return
            try:
                message_id = int(대상)
                if 방향 == "이후":
                    # 메시지 ID 이후부터 최대 길이까지
                    await self.summarize_by_id(ctx, message_id)
                else:
                    # 메시지 ID 이전은 명시적으로 시간을 지정해야 함
                    await send_discord_message(
                        ctx, f"❌ '이전'은 시간을 지정해야 합니다. 예: `!요약 123456 이전 1시간`"
                    )
            except ValueError:
                await send_discord_message(ctx, f"❌ 올바른 메시지 ID를 입력해주세요.")
        elif len(args) >= 3:
            # !요약 <메시지ID> <이후|이전> <시간>
            대상, 방향, 범위 = args[0], args[1], args[2]
            if not self._is_message_id(대상):
                await send_discord_message(ctx, f"❌ 첫 번째 인자는 메시지 ID여야 해요.")
                return
            if 방향 not in ["이후", "이전"]:
                await send_discord_message(
                    ctx, f"❌ 두 번째 인자는 '이후' 또는 '이전'이어야 합니다."
                )
                return
            try:
                message_id = int(대상)
                await self.summarize_by_range(ctx, message_id, 방향, 범위)
            except ValueError:
                await send_discord_message(
                    ctx, f"❌ 올바른 형식을 사용해주세요. 예: `!요약 123456 이후 30분`"
                )
        else:
            await send_discord_message(
                ctx,
                f"❌ 올바른 형식을 사용해주세요.\n사용법:\n- `!요약`\n- `!요약 <시간>`\n- `!요약 <메시지ID> 이후`\n- `!요약 <메시지ID> <이후|이전> <시간>`",
            )

    def _is_message_id(self, arg: str) -> bool:
        """인자가 메시지 ID인지 판단합니다."""
        if not arg.isdigit():
            return False
        # Discord 메시지 ID는 17-20자리 숫자
        return 17 <= len(arg) <= 20

    async def _summarize_by_time(self, ctx: commands.Context, minutes: int) -> None:
        """시간 기반 요약을 수행합니다."""
        async with ctx.channel.typing():
            time_limit = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            full_text, count = await self._fetch_messages(
                ctx.channel,
                limit=self.config.max_messages_per_fetch,
                after=time_limit,
                oldest_first=True,
            )

            if count == 0:
                await send_discord_message(ctx, f"ℹ️ 최근 {minutes}분 동안 메시지가 없어요.")
                return

            summary = await self.llm_service.summarize_text(full_text)
            if summary:
                await send_discord_message(
                    ctx, f"**최근 {minutes}분 {count}개 메시지 요약:**\n>>> {summary}"
                )
            else:
                await send_discord_message(ctx, GENERIC_ERROR_MESSAGE)

    async def summarize_by_id(self, ctx: commands.Context, message_id: int) -> None:
        """특정 메시지 ID 이후의 메시지를 요약합니다."""
        try:
            start_message = await ctx.channel.fetch_message(message_id)
        except discord.NotFound:
            await send_discord_message(ctx, f"❌ 메시지 ID `{message_id}`를 찾을 수 없어요.")
            return

        async with ctx.channel.typing():
            full_text, count = await self._fetch_messages(
                ctx.channel,
                limit=self.config.max_messages_per_fetch,
                after=start_message,
                oldest_first=True,
            )
            # 시작 메시지도 포함
            if not start_message.author.bot:
                full_text = f"{start_message.author.id}: {start_message.content}\n" + full_text
                count += 1

            if count == 0:
                await send_discord_message(ctx, f"ℹ️ 메시지 ID `{message_id}` 이후 메시지가 없어요.")
                return

            summary = await self.llm_service.summarize_text(full_text)
            if summary:
                await send_discord_message(
                    ctx, f"**메시지 ID `{message_id}` 이후 {count}개 메시지 요약:**\n>>> {summary}"
                )
            else:
                await send_discord_message(ctx, GENERIC_ERROR_MESSAGE)

    async def summarize_by_range(
        self,
        ctx: commands.Context,
        message_id: int,
        direction: Literal["이후", "이전"],
        time_str: str,
    ) -> None:
        """메시지 ID 기준 특정 시간 범위의 메시지를 요약합니다."""
        minutes = parse_korean_time(time_str)
        if minutes is None:
            await send_discord_message(
                ctx, f"❌ 시간 형식이 올바르지 않아요. (예: '20분', '1시간')"
            )
            return

        try:
            start_message = await ctx.channel.fetch_message(message_id)
        except discord.NotFound:
            await send_discord_message(ctx, f"❌ 메시지 ID `{message_id}`를 찾을 수 없어요.")
            return

        async with ctx.channel.typing():
            start_time = start_message.created_at
            history_args = {"limit": self.config.max_messages_per_fetch, "oldest_first": True}
            if direction == "이후":
                history_args["after"] = start_message
                history_args["before"] = start_time + timedelta(minutes=minutes)
            else:  # 이전
                history_args["after"] = start_time - timedelta(minutes=minutes)
                history_args["before"] = start_message

            full_text, count = await self._fetch_messages(ctx.channel, **history_args)

            if count == 0:
                await send_discord_message(ctx, f"ℹ️ 해당 범위에 메시지가 없어요.")
                return

            summary = await self.llm_service.summarize_text(full_text)
            if summary:
                await send_discord_message(
                    ctx,
                    f"**메시지 ID `{message_id}` {direction} {minutes}분 {count}개 메시지 요약:**\n>>> {summary}",
                )
            else:
                await send_discord_message(ctx, GENERIC_ERROR_MESSAGE)

    @summarize.error
    async def summarize_error(self, ctx, error) -> None:
        if isinstance(error, commands.BadArgument):
            await send_discord_message(
                ctx,
                "❌ 인수가 잘못되었어요. 숫자를 입력해야 하는 곳에 문자를 넣지 않았는지 확인해주세요.",
            )
        elif isinstance(error, commands.MissingRequiredArgument):
            prefix = ctx.prefix or self.config.command_prefix
            await send_discord_message(
                ctx,
                f"❌ 명령어를 완성해주세요! 예: `{prefix}요약 20분` 또는 `{prefix}요약 123456 이후 30분`",
            )
        else:
            logger.error(f"요약 명령어 에러: {error}", exc_info=True)
            await send_discord_message(ctx, GENERIC_ERROR_MESSAGE)

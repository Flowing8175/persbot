"""Command handlers for Assistant Cog."""

import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from persbot.bot.chat_handler import create_chat_reply
from persbot.bot.session import SessionManager
from persbot.config import AppConfig
from persbot.services.llm_service import LLMService
from persbot.services.prompt_service import PromptService
from persbot.tools.manager import ToolManager
from persbot.utils import GENERIC_ERROR_MESSAGE, send_discord_message

from . import utils

logger = logging.getLogger(__name__)


def register_commands(cog):
    """Register all commands to the given cog instance."""
    cog.help_command = help_command
    cog.retry_command = retry_command
    cog.abort_command = abort_command
    cog.reset_session = reset_session
    cog.set_temperature = set_temperature
    cog.set_top_p = set_top_p
    cog.toggle_break_cut = toggle_break_cut
    cog.set_thinking_budget = set_thinking_budget
    cog.set_buffer_delay = set_buffer_delay
    cog.cog_command_error = cog_command_error


@commands.hybrid_command(
    name="help",
    aliases=["도움말", "명령어", "h"],
    description="봇의 모든 명령어와 사용법을 안내합니다.",
)
async def help_command(self, ctx):
    """봇의 모든 명령어와 사용법을 안내합니다."""
    embed = discord.Embed(
        title="🤖 명령어 가이드",
        description=f"접두사: `{self.config.command_prefix}` 또는 `@mention`을 사용하여 명령을 내릴 수 있습니다.",
        color=discord.Color.blue(),
    )

    # 1. 대화 제어
    embed.add_field(
        name="💬 대화 제어",
        value=(
            "`!retry` (`!다시`): 마지막 답변을 지우고 다시 생성합니다.\n"
            "`!reset` (`!초기화`): 현재 채널의 대화 기록을 초기화합니다.\n"
            "`!undo [N]` (`!@`): 마지막 N개의 대화 쌍을 삭제합니다. (자동응답 채널 전용)\n"
            "`!abort` (`!중단`): 진행 중인 전송이나 AI 처리를 즉시 멈춥니다."
        ),
        inline=False,
    )

    # 2. 요약 및 분석
    embed.add_field(
        name="📝 요약 및 분석",
        value=(
            "`!요약`: 최근 30분 대화를 요약합니다.\n"
            "`!요약 [시간]`: 지정 시간(예: `20분`, `1시간`) 동안의 대화를 요약합니다.\n"
            "`!요약 [ID]`: 특정 메시지 이후의 대화를 요약합니다."
        ),
        inline=False,
    )

    # 3. 프롬프트 관리 (Persona)
    embed.add_field(
        name="🎭 프롬프트 (페르소나) 관리",
        value=("`!prompt`: 프롬프트 관리 UI를 엽니다. (생성, 목록, 선택, 삭제 등)\n"),
        inline=False,
    )

    # 4. 설정 및 파라미터
    embed.add_field(
        name="⚙️ 설정 및 파라미터",
        value=(
            "`!temp <0.0~2.0>`: AI의 창의성(Temperature)을 조절합니다.\n"
            "`!생각 <숫자|auto|off>`: Gemini Thinking Budget를 설정합니다.\n"
            "`!끊어치기 [on|off]`: 실시간 메시지 끊어 전송 모드를 설정합니다.\n"
            "`!대기 <초>` (`!delay`): 메시지 버퍼 대기 시간을 설정합니다 (0~60초)."
        ),
        inline=False,
    )

    embed.set_footer(text="SoyeBot | Advanced Agentic Coding Assistant")
    await send_discord_message(ctx, "", embed=embed)


@commands.hybrid_command(
    name="retry",
    aliases=["재생성", "다시"],
    description="마지막 대화를 되돌리고 응답을 다시 생성합니다.",
)
async def retry_command(self, ctx):
    """마지막 대화를 되돌리고 응답을 다시 생성합니다."""
    await ctx.defer()

    channel_id = ctx.channel.id
    session_key = f"channel:{channel_id}"

    # Cancel any active tasks
    cancelled = utils.cancel_channel_tasks(
        channel_id, self.processing_tasks, self.sending_tasks, ctx.channel.name, "Retry command"
    )

    # Undo the last exchange
    removed_messages = self.session_manager.undo_last_exchanges(session_key, 1)
    if not removed_messages:
        await ctx.send("❌ 되돌릴 대화가 없습니다.")
        return

    # Process removed messages
    user_content = await utils.process_removed_messages(ctx, removed_messages, self.llm_service)
    if not user_content:
        await ctx.send("❌ 재시도할 사용자 메시지를 찾을 수 없습니다.")
        return

    # Regenerate response
    await utils.regenerate_response(
        ctx,
        session_key,
        user_content,
        self.bot,
        self.llm_service,
        self.session_manager,
        self.tool_manager,
        lambda msg, reply: utils.send_response(
            msg, reply, self.config, self.session_manager, self._handle_break_cut_sending
        ),
        self.config,
    )


@commands.hybrid_command(
    name="abort",
    aliases=["중단", "멈춰"],
    description="진행 중인 모든 메시지 전송 및 처리를 강제로 중단합니다.",
)
async def abort_command(self, ctx):
    """진행 중인 모든 메시지 전송 및 처리를 강제로 중단합니다."""
    # Check permissions unless NO_CHECK_PERMISSION is set
    if not self.config.no_check_permission:
        if (
            not isinstance(ctx.author, discord.Member)
            or not ctx.author.guild_permissions.manage_guild
        ):
            await ctx.reply(
                "❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: manage_guild)",
                mention_author=False,
            )
            return

    channel_id = ctx.channel.id

    # Cancel tasks in both cogs
    aborted = utils.cancel_channel_tasks(
        channel_id, self.processing_tasks, self.sending_tasks, ctx.channel.name, "Abort command"
    )
    aborted = utils.cancel_auto_channel_tasks(channel_id, self.bot) or aborted

    # Send appropriate response
    if aborted:
        await utils.send_abort_success(ctx)
        logger.info("User %s requested abort in channel %s", ctx.author.name, channel_id)
    else:
        await utils.send_abort_no_tasks(ctx)


@commands.hybrid_command(
    name="초기화",
    aliases=["reset"],
    description="현재 채널의 대화 세션을 초기화합니다.",
)
async def reset_session(self, ctx):
    """현재 채널의 대화 세션을 초기화합니다."""

    try:
        self.session_manager.reset_session_by_channel(ctx.channel.id)
        if ctx.interaction:
            await ctx.reply("✅ 대화 세션이 초기화되었습니다.", ephemeral=False)
        else:
            await ctx.message.add_reaction("✅")
    except Exception as exc:
        logger.error("세션 초기화 실패: %s", exc, exc_info=True)
        await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)


@commands.hybrid_command(
    name="temp", description="LLM의 창의성(Temperature)을 설정합니다 (0.0~2.0)."
)
@app_commands.describe(value="설정할 Temperature 값 (0.0~2.0)")
async def set_temperature(self, ctx, value: Optional[float] = None):
    """LLM의 창의성(Temperature)을 설정합니다 (0.0~2.0)."""
    # Check permissions unless NO_CHECK_PERMISSION is set
    if not self.config.no_check_permission:
        if (
            not isinstance(ctx.author, discord.Member)
            or not ctx.author.guild_permissions.manage_guild
        ):
            await ctx.reply(
                "❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: manage_guild)",
                mention_author=False,
            )
            return

    if value is None:
        current_temp = getattr(self.config, "temperature", 1.0)
        await ctx.reply(f"🌡️ 현재 Temperature: {current_temp}", mention_author=False)
        return

    if not (0.0 <= value <= 2.0):
        await ctx.reply("❌ Temperature는 0.0에서 2.0 사이여야 합니다.", mention_author=False)
        return

    try:
        self.llm_service.update_parameters(temperature=value)
        if ctx.interaction:
            await ctx.reply(f"✅ Temperature가 {value}로 설정되었습니다.", ephemeral=False)
        else:
            await ctx.message.add_reaction("✅")
    except Exception as e:
        logger.error("Temperature 설정 실패: %s", e, exc_info=True)
        await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)


@commands.hybrid_command(name="topp", description="LLM의 다양성(Top-P)을 설정합니다 (0.0~1.0).")
@app_commands.describe(value="설정할 Top-P 값 (0.0~1.0)")
async def set_top_p(self, ctx, value: Optional[float] = None):
    """LLM의 다양성(Top-P)을 설정합니다 (0.0~1.0)."""
    # Check permissions unless NO_CHECK_PERMISSION is set
    if not self.config.no_check_permission:
        if (
            not isinstance(ctx.author, discord.Member)
            or not ctx.author.guild_permissions.manage_guild
        ):
            await ctx.reply(
                "❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: manage_guild)",
                mention_author=False,
            )
            return

    if value is None:
        current_top_p = getattr(self.config, "top_p", 1.0)
        await ctx.reply(f"📊 현재 Top-p: {current_top_p}", mention_author=False)
        return

    if not (0.0 <= value <= 1.0):
        await ctx.reply("❌ Top-p는 0.0에서 1.0 사이여야 합니다.", mention_author=False)
        return

    try:
        self.llm_service.update_parameters(top_p=value)
        if ctx.interaction:
            await ctx.reply(f"✅ Top-p가 {value}로 설정되었습니다.", ephemeral=False)
        else:
            await ctx.message.add_reaction("✅")
    except Exception as e:
        await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)


@commands.hybrid_command(
    name="끊어치기", description="긴 응답을 나누어 보내는 기능을 켜거나 끕니다."
)
@app_commands.describe(mode="모드 설정 (on/off)")
async def toggle_break_cut(self, ctx, mode: Optional[str] = None):
    """긴 응답을 나누어 보내는 기능을 켜거나 끕니다."""
    if mode is None:
        # Toggle
        self.config.break_cut_mode = not self.config.break_cut_mode
    else:
        cleaned = mode.lower().strip()
        if cleaned == "on":
            self.config.break_cut_mode = True
        elif cleaned == "off":
            self.config.break_cut_mode = False
        else:
            await ctx.reply("사용법: !끊어치기 [on|off] (생략 시 토글)")
            return

    status = "ON" if self.config.break_cut_mode else "OFF"
    await ctx.reply(f"✂️ 끊어치기 모드가 **{status}** 상태로 변경되었습니다.")


@commands.hybrid_command(
    name="생각",
    aliases=["think"],
    description="Gemini Thinking Budget를 설정합니다.",
)
@app_commands.describe(value="숫자(512~32768), 'auto', 또는 'off'")
async def set_thinking_budget(self, ctx, value: Optional[str] = None):
    """Gemini Thinking Budget를 설정합니다."""
    # Check permissions unless NO_CHECK_PERMISSION is set
    if not self.config.no_check_permission:
        if (
            not isinstance(ctx.author, discord.Member)
            or not ctx.author.guild_permissions.manage_guild
        ):
            await ctx.reply(
                "❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: manage_guild)",
                mention_author=False,
            )
            return

    if value is None:
        current = getattr(self.config, "thinking_budget", None)
        if current is None:
            display = "OFF"
        elif current == -1:
            display = "AUTO"
        else:
            display = str(current)
        status = f"현재 Thinking Budget: **{display}**"
        await ctx.reply(f"🧠 {status}", mention_author=False)
        return

    cleaned = value.lower().strip()
    target_value: Optional[int] = None

    if cleaned == "off":
        target_value = None
    elif cleaned == "auto":
        target_value = -1  # Special value for dynamic budget
    else:
        try:
            target_value = int(cleaned)
            if not (512 <= target_value <= 32768):
                await ctx.reply(
                    "❌ Thinking Budget은 512에서 32768 사이여야 합니다.",
                    mention_author=False,
                )
                return
        except ValueError:
            await ctx.reply(
                "❌ 올바른 숫자(512~32768), 'auto', 또는 'off'를 입력해 주세요.",
                mention_author=False,
            )
            return

    try:
        self.llm_service.update_parameters(thinking_budget=target_value)
        if ctx.interaction:
            await ctx.reply(
                f"✅ Thinking Budget가 {target_value if target_value else 'OFF'}로 설정되었습니다.",
                ephemeral=False,
            )
        else:
            await ctx.message.add_reaction("✅")

    except Exception as e:
        logger.error("Thinking Budget 설정 실패: %s", e, exc_info=True)
        await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)


@commands.hybrid_command(
    name="delay",
    aliases=["대기"],
    description="메시지 버퍼 대기 시간을 설정합니다 (초 단위).",
)
@app_commands.describe(value="설정할 대기 시간 (초, 0~60)")
async def set_buffer_delay(self, ctx, value: Optional[float] = None):
    """메시지 버퍼 대기 시간을 설정합니다 (초 단위)."""
    # Check permissions unless NO_CHECK_PERMISSION is set
    if not self.config.no_check_permission:
        if (
            not isinstance(ctx.author, discord.Member)
            or not ctx.author.guild_permissions.manage_guild
        ):
            await ctx.reply(
                "❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: manage_guild)",
                mention_author=False,
            )
            return

    if value is None:
        current_delay = self.message_buffer.default_delay
        await ctx.reply(f"⏱️ 현재 버퍼 대기 시간: {current_delay}초", mention_author=False)
        return

    if not (0.0 <= value <= 60.0):
        await ctx.reply("❌ 대기 시간은 0에서 60초 사이여야 합니다.", mention_author=False)
        return

    try:
        self.message_buffer.update_delay(value)
        self.config.message_buffer_delay = value
        if ctx.interaction:
            await ctx.reply(f"✅ 버퍼 대기 시간이 {value}초로 설정되었습니다.", ephemeral=False)
        else:
            await ctx.message.add_reaction("✅")
    except Exception as e:
        logger.error("버퍼 대기 시간 설정 실패: %s", e, exc_info=True)
        await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)


async def cog_command_error(self, ctx, error: Exception):
    """Cog 내 명령어 에러 핸들러"""
    if isinstance(error, commands.MissingPermissions):
        await ctx.reply(
            f"❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: {', '.join(error.missing_permissions)})",
            mention_author=False,
        )
    elif isinstance(error, commands.BadArgument):
        await ctx.reply(
            "❌ 잘못된 인자가 전달되었습니다. 명령어를 다시 확인해 주세요.",
            mention_author=False,
        )
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.reply(
            f"⏳ 쿨다운 중입니다. {error.retry_after:.1f}초 후에 다시 시도해 주세요.",
            mention_author=False,
        )
    else:
        logger.error(f"Command error in {ctx.command}: {error}", exc_info=True)
        # 기본 에러 메시지는 이미 globally 처리될 수도 있지만, cog 레벨에서 한번 더 확인
        if not ctx.command.has_error_handler():
            await ctx.reply(
                f"❌ 명령어 실행 중 오류가 발생했습니다: {str(error)}",
                mention_author=False,
            )

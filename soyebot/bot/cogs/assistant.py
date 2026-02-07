"""Assistant Cog for SoyeBot."""

import logging
import time
import asyncio
import re
from typing import Optional, Literal

import discord
from discord import app_commands
from discord.ext import commands

from soyebot.bot.chat_handler import (
    ChatReply,
    create_chat_reply,
    resolve_session_for_message,
    send_split_response,
)
from soyebot.bot.session import SessionManager, ResolvedSession
from soyebot.bot.cogs.base import BaseChatCog
from soyebot.config import AppConfig
from soyebot.services.llm_service import LLMService
from soyebot.services.base import ChatMessage
from soyebot.services.prompt_service import PromptService
from soyebot.utils import (
    GENERIC_ERROR_MESSAGE,
    extract_message_content,
    send_discord_message,
)
from soyebot.tools.manager import ToolManager

logger = logging.getLogger(__name__)


class AssistantCog(BaseChatCog):
    """@mention을 통한 AI 어시스턴트 기능을 처리하는 Cog"""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        session_manager: SessionManager,
        prompt_service: PromptService,
        tool_manager: Optional["ToolManager"] = None,
    ):
        super().__init__(bot, config, llm_service, session_manager)
        self.prompt_service = prompt_service
        self.tool_manager = tool_manager

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

    async def _send_response(self, message: discord.Message, reply: ChatReply) -> None:
        if not reply.text:
            logger.debug("LLM returned no text response for the mention.")
            return

        # If Break-Cut Mode is OFF, send normally (with automatic splitting)
        if not self.config.break_cut_mode:
            sent_messages = await send_discord_message(
                message, reply.text, mention_author=False
            )
            for sent_message in sent_messages:
                self.session_manager.link_message_to_session(
                    str(sent_message.id), reply.session_key
                )
            return

        # If Break-Cut Mode is ON, use shared helper
        await self._handle_break_cut_sending(message.channel.id, message.channel, reply)

    async def _handle_error(self, message: discord.Message, error: Exception):
        await message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

    async def _prepare_batch_context(self, messages: list[discord.Message]) -> str:
        # 1. Fetch recent context (10 messages before the primary message)
        primary_message = messages[0]
        context_messages = [
            msg
            async for msg in primary_message.channel.history(
                limit=10, before=primary_message
            )
        ]
        context_messages.reverse()  # Chronological order

        context_text = ""
        if context_messages:
            context_lines = []
            for msg in context_messages:
                c_content = extract_message_content(msg)
                if c_content:
                    context_lines.append(f"{msg.author.id}: {c_content}")

            if context_lines:
                context_text = (
                    "=== 이전 대화 문맥 (참고용) ===\n"
                    + "\n".join(context_lines)
                    + "\n=== 현재 메시지 ===\n"
                )

        # 2. Combine current batch contents
        combined_content = []
        for msg in messages:
            content = extract_message_content(msg)
            if content:
                if len(messages) > 1 and msg.author.id:
                    combined_content.append(f"{msg.author.id}: {content}")
                else:
                    combined_content.append(content)

        current_text = "\n".join(combined_content)

        if not current_text:
            return ""

        # Prepend context to the full text
        return context_text + current_text

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if self._should_ignore_message(message):
            return

        messages_to_prepend = self._cancel_active_tasks(
            message.channel.id, message.author.name
        )

        await self.message_buffer.add_message(
            message.channel.id, message, self._process_batch
        )

        if messages_to_prepend:
            # Ensure the list exists before prepending
            if message.channel.id in self.message_buffer.buffers:
                self.message_buffer.buffers[message.channel.id][0:0] = (
                    messages_to_prepend
                )

    # on_typing is inherited, but we might want to ensure it works for us.
    # BaseChatCog has it, checking break_cut_mode. That matches AssistantCog's logic.

    @commands.hybrid_command(
        name="help",
        aliases=["도움말", "명령어", "h"],
        description="봇의 모든 명령어와 사용법을 안내합니다.",
    )
    async def help_command(self, ctx: commands.Context):
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
            value=(
                "`!prompt`: 프롬프트 관리 UI를 엽니다. (생성, 목록, 선택, 삭제 등)\n"
            ),
            inline=False,
        )

        # 4. 설정 및 파라미터
        embed.add_field(
            name="⚙️ 설정 및 파라미터",
            value=(
                "`!temp <0.0~2.0>`: AI의 창의성(Temperature)을 조절합니다.\n"
                "`!생각 <숫자|auto|off>`: Gemini Thinking Budget를 설정합니다.\n"
                "`!끊어치기 [on|off]`: 실시간 메시지 끊어 전송 모드를 설정합니다."
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
    async def retry_command(self, ctx: commands.Context):
        """마지막 대화를 되돌리고 응답을 다시 생성합니다."""
        await ctx.defer()

        channel_id = ctx.channel.id
        session_key = f"channel:{channel_id}"

        # Cancel any active tasks
        self._cancel_channel_tasks(channel_id, ctx.channel.name, "Retry command")

        # Undo the last exchange
        removed_messages = self.session_manager.undo_last_exchanges(session_key, 1)
        if not removed_messages:
            await ctx.send("❌ 되돌릴 대화가 없습니다.")
            return

        # Process removed messages
        user_content = await self._process_removed_messages(ctx, removed_messages)
        if not user_content:
            await ctx.send("❌ 재시도할 사용자 메시지를 찾을 수 없습니다.")
            return

        # Regenerate response
        await self._regenerate_response(ctx, session_key, user_content)

    async def _process_removed_messages(
        self, ctx: commands.Context, removed_messages: list
    ) -> str:
        """Process removed messages: delete assistant messages and return user content."""
        user_role = self.llm_service.get_user_role_name()
        assistant_role = self.llm_service.get_assistant_role_name()
        user_content = ""

        for msg in removed_messages:
            if msg.role == user_role:
                user_content = msg.content
            elif msg.role == assistant_role:
                await self._delete_assistant_messages(ctx.channel, msg)

        return user_content

    async def _delete_assistant_messages(self, channel, msg) -> None:
        """Delete assistant messages from Discord."""
        if not hasattr(msg, "message_ids") or not msg.message_ids:
            return
        for mid in msg.message_ids:
            try:
                old_msg = await channel.fetch_message(int(mid))
                await old_msg.delete()
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                pass

    async def _regenerate_response(
        self, ctx: commands.Context, session_key: str, user_content: str
    ) -> None:
        """Regenerate LLM response and send it."""
        async with ctx.channel.typing():
            resolution = ResolvedSession(session_key, user_content)
            reply = await create_chat_reply(
                ctx.message,
                resolution=resolution,
                llm_service=self.llm_service,
                session_manager=self.session_manager,
                tool_manager=self.tool_manager,
            )

            if reply and reply.text:
                await self._send_response(ctx.message, reply)
                # Clean up deferred interaction in break-cut mode
                if self.config.break_cut_mode and ctx.interaction:
                    try:
                        await ctx.interaction.delete_original_response()
                    except (discord.Forbidden, discord.HTTPException):
                        pass
            else:
                await ctx.send(GENERIC_ERROR_MESSAGE)

        # Clean up command message
        try:
            await ctx.message.delete()
        except (
            discord.Forbidden,
            discord.HTTPException,
            discord.NotFound,
            AttributeError,
        ):
            pass

    def _cancel_channel_tasks(
        self, channel_id: int, channel_name: str = "", reason: str = ""
    ) -> bool:
        """Cancel active processing and sending tasks for a channel. Returns True if any cancelled."""
        cancelled = False

        if channel_id in self.processing_tasks:
            task = self.processing_tasks[channel_id]
            if not task.done():
                logger.info(
                    "%s interrupted active processing in channel #%s",
                    reason,
                    channel_name,
                )
                task.cancel()
                cancelled = True

        if channel_id in self.sending_tasks:
            task = self.sending_tasks[channel_id]
            if not task.done():
                logger.info(
                    "%s interrupted active sending in channel #%s", reason, channel_name
                )
                task.cancel()
                cancelled = True

        return cancelled

    def _cancel_auto_channel_tasks(self, channel_id: int) -> bool:
        """Cancel tasks in AutoChannelCog for a channel. Returns True if any cancelled."""
        cancelled = False
        auto_cog = self.bot.get_cog("AutoChannelCog")
        if not auto_cog:
            return False

        if channel_id in auto_cog.sending_tasks:
            task = auto_cog.sending_tasks[channel_id]
            if not task.done():
                task.cancel()
                cancelled = True

        if (
            hasattr(auto_cog, "processing_tasks")
            and channel_id in auto_cog.processing_tasks
        ):
            task = auto_cog.processing_tasks[channel_id]
            if not task.done():
                task.cancel()
                cancelled = True

        return cancelled

    @commands.hybrid_command(
        name="abort",
        aliases=["중단", "멈춰"],
        description="진행 중인 모든 메시지 전송 및 처리를 강제로 중단합니다.",
    )
    async def abort_command(self, ctx: commands.Context):
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
        aborted = self._cancel_channel_tasks(
            channel_id, ctx.channel.name, "Abort command"
        )
        aborted = self._cancel_auto_channel_tasks(channel_id) or aborted

        # Send appropriate response
        if aborted:
            await self._send_abort_success(ctx)
            logger.info(
                "User %s requested abort in channel %s", ctx.author.name, channel_id
            )
        else:
            await self._send_abort_no_tasks(ctx)

    async def _send_abort_success(self, ctx: commands.Context) -> None:
        """Send success response for abort command."""
        if ctx.interaction:
            await ctx.reply("🛑 중단되었습니다.", ephemeral=False)
        else:
            await ctx.message.add_reaction("🛑")

    async def _send_abort_no_tasks(self, ctx: commands.Context) -> None:
        """Send no-tasks response for abort command."""
        if ctx.interaction:
            await ctx.reply("❓ 중단할 작업이 없습니다.", ephemeral=True)
        else:
            await ctx.message.add_reaction("❓")

    @commands.hybrid_command(
        name="초기화",
        aliases=["reset"],
        description="현재 채널의 대화 세션을 초기화합니다.",
    )
    async def reset_session(self, ctx: commands.Context):
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
    async def set_temperature(
        self, ctx: commands.Context, value: Optional[float] = None
    ):
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
            await ctx.reply(
                "❌ Temperature는 0.0에서 2.0 사이여야 합니다.", mention_author=False
            )
            return

        try:
            self.llm_service.update_parameters(temperature=value)
            if ctx.interaction:
                await ctx.reply(
                    f"✅ Temperature가 {value}로 설정되었습니다.", ephemeral=False
                )
            else:
                await ctx.message.add_reaction("✅")
        except Exception as e:
            logger.error("Temperature 설정 실패: %s", e, exc_info=True)
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

    @commands.hybrid_command(
        name="topp", description="LLM의 다양성(Top-P)을 설정합니다 (0.0~1.0)."
    )
    @app_commands.describe(value="설정할 Top-P 값 (0.0~1.0)")
    async def set_top_p(self, ctx: commands.Context, value: Optional[float] = None):
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
            await ctx.reply(
                "❌ Top-p는 0.0에서 1.0 사이여야 합니다.", mention_author=False
            )
            return

        try:
            self.llm_service.update_parameters(top_p=value)
            if ctx.interaction:
                await ctx.reply(
                    f"✅ Top-p가 {value}로 설정되었습니다.", ephemeral=False
                )
            else:
                await ctx.message.add_reaction("✅")
        except Exception as e:
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

    @commands.hybrid_command(
        name="끊어치기", description="긴 응답을 나누어 보내는 기능을 켜거나 끕니다."
    )
    @app_commands.describe(mode="모드 설정 (on/off)")
    async def toggle_break_cut(self, ctx: commands.Context, mode: Optional[str] = None):
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
    async def set_thinking_budget(
        self, ctx: commands.Context, value: Optional[str] = None
    ):
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

    async def cog_command_error(self, ctx: commands.Context, error: Exception):
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

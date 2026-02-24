"""Help Cog for Persbot - displays comprehensive bot functionality with dropdown UI."""

import logging
from typing import Optional

import discord
from discord.ext import commands

from persbot.config import AppConfig
from persbot.utils import GENERIC_ERROR_MESSAGE, send_discord_message

logger = logging.getLogger(__name__)


class HelpView(discord.ui.View):
    """View containing the help category dropdown and close button."""

    def __init__(self, cog: "HelpCog", ctx: commands.Context) -> None:
        super().__init__(timeout=600)
        self.cog = cog
        self.ctx = ctx
        self.current_category: Optional[str] = None
        self.message: Optional[discord.Message] = None
        self.update_components()

    def update_components(self) -> None:
        """Update the view components based on current state."""
        self.clear_items()

        # Category dropdown - only show on main menu
        if self.current_category is None:
            select = discord.ui.Select(
                placeholder="도움말 카테고리를 선택하세요...",
                min_values=1,
                max_values=1,
                row=0,
            )
            select.callback = self.on_category_select
            for option in self.cog.get_category_options():
                select.add_option(**option)
            self.add_item(select)

        # Back button - only show when viewing a category
        if self.current_category is not None:
            btn_back = discord.ui.Button(
                label="← 뒤로가기", style=discord.ButtonStyle.secondary, emoji="🔙", row=0
            )
            btn_back.callback = self.on_back
            self.add_item(btn_back)

        # Close button
        btn_close = discord.ui.Button(
            label="닫기", style=discord.ButtonStyle.danger, emoji="❌", row=1
        )
        btn_close.callback = self.on_close
        self.add_item(btn_close)

    async def on_category_select(self, interaction: discord.Interaction) -> None:
        """Handle category selection from dropdown."""
        self.current_category = interaction.data["values"][0]
        await interaction.response.defer()
        await self.refresh_view(interaction)

    async def on_back(self, interaction: discord.Interaction) -> None:
        """Handle back button click."""
        self.current_category = None
        await interaction.response.defer()
        await self.refresh_view(interaction)

    async def on_close(self, interaction: discord.Interaction) -> None:
        """Handle close button click."""
        await interaction.response.defer()
        if self.message:
            await self.message.delete()
        self.stop()

    async def refresh_view(self, interaction: Optional[discord.Interaction] = None) -> None:
        """Refresh the view with updated content."""
        self.update_components()
        embed = self.build_embed()

        try:
            if interaction and not interaction.response.is_done():
                await interaction.response.edit_message(embed=embed, view=self)
            elif self.message:
                await self.message.edit(embed=embed, view=self)
        except Exception:
            logger.exception("Failed to refresh help view")

    def build_embed(self) -> discord.Embed:
        """Build the embed based on current state."""
        if self.current_category:
            return self.build_category_embed()
        return self.build_main_embed()

    def build_main_embed(self) -> discord.Embed:
        """Build the main help menu embed."""
        embed = discord.Embed(
            title="🤖 Persbot 도움말",
            description="Persbot은 다양한 AI 기능을 제공하는 디스코드 봇입니다.\n"
            "아래 드롭다운에서 카테고리를 선택하여 상세 도움말을 확인하세요.",
            color=discord.Color.blurple(),
        )

        # Quick overview of all categories
        categories = self.cog.get_category_summaries()
        for emoji, name, summary in categories:
            embed.add_field(
                name=f"{emoji} {name}",
                value=summary,
                inline=True,
            )

        # System info
        embed.add_field(
            name="🔧 시스템 정보",
            value=f"🟢 온라인 | Discord.py | **{self.cog.ai_provider_label}**",
            inline=False,
        )

        embed.set_footer(text="드롭다운에서 카테고리를 선택하세요.")
        return embed

    def build_category_embed(self) -> discord.Embed:
        """Build embed for a specific category."""
        category_data = self.cog.get_category_data(self.current_category)
        if not category_data:
            return self.build_main_embed()

        embed = discord.Embed(
            title=category_data["title"],
            description=category_data["description"],
            color=category_data["color"],
        )

        # Add tips if available
        if category_data.get("tips"):
            embed.add_field(name="💡 팁", value=category_data["tips"], inline=False)

        embed.set_footer(text="뒤로가기 버튼을 누르면 메인 메뉴로 돌아갑니다.")
        return embed


class HelpCog(commands.Cog):
    """봇의 전체 기능을 설명하는 도움말 Cog"""

    def __init__(self, bot: commands.Bot, config: AppConfig):
        self.bot = bot
        self.config = config
        provider = getattr(config, "assistant_llm_provider", "gemini")
        self.ai_provider_label = self._get_provider_label(provider)

    async def cog_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        """Handle errors in help commands."""
        error = getattr(error, 'original', error)

        if isinstance(error, commands.CommandInvokeError):
            logger.error("Help command error: %s", error, exc_info=True)
            await ctx.send("An error occurred while displaying help.")

    def _get_provider_label(self, provider: str) -> str:
        """AI 공급자 라벨을 반환합니다."""
        provider_lower = str(provider).lower()
        provider_map = {
            "openai": "OpenAI (GPT)",
            "gemini": "Google Gemini",
            "zai": "Z.AI",
        }
        return provider_map.get(provider_lower, provider)

    def get_category_options(self) -> list[dict]:
        """Get dropdown options for categories."""
        return [
            {
                "label": "💬 대화",
                "value": "대화",
                "description": "AI 대화 기능 및 명령어",
                "emoji": "💬",
            },
            {
                "label": "📊 요약",
                "value": "요약",
                "description": "채팅 내용 요약 기능",
                "emoji": "📊",
            },
            {
                "label": "🎭 페르소나",
                "value": "페르소나",
                "description": "AI 캐릭터 설정 및 관리",
                "emoji": "🎭",
            },
            {
                "label": "🤖 모델",
                "value": "모델",
                "description": "AI 모델 선택 방법",
                "emoji": "🤖",
            },
            {
                "label": "⚙️ 설정",
                "value": "설정",
                "description": "봇 동작 설정 명령어",
                "emoji": "⚙️",
            },
            {
                "label": "🔔 자동채널",
                "value": "자동채널",
                "description": "자동 응답 채널 관리",
                "emoji": "🔔",
            },
        ]

    def get_category_summaries(self) -> list[tuple[str, str, str]]:
        """Get quick summary for each category."""
        return [
            ("💬", "대화", "@멘션으로 AI와 대화\n`!retry`, `!stop`, `!초기화`"),
            ("📊", "요약", "채팅 내용 요약\n시간/메시지 ID 기반"),
            ("🎭", "페르소나", "AI 캐릭터 설정\n`!prompt`로 관리"),
            ("🤖", "모델", "LLM/이미지 모델 선택\n`!model`로 변경"),
            ("⚙️", "설정", "창의성/다양성 조절\n`!temp`, `!topp` 등"),
            ("🔔", "자동채널", "자동 응답 채널\n`!자동채널 등록`"),
        ]

    def get_category_data(self, category: str) -> Optional[dict]:
        """Get detailed data for a category."""
        category_map = {
            "대화": {
                "title": "💬 대화 기능 상세 도움말",
                "description": self._get_conversation_help(),
                "color": discord.Color.blue(),
                "tips": "• 페르소나를 설정하면 일관된 캐릭터로 대화합니다\n• 자동응답 채널에서는 멘션 없이도 대화 가능합니다",
            },
            "요약": {
                "title": "📊 요약 기능 상세 도움말",
                "description": self._get_summary_help(),
                "color": discord.Color.gold(),
                "tips": "• 메시지 우클릭 → 'ID 복사'로 메시지 ID를 쉽게 복사할 수 있습니다\n• 답글 기능을 활용하면 ID를 복사할 필요가 없습니다",
            },
            "페르소나": {
                "title": "🎭 페르소나 기능 상세 도움말",
                "description": self._get_persona_help(),
                "color": discord.Color.purple(),
                "tips": "• 잘 만든 페르소나를 .txt 파일로 저장해두면 나중에 재사용할 수 있습니다\n• 하루 최대 2개의 페르소나를 생성할 수 있습니다",
            },
            "모델": {
                "title": "🤖 모델 선택 상세 도움말",
                "description": self._get_model_help(),
                "color": discord.Color.green(),
                "tips": "• 모델 선택은 채널별로 적용됩니다\n• 일일 사용 한도를 초과하면 자동으로 대체 모델이 사용됩니다",
            },
            "설정": {
                "title": "⚙️ 설정 상세 도움말",
                "description": self._get_settings_help(),
                "color": discord.Color.orange(),
                "tips": "• 일부 설정 명령어는 관리자 권한이 필요할 수 있습니다\n• Temperature가 높을수록 더 창의적이지만 예측하기 어려워집니다",
            },
            "자동채널": {
                "title": "🔔 자동채널 상세 도움말",
                "description": self._get_auto_channel_help(),
                "color": discord.Color.red(),
                "tips": "• 자동 응답 채널에서도 `!`로 시작하는 명령어는 정상 작동합니다\n• `!@` 또는 `!undo`로 실수로 보낸 메시지를 취소할 수 있습니다",
            },
        }
        return category_map.get(category)

    def _get_conversation_help(self) -> str:
        """Get conversation help text."""
        return (
            "**기본 사용법:**\n"
            "봇을 멘션(@mention)하면 AI가 대화에 응답합니다.\n"
            "예: `@Persbot 안녕! 오늘 날씨 어때?`\n\n"
            "**명령어:**\n"
            "• `!retry` (또는 `!다시`, `!재생성`) - 마지막 답변을 다시 생성합니다.\n"
            "• `!stop` (또는 `!중단`, `!멈춰`, `!abort`) - 진행 중인 응답을 중단합니다.\n"
            "• `!초기화` (또는 `!reset`) - 현재 채널의 대화 내용을 초기화합니다.\n\n"
            "**특징:**\n"
            "• 자연스러운 한국어 대화\n"
            "• 복잡한 질문도 이해\n"
            "• 페르소나 설정 시 캐릭터 유지"
        )

    def _get_summary_help(self) -> str:
        """Get summary help text."""
        return (
            "**기본 사용법:**\n"
            "• `!요약` - 최근 30분 동안의 대화를 요약합니다.\n\n"
            "**시간 지정:**\n"
            "• `!요약 20분` - 최근 20분 요약\n"
            "• `!요약 1시간` - 최근 1시간 요약\n"
            "• `!요약 1시간30분` - 복합 시간도 지원\n\n"
            "**메시지 ID 기반:**\n"
            "• `!요약 1234567890 이후` - 해당 메시지 이후 전체 요약\n"
            "• `!요약 1234567890 이후 30분` - 메시지 이후 30분 요약\n"
            "• `!요약 1234567890 이전 1시간` - 메시지 이전 1시간 요약\n\n"
            "**답글 사용법:**\n"
            "메시지에 답글하여 `!요약 이후` 또는 `!요약 이후 30분` 사용 가능\n\n"
            "**팁:**\n"
            "• 메시지 ID는 17-20자리 숫자입니다 (메시지 우클릭 → ID 복사)"
        )

    def _get_persona_help(self) -> str:
        """Get persona help text."""
        return (
            "**기본 사용법:**\n"
            "• `!prompt` - 페르소나 관리 UI를 엽니다.\n\n"
            "**페르소나 관리 UI 기능:**\n"
            "• **새로 만들기** - AI가 자동으로 페르소나 생성\n"
            "  - ⚡ 기본 모드: 컨셉만 입력하여 빠르게 생성\n"
            "  - 🧠 AI 질문 모드: AI가 질문하고 답변으로 상세 커스텀\n"
            "• **프롬프트 추가(파일)** - .txt 파일로 페르소나 업로드\n"
            "• **채널에 적용** - 선택한 페르소나를 현재 채널에 적용\n"
            "• **이름 변경** - 페르소나 이름 수정\n"
            "• **삭제** - 페르소나 삭제 (관리자 권한 필요)\n\n"
            "**제한:**\n"
            "• 하루 최대 2개의 페르소나 생성 가능"
        )

    def _get_model_help(self) -> str:
        """Get model help text."""
        return (
            "**기본 사용법:**\n"
            "• `!model` (또는 `!모델`) - LLM 모델 선택 UI 표시\n\n"
            "**하위 명령어:**\n"
            "• `!model llm` - 대화용 LLM 모델 선택 (드롭다운 UI)\n"
            "• `!model image` (또는 `!모델 이미지`) - 이미지 생성 모델 선택\n\n"
            "**사용 가능한 모델:**\n"
            "• 🤖 Gemini 모델군\n"
            "• 🧠 OpenAI 모델군\n"
            "• ⚡ Z.AI 모델군\n\n"
            "**일일 사용 한도:**\n"
            "• 모델마다 일일 사용 횟수 제한이 있음\n"
            "• 한도 초과 시 자동으로 대체 모델 사용"
        )

    def _get_settings_help(self) -> str:
        """Get settings help text."""
        return (
            "**명령어:**\n\n"
            "• `!temp [값]` - AI 창의성 조절 (0.0~2.0)\n"
            "  • 값이 높을수록 더 창의적이고 예측 불가능\n"
            "  • 1.0이 기본값\n"
            "  • 예: `!temp 0.7`, `!temp 1.2`\n\n"
            "• `!topp [값]` - AI 다양성 조절 (0.0~1.0)\n"
            "  • 높을수록 더 다양한 단어 사용\n"
            "  • 1.0이 기본값\n\n"
            "• `!끊어치기 [on|off]` - 실시간 메시지 전송 모드\n"
            "  • ON: 긴 응답을 여러 조각으로 나누어 실시간 전송\n"
            "  • OFF: 응답 완료 후 한 번에 전송\n\n"
            "• `!delay [초]` - 메시지 버퍼 대기 시간 (0~60초)\n"
            "  • 여러 메시지를 모았다가 한 번에 처리하는 시간\n"
            "  • 기본값: 3초"
        )

    def _get_auto_channel_help(self) -> str:
        """Get auto channel help text."""
        return (
            "**자동 응답 채널이란?**\n"
            "봇을 멘션하지 않아도 자동으로 응답하는 채널입니다.\n\n"
            "**명령어:**\n"
            "• `!자동채널 등록` (또는 `!자동채널 add`, `!자동채널 register`)\n"
            "  - 현재 채널을 자동 응답 채널로 등록합니다.\n"
            "  - 관리자 권한 필요\n\n"
            "• `!자동채널 해제` (또는 `!자동채널 remove`, `!자동채널 unregister`)\n"
            "  - 현재 채널의 자동 응답을 해제합니다.\n"
            "  - 관리자 권한 필요\n\n"
            "**메시지 취소:**\n"
            "• `!@ [숫자]` (또는 `!undo [숫자]`)\n"
            "  - 마지막 N개의 대화 쌍을 취소합니다.\n"
            "  - 예: `!@ 1`, `!undo 2`\n"
            "  - 5회 이상 대화한 사용자 또는 관리자만 사용 가능"
        )

    @commands.hybrid_command(
        name="help",
        aliases=["도움말", "h"],
        description="봇의 전체 기능을 설명하는 도움말을 표시합니다.",
    )
    async def show_help(self, ctx: commands.Context) -> None:
        """봇의 전체 기능을 설명하는 도움말을 표시합니다.

        사용법: !도움말
        """
        try:
            view = HelpView(self, ctx)
            embed = view.build_main_embed()

            # Send the initial message
            sent = await send_discord_message(
                ctx, "", embed=embed, view=view, mention_author=False
            )

            if sent:
                view.message = sent[0]

        except Exception:
            logger.exception("Failed to show help")
            await ctx.reply(
                GENERIC_ERROR_MESSAGE,
                mention_author=False,
            )

    @commands.hybrid_command(
        name="features", aliases=["기능", "f"], description="봇의 주요 기능을 간단히 설명합니다."
    )
    async def show_features(self, ctx: commands.Context) -> None:
        """봇의 주요 기능을 간단히 설명합니다.

        사용법: !기능
        """
        try:
            embed = discord.Embed(
                title="🌟 Persbot의 주요 기능",
                description="Persbot은 다양한 AI 기능을 제공하는 디스코드 봇입니다.",
                color=discord.Color.green(),
            )

            features = [
                ("💬 AI 대화", f"@멘션으로 {self.ai_provider_label} AI와 자연스러운 대화"),
                ("📊 요약 기능", "다양한 조건으로 채팅 내용 요약 (시간, 메시지 ID 등)"),
                ("🎭 페르소나", "AI 캐릭터 설정으로 일관된 성격 유지"),
                ("🤖 모델 선택", "LLM/이미지 모델을 상황에 맞게 선택 가능"),
                ("⚙️ 다양한 설정", "창의성, 다양성, 전송 모드 등 세밀한 조정"),
                ("🔔 자동 응답 채널", "멘션 없이도 자동으로 응답하는 채널 설정"),
                ("🌐 한국어 지원", "편리한 한글 명령어와 상세한 안내"),
            ]

            for title, description in features:
                embed.add_field(name=title, value=description, inline=False)

            embed.add_field(
                name="📖 더 알아보기",
                value="`!도움말`로 전체 명령어를 확인하세요.\n드롭다운 메뉴로 카테고리별 상세 도움말을 볼 수 있습니다.",
                inline=False,
            )

            embed.set_footer(text="Persbot | Advanced AI Discord Bot")
            await ctx.reply(embed=embed, mention_author=False)

        except Exception:
            logger.exception("Failed to show features")
            await ctx.reply(
                GENERIC_ERROR_MESSAGE,
                mention_author=False,
            )



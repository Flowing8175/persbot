"""Help Cog for Persbot - displays comprehensive bot functionality."""

import logging
from typing import Optional

import discord
from config import AppConfig
from discord.ext import commands
from utils import GENERIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)


class HelpCog(commands.Cog):
    """봇의 전체 기능을 설명하는 도움말 Cog"""

    def __init__(self, bot: commands.Bot, config: AppConfig):
        self.bot = bot
        self.config = config
        provider = getattr(config, "assistant_llm_provider", "gemini")
        self.ai_provider_label = self._get_provider_label(provider)

    def _get_provider_label(self, provider: str) -> str:
        """AI 공급자 라벨을 반환합니다."""
        provider_lower = str(provider).lower()
        provider_map = {
            "openai": "OpenAI (GPT)",
            "gemini": "Google Gemini",
            "zai": "Z.AI",
        }
        return provider_map.get(provider_lower, provider)

    @commands.hybrid_command(
        name="help",
        aliases=["도움말", "h"],
        description="봇의 전체 기능을 설명하는 도움말을 표시합니다.",
    )
    @discord.app_commands.describe(
        category="도움말을 볼 특정 카테고리 (대화, 요약, 페르소나, 모델, 설정, 자동채널)"
    )
    async def show_help(self, ctx: commands.Context, category: Optional[str] = None) -> None:
        """봇의 전체 기능을 설명하는 도움말을 표시합니다.

        사용법: !도움말 [카테고리]
        예: !도움말 요약, !도움말 페르소나
        """
        try:
            if category:
                category = category.lower().strip()

            # Display specific help for requested category
            if category:
                await self._show_category_help(ctx, category)
                return

            # Create main help embed
            embed = discord.Embed(
                title="🤖 Persbot 도움말",
                description="Persbot은 다양한 AI 기능을 제공하는 디스코드 봇입니다.\n"
                "`!도움말 [카테고리]`로 각 카테고리의 상세 도움말을 확인하세요.",
                color=discord.Color.blurple(),
            )

            # AI Conversation Features
            embed.add_field(
                name="💬 대화",
                value="**봇을 멘션(@mention)하여 대화**\n"
                "• 자연스러운 대화\n"
                "• `!retry` - 마지막 답변 재생성\n"
                "• `!stop` - 진행 중인 응답 중단\n"
                "• `!초기화` - 대화 내용 초기화\n\n"
                "**상세:** `!도움말 대화`",
                inline=True,
            )

            # Summarization
            embed.add_field(
                name="📊 요약",
                value="**채팅 내용 요약**\n"
                "• `!요약` - 최근 30분 요약\n"
                "• `!요약 [시간]` - 지정 시간 요약\n"
                "• 메시지 ID 기반 요약 지원\n\n"
                "**상세:** `!도움말 요약`",
                inline=True,
            )

            # Persona Management
            embed.add_field(
                name="🎭 페르소나",
                value="**AI 캐릭터 설정**\n"
                "• `!prompt` - 페르소나 관리 UI\n"
                "• 캐릭터 생성/적용/관리\n"
                "• AI 질문 모드 지원\n\n"
                "**상세:** `!도움말 페르소나`",
                inline=True,
            )

            # Model Selection
            embed.add_field(
                name="🤖 모델",
                value="**AI 모델 선택**\n"
                "• `!model llm` - LLM 모델 선택\n"
                "• `!model image` - 이미지 모델 선택\n"
                "• 드롭다운 UI로 쉬운 선택\n\n"
                "**상세:** `!도움말 모델`",
                inline=True,
            )

            # Settings
            embed.add_field(
                name="⚙️ 설정",
                value="**봇 동작 설정**\n"
                "• `!temp` - 창의성 조절 (0.0~2.0)\n"
                "• `!topp` - 다양성 조절 (0.0~1.0)\n"
                "• `!끊어치기` - 실시간 전송 모드\n"
                "• `!delay` - 버퍼 대기 시간\n\n"
                "**상세:** `!도움말 설정`",
                inline=True,
            )

            # Auto Channel
            embed.add_field(
                name="🔔 자동채널",
                value="**자동 응답 채널**\n"
                "• `!자동채널 등록` - 자동응답 활성화\n"
                "• `!자동채널 해제` - 자동응답 비활성화\n"
                "• `!@` 또는 `!undo` - 메시지 취소\n\n"
                "**상세:** `!도움말 자동채널`",
                inline=True,
            )

            # Tips
            embed.add_field(
                name="💡 팁",
                value="• 명령어는 대소문자를 구분하지 않습니다\n"
                "• 대부분의 명령어는 별칭(alias)을 지원합니다\n"
                "• 자동응답 채널에서는 멘션 없이도 대화 가능합니다",
                inline=False,
            )

            # System Status
            embed.add_field(
                name="🔧 시스템 정보",
                value=f"🟢 온라인 | Discord.py | **{self.ai_provider_label}**",
                inline=False,
            )

            embed.set_footer(
                text="자세한 정보가 필요하면 !도움말 [카테고리]를 입력하세요."
            )

            await ctx.reply(embed=embed, mention_author=False)

        except Exception as e:
            logger.error(f"Failed to show help: {e}")
            await ctx.reply(
                GENERIC_ERROR_MESSAGE,
                mention_author=False,
            )

    async def _show_category_help(self, ctx: commands.Context, category: str) -> None:
        """Display help for a specific category.

        Args:
            ctx: Command context
            category: Category name (대화, 요약, 페르소나, 모델, 설정, 자동채널)
        """
        category_helps = {
            "대화": {
                "title": "💬 대화 기능 상세 도움말",
                "description": self._get_conversation_help(),
                "color": discord.Color.blue(),
            },
            "요약": {
                "title": "📊 요약 기능 상세 도움말",
                "description": self._get_summary_help(),
                "color": discord.Color.gold(),
            },
            "페르소나": {
                "title": "🎭 페르소나 기능 상세 도움말",
                "description": self._get_persona_help(),
                "color": discord.Color.purple(),
            },
            "모델": {
                "title": "🤖 모델 선택 상세 도움말",
                "description": self._get_model_help(),
                "color": discord.Color.green(),
            },
            "설정": {
                "title": "⚙️ 설정 상세 도움말",
                "description": self._get_settings_help(),
                "color": discord.Color.orange(),
            },
            "자동채널": {
                "title": "🔔 자동채널 상세 도움말",
                "description": self._get_auto_channel_help(),
                "color": discord.Color.red(),
            },
        }

        if category in category_helps:
            info = category_helps[category]
            embed = discord.Embed(
                title=info["title"],
                description=info["description"],
                color=info["color"],
            )
            embed.set_footer(text="전체 도움말은 !도움말로 확인하세요.")
            await ctx.reply(embed=embed, mention_author=False)
        else:
            # Unknown category, show available options
            available = ", ".join(category_helps.keys())
            embed = discord.Embed(
                title="❓ 알 수 없는 카테고리",
                description=f"인식할 수 없는 카테고리입니다.\n\n**사용 가능한 카테고리:**\n`{available}`",
                color=discord.Color.red(),
            )
            embed.set_footer(text="!도움말로 전체 도움말을 확인하세요.")
            await ctx.reply(embed=embed, mention_author=False)

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
            "• 하루 최대 2개의 페르소나 생성 가능\n\n"
            "**팁:**\n"
            "• 잘 만든 페르소나를 .txt로 저장해두면 나중에 재사용 가능"
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
            "• 한도 초과 시 자동으로 대체 모델 사용\n\n"
            "**팁:**\n"
            "• 모델 선택은 채널별로 적용됩니다"
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
            "  • 기본값: 3초\n\n"
            "**팁:**\n"
            "• 설정 명령어는 관리자 권한이 필요할 수 있습니다"
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
            "  - 5회 이상 대화한 사용자 또는 관리자만 사용 가능\n\n"
            "**팁:**\n"
            "• 자동 응답 채널에서도 `!`로 시작하는 명령어는 정상 작동합니다"
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
                value="`!도움말`로 전체 명령어를 확인하거나\n`!도움말 [카테고리]`로 상세 도움말을 보세요.",
                inline=False,
            )

            embed.set_footer(text="Persbot | Advanced AI Discord Bot")
            await ctx.reply(embed=embed, mention_author=False)

        except Exception as e:
            logger.error(f"Failed to show features: {e}")
            await ctx.reply(
                GENERIC_ERROR_MESSAGE,
                mention_author=False,
            )


async def setup(bot: commands.Bot) -> None:
    """Setup function for loading the cog."""
    pass

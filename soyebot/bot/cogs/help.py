"""Help Cog for SoyeBot - displays comprehensive bot functionality."""

import discord
from discord.ext import commands
import logging

from utils import GENERIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)


class HelpCog(commands.Cog):
    """봇의 전체 기능을 설명하는 도움말 Cog"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.command(name='도움말', aliases=['help', 'h'])
    async def show_help(self, ctx: commands.Context, *args):
        """봇의 전체 기능을 설명하는 도움말을 표시합니다.

        사용법: !도움말 [기능명]
        예: !도움말 요약, !도움말 ai
        """
        try:
            # Parse subcommand argument
            subcommand = ' '.join(args).lower().strip() if args else None

            # Display specific help for requested feature
            if subcommand:
                await self._show_specific_help(ctx, subcommand)
                return
            # Create main help embed
            embed = discord.Embed(
                title="🤖 SoyeBot 도움말",
                description="이 봇이 할 수 있는 모든 기능을 안내합니다.",
                color=discord.Color.blurple(),
            )

            # AI Assistant Features
            embed.add_field(
                name="💬 AI 어시스턴트 기능",
                value="봇을 멘션(@mention)하면 AI가 대화합니다.\n"
                      "• 자연스러운 대화\n"
                      "• 각 @mention마다 새로운 대화 시작 (메모리 최적화)\n"
                      "• 복잡한 질문 처리\n\n"
                      "**사용법:** `@SoyeBot 안녕! 오늘 날씨 어때?`\n"
                      "**참고:** 이전 대화 내역은 유지되지 않습니다 (1GB RAM 최적화)",
                inline=False,
            )

            # Summarization Commands
            embed.add_field(
                name="📊 요약 명령어",
                value="**`!요약`** - 최근 30분 요약\n"
                      "예: `!요약`\n\n"
                      "**`!요약 <시간>`** - 지정된 시간만큼 요약\n"
                      "예: `!요약 20분`, `!요약 1시간`\n\n"
                      "**`!요약 <메시지ID> 이후`** - 메시지 ID 이후부터 최대 길이까지 요약\n"
                      "예: `!요약 1234567890 이후`\n\n"
                      "**`!요약 <메시지ID> <이후|이전> <시간>`** - 시간 범위 요약\n"
                      "예: `!요약 1234567890 이후 30분`, `!요약 1234567890 이전 1시간`",
                inline=False,
            )

            # Advanced Features
            embed.add_field(
                name="✨ 고급 기능",
                value="**고정 프롬프트 페르소나:** 캐릭터 일관성을 유지합니다\n"
                      "**상호작용 분석:** 사용자 기본 통계를 추적합니다",
                inline=False,
            )

            # Tips and Tricks
            embed.add_field(
                name="💡 팁",
                value="• 명령어는 대소문자를 구분하지 않습니다\n"
                      "• 많은 명령어가 별칭(alias)을 지원합니다\n"
                      "• @mention 대화는 항상 새로운 세션으로 처리됩니다",
                inline=False,
            )

            # System Status
            embed.add_field(
                name="🔧 시스템 정보",
                value=f"봇 상태: 🟢 온라인\n"
                      f"프레임워크: Discord.py\n"
                      f"AI 엔진: Google Gemini API\n\n"
                      f"📊 **성능 모니터링:**\n"
                      f"실시간 메트릭 대시보드: http://localhost:5000\n"
                      f"(메모리, CPU, API 응답시간, 성공률 등)",
                inline=False,
            )

            embed.set_footer(
                text="더 자세한 정보가 필요하면 각 명령어 앞에 !도움말을 붙이세요. 예: !도움말 요약"
            )

            await ctx.reply(embed=embed, mention_author=False)
            logger.info(f"Help command requested by {ctx.author.name}")

        except Exception as e:
            logger.error(f"Failed to show help: {e}")
            await ctx.reply(
                GENERIC_ERROR_MESSAGE,
                mention_author=False,
            )

    async def _show_specific_help(self, ctx: commands.Context, feature: str):
        """Display help for a specific feature.

        Args:
            ctx: Command context
            feature: Feature name (요약, ai, etc.)
        """
        feature_helps = {
            '요약': {
                'title': '📊 요약 명령어',
                'content': (
                    "**`!요약`** - 최근 30분 요약\n"
                    "예: `!요약`\n\n"
                    "**`!요약 <시간>`** - 지정된 시간만큼 요약\n"
                    "예: `!요약 20분`, `!요약 1시간`, `!요약 1시간30분`\n\n"
                    "**`!요약 <메시지ID> 이후`** - 메시지 ID 이후부터 최대 길이까지 요약\n"
                    "예: `!요약 1234567890 이후`\n"
                    "메시지 ID는 17-20자리 숫자입니다.\n\n"
                    "**`!요약 <메시지ID> 이후 <시간>`** - 메시지 ID 이후 지정된 시간만큼 요약\n"
                    "예: `!요약 1234567890 이후 30분`\n\n"
                    "**`!요약 <메시지ID> 이전 <시간>`** - 메시지 ID 이전 지정된 시간만큼 요약\n"
                    "예: `!요약 1234567890 이전 1시간`"
                ),
                'color': discord.Color.gold(),
            },
            'ai': {
                'title': '💬 AI 어시스턴트 기능',
                'content': (
                    "봇을 멘션(@mention)하면 AI가 대화합니다.\n\n"
                    "**기능:**\n"
                    "• 자연스러운 대화\n"
                    "• 복잡한 질문 처리\n"
                    "• 고정된 캐릭터 페르소나 유지\n\n"
                    "**사용법:** `@SoyeBot 안녕! 오늘 날씨 어때?`\n\n"
                    "**팁:**\n"
                    "• 각 멘션은 독립 세션입니다\n"
                    "• 자연스러운 한국어로 대화할 수 있습니다"
                ),
                'color': discord.Color.purple(),
            },
        }

        if feature in feature_helps:
            info = feature_helps[feature]
            embed = discord.Embed(
                title=info['title'],
                description=info['content'],
                color=info['color'],
            )
            embed.set_footer(text="더 궁금한 점은 !도움말 전체로 전체 도움말을 확인하세요.")
            await ctx.reply(embed=embed, mention_author=False)
        else:
            # Unknown feature, show available options
            available = ', '.join(feature_helps.keys())
            embed = discord.Embed(
                title="❓ 알 수 없는 기능",
                description=f"인식할 수 없는 기능입니다.\n\n**사용 가능한 옵션:**\n{available}",
                color=discord.Color.red(),
            )
            await ctx.reply(embed=embed, mention_author=False)

    @commands.command(name='기능', aliases=['features', 'f'])
    async def show_features(self, ctx: commands.Context):
        """봇의 주요 기능을 간단히 설명합니다.

        사용법: !기능
        """
        try:
            embed = discord.Embed(
                title="🌟 SoyeBot의 주요 기능",
                color=discord.Color.green(),
            )

            features = [
                ("🤖 AI 대화", "봇을 멘션하면 Google Gemini API를 통한 AI와 대화"),
                ("📝 요약 기능", "채팅 내용을 자동으로 요약"),
                ("📊 통계 분석", "상호작용 패턴과 기본적인 선호 주제 분석"),
                ("🌐 다국어 지원", "한글 명령어와 안내말"),
            ]

            for title, description in features:
                embed.add_field(name=title, value=description, inline=False)

            embed.set_footer(text="전체 도움말은 !도움말 명령어로 확인하세요.")
            await ctx.reply(embed=embed, mention_author=False)

        except Exception as e:
            logger.error(f"Failed to show features: {e}")
            await ctx.reply(
                GENERIC_ERROR_MESSAGE,
                mention_author=False,
            )


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    pass

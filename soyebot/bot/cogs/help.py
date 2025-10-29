"""Help Cog for SoyeBot - displays comprehensive bot functionality."""

import discord
from discord.ext import commands
import logging

logger = logging.getLogger(__name__)


class HelpCog(commands.Cog):
    """봇의 전체 기능을 설명하는 도움말 Cog"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.command(name='도움말', aliases=['help', 'h'])
    async def show_help(self, ctx: commands.Context, *args):
        """봇의 전체 기능을 설명하는 도움말을 표시합니다.

        사용법: !도움말 [기능명]
        예: !도움말 기억, !도움말 요약, !도움말 ai
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
                      "• 기억 기능과 통합\n"
                      "• 복잡한 질문 처리\n\n"
                      "**사용법:** `@SoyeBot 안녕! 오늘 날씨 어때?`",
                inline=False,
            )

            # Memory Management Commands
            embed.add_field(
                name="📝 기억 관리 명령어",
                value="**`!기억 [내용]`** - 내용을 기억에 저장\n"
                      "예: `!기억 나는 파이썬을 좋아한다`\n\n"
                      "**`!기억목록`** - 저장된 모든 기억 조회\n"
                      "예: `!기억목록`\n\n"
                      "**`!기억삭제 [ID]`** - 특정 기억 삭제\n"
                      "예: `!기억삭제 123`\n\n"
                      "**`!기억초기화`** - 모든 기억 초기화 (확인 필요)\n"
                      "예: `!기억초기화`\n\n"
                      "**`!기억설정 [모드]`** - 검색 모드 설정\n"
                      "모드: `inject_all` 또는 `semantic_search`\n"
                      "예: `!기억설정 semantic_search`",
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
                value="**기억 통합:** AI 어시스턴트가 저장된 기억을 활용합니다\n"
                      "**의미론적 검색:** 관련된 기억을 자동으로 찾습니다\n"
                      "**상호작용 분석:** 당신의 선호와 패턴을 분석합니다",
                inline=False,
            )

            # Tips and Tricks
            embed.add_field(
                name="💡 팁",
                value="• 명령어는 대소문자를 구분하지 않습니다\n"
                      "• 많은 명령어가 별칭(alias)을 지원합니다\n"
                      "• 예: `!기억`, `!save`, `!memory` 모두 같은 명령어\n"
                      "• AI 어시스턴트와의 대화는 자동으로 기억됩니다",
                inline=False,
            )

            # System Status
            embed.add_field(
                name="🔧 시스템 정보",
                value=f"봇 상태: 🟢 온라인\n"
                      f"프레임워크: Discord.py\n"
                      f"AI 엔진: Google Gemini API",
                inline=False,
            )

            embed.set_footer(
                text="더 자세한 정보가 필요하면 각 명령어 앞에 !도움말을 붙이세요. 예: !도움말 기억"
            )

            await ctx.reply(embed=embed, mention_author=False)
            logger.info(f"Help command requested by {ctx.author.name}")

        except Exception as e:
            logger.error(f"Failed to show help: {e}")
            await ctx.reply(
                f"❌ 도움말을 표시하는 중 오류가 발생했습니다: {e}",
                mention_author=False,
            )

    async def _show_specific_help(self, ctx: commands.Context, feature: str):
        """Display help for a specific feature.

        Args:
            ctx: Command context
            feature: Feature name (기억, 요약, ai, etc.)
        """
        feature_helps = {
            '기억': {
                'title': '📝 기억 관리 명령어',
                'content': (
                    "**`!기억 [내용]`** - 내용을 기억에 저장\n"
                    "예: `!기억 나는 파이썬을 좋아한다`\n\n"
                    "**`!기억목록`** - 저장된 모든 기억 조회\n"
                    "예: `!기억목록`\n\n"
                    "**`!기억삭제 [ID]`** - 특정 기억 삭제\n"
                    "예: `!기억삭제 123`\n\n"
                    "**`!기억초기화`** - 모든 기억 초기화 (확인 필요)\n"
                    "예: `!기억초기화`\n\n"
                    "**`!기억설정 [모드]`** - 검색 모드 설정\n"
                    "모드: `inject_all` 또는 `semantic_search`\n"
                    "예: `!기억설정 semantic_search`\n\n"
                    "**`!기억통계`** - 기억 통계 조회\n"
                    "예: `!기억통계`"
                ),
                'color': discord.Color.blue(),
            },
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
                    "• 기억 기능과 통합\n"
                    "• 복잡한 질문 처리\n"
                    "• 추가 정보 학습\n\n"
                    "**사용법:** `@SoyeBot 안녕! 오늘 날씨 어때?`\n\n"
                    "**팁:**\n"
                    "• 대화는 자동으로 저장됩니다\n"
                    "• AI가 당신의 기억을 활용합니다\n"
                    "• 자연스러운 한국어로 대화할 수 있습니다"
                ),
                'color': discord.Color.purple(),
            },
            '검색': {
                'title': '🔍 의미론적 검색',
                'content': (
                    "저장된 기억 중 관련된 내용을 자동으로 찾습니다.\n\n"
                    "**기능:**\n"
                    "• 자동 의미론적 검색: AI가 관련 기억을 찾습니다\n"
                    "• 전체 기억 주입: 모든 기억을 항상 제공합니다\n\n"
                    "**설정 방법:**\n"
                    "`!기억설정 semantic_search` - 의미론적 검색 사용\n"
                    "`!기억설정 inject_all` - 모든 기억 항상 제공\n\n"
                    "**추천:**\n"
                    "• 기억이 적으면: `inject_all`\n"
                    "• 기억이 많으면: `semantic_search`"
                ),
                'color': discord.Color.green(),
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
                ("💾 기억 시스템", "사용자와의 대화 내용을 저장하고 나중에 활용"),
                ("📝 요약 기능", "채팅 내용을 자동으로 요약"),
                ("🔍 의미론적 검색", "저장된 기억 중 관련된 내용을 자동으로 찾음"),
                ("📊 통계 분석", "상호작용 패턴과 선호 주제 분석"),
                ("🌐 다국어 지원", "한글 명령어와 안내말"),
            ]

            for title, description in features:
                embed.add_field(name=title, value=description, inline=False)

            embed.set_footer(text="전체 도움말은 !도움말 명령어로 확인하세요.")
            await ctx.reply(embed=embed, mention_author=False)

        except Exception as e:
            logger.error(f"Failed to show features: {e}")
            await ctx.reply(
                f"❌ 기능 정보를 표시하는 중 오류가 발생했습니다: {e}",
                mention_author=False,
            )


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    pass

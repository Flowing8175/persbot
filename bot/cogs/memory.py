"""Memory Management Cog for SoyeBot."""

import asyncio
import discord
from discord.ext import commands
import logging
from typing import Optional

from config import AppConfig
from services.database_service import DatabaseService
from services.memory_service import MemoryService
from utils import DiscordUI

logger = logging.getLogger(__name__)


class MemoryCog(commands.Cog):
    """사용자 기억 관리 명령어를 처리하는 Cog"""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        db_service: DatabaseService,
        memory_service: MemoryService,
    ):
        self.bot = bot
        self.config = config
        self.db_service = db_service
        self.memory_service = memory_service

    @commands.command(name='기억', aliases=['memory', 'save'])
    async def save_memory(self, ctx: commands.Context, *, content: str):
        """사용자가 기억을 수동으로 저장합니다.

        사용법: !기억 [내용]
        예시: !기억 나는 프로그래밍을 좋아한다
        """
        if not self.config.enable_memory_system:
            await ctx.reply("❌ 메모리 시스템이 비활성화되어 있습니다.", mention_author=False)
            return

        try:
            if not content or not content.strip():
                await ctx.reply("❌ 기억할 내용을 입력해주세요.", mention_author=False)
                return

            # Save to unified memory with high importance
            self.memory_service.save_memory(
                memory_type='key_memory',
                content=content,
                importance_score=0.8,
            )

            await ctx.reply(f"✓ 기억했어요! 나중에 잊지 않을게요.\n> {content}", mention_author=False)
            logger.info(f"User {ctx.author.id} saved memory: {content[:50]}")

        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            await ctx.reply(f"❌ 기억을 저장하는 중 오류가 발생했습니다: {e}", mention_author=False)

    @commands.command(name='기억목록', aliases=['memories', 'list'])
    async def list_memories(self, ctx: commands.Context):
        """사용자의 기억 목록을 표시합니다.

        사용법: !기억목록
        """
        if not self.config.enable_memory_system:
            await ctx.reply("❌ 메모리 시스템이 비활성화되어 있습니다.", mention_author=False)
            return

        try:
            memories = self.db_service.get_memories(limit=50)

            if not memories:
                await ctx.reply("아직 저장된 기억이 없어요.", mention_author=False)
                return

            # Reverse to show oldest first (natural chronological order)
            memories = list(reversed(memories))

            # Group by type
            by_type = {}
            for memory in memories:
                mem_type = memory.memory_type
                if mem_type not in by_type:
                    by_type[mem_type] = []
                by_type[mem_type].append(memory)

            # Build embed
            embed = discord.Embed(
                title="📝 당신의 기억들",
                description=f"총 {len(memories)}개의 기억을 가지고 있어요!",
                color=discord.Color.blue(),
            )

            for mem_type, type_memories in by_type.items():
                memory_lines = []
                for mem in type_memories[:10]:  # Show max 10 per type
                    content_preview = mem.content[:50] + "..." if len(mem.content) > 50 else mem.content
                    memory_lines.append(f"• [{mem.id}] {content_preview}")

                if len(type_memories) > 10:
                    memory_lines.append(f"... 그리고 {len(type_memories) - 10}개 더")

                embed.add_field(
                    name=f"{mem_type.title()} ({len(type_memories)}개)",
                    value="\n".join(memory_lines),
                    inline=False,
                )

            embed.set_footer(text="!기억삭제 [ID]로 특정 기억을 삭제할 수 있어요.")
            await ctx.reply(embed=embed, mention_author=False)

        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            await ctx.reply(f"❌ 기억 목록을 불러오는 중 오류가 발생했습니다: {e}", mention_author=False)

    @commands.command(name='기억삭제', aliases=['forget', 'delete'])
    async def delete_memory(self, ctx: commands.Context, memory_id: int):
        """특정 기억을 삭제합니다.

        사용법: !기억삭제 [ID]
        예시: !기억삭제 123
        """
        if not self.config.enable_memory_system:
            await ctx.reply("❌ 메모리 시스템이 비활성화되어 있습니다.", mention_author=False)
            return

        try:
            if memory_id <= 0:
                await ctx.reply("❌ 유효한 기억 ID를 입력해주세요.", mention_author=False)
                return

            success = self.db_service.delete_memory(memory_id)
            if success:
                await ctx.reply(f"✓ 기억 #{memory_id}을 잊었어요.", mention_author=False)
                logger.info(f"User {ctx.author.id} deleted memory {memory_id}")
            else:
                await ctx.reply(f"❌ 기억 #{memory_id}를 찾을 수 없습니다.", mention_author=False)

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            await ctx.reply(f"❌ 기억을 삭제하는 중 오류가 발생했습니다: {e}", mention_author=False)

    @commands.command(name='기억초기화', aliases=['clearall', 'reset'])
    async def clear_all_memories(self, ctx: commands.Context):
        """모든 기억을 초기화합니다. (확인 필요)

        사용법: !기억초기화
        """
        if not self.config.enable_memory_system:
            await ctx.reply("❌ 메모리 시스템이 비활성화되어 있습니다.", mention_author=False)
            return

        try:
            # Send confirmation prompt
            confirm_msg = await ctx.reply(
                "⚠️ **정말로 모든 기억을 삭제하시겠어요?**\n"
                "이 작업은 되돌릴 수 없습니다.\n"
                "5초 내에 ✅ 반응을 눌러주세요.",
                mention_author=False,
            )

            await confirm_msg.add_reaction('✅')
            await confirm_msg.add_reaction('❌')

            def check(reaction, user):
                return user == ctx.author and str(reaction.emoji) in ['✅', '❌']

            try:
                reaction, _ = await self.bot.wait_for('reaction_add', timeout=5.0, check=check)

                if str(reaction.emoji) == '✅':
                    count = self.db_service.delete_all_memories(str(ctx.author.id))
                    await ctx.reply(
                        f"✅ {count}개의 기억을 모두 초기화했습니다.",
                        mention_author=False,
                    )
                    logger.info(f"User {ctx.author.id} cleared all {count} memories")
                else:
                    await ctx.reply("❌ 초기화가 취소되었습니다.", mention_author=False)

                try:
                    await confirm_msg.clear_reactions()
                except discord.errors.Forbidden:
                    pass

            except asyncio.TimeoutError:
                await ctx.reply("⏱️ 확인 시간이 초과되었습니다.", mention_author=False)
                try:
                    await confirm_msg.clear_reactions()
                except discord.errors.Forbidden:
                    pass

        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            await ctx.reply(f"❌ 기억을 초기화하는 중 오류가 발생했습니다: {e}", mention_author=False)

    @commands.command(name='기억설정', aliases=['memory_settings', 'settings'])
    async def memory_settings(self, ctx: commands.Context, mode: Optional[str] = None):
        """기억 검색 모드를 설정하거나 현재 설정을 확인합니다.

        사용법:
        - !기억설정          (현재 설정 확인)
        - !기억설정 inject_all        (전체 기억 주입 모드)
        - !기억설정 semantic_search   (의미론적 검색 모드)
        """
        if not self.config.enable_memory_system:
            await ctx.reply("❌ 메모리 시스템이 비활성화되어 있습니다.", mention_author=False)
            return

        try:
            if not mode:
                # Show current settings
                current_mode = self.memory_service.retrieval_mode
                embed = discord.Embed(
                    title="🔧 기억 설정",
                    description=f"현재 검색 모드: **{current_mode}**",
                    color=discord.Color.green(),
                )

                embed.add_field(
                    name="inject_all",
                    value="최근 기억들을 모두 주입합니다. (더 빠르고, 리소스 효율적)",
                    inline=False,
                )
                embed.add_field(
                    name="semantic_search",
                    value="대화에 관련된 기억들을 검색합니다. (더 정확하지만, 리소스 사용)",
                    inline=False,
                )

                await ctx.reply(embed=embed, mention_author=False)
                return

            if mode.lower() not in ('inject_all', 'semantic_search'):
                await ctx.reply(
                    "❌ 유효한 모드를 선택해주세요: `inject_all` 또는 `semantic_search`",
                    mention_author=False,
                )
                return

            # Change mode
            success = self.memory_service.set_retrieval_mode(mode.lower())
            if success:
                await ctx.reply(
                    f"✓ 기억 검색 모드를 **{mode.lower()}**로 변경했습니다.",
                    mention_author=False,
                )
                logger.info(f"User {ctx.author.id} changed memory retrieval mode to {mode.lower()}")
            else:
                await ctx.reply("❌ 모드 변경에 실패했습니다.", mention_author=False)

        except Exception as e:
            logger.error(f"Failed to change memory settings: {e}")
            await ctx.reply(f"❌ 설정을 변경하는 중 오류가 발생했습니다: {e}", mention_author=False)

async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    # This will be called from main.py
    pass

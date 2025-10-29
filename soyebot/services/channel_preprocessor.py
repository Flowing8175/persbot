"""Channel message preprocessor service for SoyeBot.

This service continuously reads messages from a specified channel and pre-processes them
for semantic search indexing. Only active when semantic_search mode is enabled.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

import discord
from discord.ext import commands, tasks

from services.database_service import DatabaseService
from services.memory_service import MemoryService
from config import AppConfig

logger = logging.getLogger(__name__)


class ChannelPreprocessor:
    """Preprocessor for channel messages."""

    @staticmethod
    def preprocess_message(content: str, author: str, channel: str) -> str:
        """Preprocess a message for semantic indexing.

        Args:
            content: Message content
            author: Author name
            channel: Channel name

        Returns:
            Preprocessed content
        """
        # Remove mentions and URLs
        content = content.replace("@", "")
        content = content.replace("#", "")

        # Remove extra whitespace
        content = " ".join(content.split())

        # Keep it under reasonable length for storage
        content = content[:500]

        return content.strip()

    @staticmethod
    def extract_topics(content: str) -> List[str]:
        """Extract potential topics/keywords from message.

        Args:
            content: Message content

        Returns:
            List of topic strings
        """
        # Simple keyword extraction (can be enhanced with NLP)
        topics = []

        # Common topic markers
        topic_markers = {
            "프로그래밍": ["코드", "코딩", "프로그래밍", "개발", "버그", "디버깅"],
            "게임": ["게임", "플레이", "전략", "발로란트", "마크"],
            "요리": ["요리", "밥", "음식", "먹다", "맛있다"],
            "여행": ["여행", "가다", "놀러", "관광", "여행지"],
            "공부": ["공부", "배우다", "시험", "수학", "과학"],
        }

        content_lower = content.lower()

        for topic, keywords in topic_markers.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)

        return list(set(topics))  # Remove duplicates


class ChannelPreprocessorCog(commands.Cog):
    """Cog for preprocessing channel messages for semantic search."""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        db_service: DatabaseService,
        memory_service: MemoryService,
    ):
        """Initialize channel preprocessor.

        Args:
            bot: Discord bot instance
            config: App configuration
            db_service: Database service
            memory_service: Memory service
        """
        self.bot = bot
        self.config = config
        self.db_service = db_service
        self.memory_service = memory_service
        self.preprocessor = ChannelPreprocessor()

        # Configuration for preprocessing
        self.target_channel_id: Optional[int] = None
        self.target_channel_name: Optional[str] = None
        self.is_enabled = False
        self.processed_count = 0

    def is_preprocessing_enabled(self) -> bool:
        """Check if preprocessing should be active.

        Returns:
            True if semantic search is enabled and preprocessing is active
        """
        return (
            self.config.enable_memory_system
            and self.memory_service.retrieval_mode == 'semantic_search'
            and self.is_enabled
        )

    @commands.command(name="채널설정")
    @commands.has_permissions(administrator=True)
    async def set_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """메시지를 수집할 채널을 설정합니다.

        사용법: !채널설정 #채널이름
        권한: 관리자 이상

        이 명령어 이후의 메시지들이 자동으로 수집되어 시맨틱 서치를 위해 인덱싱됩니다.
        (시맨틱 서치 모드가 활성화되어 있어야 함)
        """
        if not self.config.enable_memory_system:
            await ctx.reply("❌ 메모리 시스템이 비활성화되어 있습니다.", mention_author=False)
            return

        if self.memory_service.retrieval_mode != 'semantic_search':
            await ctx.reply(
                "❌ 시맨틱 서치 모드가 활성화되어 있지 않습니다.\n"
                "먼저 `!기억설정 semantic_search`로 시맨틱 서치를 활성화해주세요.",
                mention_author=False,
            )
            return

        self.target_channel_id = channel.id
        self.target_channel_name = channel.name
        self.is_enabled = True
        self.processed_count = 0

        await ctx.reply(
            f"✓ 채널 설정 완료!\n"
            f"#{channel.name}의 메시지들이 이제부터 시맨틱 서치를 위해 수집됩니다.\n"
            f"**주의:** 이 채널의 모든 메시지가 데이터베이스에 저장됩니다.",
            mention_author=False,
        )

        logger.info(f"Channel preprocessing enabled for #{channel.name} (ID: {channel.id})")

    @commands.command(name="채널설정해제")
    @commands.has_permissions(administrator=True)
    async def disable_channel(self, ctx: commands.Context):
        """채널 메시지 수집을 중지합니다.

        사용법: !채널설정해제
        권한: 관리자 이상
        """
        if not self.is_enabled:
            await ctx.reply("❌ 현재 수집 중인 채널이 없습니다.", mention_author=False)
            return

        channel_name = self.target_channel_name
        self.is_enabled = False
        self.target_channel_id = None
        self.target_channel_name = None

        await ctx.reply(
            f"✓ 채널 설정 해제 완료!\n"
            f"#{channel_name}의 메시지 수집이 중지되었습니다.\n"
            f"(총 {self.processed_count}개 메시지 처리됨)",
            mention_author=False,
        )

        logger.info(f"Channel preprocessing disabled")

    @commands.command(name="채널상태")
    async def channel_status(self, ctx: commands.Context):
        """현재 채널 수집 상태를 표시합니다.

        사용법: !채널상태
        """
        embed = discord.Embed(
            title="📊 채널 메시지 수집 상태",
            color=discord.Color.blue(),
        )

        if self.is_enabled and self.target_channel_name:
            embed.add_field(
                name="상태",
                value="🟢 활성화",
                inline=False,
            )
            embed.add_field(
                name="수집 중인 채널",
                value=f"#{self.target_channel_name}",
                inline=False,
            )
            embed.add_field(
                name="처리된 메시지",
                value=f"{self.processed_count}개",
                inline=False,
            )
            embed.add_field(
                name="검색 모드",
                value=f"`{self.memory_service.retrieval_mode}`",
                inline=False,
            )
        else:
            embed.add_field(
                name="상태",
                value="🔴 비활성화",
                inline=False,
            )
            embed.add_field(
                name="설정",
                value="관리자가 `!채널설정 #채널이름`으로 채널을 선택하면 수집을 시작합니다.",
                inline=False,
            )

        embed.set_footer(text="시맨틱 서치 모드에서만 작동합니다.")
        await ctx.reply(embed=embed, mention_author=False)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Listen for messages in the target channel and preprocess them.

        Args:
            message: Discord message object
        """
        # Skip if preprocessing is not enabled
        if not self.is_preprocessing_enabled():
            return

        # Skip if not in target channel
        if message.channel.id != self.target_channel_id:
            return

        # Skip bot messages and commands
        if message.author.bot or message.content.startswith("!"):
            return

        # Skip empty messages
        if not message.content.strip():
            return

        try:
            await self._preprocess_and_save(message)
        except Exception as e:
            logger.error(f"Error preprocessing message: {e}", exc_info=True)

    async def _preprocess_and_save(self, message: discord.Message) -> None:
        """Preprocess and save a message.

        Args:
            message: Discord message to preprocess
        """
        # Preprocess content
        processed_content = self.preprocessor.preprocess_message(
            content=message.content,
            author=message.author.name,
            channel=message.channel.name,
        )

        # Extract topics
        topics = self.preprocessor.extract_topics(processed_content)

        # Format as memory
        topic_str = ", ".join(topics) if topics else "일반"
        memory_content = f"[{self.target_channel_name}] {processed_content}"

        # Save as memory for the channel (not user-specific)
        # Use channel ID as "user_id" for channel-level memories
        channel_user_id = f"channel_{self.target_channel_id}"

        try:
            # Ensure "channel user" exists
            self.db_service.get_or_create_user(
                user_id=channel_user_id,
                username=f"Channel: {self.target_channel_name}",
            )

            # Save as fact memory (lower importance than user-provided memories)
            self.memory_service.save_memory(
                user_id=channel_user_id,
                memory_type='fact',
                content=memory_content,
                importance_score=0.3,  # Lower importance for auto-indexed messages
            )

            # Update interaction pattern with topics
            for topic in topics:
                self.db_service.update_interaction_pattern(
                    user_id=channel_user_id,
                    topic=topic,
                    sentiment=None,
                )

            self.processed_count += 1

            logger.debug(
                f"Preprocessed message from {message.author.name}: "
                f"{processed_content[:50]}... (Topics: {topic_str})"
            )

        except Exception as e:
            logger.error(f"Failed to save preprocessed message: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessor statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'enabled': self.is_enabled,
            'target_channel': self.target_channel_name,
            'target_channel_id': self.target_channel_id,
            'processed_count': self.processed_count,
            'retrieval_mode': self.memory_service.retrieval_mode,
            'can_preprocess': self.is_preprocessing_enabled(),
        }


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    pass

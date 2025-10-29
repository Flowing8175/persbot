"""Main entry point for the SoyeBot Discord bot."""

import discord
from discord.ext import commands
import asyncio
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import load_config
from services.gemini_service import GeminiService
from services.database_service import DatabaseService
from services.memory_service import MemoryService
from services.optimization_service import OptimizationService
from services.channel_preprocessor import ChannelPreprocessorCog
from bot.session import SessionManager
from bot.cogs.summarizer import SummarizerCog
from bot.cogs.assistant import AssistantCog
from bot.cogs.memory import MemoryCog
from bot.cogs.help import HelpCog

logger = logging.getLogger(__name__)

async def main():
    """Initializes and runs the bot."""
    config = load_config()

    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    intents.message_content = True

    bot = commands.Bot(command_prefix=config.command_prefix, intents=intents, help_command=None)

    # Initialize services
    gemini_service = GeminiService(config)
    db_service = DatabaseService(config.database_path)
    memory_service = MemoryService(
        db_service=db_service,
        retrieval_mode=config.memory_retrieval_mode,
        embedding_model_name=config.embedding_model_name,
        cache_size=config.memory_cache_size,
    )
    session_manager = SessionManager(config, gemini_service, db_service, memory_service)
    optimization_service = OptimizationService(
        db_service=db_service,
        cache_size=config.memory_cache_size,
    )

    @bot.event
    async def on_ready():
        logger.info(f'로그인 완료: {bot.user.name} ({bot.user.id})')
        logger.info(f"봇이 준비되었습니다! '{config.command_prefix}' 또는 @mention으로 상호작용할 수 있습니다.")

        # Initialize cogs
        await bot.add_cog(HelpCog(bot))
        await bot.add_cog(SummarizerCog(bot, config, gemini_service, memory_service))
        await bot.add_cog(AssistantCog(bot, config, gemini_service, session_manager, memory_service))

        # Add memory cog if memory system is enabled
        if config.enable_memory_system:
            await bot.add_cog(MemoryCog(bot, config, db_service, memory_service))
            await bot.add_cog(ChannelPreprocessorCog(bot, config, db_service, memory_service))
            logger.info("Memory system initialized")

            # Start optimization service
            await optimization_service.start_cleanup_loop()
            logger.info("Optimization service started")

        logger.info("Cogs 로드 완료.")

    @bot.event
    async def on_close():
        """Cleanup on bot close."""
        try:
            memory_service.cleanup()
            await optimization_service.stop()
            db_service.close()
            logger.info("Services cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


    try:
        logger.info("Discord 봇을 시작합니다...")
        await bot.start(config.discord_token)
    except discord.LoginFailure:
        logger.error("에러: 로그인 실패. Discord 봇 토큰을 확인하세요.")
    except Exception as e:
        logger.error(f"봇 실행 중 에러 발생: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

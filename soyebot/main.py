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
from bot.session import SessionManager
from bot.cogs.summarizer import SummarizerCog
from bot.cogs.assistant import AssistantCog
from bot.cogs.help import HelpCog

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration with suppressed discord.py spam."""
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    # Suppress discord.py debug logs
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('discord.http').setLevel(logging.WARNING)
    logging.getLogger('discord.client').setLevel(logging.WARNING)
    logging.getLogger('discord.gateway').setLevel(logging.WARNING)

    logger.debug("Logging configuration initialized")

async def periodic_session_cleanup(session_manager: SessionManager):
    """Background task to periodically clean up expired sessions.

    This prevents cleanup overhead in the message handling hot path.
    """
    while True:
        try:
            await asyncio.sleep(session_manager.config.session_cleanup_interval)
            session_manager.cleanup_expired()
            logger.debug(f"Background cleanup completed: {len(session_manager.sessions)} active sessions")
        except Exception as e:
            logger.error(f"Error in background session cleanup: {e}", exc_info=True)

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
    session_manager = SessionManager(config, gemini_service, db_service)

    @bot.event
    async def on_ready():
        logger.info(f'로그인 완료: {bot.user.name} ({bot.user.id})')
        logger.info(f"봇이 준비되었습니다! '{config.command_prefix}' 또는 @mention으로 상호작용할 수 있습니다.")

        # Initialize cogs
        await bot.add_cog(HelpCog(bot))
        await bot.add_cog(SummarizerCog(bot, config, gemini_service))
        await bot.add_cog(AssistantCog(bot, config, gemini_service, session_manager))
        logger.info("Cogs 로드 완료.")

        # Start background cleanup task
        asyncio.create_task(periodic_session_cleanup(session_manager))
        logger.info("Background session cleanup task started")

    @bot.event
    async def on_close():
        """Cleanup on bot close."""
        try:
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
    setup_logging()
    asyncio.run(main())

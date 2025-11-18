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
from services.llm_service import LLMService
from services.database_service import DatabaseService
from bot.session import SessionManager
from bot.cogs.summarizer import SummarizerCog
from bot.cogs.assistant import AssistantCog
from bot.cogs.help import HelpCog
from web.metrics_server import start_metrics_server_background

logger = logging.getLogger(__name__)


def setup_logging(log_level: int) -> None:
    """Setup logging configuration with suppressed discord.py spam."""

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(log_level)

    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('discord.http').setLevel(logging.WARNING)
    logging.getLogger('discord.client').setLevel(logging.WARNING)
    logging.getLogger('discord.gateway').setLevel(logging.WARNING)


async def main(config):
    """Initializes and runs the bot."""
    auto_channel_cog_cls = None
    if config.auto_reply_channel_ids:
        try:
            from bot.cogs.auto_channel import AutoChannelCog
            auto_channel_cog_cls = AutoChannelCog
        except ModuleNotFoundError as exc:
            logger.error(
                "AUTO_REPLY_CHANNEL_IDS가 설정되었지만 auto_channel Cog를 불러올 수 없습니다: %s",
                exc,
            )

    # Start metrics web UI server in background thread
    start_metrics_server_background(host='0.0.0.0', port=5000)

    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    intents.message_content = True

    bot = commands.Bot(command_prefix=config.command_prefix, intents=intents, help_command=None)

    # Initialize services
    llm_service = LLMService(config)
    db_service = DatabaseService(config.database_path)
    session_manager = SessionManager(config, llm_service, db_service)

    @bot.event
    async def on_ready():
        logger.info(f'로그인 완료: {bot.user.name} ({bot.user.id})')
        logger.info(f"봇이 준비되었습니다! '{config.command_prefix}' 또는 @mention으로 상호작용할 수 있습니다.")

        if config.auto_reply_channel_ids:
            logger.info("channel registered to reply: %s", list(config.auto_reply_channel_ids))
        else:
            logger.info("channel registered to reply: []")

        # Initialize cogs
        await bot.add_cog(HelpCog(bot, config))
        await bot.add_cog(SummarizerCog(bot, config, llm_service))
        await bot.add_cog(AssistantCog(bot, config, llm_service, session_manager))
        if auto_channel_cog_cls:
            await bot.add_cog(auto_channel_cog_cls(bot, config, llm_service, session_manager))
        logger.info("Cogs 로드 완료.")

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
    config = load_config()
    setup_logging(config.log_level)
    asyncio.run(main(config))

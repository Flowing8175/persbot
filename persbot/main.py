"""Main entry point for the SoyeBot Discord bot."""

import asyncio
import logging
import os
import sys

import discord
from discord.ext import commands

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from persbot.bot.cogs.assistant import AssistantCog
from persbot.bot.cogs.help import HelpCog
from persbot.bot.cogs.model_selector import ModelSelectorCog
from persbot.bot.cogs.persona import PersonaCog
from persbot.bot.cogs.summarizer import SummarizerCog
from persbot.bot.session import SessionManager
from persbot.config import load_config
from persbot.services.llm_service import LLMService
from persbot.services.prompt_service import PromptService
from persbot.tools.manager import ToolManager

logger = logging.getLogger(__name__)


def setup_logging(log_level: int) -> None:
    """Setup logging configuration with suppressed discord.py spam."""

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(log_level)

    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.http").setLevel(logging.WARNING)
    logging.getLogger("discord.client").setLevel(logging.WARNING)
    logging.getLogger("discord.gateway").setLevel(logging.WARNING)


async def main(config) -> None:
    """Initializes and runs the bot."""
    auto_channel_cog_cls = None
    if config.auto_reply_channel_ids:
        try:
            from persbot.bot.cogs.auto_channel import AutoChannelCog

            auto_channel_cog_cls = AutoChannelCog
        except ModuleNotFoundError as exc:
            logger.error(
                "AUTO_REPLY_CHANNEL_IDS가 설정되었지만 auto_channel Cog를 불러올 수 없습니다: %s",
                exc,
            )

    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    intents.members = True  # Required for get_member_info and get_member_roles tools
    intents.message_content = True

    # Optimize member caching to reduce memory footprint
    # Only cache members when they become active (send message, react, etc.)
    # rather than caching all members on guild join
    # from_intents() automatically configures appropriate cache flags based on intents
    member_cache_flags = discord.MemberCacheFlags.from_intents(intents)

    # Limit internal message cache to reduce memory (default is 1000)
    max_messages = getattr(config, 'max_message_cache', 100)

    bot = commands.Bot(
        command_prefix=config.command_prefix,
        intents=intents,
        help_command=None,
        member_cache_flags=member_cache_flags,
        max_messages=max_messages,
    )

    # Initialize services
    llm_service = LLMService(config)
    session_manager = SessionManager(config, llm_service)
    prompt_service = PromptService()
    tool_manager = ToolManager(config)

    # Start background cache warmup to reduce first-message latency
    # This pre-creates commonly used Gemini caches asynchronously
    llm_service.start_background_cache_warmup()

    # Register cogs before starting so listeners are ready on first connect
    # and won't raise on reconnect (on_ready can fire multiple times).
    await bot.add_cog(HelpCog(bot, config))
    await bot.add_cog(SummarizerCog(bot, config, llm_service))
    await bot.add_cog(
        AssistantCog(bot, config, llm_service, session_manager, prompt_service, tool_manager)
    )
    await bot.add_cog(PersonaCog(bot, config, llm_service, session_manager, prompt_service))
    await bot.add_cog(ModelSelectorCog(bot, session_manager))
    if auto_channel_cog_cls:
        await bot.add_cog(
            auto_channel_cog_cls(bot, config, llm_service, session_manager, tool_manager)
        )

    tree_synced = False

    @bot.event
    async def on_ready() -> None:
        nonlocal tree_synced
        # Sync Command Tree (only once, on_ready can fire multiple times on reconnect)
        if not tree_synced:
            try:
                await bot.tree.sync()
                tree_synced = True
            except Exception:
                logger.exception("Failed to sync command tree")

    @bot.event
    async def on_close() -> None:
        """Cleanup on bot close."""
        pass

    try:
        await bot.start(config.discord_token)
    except discord.LoginFailure:
        logger.error("에러: 로그인 실패. Discord 봇 토큰을 확인하세요.")
    except Exception as e:
        logger.error(f"봇 실행 중 에러 발생: {e}", exc_info=True)


if __name__ == "__main__":
    config = load_config()
    setup_logging(config.log_level)
    asyncio.run(main(config))

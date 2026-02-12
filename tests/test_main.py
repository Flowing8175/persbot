"""Tests for the main bot entry point."""

import asyncio
import logging
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from discord.ext import commands

from persbot.config import AppConfig
from persbot.main import main, setup_logging
from persbot.services.llm_service import LLMService
from persbot.services.prompt_service import PromptService
from persbot.bot.session import SessionManager
from persbot.tools.manager import ToolManager


class TestSetupLogging:
    """Test logging setup functionality."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        log_level = logging.INFO

        with patch("persbot.main.logging") as mock_logging:
            mock_root_logger = Mock()
            mock_discord_logger = Mock()
            
            def get_logger_side_effect(name):
                if name == "discord":
                    return mock_discord_logger
                return mock_root_logger
            
            mock_logging.getLogger.side_effect = get_logger_side_effect

            setup_logging(log_level)

            mock_logging.getLogger.assert_called()
            mock_root_logger.setLevel.assert_called_once_with(log_level)
            mock_logging.getLogger.assert_called_with("discord")
            mock_discord_logger.setLevel.assert_called_with(logging.WARNING)

    def test_setup_logging_existing_handlers(self, tmp_path):
        """Test logging setup with existing handlers."""
        with patch("persbot.main.logging") as mock_logging:
            mock_logger = Mock()
            mock_logger.handlers = [Mock()]  # Existing handler

            mock_logging.getLogger.return_value = mock_logger

            setup_logging(logging.DEBUG)

            # Should not add new handler if existing
            mock_logger.addHandler.assert_not_called()


class TestMainFunction:
    """Test main function initialization and execution."""

    @pytest.mark.asyncio
    async def test_main_initialization(self, mock_app_config):
        """Test main function initializes bot correctly."""
        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
            mock_bot.add_cog = AsyncMock()

            # Mock bot events
            mock_on_ready = AsyncMock()
            mock_on_close = AsyncMock()
            mock_bot.event = Mock(
                side_effect=lambda func: (
                    setattr(mock_bot, "on_ready", func)
                    if func.__name__ == "on_ready"
                    else setattr(mock_bot, "on_close", func)
                )
            )

            # Mock tree sync
            mock_bot.tree.sync = AsyncMock(return_value=[])

# Mock AutoChannelCog import (not available)
            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Verify bot initialization
            mock_bot_class.assert_called_once()
            assert mock_bot_class.call_args[1]["command_prefix"] == mock_app_config.command_prefix
            assert mock_bot_class.call_args[1]["intents"].messages == True
            assert mock_bot_class.call_args[1]["intents"].guilds == True
            assert mock_bot_class.call_args[1]["intents"].members == True
            assert mock_bot_class.call_args[1]["intents"].message_content == True

            # Verify services initialization
            mock_llm_service.assert_called_once_with(mock_app_config)
            mock_session_manager.assert_called_once_with(
                mock_app_config, mock_llm_service.return_value
            )
            mock_prompt_service.assert_called_once()
            mock_tool_manager.assert_called_once_with(mock_app_config)

            # Verify cogs are added
            assert mock_bot.add_cog.await_count == 4  # 4 cogs should be added
            mock_bot.add_cog.assert_any_call(mock_summarizer_cog.return_value)
            mock_bot.add_cog.assert_any_call(mock_assistant_cog.return_value)
            mock_bot.add_cog.assert_any_call(mock_persona_cog.return_value)
            mock_bot.add_cog.assert_any_call(mock_model_selector_cog.return_value)

    @pytest.mark.asyncio
    async def test_main_with_auto_channel_cog(self, mock_app_config):
        """Test main function with AutoChannelCog available."""
        mock_app_config.auto_reply_channel_ids = (123456789, 987654321)

        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
            mock_bot.add_cog = AsyncMock()

            # Mock bot events
            mock_on_ready = AsyncMock()
            mock_on_close = AsyncMock()
            mock_bot.event = Mock(
                side_effect=lambda func: (
                    setattr(mock_bot, "on_ready", func)
                    if func.__name__ == "on_ready"
                    else setattr(mock_bot, "on_close", func)
                )
            )

            # Mock tree sync
            mock_bot.tree.sync = AsyncMock(return_value=[])

# Mock AutoChannelCog import (available)
            mock_auto_channel_module = Mock()
            mock_auto_channel_module.AutoChannelCog = Mock()
            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": mock_auto_channel_module}):
                await main(mock_app_config)

            # Verify all 5 cogs are added including AutoChannelCog
            assert mock_bot.add_cog.await_count == 5

    @pytest.mark.asyncio
    async def test_main_on_ready_event(self, mock_app_config):
        """Test on_ready event handler."""
        tree_synced = False

        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
            mock_bot.add_cog = AsyncMock()

            # Mock bot events
            mock_tree_sync = AsyncMock(return_value=[])
            mock_bot.tree.sync = mock_tree_sync

            # Mock event handler
            event_handler_called = False

            def mock_event_handler(func):
                nonlocal event_handler_called
                event_handler_called = True
                if func.__name__ == "on_ready":
                    # Create a real on_ready function that can be called
                    async def on_ready():
                        nonlocal tree_synced
                        if mock_bot.user:
                            logging.info(f"로그인 완료: {mock_bot.user.name} ({mock_bot.user.id})")
                        logging.info(
                            f"봇이 준비되었습니다! '{mock_app_config.command_prefix}' 또는 @mention으로 상호작용할 수 있습니다."
                        )
                        if mock_app_config.auto_reply_channel_ids:
                            logging.info(
                                "channel registered to reply: %s",
                                list(mock_app_config.auto_reply_channel_ids),
                            )
                        else:
                            logging.info("channel registered to reply: []")
                        # Sync Command Tree (only once, on_ready can fire multiple times on reconnect)
                        if not tree_synced:
                            try:
                                synced = await mock_bot.tree.sync()
                                tree_synced = True
                                logging.info(f"Command Tree Synced: {len(synced)} commands.")
                            except Exception as e:
                                logging.error(f"Failed to sync command tree: {e}")

                    return on_ready
                return func

            mock_bot.event = Mock(side_effect=mock_event_handler)

# Run main
            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Call on_ready event
            await mock_bot.on_ready()

            # Verify tree sync was called
            mock_tree_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_on_close_event(self, mock_app_config):
        """Test on_close event handler."""
        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
            mock_bot.add_cog = AsyncMock()

            # Mock event handler
            close_handler_called = False

            def mock_event_handler(func):
                nonlocal close_handler_called
                if func.__name__ == "on_close":
                    close_handler_called = True

                    # Create a real on_close function
                    async def on_close():
                        logging.info("Services cleaned up successfully")

                    return on_close
                return func

                mock_bot.event = Mock(side_effect=mock_event_handler)

            # Run main
            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Call on_close event
            await mock_bot.on_close()

            # Verify close handler was called
            assert close_handler_called

    @pytest.mark.asyncio
    async def test_main_login_failure(self, mock_app_config):
        """Test main function handles login failure."""
        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
            patch("persbot.main.logger") as mock_logger,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
            mock_bot.add_cog = AsyncMock()

            # Mock bot start to raise LoginFailure
            import discord

            mock_bot.start = AsyncMock(side_effect=discord.LoginFailure("Invalid token"))

            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "로그인 실패" in mock_logger.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_main_general_exception(self, mock_app_config):
        """Test main function handles general exceptions."""
        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
            patch("persbot.main.logger") as mock_logger,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
            mock_bot.add_cog = AsyncMock()

            # Mock bot start to raise general exception
mock_bot.start = AsyncMock(side_effect=Exception("General error"))

            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "봇 실행 중 에러 발생" in mock_logger.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_main_bot_start_calls(self, mock_app_config):
        """Test main function properly calls bot start."""
        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
            mock_bot.add_cog = AsyncMock()

            # Mock bot events
            mock_tree_sync = AsyncMock(return_value=[])
mock_bot.tree.sync = mock_tree_sync

            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Verify bot.start was called with correct token
            mock_bot.start.assert_called_once_with(mock_app_config.discord_token)


class TestMainModuleIntegration:
    """Test main module integration components."""

    @pytest.mark.asyncio
    async def test_cog_loading_order(self, mock_app_config):
        """Test cogs are loaded in correct order."""
        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
mock_bot.add_cog = AsyncMock()

            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Verify cogs are called with correct arguments
            call_args_list = mock_bot.add_cog.await_args_list

            # Check each cog was called with correct parameters
            expected_calls = [
                call(mock_summarizer_cog.return_value),
                call(mock_assistant_cog.return_value),
                call(mock_persona_cog.return_value),
                call(mock_model_selector_cog.return_value),
            ]

            # Verify all expected calls were made
            for expected_call in expected_calls:
                assert expected_call in call_args_list

    @pytest.mark.asyncio
    async def test_intent_configuration(self, mock_app_config):
        """Test bot intents are configured correctly."""
        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
mock_bot.add_cog = AsyncMock()

            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Verify bot initialization calls
            bot_call_args = mock_bot_class.call_args
            assert bot_call_args[1]["command_prefix"] == mock_app_config.command_prefix
            assert bot_call_args[1]["help_command"] is None

            # Verify intents
            intents = bot_call_args[1]["intents"]
            assert intents.messages == True
            assert intents.guilds == True
            assert intents.members == True
            assert intents.message_content == True

    @pytest.mark.asyncio
    async def test_tree_sync_error_handling(self, mock_app_config):
        """Test error handling during tree sync."""
        tree_synced = False

        with (
            patch("persbot.main.LLMService") as mock_llm_service,
            patch("persbot.main.SessionManager") as mock_session_manager,
            patch("persbot.main.PromptService") as mock_prompt_service,
            patch("persbot.main.ToolManager") as mock_tool_manager,
            patch("persbot.main.commands.Bot") as mock_bot_class,
            patch("persbot.main.SummarizerCog") as mock_summarizer_cog,
            patch("persbot.main.AssistantCog") as mock_assistant_cog,
            patch("persbot.main.PersonaCog") as mock_persona_cog,
            patch("persbot.main.ModelSelectorCog") as mock_model_selector_cog,
            patch("persbot.main.logger") as mock_logger,
        ):
            # Setup mocks
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            mock_bot.user = Mock(name="TestBot", id=123456789)
            mock_bot.add_cog = AsyncMock()

            # Mock bot events
            def mock_event_handler(func):
                if func.__name__ == "on_ready":
                    # Create on_ready function with error handling
                    async def on_ready():
                        nonlocal tree_synced
                        if mock_bot.user:
                            logging.info(f"로그인 완료: {mock_bot.user.name} ({mock_bot.user.id})")
                        logging.info(
                            f"봇이 준비되었습니다! '{mock_app_config.command_prefix}' 또는 @mention으로 상호작용할 수 있습니다."
                        )
                        if mock_app_config.auto_reply_channel_ids:
                            logging.info(
                                "channel registered to reply: %s",
                                list(mock_app_config.auto_reply_channel_ids),
                            )
                        else:
                            logging.info("channel registered to reply: []")
                        # Sync Command Tree (only once, on_ready can fire multiple times on reconnect)
                        if not tree_synced:
                            try:
                                synced = await mock_bot.tree.sync()
                                tree_synced = True
                                logging.info(f"Command Tree Synced: {len(synced)} commands.")
                            except Exception as e:
                                logging.error(f"Failed to sync command tree: {e}")

                    return on_ready
                return func

            mock_bot.event = Mock(side_effect=mock_event_handler)

            # Make tree sync raise an exception
            import discord

mock_bot.tree.sync = AsyncMock(side_effect=Exception("Sync failed"))

            with patch.dict("sys.modules", {"persbot.bot.cogs.auto_channel": None}):
                await main(mock_app_config)

            # Call on_ready to trigger sync
            await mock_bot.on_ready()

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Failed to sync command tree" in mock_logger.error.call_args[0][0]

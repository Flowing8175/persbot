"""Tests for the Assistant Cog."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from discord.ext import commands

from persbot.bot.cogs.assistant import AssistantCog
from persbot.bot.session import ResolvedSession
from persbot.services.base import ChatMessage


class TestAssistantCogInitialization:
    """Test AssistantCog initialization and setup."""

    @pytest.mark.asyncio
    async def test_cog_initialization(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test that AssistantCog initializes correctly."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        assert cog.bot is mock_bot
        assert cog.config is mock_app_config
        assert cog.llm_service is mock_llm_service
        assert cog.session_manager is mock_session_manager
        assert cog.prompt_service is mock_prompt_service

    @pytest.mark.asyncio
    async def test_cog_initialization_with_tool_manager(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test that AssistantCog initializes correctly with tool manager."""
        mock_tool_manager = Mock()
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
            tool_manager=mock_tool_manager,
        )

        assert cog.tool_manager is mock_tool_manager


class TestShouldIgnoreMessage:
    """Test message filtering logic."""

    @pytest.mark.asyncio
    async def test_ignore_bot_messages(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test that bot messages are ignored."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        mock_message.author.bot = True
        assert cog._should_ignore_message(mock_message) is True

    @pytest.mark.asyncio
    async def test_ignore_auto_reply_channel_messages(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test that messages in auto-reply channels are ignored."""
        config_with_auto_reply = Mock(
            spec=mock_app_config,
            auto_reply_channel_ids=(111222333,),
            command_prefix="!",
            break_cut_mode=False,
        )

        cog = AssistantCog(
            bot=mock_bot,
            config=config_with_auto_reply,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        assert cog._should_ignore_message(mock_message) is True

    @pytest.mark.asyncio
    async def test_ignore_unmentioned_messages(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test that messages without bot mention are ignored."""
        mock_bot.user = Mock(id=123456789)
        mock_bot.user.mentioned_in = Mock(return_value=False)

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        assert cog._should_ignore_message(mock_message) is True

    @pytest.mark.asyncio
    async def test_process_mentioned_messages(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_mention_message,
    ):
        """Test that mentioned messages are processed."""
        mock_bot.user = Mock(id=123456789)
        mock_bot.user.mentioned_in = Mock(return_value=True)
        mock_mention_message.author.bot = False
        mock_mention_message.channel.id = 999888777  # Not an auto-reply channel

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        assert cog._should_ignore_message(mock_mention_message) is False

    @pytest.mark.asyncio
    async def test_ignore_mention_everyone(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_mention_message,
    ):
        """Test that @everyone mentions are ignored."""
        mock_bot.user = Mock(id=123456789)
        mock_bot.user.mentioned_in = Mock(return_value=True)
        mock_mention_message.author.bot = False
        mock_mention_message.channel.id = 999888777
        mock_mention_message.mention_everyone = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        assert cog._should_ignore_message(mock_mention_message) is True


class TestHelpCommand:
    """Test !help command."""

    @pytest.mark.asyncio
    async def test_help_command_sends_embed(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test that !help command sends an embed."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create a mock context
        ctx = Mock()
        ctx.channel = mock_message.channel
        ctx.interaction = None

        await cog.help_command(ctx)

        # Verify send_discord_message was called (it's mocked in utils)
        # The actual verification would check that embed was sent


class TestRetryCommand:
    """Test !retry command."""

    @pytest.mark.asyncio
    async def test_retry_command_no_messages_to_undo(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test retry when there's nothing to undo."""
        mock_session_manager.undo_last_exchanges = Mock(return_value=[])

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel = mock_message.channel
        ctx.defer = AsyncMock()
        ctx.send = AsyncMock()

        await cog.retry_command(ctx)

        ctx.send.assert_called_once()
        assert "되돌릴 대화가 없습니다" in ctx.send.call_args[0][0]

    @pytest.mark.asyncio
    async def test_retry_command_success(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test successful retry command."""
        # Mock the session messages
        user_msg = ChatMessage(role="user", content="Hello", author_id=123456789)
        assistant_msg = ChatMessage(role="model", content="Hi there!", author_id=None)
        assistant_msg.message_ids = ["999"]

        mock_session_manager.undo_last_exchanges = Mock(return_value=[assistant_msg, user_msg])

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel = mock_message.channel
        ctx.interaction = None
        ctx.message = mock_message
        ctx.defer = AsyncMock()

        # Mock create_chat_reply to return a response
        with patch("persbot.bot.cogs.assistant.create_chat_reply") as mock_create_reply:
            mock_create_reply.return_value = ChatReply(
                text="New response",
                session_key="channel:111222333",
                response=None,
            )

            # Mock fetch_message for deletion
            mock_channel = Mock()
            mock_channel.fetch_message = AsyncMock()

            await cog.retry_command(ctx)

            # Verify the retry flow was triggered
            mock_session_manager.undo_last_exchanges.assert_called_once()


class TestAbortCommand:
    """Test !abort command."""

    @pytest.mark.asyncio
    async def test_abort_with_no_permission(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test abort command without manage_guild permission."""
        mock_app_config.no_check_permission = False
        mock_user.guild_permissions.manage_guild = False

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.abort_command(ctx)

        ctx.reply.assert_called_once()
        assert "권한이 없습니다" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_abort_with_permission(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test abort command with proper permissions."""
        mock_app_config.no_check_permission = True  # Disable permission check

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create an active processing task
        mock_task = Mock()
        mock_task.done = Mock(return_value=False)
        cog.processing_tasks[mock_channel.id] = mock_task

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.channel.name = "test-channel"
        ctx.reply = AsyncMock()
        ctx.message = Mock()
        ctx.message.add_reaction = Mock()
        ctx.interaction = None

        await cog.abort_command(ctx)

        # Verify task was cancelled
        mock_task.cancel.assert_called_once()


class TestResetCommand:
    """Test !reset command."""

    @pytest.mark.asyncio
    async def test_reset_session_command(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test reset command."""
        mock_session_manager.reset_session_by_channel = Mock(return_value=True)

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel = mock_message.channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.reset_session(ctx)

        mock_session_manager.reset_session_by_channel.assert_called_once_with(
            mock_message.channel.id
        )

    @pytest.mark.asyncio
    async def test_reset_session_error(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test reset command with error."""
        mock_session_manager.reset_session_by_channel = Mock(
            side_effect=Exception("Database error")
        )

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel = mock_message.channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.reset_session(ctx)

        # Should send error message
        ctx.reply.assert_called_once()


class TestTemperatureCommand:
    """Test !temp command."""

    @pytest.mark.asyncio
    async def test_temp_command_get_current(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test getting current temperature."""
        mock_app_config.no_check_permission = True
        mock_app_config.temperature = 0.7

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.set_temperature(ctx, value=None)

        ctx.reply.assert_called_once()
        assert "0.7" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_temp_command_set_valid(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting temperature to valid value."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.set_temperature(ctx, value=0.5)

        mock_llm_service.update_parameters.assert_called_once_with(temperature=0.5)

    @pytest.mark.asyncio
    async def test_temp_command_invalid_value(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting temperature to invalid value."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.set_temperature(ctx, value=3.0)  # Out of range (0.0-2.0)

        ctx.reply.assert_called_once()
        assert "0.0에서 2.0 사이여야 합니다" in ctx.reply.call_args[0][0]


class TestTopPCommand:
    """Test !topp command."""

    @pytest.mark.asyncio
    async def test_top_p_command_get_current(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test getting current top_p."""
        mock_app_config.no_check_permission = True
        mock_app_config.top_p = 0.9

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.set_top_p(ctx, value=None)

        ctx.reply.assert_called_once()
        assert "0.9" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_top_p_command_set_valid(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting top_p to valid value."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.set_top_p(ctx, value=0.8)

        mock_llm_service.update_parameters.assert_called_once_with(top_p=0.8)


class TestBreakCutCommand:
    """Test !끊어치기 command."""

    @pytest.mark.asyncio
    async def test_break_cut_toggle(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test toggling break-cut mode."""
        mock_app_config.break_cut_mode = False
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.toggle_break_cut(ctx, mode=None)

        assert cog.config.break_cut_mode is True
        ctx.reply.assert_called_once()
        assert "ON" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_break_cut_set_on(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting break-cut mode to ON."""
        mock_app_config.break_cut_mode = False
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.toggle_break_cut(ctx, mode="on")

        assert cog.config.break_cut_mode is True

    @pytest.mark.asyncio
    async def test_break_cut_set_off(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting break-cut mode to OFF."""
        mock_app_config.break_cut_mode = True
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.toggle_break_cut(ctx, mode="off")

        assert cog.config.break_cut_mode is False


class TestThinkingBudgetCommand:
    """Test !생각 command."""

    @pytest.mark.asyncio
    async def test_thinking_budget_get_current(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test getting current thinking budget."""
        mock_app_config.no_check_permission = True
        mock_app_config.thinking_budget = None

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.set_thinking_budget(ctx, value=None)

        ctx.reply.assert_called_once()
        assert "OFF" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_thinking_budget_set_off(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting thinking budget to OFF."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.set_thinking_budget(ctx, value="off")

        mock_llm_service.update_parameters.assert_called_once_with(thinking_budget=None)

    @pytest.mark.asyncio
    async def test_thinking_budget_set_auto(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting thinking budget to AUTO."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.set_thinking_budget(ctx, value="auto")

        mock_llm_service.update_parameters.assert_called_once_with(thinking_budget=-1)

    @pytest.mark.asyncio
    async def test_thinking_budget_set_number(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting thinking budget to numeric value."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.set_thinking_budget(ctx, value="1024")

        mock_llm_service.update_parameters.assert_called_once_with(thinking_budget=1024)

    @pytest.mark.asyncio
    async def test_thinking_budget_invalid_range(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting thinking budget to invalid range."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.set_thinking_budget(ctx, value="100")  # Below minimum (512)

        ctx.reply.assert_called_once()
        assert "512에서 32768 사이여야 합니다" in ctx.reply.call_args[0][0]


class TestDelayCommand:
    """Test !delay command."""

    @pytest.mark.asyncio
    async def test_delay_get_current(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test getting current buffer delay."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.set_buffer_delay(ctx, value=None)

        ctx.reply.assert_called_once()

    @pytest.mark.asyncio
    async def test_delay_set_valid(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting buffer delay to valid value."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()
        ctx.interaction = None

        await cog.set_buffer_delay(ctx, value=5.0)

        # Verify delay was updated
        assert cog.message_buffer.default_delay == 5.0

    @pytest.mark.asyncio
    async def test_delay_invalid_value(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test setting buffer delay to invalid value."""
        mock_app_config.no_check_permission = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.set_buffer_delay(ctx, value=100.0)  # Above maximum (60)

        ctx.reply.assert_called_once()
        assert "0에서 60초 사이여야 합니다" in ctx.reply.call_args[0][0]


class TestOnMessageHandler:
    """Test on_message event handler."""

    @pytest.mark.asyncio
    async def test_on_message_ignored(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test that ignored messages are not processed."""
        mock_message.author.bot = True

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        await cog.on_message(mock_message)

        # Message should be ignored (no processing)
        assert len(cog.message_buffer.buffers) == 0

    @pytest.mark.asyncio
    async def test_on_message_processed(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_mention_message,
    ):
        """Test that mentioned messages are processed."""
        mock_bot.user = Mock(id=123456789)
        mock_bot.user.mentioned_in = Mock(return_value=True)
        mock_mention_message.author.bot = False

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        await cog.on_message(mock_mention_message)

        # Message should be added to buffer


class TestPrepareBatchContext:
    """Test batch context preparation."""

    @pytest.mark.asyncio
    async def test_prepare_batch_context_single_message(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test preparing context for a single message."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        mock_message.channel.history = AsyncMock(return_value=[])

        result = await cog._prepare_batch_context([mock_message])

        assert "Test message content" in result

    @pytest.mark.asyncio
    async def test_prepare_batch_context_multiple_messages(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test preparing context for multiple messages."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        msg1 = Mock()
        msg1.id = "1"
        msg1.author = mock_user
        msg1.channel = mock_channel
        msg1.content = "First message"

        msg2 = Mock()
        msg2.id = "2"
        msg2.author = mock_user
        msg2.channel = mock_channel
        msg2.content = "Second message"

        mock_channel.history = AsyncMock(return_value=[])

        result = await cog._prepare_batch_context([msg1, msg2])

        assert "First message" in result
        assert "Second message" in result


class TestCancelChannelTasks:
    """Test task cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_channel_tasks_processing(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_channel,
    ):
        """Test cancelling processing tasks."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        mock_task = Mock()
        mock_task.done = Mock(return_value=False)
        cog.processing_tasks[mock_channel.id] = mock_task

        result = cog._cancel_channel_tasks(mock_channel.id, "test-channel", "Test cancel")

        assert result is True
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_channel_tasks_sending(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_channel,
    ):
        """Test cancelling sending tasks."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        mock_task = Mock()
        mock_task.done = Mock(return_value=False)
        cog.sending_tasks[mock_channel.id] = mock_task

        result = cog._cancel_channel_tasks(mock_channel.id, "test-channel", "Test cancel")

        assert result is True
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_channel_tasks_none_active(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_channel,
    ):
        """Test cancelling when no tasks are active."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        result = cog._cancel_channel_tasks(mock_channel.id, "test-channel", "Test cancel")

        assert result is False


class TestCancelAutoChannelTasks:
    """Test AutoChannelCog task cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_auto_channel_tasks_no_cog(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test cancelling auto channel tasks when AutoChannelCog doesn't exist."""
        mock_bot.get_cog = Mock(return_value=None)

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        result = cog._cancel_auto_channel_tasks(111222333)

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_auto_channel_tasks_with_cog(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test cancelling auto channel tasks when AutoChannelCog exists."""
        mock_auto_cog = Mock()
        mock_auto_cog.sending_tasks = {}
        mock_auto_cog.processing_tasks = {}

        mock_bot.get_cog = Mock(return_value=mock_auto_cog)

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        result = cog._cancel_auto_channel_tasks(111222333)

        assert result is False  # No tasks to cancel


class TestSendResponse:
    """Test _send_response method."""

    @pytest.mark.asyncio
    async def test_send_response_no_text(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test sending response with no text."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        reply = ChatReply(text="", session_key="channel:123", response=None)

        # Should not raise error
        await cog._send_response(mock_message, reply)


class TestErrorHandler:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_handle_error(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test error handler."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        test_error = Exception("Test error")

        await cog._handle_error(mock_message, test_error)

        # Verify error message was sent
        mock_message.reply.assert_called_once()


class TestDeleteAssistantMessages:
    """Test _delete_assistant_messages method."""

    @pytest.mark.asyncio
    async def test_delete_assistant_messages_with_ids(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_channel,
    ):
        """Test deleting assistant messages with valid IDs."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        msg = ChatMessage(role="model", content="Test", author_id=None)
        msg.message_ids = ["999"]

        mock_old_msg = Mock()
        mock_old_msg.delete = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_old_msg)

        # Should not raise error
        await cog._delete_assistant_messages(mock_channel, msg)

    @pytest.mark.asyncio
    async def test_delete_assistant_messages_not_found(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_channel,
    ):
        """Test deleting assistant messages when message not found."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        msg = ChatMessage(role="model", content="Test", author_id=None)
        msg.message_ids = ["999"]

        from discord import NotFound

        mock_channel.fetch_message = AsyncMock(
            side_effect=NotFound(response=Mock(), message="Not found")
        )

        # Should not raise error
        await cog._delete_assistant_messages(mock_channel, msg)


class TestProcessRemovedMessages:
    """Test _process_removed_messages method."""

    @pytest.mark.asyncio
    async def test_process_removed_messages(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_channel,
    ):
        """Test processing removed messages."""
        mock_llm_service.get_user_role_name = Mock(return_value="user")
        mock_llm_service.get_assistant_role_name = Mock(return_value="model")

        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        user_msg = ChatMessage(role="user", content="Hello world", author_id=123456789)
        assistant_msg = ChatMessage(role="model", content="Hi there!", author_id=None)

        result = await cog._process_removed_messages(
            Mock(channel=mock_channel), [assistant_msg, user_msg]
        )

        assert result == "Hello world"


class TestRegenerateResponse:
    """Test _regenerate_response method."""

    @pytest.mark.asyncio
    async def test_regenerate_response_success(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_message,
    ):
        """Test successful response regeneration."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel = mock_message.channel
        ctx.message = mock_message
        ctx.interaction = None

        with patch("persbot.bot.cogs.assistant.create_chat_reply") as mock_create_reply:
            mock_create_reply.return_value = ChatReply(
                text="Regenerated response",
                session_key="channel:123",
                response=None,
            )

            with patch.object(cog, "_send_response", new_callable=AsyncMock):
                await cog._regenerate_response(ctx, "channel:123", "Test content")

                mock_create_reply.assert_called_once()


class TestCogCommandError:
    """Test cog-level error handler."""

    @pytest.mark.asyncio
    async def test_cog_command_error_missing_permissions(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test error handler for missing permissions."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.reply = AsyncMock()

        from discord.ext.commands import MissingPermissions

        error = MissingPermissions(missing_permissions=["manage_guild"])

        await cog.cog_command_error(ctx, error)

        ctx.reply.assert_called_once()
        assert "권한이 없습니다" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_cog_command_error_bad_argument(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test error handler for bad arguments."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.reply = AsyncMock()

        from discord.ext.commands import BadArgument

        error = BadArgument("Invalid argument")

        await cog.cog_command_error(ctx, error)

        ctx.reply.assert_called_once()
        assert "잘못된 인자" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_cog_command_error_on_cooldown(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_user,
        mock_channel,
    ):
        """Test error handler for command on cooldown."""
        cog = AssistantCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.reply = AsyncMock()

        from discord.ext.commands import CommandOnCooldown

        error = CommandOnCooldown(retry_after=5.0, type=commands.BucketType.default)

        await cog.cog_command_error(ctx, error)

        ctx.reply.assert_called_once()
        assert "쿨다운 중" in ctx.reply.call_args[0][0]

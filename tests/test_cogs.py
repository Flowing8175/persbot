"""Tests for Discord Cogs.

Tests focus on:
- ActiveAPICall: cancel() method
- BaseChatCog: initialization, check_guild_admin_permission, _should_use_streaming, _cancel_active_tasks
- AutoChannelCog: _load_dynamic_channels, _save_dynamic_channels, _validate_undo_arg, _check_undo_permission
- HelpCog: _get_provider_label, get_category_options, get_category_summaries, get_category_data
- HelpView: update_components, build_embed
"""

import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from collections import deque

import pytest
import discord
from discord.ext import commands

from persbot.bot.cogs.base import BaseChatCog, ActiveAPICall
from persbot.bot.cogs.auto_channel import AutoChannelCog
from persbot.bot.cogs.help import HelpCog, HelpView
from persbot.config import AppConfig
from persbot.bot.session import SessionManager
from persbot.services.llm_service import LLMService
from persbot.tools.manager import ToolManager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_bot():
    """Create a mock Discord bot."""
    bot = Mock(spec=commands.Bot)
    bot.wait_until_ready = AsyncMock()
    return bot


@pytest.fixture
def mock_config():
    """Create a mock AppConfig."""
    config = Mock(spec=AppConfig)
    config.message_buffer_delay = 2.0
    config.break_cut_mode = False
    config.no_check_permission = False
    config.command_prefix = "!"
    config.auto_reply_channel_ids = (123, 456)
    config.assistant_llm_provider = "gemini"
    return config


@pytest.fixture
def mock_llm_service():
    """Create a mock LLMService."""
    service = Mock(spec=LLMService)
    service.get_user_role_name = Mock(return_value="user")
    service.get_assistant_role_name = Mock(return_value="assistant")
    return service


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager."""
    manager = Mock(spec=SessionManager)
    manager.sessions = {}
    manager.undo_last_exchanges = Mock(return_value=[])
    return manager


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager."""
    return Mock(spec=ToolManager)


@pytest.fixture
def mock_context():
    """Create a mock Discord context."""
    ctx = Mock(spec=commands.Context)
    ctx.author = Mock(spec=discord.Member)
    ctx.author.id = 789
    ctx.author.display_name = "TestUser"
    ctx.author.guild_permissions = Mock()
    ctx.author.guild_permissions.manage_guild = True
    ctx.channel = Mock()
    ctx.channel.id = 123
    ctx.channel.name = "test-channel"
    ctx.message = Mock()
    ctx.message.add_reaction = AsyncMock()
    ctx.message.delete = AsyncMock()
    ctx.reply = AsyncMock()
    ctx.send = AsyncMock()
    ctx.send_help = Mock()
    return ctx


@pytest.fixture
def mock_message():
    """Create a mock Discord message."""
    msg = Mock(spec=discord.Message)
    msg.id = 999
    msg.content = "test message"
    msg.author = Mock()
    msg.author.id = 789
    msg.author.bot = False
    msg.channel = Mock()
    msg.channel.id = 123
    msg.channel.name = "test-channel"
    msg.channel.typing = Mock()
    msg.channel.typing.return_value.__aenter__ = AsyncMock()
    msg.channel.typing.return_value.__aexit__ = AsyncMock()
    msg.attachments = []
    return msg


# =============================================================================
# ActiveAPICall Tests
# =============================================================================

class TestActiveAPICall:
    """Tests for ActiveAPICall class."""

    def test_initialization(self):
        """ActiveAPICall initializes with task and cancel_event."""
        task = Mock()
        event = asyncio.Event()

        api_call = ActiveAPICall(task, event)

        assert api_call.task == task
        assert api_call.cancel_event == event

    def test_cancel_sets_event(self):
        """cancel() sets the cancel_event."""
        event = asyncio.Event()
        task = Mock()
        task.done = Mock(return_value=False)

        api_call = ActiveAPICall(task, event)
        api_call.cancel()

        assert event.is_set()

    def test_cancel_cancels_task(self):
        """cancel() cancels the task if not done."""
        task = Mock()
        task.done = Mock(return_value=False)
        task.cancel = Mock()
        event = asyncio.Event()

        api_call = ActiveAPICall(task, event)
        api_call.cancel()

        task.cancel.assert_called_once()

    def test_cancel_skips_done_task(self):
        """cancel() does not cancel already done tasks."""
        task = Mock()
        task.done = Mock(return_value=True)
        task.cancel = Mock()
        event = asyncio.Event()

        api_call = ActiveAPICall(task, event)
        api_call.cancel()

        task.cancel.assert_not_called()

    def test_cancel_handles_none_task(self):
        """cancel() handles None task gracefully."""
        event = asyncio.Event()
        api_call = ActiveAPICall(None, event)

        # Should not raise
        api_call.cancel()
        assert event.is_set()

    def test_cancel_handles_none_event(self):
        """cancel() handles None event gracefully."""
        task = Mock()
        task.done = Mock(return_value=False)
        task.cancel = Mock()

        api_call = ActiveAPICall(task, None)

        # Should not raise
        api_call.cancel()
        task.cancel.assert_called_once()


# =============================================================================
# BaseChatCog Tests
# =============================================================================

class TestBaseChatCogInit:
    """Tests for BaseChatCog initialization."""

    def test_initialization(self, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """BaseChatCog initializes with all required attributes."""
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        assert cog.bot == mock_bot
        assert cog.config == mock_config
        assert cog.llm_service == mock_llm_service
        assert cog.session_manager == mock_session_manager
        assert cog.message_buffer is not None
        assert cog.message_buffer.default_delay == mock_config.message_buffer_delay
        assert cog.sending_tasks == {}
        assert cog.processing_tasks == {}
        assert cog.active_batches == {}
        assert cog.cancellation_signals == {}
        assert cog.active_api_calls == {}

    def test_initialization_with_tool_manager(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_tool_manager):
        """BaseChatCog initializes with tool_manager."""
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_tool_manager)

        assert cog.tool_manager == mock_tool_manager

    def test_initialization_without_tool_manager(self, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """BaseChatCog initializes without tool_manager (optional)."""
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        assert cog.tool_manager is None


class TestBaseChatCogPermissions:
    """Tests for BaseChatCog permission checking."""

    @pytest.mark.asyncio
    async def test_check_guild_admin_with_no_check_permission(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """check_guild_admin_permission returns True when no_check_permission is True."""
        mock_config.no_check_permission = True
        mock_context.author.guild_permissions.manage_guild = False

        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        result = await cog.check_guild_admin_permission(mock_context)

        assert result is True
        mock_context.reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_guild_admin_with_non_member_author(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """check_guild_admin_permission returns False and sends error for non-Member authors."""
        mock_config.no_check_permission = False
        mock_context.author = Mock(spec=discord.User)  # Not a Member

        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        result = await cog.check_guild_admin_permission(mock_context)

        assert result is False
        mock_context.reply.assert_called_once()
        assert "권한이 없습니다" in mock_context.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_check_guild_admin_without_permission(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """check_guild_admin_permission returns False when user lacks manage_guild permission."""
        mock_config.no_check_permission = False
        mock_context.author.guild_permissions.manage_guild = False

        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        result = await cog.check_guild_admin_permission(mock_context)

        assert result is False
        mock_context.reply.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_guild_admin_with_permission(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """check_guild_admin_permission returns True when user has manage_guild permission."""
        mock_config.no_check_permission = False
        mock_context.author.guild_permissions.manage_guild = True

        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        result = await cog.check_guild_admin_permission(mock_context)

        assert result is True
        mock_context.reply.assert_not_called()


class TestBaseChatCogStreaming:
    """Tests for BaseChatCog streaming behavior."""

    def test_should_use_streaming_with_break_cut_mode(self, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_should_use_streaming returns True when break_cut_mode is enabled."""
        mock_config.break_cut_mode = True

        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        result = cog._should_use_streaming()

        assert result is True

    def test_should_use_streaming_without_break_cut_mode(self, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_should_use_streaming returns False when break_cut_mode is disabled."""
        mock_config.break_cut_mode = False

        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        result = cog._should_use_streaming()

        assert result is False


class TestBaseChatCogCancelTasks:
    """Tests for BaseChatCog _cancel_active_tasks."""

    def test_cancel_active_tasks_returns_empty_when_no_tasks(self, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_cancel_active_tasks returns empty list when no tasks exist."""
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        result = cog._cancel_active_tasks(123, "TestUser")

        assert result == []

    def test_cancel_active_tasks_with_api_call(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_message):
        """_cancel_active_tasks cancels active API call but does NOT return messages.

        When API call is cancelled, the LLM service saves partial response to history,
        so we don't prepend old messages to avoid duplication.
        """
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        # Setup active API call
        event = asyncio.Event()
        task = Mock()
        task.done = Mock(return_value=False)
        task.cancel = Mock()
        cog.active_api_calls[123] = ActiveAPICall(task, event)
        cog.active_batches[123] = [mock_message]

        result = cog._cancel_active_tasks(123, "TestUser")

        # Should NOT return messages - partial response is saved to history
        assert result == []
        assert event.is_set()
        task.cancel.assert_called_once()

    def test_cancel_active_tasks_with_sending_task_in_break_cut_mode(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_message):
        """_cancel_active_tasks cancels sending task in break-cut mode."""
        mock_config.break_cut_mode = True
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        # Setup sending task
        task = Mock()
        task.done = Mock(return_value=False)
        task.cancel = Mock()
        cog.sending_tasks[123] = task
        cog.active_batches[123] = [mock_message]

        result = cog._cancel_active_tasks(123, "TestUser")

        assert result == [mock_message]
        task.cancel.assert_called_once()

    def test_cancel_active_tasks_with_processing_task(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_message):
        """_cancel_active_tasks cancels processing task."""
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        # Setup processing task
        task = Mock()
        task.done = Mock(return_value=False)
        task.cancel = Mock()
        cog.processing_tasks[123] = task
        cog.active_batches[123] = [mock_message]

        result = cog._cancel_active_tasks(123, "TestUser")

        assert result == [mock_message]
        task.cancel.assert_called_once()

    def test_cancel_active_tasks_sets_cancellation_signal(self, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_cancel_active_tasks sets cancellation signal."""
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        event = asyncio.Event()
        cog.cancellation_signals[123] = event

        cog._cancel_active_tasks(123, "TestUser")

        assert event.is_set()

    def test_cancel_active_tasks_no_prepend_when_already_done(self, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_cancel_active_tasks returns empty list when tasks are already done."""
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        # Setup done task
        task = Mock()
        task.done = Mock(return_value=True)
        task.cancel = Mock()
        cog.active_api_calls[123] = ActiveAPICall(task, asyncio.Event())

        result = cog._cancel_active_tasks(123, "TestUser")

        assert result == []
        task.cancel.assert_not_called()

    def test_cancel_active_tasks_prevents_duplicate_prepends(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_message):
        """_cancel_active_tasks does NOT return messages when API call is cancelled.

        When API call is cancelled, partial response is saved to history, so we
        don't prepend old messages. This prevents duplicate user messages in API calls.
        """
        mock_config.break_cut_mode = True
        cog = BaseChatCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        # Setup multiple task sources with same batch
        api_task = Mock()
        api_task.done = Mock(return_value=False)
        api_task.cancel = Mock()

        send_task = Mock()
        send_task.done = Mock(return_value=False)
        send_task.cancel = Mock()

        cog.active_api_calls[123] = ActiveAPICall(api_task, asyncio.Event())
        cog.sending_tasks[123] = send_task
        cog.active_batches[123] = [mock_message]

        result = cog._cancel_active_tasks(123, "TestUser")

        # Should NOT return messages because API call was cancelled
        # (partial response is saved to history)
        assert result == []
        api_task.cancel.assert_called_once()
        send_task.cancel.assert_called_once()


# =============================================================================
# AutoChannelCog Tests
# =============================================================================

class TestAutoChannelCogInit:
    """Tests for AutoChannelCog initialization."""

    @patch('asyncio.create_task')
    def test_initialization(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """AutoChannelCog initializes correctly."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        assert cog.json_file_path == Path("data/auto_channels.json")
        assert isinstance(cog.dynamic_channel_ids, set)
        assert cog.env_channel_ids == set(mock_config.auto_reply_channel_ids)

    @patch('asyncio.create_task')
    def test_initialization_creates_load_task(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """AutoChannelCog creates async task to load dynamic channels."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        mock_create_task.assert_called_once()


class TestAutoChannelCogLoadChannels:
    """Tests for AutoChannelCog._load_dynamic_channels."""

    @pytest.mark.asyncio
    @patch('asyncio.create_task')
    async def test_load_dynamic_channels_waits_for_bot(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_load_dynamic_channels waits for bot to be ready."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        await cog._load_dynamic_channels()

        mock_bot.wait_until_ready.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncio.create_task')
    async def test_load_dynamic_channels_with_valid_file(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, tmp_path):
        """_load_dynamic_channels loads channels from JSON file."""
        # Create temporary JSON file
        json_file = tmp_path / "auto_channels.json"
        json_file.write_text(json.dumps([111, 222, 333]))

        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.json_file_path = json_file

        await cog._load_dynamic_channels()

        assert cog.dynamic_channel_ids == {111, 222, 333}
        # Should merge with env channels
        assert 123 in mock_config.auto_reply_channel_ids
        assert 456 in mock_config.auto_reply_channel_ids

    @pytest.mark.asyncio
    @patch('asyncio.create_task')
    async def test_load_dynamic_channels_with_nonexistent_file(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_load_dynamic_channels handles nonexistent file gracefully."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.json_file_path = Path("nonexistent_file.json")

        # Should not raise
        await cog._load_dynamic_channels()

        assert cog.dynamic_channel_ids == set()

    @pytest.mark.asyncio
    @patch('asyncio.create_task')
    async def test_load_dynamic_channels_with_invalid_json(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, tmp_path):
        """_load_dynamic_channels handles invalid JSON gracefully."""
        # Create invalid JSON file
        json_file = tmp_path / "auto_channels.json"
        json_file.write_text("invalid json")

        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.json_file_path = json_file

        # Should not raise
        await cog._load_dynamic_channels()

        assert cog.dynamic_channel_ids == set()

    @pytest.mark.asyncio
    @patch('asyncio.create_task')
    async def test_load_dynamic_channels_with_non_list_data(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, tmp_path):
        """_load_dynamic_channels handles non-list data in JSON."""
        # Create JSON with dict instead of list
        json_file = tmp_path / "auto_channels.json"
        json_file.write_text(json.dumps({"channels": [111, 222]}))

        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.json_file_path = json_file

        await cog._load_dynamic_channels()

        assert cog.dynamic_channel_ids == set()


class TestAutoChannelCogSaveChannels:
    """Tests for AutoChannelCog._save_dynamic_channels."""

    @pytest.mark.asyncio
    @patch('asyncio.create_task')
    async def test_save_dynamic_channels_creates_directory(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, tmp_path):
        """_save_dynamic_channels creates parent directory if needed."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.json_file_path = tmp_path / "subdir" / "auto_channels.json"
        cog.dynamic_channel_ids = {111, 222}

        await cog._save_dynamic_channels()

        assert cog.json_file_path.parent.exists()
        assert cog.json_file_path.exists()

    @pytest.mark.asyncio
    @patch('asyncio.create_task')
    async def test_save_dynamic_channels_writes_json(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, tmp_path):
        """_save_dynamic_channels writes channels to JSON file."""
        json_file = tmp_path / "auto_channels.json"
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.json_file_path = json_file
        cog.dynamic_channel_ids = {111, 222, 333}
        cog.env_channel_ids = set()

        await cog._save_dynamic_channels()

        content = json_file.read_text()
        saved_ids = json.loads(content)
        # The saved IDs should be the dynamic channels (a list)
        assert set(saved_ids) == {111, 222, 333}

    @pytest.mark.asyncio
    @patch('asyncio.create_task')
    async def test_save_dynamic_channels_updates_config(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, tmp_path):
        """_save_dynamic_channels updates config with merged channels."""
        json_file = tmp_path / "auto_channels.json"
        mock_config.auto_reply_channel_ids = (123, 456)
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.json_file_path = json_file
        cog.dynamic_channel_ids = {111, 222}
        cog.env_channel_ids = {123, 456}

        await cog._save_dynamic_channels()

        # Config should have union of env and dynamic channels
        assert set(mock_config.auto_reply_channel_ids) == {111, 222, 123, 456}


class TestAutoChannelCogValidateUndoArg:
    """Tests for AutoChannelCog._validate_undo_arg."""

    @patch('asyncio.create_task')
    def test_validate_undo_arg_valid_positive_integer(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_validate_undo_arg returns int for valid positive integer."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        result = cog._validate_undo_arg("5")

        assert result == 5

    @patch('asyncio.create_task')
    def test_validate_undo_arg_with_string_number(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_validate_undo_arg returns int for string number."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        result = cog._validate_undo_arg("10")

        assert result == 10

    @patch('asyncio.create_task')
    def test_validate_undo_arg_defaults_to_one(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_validate_undo_arg defaults to 1 when "1" is passed."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        result = cog._validate_undo_arg("1")

        assert result == 1

    @patch('asyncio.create_task')
    def test_validate_undo_arg_rejects_zero(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_validate_undo_arg returns None for zero."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        result = cog._validate_undo_arg("0")

        assert result is None

    @patch('asyncio.create_task')
    def test_validate_undo_arg_rejects_negative(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_validate_undo_arg returns None for negative numbers."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        result = cog._validate_undo_arg("-5")

        assert result is None

    @patch('asyncio.create_task')
    def test_validate_undo_arg_rejects_invalid_string(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager):
        """_validate_undo_arg returns None for invalid string."""
        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)

        result = cog._validate_undo_arg("abc")

        assert result is None


class TestAutoChannelCogCheckUndoPermission:
    """Tests for AutoChannelCog._check_undo_permission."""

    @patch('asyncio.create_task')
    def test_check_undo_permission_with_admin(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """_check_undo_permission returns True for admins."""
        mock_context.author.guild_permissions.manage_guild = True

        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.session_manager.sessions = {}

        result = cog._check_undo_permission(mock_context, "channel:123")

        assert result is True

    @patch('asyncio.create_task')
    def test_check_undo_permission_with_user_with_enough_messages(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """_check_undo_permission returns True for users with 5+ messages."""
        mock_context.author.guild_permissions.manage_guild = False
        mock_context.author.id = 789

        # Create mock session with user messages
        mock_session = Mock()
        user_msgs = [Mock(role="user", author_id=789) for _ in range(5)]
        mock_session.chat.history = user_msgs

        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.session_manager.sessions = {"channel:123": mock_session}

        result = cog._check_undo_permission(mock_context, "channel:123")

        assert result is True

    @patch('asyncio.create_task')
    def test_check_undo_permission_with_user_insufficient_messages(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """_check_undo_permission returns False for users with fewer than 5 messages."""
        mock_context.author.guild_permissions.manage_guild = False
        mock_context.author.id = 789

        # Create mock session with fewer messages
        mock_session = Mock()
        user_msgs = [Mock(role="user", author_id=789) for _ in range(2)]
        mock_session.chat.history = user_msgs

        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.session_manager.sessions = {"channel:123": mock_session}

        result = cog._check_undo_permission(mock_context, "channel:123")

        assert result is False

    @patch('asyncio.create_task')
    def test_check_undo_permission_counts_only_user_messages(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """_check_undo_permission only counts messages from the user."""
        mock_context.author.guild_permissions.manage_guild = False
        mock_context.author.id = 789

        # Create mock session with mixed messages
        mock_session = Mock()
        user_msgs = [Mock(role="user", author_id=789) for _ in range(5)]
        other_msgs = [Mock(role="user", author_id=888), Mock(role="assistant", author_id=None)]
        mock_session.chat.history = user_msgs + other_msgs

        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.session_manager.sessions = {"channel:123": mock_session}

        result = cog._check_undo_permission(mock_context, "channel:123")

        assert result is True  # Has 5 user messages

    @patch('asyncio.create_task')
    def test_check_undo_permission_with_no_check_permission(self, mock_create_task, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_context):
        """_check_undo_permission bypasses check when no_check_permission is True."""
        mock_config.no_check_permission = True
        mock_context.author.guild_permissions.manage_guild = False

        cog = AutoChannelCog(mock_bot, mock_config, mock_llm_service, mock_session_manager)
        cog.session_manager.sessions = {}

        result = cog._check_undo_permission(mock_context, "channel:123")

        assert result is True


# =============================================================================
# HelpCog Tests
# =============================================================================

class TestHelpCogInit:
    """Tests for HelpCog initialization."""

    def test_initialization(self, mock_bot, mock_config):
        """HelpCog initializes correctly."""
        mock_config.assistant_llm_provider = "gemini"

        cog = HelpCog(mock_bot, mock_config)

        assert cog.bot == mock_bot
        assert cog.config == mock_config
        assert cog.ai_provider_label == "Google Gemini"

    def test_initialization_with_openai_provider(self, mock_bot, mock_config):
        """HelpCog initializes with OpenAI provider."""
        mock_config.assistant_llm_provider = "openai"

        cog = HelpCog(mock_bot, mock_config)

        assert cog.ai_provider_label == "OpenAI (GPT)"

    def test_initialization_with_zai_provider(self, mock_bot, mock_config):
        """HelpCog initializes with Z.AI provider."""
        mock_config.assistant_llm_provider = "zai"

        cog = HelpCog(mock_bot, mock_config)

        assert cog.ai_provider_label == "Z.AI"


class TestHelpCogProviderLabel:
    """Tests for HelpCog._get_provider_label."""

    def test_get_provider_label_gemini(self, mock_bot, mock_config):
        """_get_provider_label returns correct label for Gemini."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog._get_provider_label("gemini")

        assert result == "Google Gemini"

    def test_get_provider_label_openai(self, mock_bot, mock_config):
        """_get_provider_label returns correct label for OpenAI."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog._get_provider_label("openai")

        assert result == "OpenAI (GPT)"

    def test_get_provider_label_zai(self, mock_bot, mock_config):
        """_get_provider_label returns correct label for Z.AI."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog._get_provider_label("zai")

        assert result == "Z.AI"

    def test_get_provider_label_case_insensitive(self, mock_bot, mock_config):
        """_get_provider_label is case-insensitive."""
        cog = HelpCog(mock_bot, mock_config)

        assert cog._get_provider_label("GEMINI") == "Google Gemini"
        assert cog._get_provider_label("OpenAI") == "OpenAI (GPT)"
        assert cog._get_provider_label("ZAI") == "Z.AI"

    def test_get_provider_label_unknown_provider(self, mock_bot, mock_config):
        """_get_provider_label returns original string for unknown providers."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog._get_provider_label("unknown_provider")

        assert result == "unknown_provider"


class TestHelpCogCategoryOptions:
    """Tests for HelpCog.get_category_options."""

    def test_get_category_options_returns_list(self, mock_bot, mock_config):
        """get_category_options returns a list of option dicts."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_options()

        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_category_options_has_required_fields(self, mock_bot, mock_config):
        """get_category_options returns dicts with required fields."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_options()

        for option in result:
            assert "label" in option
            assert "value" in option
            assert "description" in option
            assert "emoji" in option

    def test_get_category_options_expected_categories(self, mock_bot, mock_config):
        """get_category_options returns expected categories."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_options()

        values = [opt["value"] for opt in result]
        assert "대화" in values
        assert "요약" in values
        assert "페르소나" in values
        assert "모델" in values
        assert "설정" in values
        assert "자동채널" in values


class TestHelpCogCategorySummaries:
    """Tests for HelpCog.get_category_summaries."""

    def test_get_category_summaries_returns_list(self, mock_bot, mock_config):
        """get_category_summaries returns a list of tuples."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_summaries()

        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_category_summaries_has_three_elements(self, mock_bot, mock_config):
        """get_category_summaries returns tuples with (emoji, name, summary)."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_summaries()

        for summary in result:
            assert isinstance(summary, tuple)
            assert len(summary) == 3

    def test_get_category_summaries_expected_categories(self, mock_bot, mock_config):
        """get_category_summaries includes all expected categories."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_summaries()

        names = [s[1] for s in result]
        assert "대화" in names
        assert "요약" in names
        assert "페르소나" in names
        assert "모델" in names
        assert "설정" in names
        assert "자동채널" in names


class TestHelpCogCategoryData:
    """Tests for HelpCog.get_category_data."""

    def test_get_category_data_returns_dict(self, mock_bot, mock_config):
        """get_category_data returns dict for valid category."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_data("대화")

        assert isinstance(result, dict)

    def test_get_category_data_has_required_fields(self, mock_bot, mock_config):
        """get_category_data dict has required fields."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_data("대화")

        assert "title" in result
        assert "description" in result
        assert "color" in result
        assert isinstance(result["color"], discord.Color)

    def test_get_category_data_has_tips(self, mock_bot, mock_config):
        """get_category_data includes tips field."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_data("대화")

        assert "tips" in result

    def test_get_category_data_all_categories(self, mock_bot, mock_config):
        """get_category_data returns data for all categories."""
        cog = HelpCog(mock_bot, mock_config)

        categories = ["대화", "요약", "페르소나", "모델", "설정", "자동채널"]

        for category in categories:
            result = cog.get_category_data(category)
            assert result is not None
            assert "title" in result

    def test_get_category_data_invalid_category(self, mock_bot, mock_config):
        """get_category_data returns None for invalid category."""
        cog = HelpCog(mock_bot, mock_config)

        result = cog.get_category_data("invalid_category")

        assert result is None


# =============================================================================
# HelpView Tests
# =============================================================================

class TestHelpViewInit:
    """Tests for HelpView initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_bot, mock_config, mock_context):
        """HelpView initializes correctly."""
        cog = HelpCog(mock_bot, mock_config)

        view = HelpView(cog, mock_context)

        assert view.cog == cog
        assert view.ctx == mock_context
        assert view.current_category is None
        assert view.message is None
        assert view.timeout == 600

    @pytest.mark.asyncio
    async def test_initialization_adds_items(self, mock_bot, mock_config, mock_context):
        """HelpView adds components on initialization."""
        cog = HelpCog(mock_bot, mock_config)

        view = HelpView(cog, mock_context)

        # Should have children (discord.ui.View uses children property)
        assert len(view.children) > 0


class TestHelpViewUpdateComponents:
    """Tests for HelpView.update_components."""

    @pytest.mark.asyncio
    async def test_update_components_clears_items(self, mock_bot, mock_config, mock_context):
        """update_components clears existing items."""
        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)

        view.update_components()

        # Items should be cleared and re-added
        assert len(view.children) > 0

    @pytest.mark.asyncio
    async def test_update_components_main_menu(self, mock_bot, mock_config, mock_context):
        """update_components shows category select on main menu."""
        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = None

        view.update_components()

        # Should have select dropdown and close button
        item_types = [type(item) for item in view.children]
        assert discord.ui.Select in item_types
        assert discord.ui.Button in item_types

    @pytest.mark.asyncio
    async def test_update_components_category_view(self, mock_bot, mock_config, mock_context):
        """update_components shows back button when viewing category."""
        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = "대화"

        view.update_components()

        # Should have back button and close button
        item_types = [type(item) for item in view.children]
        assert discord.ui.Button in item_types
        # No select on category view
        select_items = [item for item in view.children if isinstance(item, discord.ui.Select)]
        assert len(select_items) == 0


class TestHelpViewBuildEmbed:
    """Tests for HelpView.build_embed."""

    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_build_embed_main_menu(self, mock_get_loop, mock_bot, mock_config, mock_context):
        """build_embed returns main embed when on main menu."""
        # Mock the event loop
        mock_loop = Mock()
        mock_loop.time = Mock(return_value=0)
        mock_get_loop.return_value = mock_loop

        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = None

        embed = view.build_embed()

        assert isinstance(embed, discord.Embed)
        assert "Persbot 도움말" in embed.title

    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_build_embed_category_view(self, mock_get_loop, mock_bot, mock_config, mock_context):
        """build_embed returns category embed when viewing category."""
        # Mock the event loop
        mock_loop = Mock()
        mock_loop.time = Mock(return_value=0)
        mock_get_loop.return_value = mock_loop

        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = "대화"

        embed = view.build_embed()

        assert isinstance(embed, discord.Embed)
        assert "대화" in embed.title

    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_build_embed_main_menu_has_summaries(self, mock_get_loop, mock_bot, mock_config, mock_context):
        """build_embed main menu includes category summaries."""
        # Mock the event loop
        mock_loop = Mock()
        mock_loop.time = Mock(return_value=0)
        mock_get_loop.return_value = mock_loop

        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = None

        embed = view.build_embed()

        assert len(embed.fields) > 0
        assert embed.footer

    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_build_embed_category_embed_has_color(self, mock_get_loop, mock_bot, mock_config, mock_context):
        """build_embed category embed has appropriate color."""
        # Mock the event loop
        mock_loop = Mock()
        mock_loop.time = Mock(return_value=0)
        mock_get_loop.return_value = mock_loop

        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = "대화"

        embed = view.build_embed()

        assert embed.color == discord.Color.blue()

    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_build_main_embed_system_info(self, mock_get_loop, mock_bot, mock_config, mock_context):
        """build_main_embed includes system info field."""
        # Mock the event loop
        mock_loop = Mock()
        mock_loop.time = Mock(return_value=0)
        mock_get_loop.return_value = mock_loop

        mock_config.assistant_llm_provider = "gemini"
        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = None

        embed = view.build_main_embed()

        # Check for system info field
        system_fields = [f for f in embed.fields if "시스템 정보" in f.name]
        assert len(system_fields) > 0
        assert "Google Gemini" in system_fields[0].value

    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_build_category_embed_has_tips(self, mock_get_loop, mock_bot, mock_config, mock_context):
        """build_category_embed includes tips when available."""
        # Mock the event loop
        mock_loop = Mock()
        mock_loop.time = Mock(return_value=0)
        mock_get_loop.return_value = mock_loop

        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = "대화"

        embed = view.build_category_embed()

        # Should have tips field
        assert len(embed.fields) > 0
        assert embed.footer

    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_build_category_embed_invalid_category_fallback(self, mock_get_loop, mock_bot, mock_config, mock_context):
        """build_category_embed falls back to main embed for invalid category."""
        # Mock the event loop
        mock_loop = Mock()
        mock_loop.time = Mock(return_value=0)
        mock_get_loop.return_value = mock_loop

        cog = HelpCog(mock_bot, mock_config)
        view = HelpView(cog, mock_context)
        view.current_category = "invalid_category"

        embed = view.build_category_embed()

        # Should return main embed
        assert "Persbot 도움말" in embed.title

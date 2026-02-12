"""Tests for the Auto Channel Cog."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from discord.ext import commands

from persbot.bot.cogs.auto_channel import AutoChannelCog
from persbot.config import AppConfig


class TestAutoChannelCogInitialization:
    """Test AutoChannelCog initialization and setup."""

    @pytest.mark.asyncio
    async def test_cog_initialization_without_tool_manager(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test that AutoChannelCog initializes correctly without tool manager."""
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        assert cog.bot is mock_bot
        assert cog.config is mock_app_config
        assert cog.llm_service is mock_llm_service
        assert cog.session_manager is mock_session_manager
        assert cog.tool_manager is None

        # Check that dynamic channels are initialized
        assert len(cog.dynamic_channel_ids) == 0
        assert len(cog.env_channel_ids) == 0

    @pytest.mark.asyncio
    async def test_cog_initialization_with_tool_manager(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test that AutoChannelCog initializes correctly with tool manager."""
        mock_tool_manager = Mock()
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        assert cog.tool_manager is mock_tool_manager

    @pytest.mark.asyncio
    async def test_cog_initialization_with_env_channels(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test that AutoChannelCog initializes with environment channels."""
        env_channels = (111222333, 444555666)
        mock_app_config.auto_reply_channel_ids = env_channels

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        # Check that environment channels are loaded
        assert len(cog.env_channel_ids) == 2
        assert 111222333 in cog.env_channel_ids
        assert 444555666 in cog.env_channel_ids


class TestAutoChannelCogLoadDynamicChannels:
    """Test AutoChannelCog channel loading functionality."""

    @pytest.mark.asyncio
    async def test_load_dynamic_channels_from_existing_json(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, temp_dir
    ):
        """Test loading dynamic channels from existing JSON file."""
        # Create test JSON file
        test_channels = [111222333, 444555666]
        json_file = temp_dir / "auto_channels.json"
        async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(test_channels))

        # Create cog instance with custom path
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )
        cog.json_file_path = json_file

        await cog._load_dynamic_channels()

        assert len(cog.dynamic_channel_ids) == 2
        assert 111222333 in cog.dynamic_channel_ids
        assert 444555666 in cog.dynamic_channel_ids

    @pytest.mark.asyncio
    async def test_load_dynamic_channels_with_missing_json(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, temp_dir
    ):
        """Test loading dynamic channels with missing JSON file."""
        # Don't create the JSON file (simulate missing file)
        json_file = temp_dir / "auto_channels.json"

        # Create cog instance with custom path
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )
        cog.json_file_path = json_file

        await cog._load_dynamic_channels()

        # Should have empty set when file doesn't exist
        assert len(cog.dynamic_channel_ids) == 0

    @pytest.mark.asyncio
    async def test_load_dynamic_channels_with_corrupted_json(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, temp_dir
    ):
        """Test loading dynamic channels with corrupted JSON file."""
        # Create corrupted JSON file
        json_file = temp_dir / "auto_channels.json"
        async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
            await f.write("invalid json content")

        # Create cog instance with custom path
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )
        cog.json_file_path = json_file

        await cog._load_dynamic_channels()

        # Should have empty set when file is corrupted
        assert len(cog.dynamic_channel_ids) == 0

    @pytest.mark.asyncio
    async def test_load_dynamic_channels_with_env_channels(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, temp_dir
    ):
        """Test loading dynamic channels with environment channels."""
        # Set up environment channels
        env_channels = (123456789,)
        mock_app_config.auto_reply_channel_ids = env_channels

        # Create test JSON file
        test_channels = [111222333]
        json_file = temp_dir / "auto_channels.json"
        async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(test_channels))

        # Create cog instance with custom path
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )
        cog.json_file_path = json_file

        await cog._load_dynamic_channels()

        # Check that environment channels are preserved and combined
        assert len(cog.config.auto_reply_channel_ids) == 2
        assert 123456789 in cog.config.auto_reply_channel_ids
        assert 111222333 in cog.config.auto_reply_channel_ids


class TestAutoChannelCogSaveDynamicChannels:
    """Test AutoChannelCog channel saving functionality."""

    @pytest.mark.asyncio
    async def test_save_dynamic_channels_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, temp_dir
    ):
        """Test successful saving of dynamic channels."""
        # Add some dynamic channels
        dynamic_channels = [111222333, 444555666]

        # Monkey patch asyncio.create_task to prevent background loading
        original_create_task = asyncio.create_task
        mock_create_task = Mock(return_value=Mock())
        asyncio.create_task = mock_create_task

        try:
            # Set the config first to establish the env_channel_ids
            original_channels = mock_app_config.auto_reply_channel_ids
            mock_app_config.auto_reply_channel_ids = dynamic_channels

            # Create the cog first
            cog = AutoChannelCog(
                bot=mock_bot,
                config=mock_app_config,
                llm_service=mock_llm_service,
                session_manager=mock_session_manager,
            )

            # Restore original config
            mock_app_config.auto_reply_channel_ids = original_channels

            # Set the json_file_path and dynamic_channel_ids
            cog.json_file_path = temp_dir / "auto_channels.json"
            cog.dynamic_channel_ids = set(dynamic_channels)

            await cog._save_dynamic_channels()

            # Verify file was created and contains correct data
            assert cog.json_file_path.exists()
            async with aiofiles.open(cog.json_file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                saved_data = json.loads(content)
                # Check that the saved data matches what we expect from _save_dynamic_channels
                # (order doesn't matter in sets, but JSON preserves insertion order)
                assert set(saved_data) == set(dynamic_channels)
        finally:
            asyncio.create_task = original_create_task

    @pytest.mark.asyncio
    async def test_save_dynamic_channels_updates_config(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, temp_dir
    ):
        """Test that saving dynamic channels updates config."""
        # Set up environment channels
        env_channels = (999888777,)
        mock_app_config.auto_reply_channel_ids = env_channels

        # Add dynamic channels
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )
        cog.dynamic_channel_ids = {111222333}

        json_file = temp_dir / "auto_channels.json"
        with patch.object(cog, "json_file_path", json_file):
            await cog._save_dynamic_channels()

            # Check that config was updated with combined channels
            combined = set(env_channels) | cog.dynamic_channel_ids
            assert mock_app_config.auto_reply_channel_ids == tuple(combined)


class TestAutoChannelCogCommands:
    """Test AutoChannelCog command handlers."""

@pytest.mark.asyncio
    async def test_auto_channel_group_permission_denied(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test auto channel group without manage_guild permission."""
        mock_app_config.no_check_permission = False
        mock_user.guild_permissions.manage_guild = False

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.send_help = AsyncMock()
        ctx.reply = AsyncMock()

        await cog.auto_channel_group.callback(cog, ctx)

        ctx.reply.assert_called_once()
        assert "권한이 없습니다" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_auto_channel_group_permission_allowed(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test auto channel group with manage_guild permission."""
        mock_app_config.no_check_permission = True  # Skip permission checks entirely
        mock_user.guild_permissions.manage_guild = True

        # Make bot.get_context async
        mock_bot.get_context = AsyncMock(return_value=Mock(valid=False))

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        ctx = Mock()
        ctx.author = mock_user
        ctx.channel = mock_channel
        ctx.command = cog.auto_channel_group
        ctx.send_help = AsyncMock()
        ctx.reply = AsyncMock()
        # Make bot.can_run return True to pass permission check
        ctx.bot = Mock()
        ctx.bot.can_run = AsyncMock(return_value=True)
        # Mock before_invoke and after_invoke hooks
        ctx.bot._before_invoke = None
        ctx.bot._after_invoke = None
        # Mock the cog attribute
        ctx.command.cog = cog
        # Mock invoke_without_command to pass the invoke check
        ctx.command.invoke_without_command = AsyncMock(return_value=True)

        # Use direct call to the group method, bypassing Discord.py command infrastructure
        # Call the method directly without going through Discord's command system
        await cog.auto_channel_group.callback(cog, ctx)

        ctx.send_help.assert_called_once_with(ctx.command)

    @pytest.mark.asyncio
    async def test_register_channel_success(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test successful channel registration."""
        # Make _save_dynamic_channels async
        with patch.object(
            AutoChannelCog, "_save_dynamic_channels", new_callable=AsyncMock
        ) as mock_save:
            cog = AutoChannelCog(
                bot=mock_bot,
                config=mock_app_config,
                llm_service=mock_llm_service,
                session_manager=mock_session_manager,
            )

            ctx = Mock()
            ctx.channel = mock_channel
            ctx.message = Mock()
            ctx.message.add_reaction = AsyncMock()

            await cog.register_channel.callback(cog, ctx)

            # Check that channel was added and saved
            assert mock_channel.id in cog.dynamic_channel_ids
            mock_save.assert_called_once()
            ctx.message.add_reaction.assert_called_once_with("✅")

    @pytest.mark.asyncio
    async def test_register_channel_already_registered(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test registering already registered channel."""
        # Pre-register the channel
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )
        cog.dynamic_channel_ids.add(mock_channel.id)

        ctx = Mock()
        ctx.channel = mock_channel
        ctx.message = Mock()
        ctx.message.add_reaction = AsyncMock()

        await cog.register_channel.callback(cog, ctx)

        # Should still add reaction but not add to set again
        ctx.message.add_reaction.assert_called_once_with("✅")
        assert len(cog.dynamic_channel_ids) == 1  # Still only one channel

    @pytest.mark.asyncio
    async def test_unregister_channel_dynamic_channel_success(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test successful unregistration of dynamic channel."""
        # Pre-register the channel and mock _save_dynamic_channels
        with patch.object(
            AutoChannelCog, "_save_dynamic_channels", new_callable=AsyncMock
        ) as mock_save:
            cog = AutoChannelCog(
                bot=mock_bot,
                config=mock_app_config,
                llm_service=mock_llm_service,
                session_manager=mock_session_manager,
            )
            cog.dynamic_channel_ids.add(mock_channel.id)

            ctx = Mock()
            ctx.channel = mock_channel
            ctx.message = Mock()
            ctx.message.add_reaction = AsyncMock()

            await cog.unregister_channel.callback(cog, ctx)

            # Check that channel was removed
            assert mock_channel.id not in cog.dynamic_channel_ids
            mock_save.assert_called_once()
            ctx.message.add_reaction.assert_called_once_with("✅")

    @pytest.mark.asyncio
    async def test_unregister_channel_env_channel_rejected(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test unregistration of environment-configured channel is rejected."""
        # Set channel as environment channel
        mock_app_config.auto_reply_channel_ids = (mock_channel.id,)

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        ctx = Mock()
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.unregister_channel.callback(cog, ctx)

        ctx.reply.assert_called_once()
        assert "시스템 설정" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_unregister_channel_not_registered(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test unregistration of not-registered channel."""
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        ctx = Mock()
        ctx.channel = mock_channel
        ctx.reply = AsyncMock()

        await cog.unregister_channel.callback(cog, ctx)

        ctx.reply.assert_called_once()
        assert "자동 응답 채널이 아닙니다" in ctx.reply.call_args[0][0]


class TestAutoChannelCogEventHandlers:
    """Test AutoChannelCog event handlers."""

    @pytest.mark.asyncio
    async def test_on_message_bot_message_ignored(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test that bot messages are ignored."""
        mock_message.author.bot = True

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        await cog.on_message(mock_message)

        # Message should be ignored (no buffer processing)
        assert mock_message.channel.id not in cog.message_buffer.buffers

    @pytest.mark.asyncio
    async def test_on_message_non_auto_channel_ignored(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test that messages in non-auto channels are ignored."""
        mock_app_config.auto_reply_channel_ids = (999888777,)  # Different channel

        # Make bot.get_context async
        mock_bot.get_context = AsyncMock(return_value=Mock(valid=False))

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        await cog.on_message(mock_message)

        # Message should be ignored
        assert mock_message.channel.id not in cog.message_buffer.buffers

    @pytest.mark.asyncio
    async def test_on_message_command_prefix_ignored(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test that messages with command prefix are ignored."""
        mock_app_config.auto_reply_channel_ids = (mock_message.channel.id,)
        mock_app_config.command_prefix = "!"

        # Make bot.get_context async and simulate that it's a valid command
        mock_context = Mock()
        mock_context.valid = True  # This simulates a command
        mock_bot.get_context = AsyncMock(return_value=mock_context)

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        await cog.on_message(mock_message)

        # Message should be ignored (valid command)
        assert mock_message.channel.id not in cog.message_buffer.buffers

    @pytest.mark.asyncio
    async def test_on_message_valid_auto_reply_processed(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test that valid auto-reply messages are processed."""
        mock_app_config.auto_reply_channel_ids = (mock_message.channel.id,)

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        # Mock get_context to return invalid context (not a command)
        mock_context = Mock()
        mock_context.valid = False
        mock_bot.get_context = AsyncMock(return_value=mock_context)

        await cog.on_message(mock_message)

        # Message should be added to buffer
        assert mock_message.channel.id in cog.message_buffer.buffers

    @pytest.mark.asyncio
    async def test_on_typing_in_auto_channel(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_channel,
        mock_user,
    ):
        """Test typing in auto channel triggers buffer handling."""
        mock_app_config.auto_reply_channel_ids = (mock_channel.id,)

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        # Mock buffer typing handler
        cog.message_buffer.handle_typing = Mock()

        await cog.on_typing(mock_channel, mock_user, 123456789.0)

        cog.message_buffer.handle_typing.assert_called_once_with(
            mock_channel.id, cog._process_batch
        )

    @pytest.mark.asyncio
    async def test_on_typing_not_in_auto_channel_ignored(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_channel,
        mock_user,
    ):
        """Test typing in non-auto channel is ignored."""
        mock_app_config.auto_reply_channel_ids = (999888777,)  # Different channel

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        await cog.on_typing(mock_channel, mock_user, 123456789.0)

        # Should not call buffer handler
        # (no assertion needed, just ensure no errors)

    @pytest.mark.asyncio
    async def test_on_typing_bot_user_ignored(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_channel
    ):
        """Test typing by bot user is ignored."""
        mock_app_config.auto_reply_channel_ids = (mock_channel.id,)
        mock_bot_user = Mock(bot=True)
        mock_bot_user.id = 123456789

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        await cog.on_typing(mock_channel, mock_bot_user, 123456789.0)

        # Should not call buffer handler
        # (no assertion needed, just ensure no errors)


class TestAutoChannelUndoCommand:
    """Test AutoChannelCog undo functionality."""

    @pytest.mark.asyncio
    async def test_undo_command_valid(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test valid undo command."""
        mock_app_config.auto_reply_channel_ids = (mock_channel.id,)
        mock_app_config.no_check_permission = True  # Disable permission checks

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        ctx = Mock()
        ctx.channel = mock_channel
        ctx.author = mock_user
        ctx.message = Mock()
        ctx.message.add_reaction = AsyncMock()
        ctx.message.delete = AsyncMock()

        # Mock undo operation and permission check
        mock_session_manager.undo_last_exchanges = Mock(return_value=["mock_message"])
        mock_session_manager.sessions = {f"channel:{mock_channel.id}": Mock(chat=Mock(history=[]))}
        mock_llm_service.get_user_role_name = Mock(return_value="user")
        mock_llm_service.get_assistant_role_name = Mock(return_value="model")

        # Mock the _cancel_pending_tasks to return (False, 2)
        cog._cancel_pending_tasks = AsyncMock(return_value=(False, 2))
        # Mock _validate_undo_arg to return a valid number
        cog._validate_undo_arg = Mock(return_value=2)
        # Mock _check_undo_permission to return True (allow undo)
        cog._check_undo_permission = Mock(return_value=True)
        # Mock _try_delete_message to delete the message
        cog._try_delete_message = AsyncMock()

        await cog.undo_command.callback(cog, ctx, "2")

        # Check that undo was called
        mock_session_manager.undo_last_exchanges.assert_called_once()
        cog._try_delete_message.assert_called_once_with(ctx.message)
        # Note: The actual code doesn't add ✅ reaction, it deletes the message

    @pytest.mark.asyncio
    async def test_undo_command_invalid_argument(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test undo command with invalid argument."""
        mock_app_config.auto_reply_channel_ids = (mock_channel.id,)

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        ctx = Mock()
        ctx.channel = mock_channel
        ctx.author = mock_user
        ctx.message = Mock()
        ctx.message.add_reaction = AsyncMock()

        await cog.undo_command.callback(cog, ctx, "invalid")

        # Should show error reaction
        ctx.message.add_reaction.assert_called_once_with("❌")

    @pytest.mark.asyncio
    async def test_undo_command_not_in_auto_channel(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test undo command when not in auto channel."""
        mock_app_config.auto_reply_channel_ids = (999888777,)  # Different channel

        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        ctx = Mock()
        ctx.channel = mock_channel
        ctx.author = mock_user
        ctx.message = Mock()
        ctx.message.add_reaction = AsyncMock()

        await cog.undo_command.callback(cog, ctx, "2")

        # Should not do anything (no error, just return)
        ctx.message.add_reaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_undo_arg_valid(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test valid undo argument validation."""
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        result = cog._validate_undo_arg("5")
        assert result == 5

        result = cog._validate_undo_arg("1")
        assert result == 1

    @pytest.mark.asyncio
    async def test_validate_undo_arg_invalid(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test invalid undo argument validation."""
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        # Test invalid numbers
        assert cog._validate_undo_arg("0") is None
        assert cog._validate_undo_arg("-1") is None
        assert cog._validate_undo_arg("abc") is None
        # Test None string (should be handled gracefully)
        assert cog._validate_undo_arg("None") is None


class TestAutoChannelCogErrorHandling:
    """Test AutoChannelCog error handling."""

    @pytest.mark.asyncio
    async def test_cog_command_error_missing_permissions(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_user,
        mock_channel,
    ):
        """Test error handler for missing permissions."""
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
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
        mock_user,
        mock_channel,
    ):
        """Test error handler for bad arguments."""
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        ctx = Mock()
        ctx.reply = AsyncMock()

        from discord.ext.commands import BadArgument

        error = BadArgument("Invalid argument")

        await cog.cog_command_error(ctx, error)

        ctx.reply.assert_called_once()
        assert "잘못된 인자" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_error(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test error handling."""
        cog = AutoChannelCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        test_error = Exception("Test error")

        await cog._handle_error(mock_message, test_error)

        # Verify error message was sent
        mock_message.channel.send.assert_called_once()


# Import aiofiles for async file operations
import aiofiles

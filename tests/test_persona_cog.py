"""Tests for the Persona Cog."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, List, Optional

import pytest
import pytest_asyncio
import discord
from discord.ext import commands

# Configure pytest-asyncio mode
pytest_asyncio_modes = "auto"

# Mock external dependencies
sys.modules["bot"] = MagicMock()
sys.modules["bot.session"] = MagicMock()
sys.modules["config"] = MagicMock()
sys.modules["services"] = MagicMock()
sys.modules["services.llm_service"] = MagicMock()
sys.modules["services.prompt_service"] = MagicMock()
sys.modules["utils"] = MagicMock()

# Import the classes we need to test
from persbot.bot.cogs.persona import (
    PersonaCog,
    PromptManagerView,
    PromptRenameModal,
    ShowModalButton,
    PromptCreateModal,
    PromptModeSelectView,
)


# Helper function to create UI components with proper event loop
async def create_ui_component_in_context(component_class, *args, **kwargs):
    """Create a UI component within an async context to avoid event loop issues."""
    return component_class(*args, **kwargs)


# Context manager for mocking send_discord_message in tests
@asynccontextmanager
async def mock_send_discord_message():
    """Mock send_discord_message for tests."""
    with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:

        async def mock_awaitable(*args, **kwargs):
            return []

        mock_send.side_effect = mock_awaitable
        yield mock_send


class TestPromptManagerView:
    """Test PromptManagerView functionality."""

    @pytest.mark.asyncio
    async def test_on_submit_failed(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test on_submit with failed rename."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        parent_view = await create_ui_component_in_context(PromptManagerView, cog, ctx)
        cog.prompt_service.rename_prompt = AsyncMock(return_value=False)

        modal = PromptRenameModal(parent_view, 0, "Old Name")

        # Create mock interaction
        interaction = Mock()
        send_discord_message = AsyncMock()
        interaction.send = send_discord_message

        # Test the modal submission
        await modal.on_submit(interaction)

        # Verify that rename was attempted but failed
        cog.prompt_service.rename_prompt.assert_called_once_with(0, "Old Name")

        # Verify error message was sent
        send_discord_message.assert_called_once()
        assert "변경 실패" in send_discord_message.call_args[0][0]


class TestPersonaCogCommands:
    """Test PersonaCog command functionality."""

    @pytest.mark.asyncio
    async def test_prompt_command(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test the prompt command."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create mock context
        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

            # Execute the prompt command
            await cog.prompt_command(ctx)

            # Verify send_discord_message was called
            mock_send.assert_called_once()
            args = mock_send.call_args
            assert args[0][0] == ""  # Empty content
            assert "view" in args[1]  # Should have a view argument

    @pytest.mark.asyncio
    async def test_cog_command_error_missing_permissions(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test error handler for missing permissions."""
        cog = PersonaCog(
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

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

            # Execute the error handler
            await cog.cog_command_error(ctx, error)

            # Verify error message was sent
            ctx.reply.assert_called_once()
            assert "권한이 없습니다" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_cog_command_error_bad_argument(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test error handler for bad arguments."""
        cog = PersonaCog(
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

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

            # Execute the error handler
            await cog.cog_command_error(ctx, error)

            # Verify error message was sent
            ctx.reply.assert_called_once()
            assert "잘못된 인자" in ctx.reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_cog_command_error_on_cooldown(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test error handler for command on cooldown."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.reply = AsyncMock()

        from discord.ext.commands import CommandOnCooldown, Cooldown

        cooldown = Cooldown(rate=1, per=10.0)
        error = CommandOnCooldown(
            cooldown=cooldown, retry_after=5.0, type=commands.BucketType.default
        )

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

            # Execute the error handler
            await cog.cog_command_error(ctx, error)

            # Verify cooldown message was sent
            ctx.reply.assert_called_once()
            assert "쿨다운 중" in ctx.reply.call_args[0][0]


class TestPersonaCogIntegration:
    """Test PersonaCog integration with other services."""

    @pytest.mark.asyncio
    async def test_session_manager_integration(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test integration with SessionManager."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Test setting channel prompt
        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = [
            {"name": "Default", "content": "Default prompt", "path": "/default.md"},
        ]
        cog.prompt_service.list_prompts = Mock(return_value=prompts)
        cog.prompt_service.get_prompt = Mock(return_value=prompts[0])

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0

        # Mock interaction for applying prompt
        interaction = Mock()
        interaction.response.defer = AsyncMock()
        send_discord_message = AsyncMock()

        with patch("persbot.bot.cogs.persona.send_discord_message", send_discord_message):
            # Test applying prompt to channel
            await view.on_apply(interaction)

            # Verify session manager was called to set channel prompt
            mock_session_manager.set_channel_prompt.assert_called_once_with(
                ctx.channel.id, "Default prompt"
            )

            # Verify success message was sent
            send_discord_message.assert_called_once()
            assert "Default" in send_discord_message.call_args[0][0]

    @pytest.mark.asyncio
    async def test_prompt_service_integration(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test integration with PromptService."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Test listing prompts
        prompts = [
            {"name": "Default", "content": "Default prompt", "path": "/default.md"},
            {"name": "Custom", "content": "Custom prompt", "path": "/custom.md"},
        ]
        cog.prompt_service.list_prompts = Mock(return_value=prompts)

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        view = PromptManagerView(cog, ctx)
        view.update_components()

        # Verify prompt service was called (once during init, once during refresh_view)
        cog.prompt_service.list_prompts.assert_called()
        assert len(view.children) == 7  # 1 select + 6 buttons

    @pytest.mark.asyncio
    async def test_llm_service_integration(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test integration with LLMService (concept generation)."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Test LLM service call for prompt generation
        cog.llm_service.generate_prompt_from_concept = AsyncMock(
            return_value='Project "Test Persona" Generated Prompt Content'
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = []
        cog.prompt_service.list_prompts = Mock(return_value=prompts)
        cog.prompt_service.add_prompt = AsyncMock(return_value=1)
        cog.prompt_service.increment_today_usage = AsyncMock()

        view = PromptManagerView(cog, ctx)

        # Test direct prompt generation (without questions)
        # Import PromptCreateModal for testing
        from persbot.bot.cogs.persona import PromptCreateModal

        # Create a modal and test it
        modal = PromptCreateModal(view, use_questions=False)
        modal.concept = Mock()
        modal.concept.value = "test concept"

        interaction = Mock()
        interaction.response.defer = AsyncMock()
        interaction.followup.send = AsyncMock()

        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

            # Test modal submission
            await modal.on_submit(interaction)

            # Verify LLM service was called
            cog.llm_service.generate_prompt_from_concept.assert_called_once_with("test concept")

            # Verify prompt service was called
            cog.prompt_service.add_prompt.assert_called_once()
            cog.prompt_service.increment_today_usage.assert_called_once()


class TestPersonaCRUDOperations:
    """Test Persona CRUD operations through the UI."""

    @pytest.mark.asyncio
    async def test_create_persona_file_upload(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test persona creation via file upload."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = []
        cog.prompt_service.list_prompts = Mock(return_value=prompts)
        cog.prompt_service.add_prompt = AsyncMock(return_value=1)
        cog.config.no_check_permission = True

        view = PromptManagerView(cog, ctx)

        # Create mock attachment
        mock_attachment = Mock()
        mock_attachment.filename = "test_person.txt"
        mock_attachment.content_type = "text/plain"
        mock_attachment.read = AsyncMock(return_value=b"Test persona content")

        # Create mock message with attachment
        mock_message = Mock()
        mock_message.author.id = 123456789
        mock_message.channel.id = 123456789
        mock_message.attachments = [mock_attachment]

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

            # Mock bot wait_for to return our message with attachment
            mock_cog.bot.wait_for = AsyncMock(return_value=mock_message)

            # Execute file upload
            await view.on_file_add(interaction)

            # Verify prompt service was called with correct data
            mock_prompt_service.add_prompt.assert_called_once_with("test", "Test persona content")

            # Verify success message was sent
            mock_send.assert_called()
            success_calls = [
                call for call in mock_send.call_args_list if "추가되었습니다" in str(call)
            ]
            assert len(success_calls) > 0

    @pytest.mark.asyncio
    async def test_create_persona_invalid_file(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test persona creation with invalid file type."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = []
        cog.prompt_service.list_prompts = Mock(return_value=prompts)
        cog.config.no_check_permission = True

        view = PromptManagerView(cog, ctx)

        # Create mock attachment (invalid file type)
        mock_attachment = Mock()
        mock_attachment.filename = "test_person.pdf"
        mock_attachment.content_type = "application/pdf"
        mock_attachment.read = AsyncMock(return_value=b"PDF content")

        # Create mock message with attachment
        mock_message = Mock()
        mock_message.author.id = 123456789
        mock_message.channel.id = 123456789
        mock_message.attachments = [mock_attachment]

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

    @pytest.mark.asyncio
    async def test_delete_persona(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test persona deletion."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

    @pytest.mark.asyncio
    async def test_service_layer_integration(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test integration with service layer."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Test prompt service integration
        test_prompts = [
            {"name": "Test Persona", "content": "Test content", "path": "/test.md"},
            {"name": "Another Persona", "content": "Another content", "path": "/another.md"},
        ]

        cog.prompt_service.list_prompts = Mock(return_value=test_prompts)

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        view = PromptManagerView(cog, ctx)
        view.update_components()

        # Verify prompts are properly loaded (called once during init)
        cog.prompt_service.list_prompts.assert_called()
        assert len(view.children) == 7  # 1 select + 6 buttons

    @pytest.mark.asyncio
    async def test_refresh_view_integration(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test view refresh integration."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = [
            {"name": "Test Persona", "content": "Test content", "path": "/test.md"},
        ]
        cog.prompt_service.list_prompts = Mock(return_value=prompts)

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0

        # Create mock interaction
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.is_done = Mock(return_value=False)
        interaction.response.edit_message = AsyncMock()

        await view.refresh_view(interaction)

        # Verify the view was refreshed
        interaction.response.edit_message.assert_called_once()
        # Check that the embed argument is passed correctly (using kwargs since edit_message is called with keyword args)
        assert "embed" in interaction.response.edit_message.call_args.kwargs
        assert isinstance(
            interaction.response.edit_message.call_args.kwargs["embed"], discord.Embed
        )


class TestPersonaPersistence:
    """Test persona persistence through the UI."""

    @pytest.mark.asyncio
    async def test_persistence_after_creation(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test persona persistence after creation."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        # Mock prompt creation
        cog.prompt_service.add_prompt = AsyncMock(return_value=1)
        cog.prompt_service.increment_today_usage = AsyncMock()

        prompts = []
        cog.prompt_service.list_prompts = Mock(return_value=prompts)

        view = PromptManagerView(cog, ctx)

        # Test prompt generation through LLM service
        cog.llm_service.generate_prompt_from_concept = AsyncMock(
            return_value='Project "Test Persona" Generated Prompt Content'
        )

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

    @pytest.mark.asyncio
    async def test_persistence_after_removal(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test persistence after persona removal."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = [
            {"name": "Test Persona", "content": "Test content", "path": "/test.md"},
        ]
        cog.prompt_service.list_prompts = Mock(return_value=prompts)
        cog.prompt_service.get_prompt = Mock(return_value=prompts[0])
        cog.prompt_service.delete_prompt = AsyncMock(return_value=True)
        cog.config.no_check_permission = True

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

    @pytest.mark.asyncio
    async def test_persistence_after_renaming(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test persistence after persona renaming."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = [
            {"name": "Old Name", "content": "Test content", "path": "/test.md"},
        ]
        cog.prompt_service.list_prompts = Mock(return_value=prompts)
        cog.prompt_service.get_prompt = Mock(return_value=prompts[0])
        cog.prompt_service.rename_prompt = AsyncMock(return_value=True)

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []


class TestPersonaUIComponents:
    """Test Persona UI component interactions."""

    @pytest.mark.asyncio
    async def test_select_menu_interaction(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test select menu interaction."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = [
            {"name": "First", "content": "First content", "path": "/first.md"},
            {"name": "Second", "content": "Second content", "path": "/second.md"},
        ]
        cog.prompt_service.list_prompts = Mock(return_value=prompts)

        view = PromptManagerView(cog, ctx)

        # Test selecting the first prompt
        interaction = Mock()
        interaction.data = {"values": ["0"]}
        interaction.response = Mock()
        interaction.response.defer = AsyncMock()

        await view.on_select(interaction)

        assert view.selected_index == 0

    @pytest.mark.asyncio
    async def test_button_state_management(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test button state management."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        prompts = [
            {"name": "Test", "content": "Test content", "path": "/test.md"},
        ]
        cog.prompt_service.list_prompts = Mock(return_value=prompts)

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0
        view.update_components()

        # Find the rename button
        rename_button = None
        for child in view.children:
            if hasattr(child, "label") and child.label == "이름 변경":
                rename_button = child
                break

        assert rename_button is not None
        assert not rename_button.disabled  # Should be enabled when a prompt is selected

        # Test with no selection
        view.selected_index = None
        view.update_components()

        rename_button = None
        for child in view.children:
            if hasattr(child, "label") and child.label == "이름 변경":
                rename_button = child
                break

        assert rename_button is not None
        assert rename_button.disabled  # Should be disabled when no prompt is selected

    @pytest.mark.asyncio
    async def test_modal_callback_integration(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test modal callback integration."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789
        ctx.message = Mock()

        parent_view = PromptManagerView(cog, ctx)
        cog.prompt_service.rename_prompt = AsyncMock(return_value=True)

        modal = PromptRenameModal(parent_view, 0, "Old Name")

        # Create mock interaction
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.send_message = AsyncMock()

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

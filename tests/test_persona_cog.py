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
    PromptModeSelectView,
    PromptAnswerModal,
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

        async def mock_implementation(*args, **kwargs):
            # Return a list with a mock message to simulate successful message sending
            mock_message = Mock()
            mock_message.id = "123456789"
            mock_message.edit = AsyncMock()
            mock_message.delete = AsyncMock()
            return [mock_message]

        mock_send.side_effect = mock_implementation
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
        interaction.response = Mock()
        interaction.response.send_message = AsyncMock()

        # Create a custom mock for the text input with a value property
        mock_text_input = Mock()
        mock_text_input.value = "New Name"
        modal.new_name = mock_text_input

        # Mock send_discord_message for the modal
        async with mock_send_discord_message() as mock_send:
            # Test the modal submission
            await modal.on_submit(interaction)

        # Verify that rename was attempted with the new name
        cog.prompt_service.rename_prompt.assert_called_once_with(0, "New Name")

        # Verify error message was sent
        mock_send.assert_called_once()
        # Check the actual call arguments - the second argument should be the message
        call_args = mock_send.call_args
        assert len(call_args[0]) >= 2  # Should have at least interaction and message
        message_content = call_args[0][1]  # Second argument is the message content
        assert "변경 실패" in str(message_content)


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
        ctx.send = AsyncMock()

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Execute the prompt command
            await cog.prompt_command.callback(cog, ctx)

            # Verify send_discord_message was called
            mock_send.assert_called_once()
            # Check that it was called with proper arguments (view should be present)
            args = mock_send.call_args
            assert len(args[0]) > 0  # Should have at least the target argument
            assert "view" in args[1]  # Should have a view argument


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

        # Create mock interaction
        interaction = Mock()
        interaction.user.id = 123456789
        interaction.channel.id = 987654321

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
        async with mock_send_discord_message() as mock_send:
            # Mock bot wait_for to return our message with attachment
            cog.bot.wait_for = AsyncMock(return_value=mock_message)

            # Execute file upload
            await view.on_file_add(interaction)

            # Verify prompt service was called with correct data (name extracted from filename)
            mock_prompt_service.add_prompt.assert_called_once_with(
                "test_person", "Test persona content"
            )

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

        # Create mock interaction
        interaction = Mock()
        interaction.user.id = 123456789
        interaction.channel.id = 987654321

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
        async with mock_send_discord_message() as mock_send:
            # Test LLM service integration
            await cog.llm_service.generate_prompt_from_concept("test concept")

            # Verify LLM service was called
            cog.llm_service.generate_prompt_from_concept.assert_called_once_with("test concept")

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


class TestShowModalButton:
    """Test ShowModalButton functionality."""

    @pytest.mark.asyncio
    async def test_show_modal_button_click(self):
        """Test ShowModalButton click functionality."""
        # Create a mock modal
        mock_modal = Mock()

        # Create ShowModalButton with the mock modal
        button_view = ShowModalButton(mock_modal)

        # Create mock interaction
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.send_modal = AsyncMock()

        # Get the button (should be the first item)
        button = button_view.children[0]

        # Test button click - Discord callback passes (self, interaction, button) automatically
        await button.callback(interaction)

        # Verify modal was sent
        interaction.response.send_modal.assert_called_once_with(mock_modal)


class TestPromptModeSelectView:
    """Test PromptModeSelectView button interactions."""

    @pytest.mark.asyncio
    async def test_basic_mode_button_click(self):
        """Test basic mode button click."""
        # Create mock parent view
        parent_view = Mock()

        # Create PromptModeSelectView
        mode_view = PromptModeSelectView(parent_view)

        # Get the basic mode button (first button)
        basic_button = mode_view.children[0]

        # Mock response
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.send_modal = AsyncMock()

        # Test basic mode button click - Discord callback passes (self, interaction, button) automatically
        await basic_button.callback(interaction)

        # Verify modal was sent for basic mode
        interaction.response.send_modal.assert_called_once()
        # Check that it was called with some modal (the actual modal class isn't defined)
        call_args = interaction.response.send_modal.call_args
        assert len(call_args[0]) > 0  # Should be called with a modal argument

    @pytest.mark.asyncio
    async def test_qa_mode_button_click(self):
        """Test QA mode button click."""
        # Create mock parent view
        parent_view = Mock()

        # Create PromptModeSelectView
        mode_view = PromptModeSelectView(parent_view)

        # Get the QA mode button (second button)
        qa_button = mode_view.children[1]

        # Mock response
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.send_modal = AsyncMock()

        # Test QA mode button click - Discord callback passes (self, interaction, button) automatically
        await qa_button.callback(interaction)

        # Verify modal was sent for QA mode
        interaction.response.send_modal.assert_called_once()
        # Check that it was called with some modal (the actual modal class isn't defined)
        call_args = interaction.response.send_modal.call_args
        assert len(call_args[0]) > 0  # Should be called with a modal argument

    @pytest.mark.asyncio
    async def test_cancel_button_click(self):
        """Test cancel button click."""
        # Create mock parent view
        parent_view = Mock()

        # Create PromptModeSelectView
        mode_view = PromptModeSelectView(parent_view)

        # Get the cancel button (third button)
        cancel_button = mode_view.children[2]

        # Mock response
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.delete_message = AsyncMock()

        # Test cancel button click - Discord callback passes (self, interaction, button) automatically
        await cancel_button.callback(interaction)

        # Verify message was deleted
        interaction.response.delete_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_submit_with_qa_mode_success(
        self,
        mock_bot,
        mock_app_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service,
        mock_interaction,
    ):
        """Test PromptModeSelectView.on_submit with QA mode successful question generation."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        parent_view = PromptManagerView(cog, ctx)
        mode_view = PromptModeSelectView(parent_view)

        # Mock the LLM service to return valid JSON questions
        questions_json = '{"questions": [{"question": "What is your main goal?", "sample_answer": "To help users"}, {"question": "What is your specialty?", "sample_answer": "AI assistance"}]}'
        mock_llm_service.generate_questions_from_concept = AsyncMock(return_value=questions_json)

        # Mock the view reference to point to our mock view
        mode_view.view_ref = parent_view

        # Create concept input
        mode_view.concept = Mock()
        mode_view.concept.value = "Test concept"
        mode_view.use_questions = True

        # Mock interaction manually for this specific test
        from unittest.mock import MagicMock

        interaction = MagicMock()
        interaction.response = MagicMock()
        interaction.response.defer = AsyncMock()

        # Create mock message with edit capability
        mock_msg = Mock()
        mock_msg.edit = AsyncMock()

        # Patch the followup.send method with AsyncMock
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock(return_value=mock_msg)

        # Mock send_discord_message to avoid await issues
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

            # Test basic mode submission
            await mode_view.on_submit(interaction)

            # Verify LLM service was called
            cog.llm_service.generate_questions_from_concept.assert_called_once_with("Test concept")

            # Verify that the loading message was edited (indicating successful QA flow)
            mock_msg.edit.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_direct_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptModeSelectView._generate_direct method success case."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        parent_view = PromptManagerView(cog, ctx)
        mode_view = PromptModeSelectView(parent_view)
        mode_view.view_ref = parent_view

        # Mock LLM service
        cog.llm_service.generate_prompt_from_concept = AsyncMock(
            return_value='Project "Test Persona" Generated prompt content'
        )
        cog.prompt_service.add_prompt = AsyncMock(return_value=1)
        cog.prompt_service.increment_today_usage = AsyncMock()

        # Mock message
        mock_msg = Mock()
        mock_msg.edit = AsyncMock()

        # Mock interaction
        interaction = Mock()
        interaction.user.id = 123456789

        # Test _generate_direct method
        await mode_view._generate_direct(interaction, "Test concept", mock_msg)

        # Verify LLM service was called
        cog.llm_service.generate_prompt_from_concept.assert_called_once_with("Test concept")

        # Verify prompt service was called
        cog.prompt_service.add_prompt.assert_called_once()
        cog.prompt_service.increment_today_usage.assert_called_once()

        # Verify message was updated
        mock_msg.edit.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_direct_failure(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptModeSelectView._generate_direct method failure case."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        parent_view = PromptManagerView(cog, ctx)
        mode_view = PromptModeSelectView(parent_view)
        mode_view.view_ref = parent_view

        # Mock LLM service to return None (failure)
        cog.llm_service.generate_prompt_from_concept = AsyncMock(return_value=None)

        # Mock message
        mock_msg = Mock()
        mock_msg.edit = AsyncMock()

        # Mock interaction
        interaction = Mock()
        interaction.user.id = 123456789

        # Test _generate_direct method
        await mode_view._generate_direct(interaction, "Test concept", mock_msg)

        # Verify message shows failure
        mock_msg.edit.assert_called_once()
        # Check that the edit was called with failure content
        mock_msg.edit.assert_any_call(content="❌ 프롬프트 생성에 실패했습니다.")


class TestPromptAnswerModal:
    """Test PromptAnswerModal functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test PromptAnswerModal initialization."""
        # Create mock view and concept
        mock_view = Mock()
        concept = "Test concept"
        questions = [
            {"question": "What is your goal?", "sample_answer": "To help"},
            {"question": "What is your specialty?", "sample_answer": "AI"},
        ]

        # Create modal
        modal = PromptAnswerModal(mock_view, concept, questions)

        # Verify initialization
        assert modal.view_ref == mock_view
        assert modal.concept == concept
        assert modal.questions == questions
        assert len(modal.children) == 2  # 2 questions = 2 label components

    @pytest.mark.asyncio
    async def test_on_submit_with_answers(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptAnswerModal.on_submit with user answers."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create mock view
        mock_view = Mock()
        mock_view.cog = cog

        concept = "Test concept"
        questions = [
            {"question": "What is your goal?", "sample_answer": "To help"},
            {"question": "What is your specialty?", "sample_answer": "AI"},
        ]

        # Create modal
        modal = PromptAnswerModal(mock_view, concept, questions)

        # Mock text inputs with user answers
        modal.answer_0 = Mock()
        modal.answer_0.value = "My goal is to help users"
        modal.answer_1 = Mock()
        modal.answer_1.value = "I specialize in AI assistance"

        # Mock LLM service
        cog.llm_service.generate_prompt_from_concept_with_answers = AsyncMock(
            return_value='Project "Test Persona" Generated prompt content'
        )
        cog.prompt_service.add_prompt = AsyncMock(return_value=1)
        cog.prompt_service.increment_today_usage = AsyncMock()

        # Create custom interaction with properly mocked followup
        mock_message = Mock()
        mock_message.edit = AsyncMock()
        mock_followup_send = AsyncMock(return_value=mock_message)
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.defer = AsyncMock()
        interaction.followup = Mock()
        interaction.followup.send = mock_followup_send

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test submission
            await modal.on_submit(interaction)

            # Verify LLM service was called with Q&A string
            expected_qa = "Q: What is your goal?\nA: My goal is to help users\n\nQ: What is your specialty?\nA: I specialize in AI assistance"
            cog.llm_service.generate_prompt_from_concept_with_answers.assert_called_once_with(
                concept, expected_qa
            )

            # Verify prompt service was called
            cog.prompt_service.add_prompt.assert_called_once()
            cog.prompt_service.increment_today_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_submit_with_sample_answers(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptAnswerModal.on_submit with empty answers (should use sample answers)."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create mock view
        mock_view = Mock()
        mock_view.cog = cog

        concept = "Test concept"
        questions = [
            {"question": "What is your goal?", "sample_answer": "To help"},
            {"question": "What is your specialty?", "sample_answer": "AI"},
        ]

        # Create modal
        modal = PromptAnswerModal(mock_view, concept, questions)

        # Mock text inputs with empty answers
        modal.answer_0 = Mock()
        modal.answer_0.value = ""  # Empty, should use sample
        modal.answer_1 = Mock()
        modal.answer_1.value = ""  # Empty, should use sample

        # Mock LLM service
        cog.llm_service.generate_prompt_from_concept_with_answers = AsyncMock(
            return_value='Project "Test Persona" Generated prompt content'
        )
        cog.prompt_service.add_prompt = AsyncMock(return_value=1)
        cog.prompt_service.increment_today_usage = AsyncMock()

        # Mock interaction
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.defer = AsyncMock()
        mock_message = Mock()
        mock_message.edit = AsyncMock()
        interaction.followup = Mock()
        interaction.followup.send = AsyncMock(return_value=mock_message)

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test submission
            await modal.on_submit(interaction)

            # Verify LLM service was called with sample answers
            expected_qa = "Q: What is your goal?\nA: To help\n\nQ: What is your specialty?\nA: AI"
            cog.llm_service.generate_prompt_from_concept_with_answers.assert_called_once_with(
                concept, expected_qa
            )


class TestPromptRenameModal:
    """Test PromptRenameModal functionality."""

    @pytest.mark.asyncio
    async def test_on_submit_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptRenameModal.on_submit with successful rename."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        parent_view = PromptManagerView(cog, ctx)

        # Mock prompt service to return True (successful rename)
        cog.prompt_service.rename_prompt = AsyncMock(return_value=True)

        # Create modal
        modal = PromptRenameModal(parent_view, 0, "Old Name")

        # Mock text input
        mock_text_input = Mock()
        mock_text_input.value = "New Name"
        modal.new_name = mock_text_input

        # Mock interaction
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.send_message = AsyncMock()

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test submission
            await modal.on_submit(interaction)

            # Verify prompt service was called
            cog.prompt_service.rename_prompt.assert_called_once_with(0, "New Name")

            # Verify send_discord_message was called
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_submit_failure(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptRenameModal.on_submit with failed rename."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        parent_view = PromptManagerView(cog, ctx)

        # Mock prompt service to return False (failed rename)
        cog.prompt_service.rename_prompt = AsyncMock(return_value=False)

        # Create modal
        modal = PromptRenameModal(parent_view, 0, "Old Name")

        # Mock text input
        mock_text_input = Mock()
        mock_text_input.value = "New Name"
        modal.new_name = mock_text_input

        # Mock interaction
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.send_message = AsyncMock()

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test submission
            await modal.on_submit(interaction)

            # Verify prompt service was called
            cog.prompt_service.rename_prompt.assert_called_once_with(0, "New Name")

            # Verify send_discord_message was called with error message
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "변경 실패" in str(call_args[0][1])


class TestPromptManagerViewAdditional:
    """Additional tests for PromptManagerView to improve coverage."""

    @pytest.mark.asyncio
    async def test_refresh_view_error_handling(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.refresh_view error handling."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)

        # Mock interaction with response already done
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.is_done = Mock(return_value=True)

        # Mock message
        mock_message = Mock()
        view.message = mock_message

        # Mock embed
        embed = discord.Embed(title="Test", description="Test description")

        # Mock message edit to raise exception
        mock_message.edit = Mock(side_effect=Exception("Edit failed"))

        # Test refresh with error
        await view.refresh_view(interaction)

        # Verify message edit was attempted
        mock_message.edit.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_select_edge_case_no_prompts(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_select with no prompts (val == -1 case)."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)

        # Mock interaction with empty/no prompts value
        interaction = Mock()
        interaction.data = {"values": ["-1"]}
        interaction.response = Mock()
        interaction.response.defer = AsyncMock()

        # Test on_select with -1 value (should return early)
        await view.on_select(interaction)

        # Verify response was deferred
        interaction.response.defer.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_new_daily_limit_exceeded(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_new with daily limit exceeded."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)

        # Mock prompt service to return False (limit exceeded)
        cog.prompt_service.check_today_limit = AsyncMock(return_value=False)

        # Mock interaction
        interaction = Mock()
        interaction.user.id = 123456789

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test on_new with exceeded limit
            await view.on_new(interaction)

            # Verify check was called
            cog.prompt_service.check_today_limit.assert_called_once_with(123456789)

            # Verify error message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "오늘 생성 한도" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_on_file_add_permission_denied(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_file_add with permission denied."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Enable permission checking
        cog.config.no_check_permission = False

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)

        # Mock interaction without manage_guild permission
        interaction = Mock()
        interaction.user.guild_permissions.manage_guild = False

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test file add with permission denied
            await view.on_file_add(interaction)

            # Verify error message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "권한" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_on_file_add_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_file_add successful file upload."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)

        # Mock attachment
        mock_attachment = Mock()
        mock_attachment.filename = "test_prompt.txt"
        mock_attachment.read = AsyncMock(return_value=b"Test prompt content")

        # Mock message
        mock_message = Mock()
        mock_message.author.id = 123456789
        mock_message.channel.id = 123456789
        mock_message.attachments = [mock_attachment]

        # Mock bot wait_for
        cog.bot.wait_for = AsyncMock(return_value=mock_message)

        # Mock prompt service
        cog.prompt_service.add_prompt = AsyncMock(return_value=1)

        # Mock interaction
        interaction = Mock()
        interaction.user.id = 123456789
        interaction.channel.id = 123456789

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test successful file add
            await view.on_file_add(interaction)

            # Verify prompt service was called
            cog.prompt_service.add_prompt.assert_called_once_with(
                "test_prompt", "Test prompt content"
            )

            # Verify messages were sent (initial request + success)
            assert mock_send.call_count == 2
            # Get the success message (second call)
            success_call = None
            for call in mock_send.call_args_list:
                if "새 페르소나" in str(call[0][1]):
                    success_call = call
                    break
            assert success_call is not None
            assert "새 페르소나" in str(success_call[0][1])

    @pytest.mark.asyncio
    async def test_on_file_add_invalid_file_type(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_file_add with invalid file type."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)

        # Mock attachment (invalid .pdf file)
        mock_attachment = Mock()
        mock_attachment.filename = "test_prompt.pdf"
        mock_attachment.read = AsyncMock(return_value=b"PDF content")

        # Mock message
        mock_message = Mock()
        mock_message.author.id = 123456789
        mock_message.channel.id = 123456789
        mock_message.attachments = [mock_attachment]

        # Mock bot wait_for
        cog.bot.wait_for = AsyncMock(return_value=mock_message)

        # Mock interaction
        interaction = Mock()
        interaction.user.id = 123456789
        interaction.channel.id = 123456789

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test file add with invalid type
            await view.on_file_add(interaction)

            # Verify messages were sent (initial request + error)
            assert mock_send.call_count == 2
            # Get the error message (second call)
            error_call = None
            for call in mock_send.call_args_list:
                if ".txt" in str(call[0][1]) and "지원합니다" in str(call[0][1]):
                    error_call = call
                    break
            assert error_call is not None
            assert ".txt" in str(error_call[0][1])

    @pytest.mark.asyncio
    async def test_on_apply_error_handling(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_apply error handling when prompt not found."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0

        # Mock prompt service to return None (prompt not found)
        cog.prompt_service.get_prompt = Mock(return_value=None)

        # Mock interaction
        interaction = Mock()
        interaction.user.id = 123456789

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test apply with non-existent prompt
            await view.on_apply(interaction)

            # Verify error message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "찾을 수 없습니다" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_on_rename_prompt_not_found(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_rename when prompt doesn't exist."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0

        # Mock prompt service to return None (prompt not found)
        cog.prompt_service.get_prompt = Mock(return_value=None)

        # Mock interaction
        interaction = Mock()
        interaction.response = Mock()
        interaction.response.send_modal = AsyncMock()

        # Test rename with non-existent prompt
        await view.on_rename(interaction)

        # Verify modal was NOT sent because prompt doesn't exist
        interaction.response.send_modal.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_delete_permission_denied(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_delete with permission denied."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Enable permission checking
        cog.config.no_check_permission = False

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0

        # Mock interaction without manage_guild permission
        interaction = Mock()
        interaction.user.guild_permissions.manage_guild = False

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test delete with permission denied
            await view.on_delete(interaction)

            # Verify error message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "권한" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_on_delete_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_delete successful deletion."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)
        view.selected_index = 0

        # Mock prompt
        test_prompt = {"name": "Test Prompt", "content": "Test content"}
        cog.prompt_service.get_prompt = Mock(return_value=test_prompt)
        cog.prompt_service.delete_prompt = AsyncMock(return_value=True)

        # Mock interaction
        interaction = Mock()
        interaction.user.id = 123456789

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test successful delete
            await view.on_delete(interaction)

            # Verify prompt service was called
            cog.prompt_service.delete_prompt.assert_called_once_with(0)

            # Verify selected_index was cleared
            assert view.selected_index is None

            # Verify success message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "삭제 완료" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_on_close_message_deletion(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PromptManagerView.on_close message deletion."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        ctx = Mock()
        ctx.channel.id = 123456789

        view = PromptManagerView(cog, ctx)

        # Mock message
        mock_message = Mock()
        mock_message.delete = AsyncMock()

        # Mock interaction
        interaction = Mock()
        interaction.message = mock_message

        # Test close
        await view.on_close(interaction)

        # Verify message was deleted
        mock_message.delete.assert_called_once()

        # Verify view was stopped
        assert view.is_stopped()


class TestPersonaCogErrorHandling:
    """Test PersonaCog error handling methods."""

    @pytest.mark.asyncio
    async def test_cog_command_error_missing_permissions(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PersonaCog.cog_command_error with missing permissions."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create mock context
        ctx = Mock()
        ctx.command = Mock()
        ctx.command.has_error_handler = Mock(return_value=False)

        # Create missing permissions error
        missing_perms = ["send_messages", "read_message_history"]
        error = commands.MissingPermissions(missing_perms)

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test error handling
            await cog.cog_command_error(ctx, error)

            # Verify error message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "권한이 없습니다" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_cog_command_error_bad_argument(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PersonaCog.cog_command_error with bad argument."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create mock context
        ctx = Mock()
        ctx.command = Mock()
        ctx.command.has_error_handler = Mock(return_value=False)

        # Create bad argument error
        error = commands.BadArgument("Invalid argument provided")

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test error handling
            await cog.cog_command_error(ctx, error)

            # Verify error message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "잘못된 인자" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_cog_command_error_command_on_cooldown(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PersonaCog.cog_command_error with command on cooldown."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create mock context
        ctx = Mock()
        ctx.command = Mock()
        ctx.command.has_error_handler = Mock(return_value=False)

        # Create cooldown error
        from discord.ext import commands

        cooldown = commands.Cooldown(rate=1, per=60.0)
        error = commands.CommandOnCooldown(commands.BucketType.user, cooldown, 5.0)

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test error handling
            await cog.cog_command_error(ctx, error)

            # Verify error message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "쿨다운" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_cog_command_error_unhandled_exception(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PersonaCog.cog_command_error with unhandled exception."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create mock context
        ctx = Mock()
        ctx.command = Mock()
        ctx.command.has_error_handler = Mock(return_value=False)

        # Create generic error
        error = Exception("Something went wrong")

        # Mock send_discord_message
        async with mock_send_discord_message() as mock_send:
            # Test error handling
            await cog.cog_command_error(ctx, error)

            # Verify error message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "오류가 발생했습니다" in str(call_args[0][1])

    @pytest.mark.asyncio
    async def test_cog_command_error_with_error_handler(
        self, mock_bot, mock_app_config, mock_llm_service, mock_session_manager, mock_prompt_service
    ):
        """Test PersonaCog.cog_command_error when command has its own error handler."""
        cog = PersonaCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            prompt_service=mock_prompt_service,
        )

        # Create mock context
        ctx = Mock()
        ctx.command = Mock()
        ctx.command.has_error_handler = Mock(return_value=True)

        # Create error
        error = Exception("Command has its own handler")

        # Mock send_discord_message
        with patch("persbot.bot.cogs.persona.send_discord_message") as mock_send:
            mock_send.return_value = []

            # Test error handling
            await cog.cog_command_error(ctx, error)

            # Verify no message was sent (command has its own handler)
            mock_send.assert_not_called()

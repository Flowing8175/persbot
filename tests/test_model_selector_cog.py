"""Tests for ModelSelectorCog.

Tests focus on:
- ModelSelectorView: initialization, options creation
- ModelSelect: callback interaction, session update, message handling
- ModelSelectorCog: initialization, model_command, llm_subcommand, image_subcommand
- ImageModelSelectorView: initialization, options creation
- ImageModelSelect: callback interaction, model update, message handling
"""

import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest
import discord
from discord.ext import commands

from persbot.bot.cogs.model_selector import (
    ModelSelectorView,
    ModelSelect,
    ModelSelectorCog,
    ImageModelSelectorView,
    ImageModelSelect,
)
from persbot.bot.session import SessionManager
from persbot.services.model_usage_service import ModelUsageService, ModelDefinition
from persbot.services.image_model_service import (
    get_available_image_models,
    get_channel_image_model,
    set_channel_image_model,
    ImageModelDefinition,
)
from persbot.config import AppConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_bot():
    """Create a mock Discord bot."""
    bot = Mock(spec=commands.Bot)
    return bot


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager."""
    manager = Mock(spec=SessionManager)
    manager.session_contexts = {}
    manager.channel_model_preferences = {}
    manager.set_session_model = Mock()
    return manager


@pytest.fixture
def mock_context():
    """Create a mock Discord context."""
    ctx = Mock(spec=commands.Context)
    ctx.author = Mock(spec=discord.Member)
    ctx.author.id = 789
    ctx.author.display_name = "TestUser"
    ctx.channel = Mock()
    ctx.channel.id = 123
    ctx.channel.name = "test-channel"
    ctx.message = Mock()
    ctx.message.id = 999
    ctx.message.delete = AsyncMock()
    ctx.reply = AsyncMock()
    ctx.send = AsyncMock()
    return ctx


@pytest.fixture
def mock_interaction():
    """Create a mock Discord interaction."""
    interaction = Mock(spec=discord.Interaction)
    interaction.channel_id = 123
    interaction.response = Mock()
    interaction.response.defer = AsyncMock()
    interaction.message = Mock()
    interaction.message.delete = AsyncMock()
    return interaction


@pytest.fixture
def sample_model_definitions():
    """Create sample model definitions for testing."""
    return {
        "Gemini 2.5 flash": ModelDefinition(
            display_name="Gemini 2.5 flash",
            api_model_name="gemini-2.5-flash",
            daily_limit=1500,
            scope="guild",
            provider="gemini",
            description="빠르고 효율적인 Gemini 모델"
        ),
        "GPT 5 mini": ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=1000,
            scope="guild",
            provider="openai",
            description="가벼운 GPT-5 모델"
        ),
        "GLM 4.7": ModelDefinition(
            display_name="GLM 4.7",
            api_model_name="glm-4.7",
            daily_limit=200,
            scope="guild",
            provider="zai",
            description="Z.AI의 효율적인 모델"
        ),
    }


@pytest.fixture
def sample_image_models():
    """Create sample image model definitions for testing."""
    return [
        ImageModelDefinition(
            display_name="Flux 2 Klein",
            api_model_name="black-forest-labs/flux.2-klein-4b",
            description="Fast and efficient image generation model",
            default=True,
        ),
        ImageModelDefinition(
            display_name="Riverflow Pro",
            api_model_name="sourceful/riverflow-v2-pro",
            description="High quality image generation model",
            default=False,
        ),
    ]


# =============================================================================
# ModelSelectorView Tests
# =============================================================================

class TestModelSelectorViewInit:
    """Tests for ModelSelectorView initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_session_manager):
        """ModelSelectorView initializes with required attributes."""
        view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")

        assert view.session_manager == mock_session_manager
        assert view.original_message is None
        assert view.timeout == 60

    @pytest.mark.asyncio
    async def test_initialization_with_original_message(self, mock_session_manager, mock_context):
        """ModelSelectorView initializes with original_message."""
        view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash", original_message=mock_context.message)

        assert view.original_message == mock_context.message

    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @pytest.mark.asyncio
    async def test_initialization_creates_options(self, mock_service, mock_session_manager, sample_model_definitions):
        """ModelSelectorView creates select options from model definitions."""
        mock_service.MODEL_DEFINITIONS = sample_model_definitions

        view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")

        # Should have one child (the select dropdown)
        assert len(view.children) == 1
        assert isinstance(view.children[0], ModelSelect)

    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @pytest.mark.asyncio
    async def test_option_has_emoji_gemini(self, mock_service, mock_session_manager, sample_model_definitions):
        """Options have correct emoji for Gemini provider."""
        mock_service.MODEL_DEFINITIONS = sample_model_definitions

        view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")
        select = view.children[0]

        gemini_option = [o for o in select.options if "Gemini" in o.label][0]
        assert str(gemini_option.emoji) == "🤖"

    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @pytest.mark.asyncio
    async def test_option_has_emoji_zai(self, mock_service, mock_session_manager, sample_model_definitions):
        """Options have correct emoji for Z.AI provider."""
        mock_service.MODEL_DEFINITIONS = sample_model_definitions

        view = ModelSelectorView(mock_session_manager, "GLM 4.7")
        select = view.children[0]

        zai_option = [o for o in select.options if "GLM" in o.label][0]
        assert str(zai_option.emoji) == "⚡"

    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @pytest.mark.asyncio
    async def test_option_has_emoji_openai(self, mock_service, mock_session_manager, sample_model_definitions):
        """Options have correct emoji for OpenAI provider."""
        mock_service.MODEL_DEFINITIONS = sample_model_definitions

        view = ModelSelectorView(mock_session_manager, "GPT 5 mini")
        select = view.children[0]

        openai_option = [o for o in select.options if "GPT" in o.label][0]
        assert str(openai_option.emoji) == "🧠"

    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @pytest.mark.asyncio
    async def test_option_description_includes_limit(self, mock_service, mock_session_manager, sample_model_definitions):
        """Options include daily limit in description."""
        mock_service.MODEL_DEFINITIONS = sample_model_definitions

        view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")
        select = view.children[0]

        for option in select.options:
            assert "1일 한도:" in option.description
            assert "회 (서버 공통)" in option.description

    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @pytest.mark.asyncio
    async def test_option_marks_current_as_default(self, mock_service, mock_session_manager, sample_model_definitions):
        """Current model is marked as default in options."""
        mock_service.MODEL_DEFINITIONS = sample_model_definitions

        view = ModelSelectorView(mock_session_manager, "GPT 5 mini")
        select = view.children[0]

        gpt_option = [o for o in select.options if "GPT" in o.label][0]
        assert gpt_option.default is True

        gemini_option = [o for o in select.options if "Gemini" in o.label][0]
        assert gemini_option.default is False


# =============================================================================
# ModelSelect Tests
# =============================================================================

class TestModelSelectInit:
    """Tests for ModelSelect initialization."""

    def test_initialization(self):
        """ModelSelect initializes with correct configuration."""
        options = [
            discord.SelectOption(label="Test Model", description="Test description"),
        ]

        select = ModelSelect(options)

        assert select.placeholder == "모델을 선택하세요..."
        assert select.min_values == 1
        assert select.max_values == 1
        assert select.options == options


class TestModelSelectCallback:
    """Tests for ModelSelect callback."""

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @pytest.mark.asyncio
    async def test_callback_defers_interaction(self, mock_send, mock_session_manager, mock_interaction, sample_model_definitions):
        """callback defers the interaction."""
        with patch.object(ModelUsageService, 'MODEL_DEFINITIONS', sample_model_definitions):
            view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")
            select = view.children[0]

            # Mock the values property
            with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
                mock_values.return_value = ["GPT 5 mini"]

                await select.callback(mock_interaction)

                mock_interaction.response.defer.assert_called_once()

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @pytest.mark.asyncio
    async def test_callback_updates_session_model(self, mock_send, mock_session_manager, mock_interaction, sample_model_definitions):
        """callback updates the session model."""
        with patch.object(ModelUsageService, 'MODEL_DEFINITIONS', sample_model_definitions):
            view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")
            select = view.children[0]

            with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
                mock_values.return_value = ["GLM 4.7"]

                await select.callback(mock_interaction)

                mock_session_manager.set_session_model.assert_called_once_with(123, "GLM 4.7")

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @pytest.mark.asyncio
    async def test_callback_sends_confirmation_without_original(self, mock_send, mock_session_manager, mock_interaction, sample_model_definitions):
        """callback sends confirmation when no original message."""
        with patch.object(ModelUsageService, 'MODEL_DEFINITIONS', sample_model_definitions):
            view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")
            select = view.children[0]

            with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
                mock_values.return_value = ["GPT 5 mini"]

                await select.callback(mock_interaction)

                mock_send.assert_called_once()
                # Just verify the message content is correct
                args, kwargs = mock_send.call_args
                assert "✅ 모델이 **GPT 5 mini**로 변경되었습니다." in args[1]

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @pytest.mark.asyncio
    async def test_callback_sends_to_original_message(self, mock_send, mock_session_manager, mock_interaction, mock_context, sample_model_definitions):
        """callback sends confirmation to original message when available."""
        with patch.object(ModelUsageService, 'MODEL_DEFINITIONS', sample_model_definitions):
            view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash", original_message=mock_context.message)
            select = view.children[0]

            with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
                mock_values.return_value = ["GLM 4.7"]

                await select.callback(mock_interaction)

                # Should send to original message
                args, kwargs = mock_send.call_args
                assert args[0] == mock_context.message
                assert "✅ 모델이 **GLM 4.7**로 변경되었습니다." in args[1]

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @pytest.mark.asyncio
    async def test_callback_deletes_interaction_message(self, mock_send, mock_session_manager, mock_interaction, sample_model_definitions):
        """callback deletes the interaction message."""
        with patch.object(ModelUsageService, 'MODEL_DEFINITIONS', sample_model_definitions):
            view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")
            select = view.children[0]

            with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
                mock_values.return_value = ["GPT 5 mini"]

                await select.callback(mock_interaction)

                mock_interaction.message.delete.assert_called_once()

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @pytest.mark.asyncio
    async def test_callback_handles_deletion_gracefully(self, mock_send, mock_session_manager, mock_interaction, sample_model_definitions):
        """callback handles message deletion errors gracefully."""
        # Create a proper NotFound exception
        not_found = discord.NotFound(response=Mock(), message="Not found")
        mock_interaction.message.delete = AsyncMock(side_effect=not_found)

        with patch.object(ModelUsageService, 'MODEL_DEFINITIONS', sample_model_definitions):
            view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash")
            select = view.children[0]

            with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
                mock_values.return_value = ["GPT 5 mini"]

                # Should not raise
                await select.callback(mock_interaction)

                # Confirmation should still be sent
                assert mock_send.called

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @pytest.mark.asyncio
    async def test_callback_fallback_when_original_deleted(self, mock_send, mock_session_manager, mock_interaction, mock_context, sample_model_definitions):
        """callback falls back to interaction when original message deleted."""
        # Create a proper NotFound exception
        not_found = discord.NotFound(response=Mock(), message="Not found")

        # Make sending to original message fail
        async def failing_send(*args, **kwargs):
            if args and args[0] == mock_context.message:
                raise not_found

        mock_send.side_effect = failing_send

        with patch.object(ModelUsageService, 'MODEL_DEFINITIONS', sample_model_definitions):
            view = ModelSelectorView(mock_session_manager, "Gemini 2.5 flash", original_message=mock_context.message)
            select = view.children[0]

            with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
                mock_values.return_value = ["GPT 5 mini"]

                await select.callback(mock_interaction)

                # Should have called send twice (original failed, then fallback)
                assert mock_send.call_count == 2


# =============================================================================
# ModelSelectorCog Tests
# =============================================================================

class TestModelSelectorCogInit:
    """Tests for ModelSelectorCog initialization."""

    @patch('persbot.bot.cogs.model_selector.load_config')
    def test_initialization(self, mock_load_config, mock_bot, mock_session_manager):
        """ModelSelectorCog initializes with required attributes."""
        mock_config = Mock(spec=AppConfig)
        mock_load_config.return_value = mock_config

        cog = ModelSelectorCog(mock_bot, mock_session_manager)

        assert cog.bot == mock_bot
        assert cog.session_manager == mock_session_manager
        assert cog.config == mock_config
        mock_load_config.assert_called_once()


class TestModelSelectorCogModelCommand:
    """Tests for ModelSelectorCog.model_command."""

    # Note: The model_command invokes llm_subcommand internally, which is tested
    # thoroughly in TestModelSelectorCogLLMSubcommand. Direct testing of the
    # command decorator wrapper is complex due to Discord's command system internals.
    # The integration is verified indirectly through the llm_subcommand tests.


class TestModelSelectorCogLLMSubcommand:
    """Tests for ModelSelectorCog.llm_subcommand."""

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_uses_default_model_no_session(self, mock_load_config, mock_service, mock_send, mock_bot, mock_session_manager, mock_context, sample_model_definitions):
        """llm_subcommand uses default model when no session exists."""
        mock_load_config.return_value = Mock()
        mock_service.MODEL_DEFINITIONS = sample_model_definitions
        mock_service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash"
        mock_session_manager.session_contexts = {}
        mock_session_manager.channel_model_preferences = {}

        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        # Call the underlying callback method
        await cog.llm_subcommand.callback(cog, mock_context)

        args, kwargs = mock_send.call_args
        assert "현재 LLM 모델: **Gemini 2.5 flash**" in args[1]

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_uses_channel_preference(self, mock_load_config, mock_service, mock_send, mock_bot, mock_session_manager, mock_context, sample_model_definitions):
        """llm_subcommand uses channel preference when set."""
        mock_load_config.return_value = Mock()
        mock_service.MODEL_DEFINITIONS = sample_model_definitions
        mock_service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash"
        mock_session_manager.session_contexts = {}
        mock_session_manager.channel_model_preferences = {123: "GPT 5 mini"}

        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        await cog.llm_subcommand.callback(cog, mock_context)

        args, kwargs = mock_send.call_args
        assert "현재 LLM 모델: **GPT 5 mini**" in args[1]

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_uses_session_model(self, mock_load_config, mock_service, mock_send, mock_bot, mock_session_manager, mock_context, sample_model_definitions):
        """llm_subcommand uses session model when available."""
        mock_load_config.return_value = Mock()
        mock_service.MODEL_DEFINITIONS = sample_model_definitions
        mock_service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash"

        # Create mock session context
        mock_session = Mock()
        mock_session.model_alias = "GLM 4.7"
        mock_session_manager.session_contexts = {"channel:123": mock_session}
        mock_session_manager.channel_model_preferences = {}

        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        await cog.llm_subcommand.callback(cog, mock_context)

        args, kwargs = mock_send.call_args
        assert "현재 LLM 모델: **GLM 4.7**" in args[1]

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_session_takes_priority_over_channel(self, mock_load_config, mock_service, mock_send, mock_bot, mock_session_manager, mock_context, sample_model_definitions):
        """llm_subcommand prioritizes session model over channel preference."""
        mock_load_config.return_value = Mock()
        mock_service.MODEL_DEFINITIONS = sample_model_definitions
        mock_service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash"

        # Create mock session context
        mock_session = Mock()
        mock_session.model_alias = "GPT 5 mini"
        mock_session_manager.session_contexts = {"channel:123": mock_session}
        mock_session_manager.channel_model_preferences = {123: "GLM 4.7"}

        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        await cog.llm_subcommand.callback(cog, mock_context)

        args, kwargs = mock_send.call_args
        assert "현재 LLM 모델: **GPT 5 mini**" in args[1]

    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.ModelUsageService')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_sends_view_with_message(self, mock_load_config, mock_service, mock_send, mock_bot, mock_session_manager, mock_context, sample_model_definitions):
        """llm_subcommand sends view with selection dropdown."""
        mock_load_config.return_value = Mock()
        mock_service.MODEL_DEFINITIONS = sample_model_definitions
        mock_service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash"
        mock_session_manager.session_contexts = {}
        mock_session_manager.channel_model_preferences = {}

        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        await cog.llm_subcommand.callback(cog, mock_context)

        args, kwargs = mock_send.call_args
        assert "view" in kwargs
        assert isinstance(kwargs["view"], ModelSelectorView)
        assert kwargs["view"].original_message == mock_context.message
        assert kwargs.get("mention_author") is False


class TestModelSelectorCogImageSubcommand:
    """Tests for ModelSelectorCog.image_subcommand."""

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.get_channel_image_model')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_gets_current_image_model(self, mock_load_config, mock_send, mock_get_channel, mock_get_available, mock_bot, mock_session_manager, mock_context, sample_image_models):
        """image_subcommand gets current image model for channel."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        mock_get_channel.return_value = "black-forest-labs/flux.2-klein-4b"

        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        await cog.image_subcommand.callback(cog, mock_context)

        mock_get_channel.assert_called_once_with(123)

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.get_channel_image_model')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_sends_view_with_current_model(self, mock_load_config, mock_send, mock_get_channel, mock_get_available, mock_bot, mock_session_manager, mock_context, sample_image_models):
        """image_subcommand sends view with current image model."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        mock_get_channel.return_value = "black-forest-labs/flux.2-klein-4b"

        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        await cog.image_subcommand.callback(cog, mock_context)

        args, kwargs = mock_send.call_args
        assert "현재 이미지 모델: **Flux 2 Klein**" in args[1]
        assert "black-forest-labs/flux.2-klein-4b" in args[1]
        assert "view" in kwargs
        assert isinstance(kwargs["view"], ImageModelSelectorView)

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.get_channel_image_model')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_shows_api_model_name(self, mock_load_config, mock_send, mock_get_channel, mock_get_available, mock_bot, mock_session_manager, mock_context, sample_image_models):
        """image_subcommand shows API model name in parentheses."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        mock_get_channel.return_value = "sourceful/riverflow-v2-pro"

        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        await cog.image_subcommand.callback(cog, mock_context)

        args, kwargs = mock_send.call_args
        assert "Riverflow Pro" in args[1]
        assert "sourceful/riverflow-v2-pro" in args[1]


# =============================================================================
# ImageModelSelectorView Tests
# =============================================================================

class TestImageModelSelectorViewInit:
    """Tests for ImageModelSelectorView initialization."""

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_initialization(self, mock_load_config, mock_get_available, mock_bot, mock_session_manager, sample_image_models):
        """ImageModelSelectorView initializes with required attributes."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)

        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")

        assert view.cog == cog
        assert view.original_message is None
        assert view.timeout == 60

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_initialization_with_original_message(self, mock_load_config, mock_get_available, mock_bot, mock_session_manager, mock_context, sample_image_models):
        """ImageModelSelectorView initializes with original_message."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)

        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b", original_message=mock_context.message)

        assert view.original_message == mock_context.message

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_creates_select_options(self, mock_load_config, mock_get_available, mock_bot, mock_session_manager, sample_image_models):
        """ImageModelSelectorView creates select options from available models."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)

        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")

        assert len(view.children) == 1
        assert isinstance(view.children[0], ImageModelSelect)

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_options_have_emoji(self, mock_load_config, mock_get_available, mock_bot, mock_session_manager, sample_image_models):
        """Options have art palette emoji."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)

        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")
        select = view.children[0]

        for option in select.options:
            assert str(option.emoji) == "🎨"

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_options_include_description(self, mock_load_config, mock_get_available, mock_bot, mock_session_manager, sample_image_models):
        """Options include model description."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)

        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")
        select = view.children[0]

        for option in select.options:
            assert option.description

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_marks_current_as_default(self, mock_load_config, mock_get_available, mock_bot, mock_session_manager, sample_image_models):
        """Current model is marked as default."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)

        view = ImageModelSelectorView(cog, "sourceful/riverflow-v2-pro")
        select = view.children[0]

        riverflow_option = [o for o in select.options if "Riverflow" in o.label][0]
        assert riverflow_option.default is True

        flux_option = [o for o in select.options if "Flux" in o.label][0]
        assert flux_option.default is False


# =============================================================================
# ImageModelSelect Tests
# =============================================================================

class TestImageModelSelectInit:
    """Tests for ImageModelSelect initialization."""

    def test_initialization(self):
        """ImageModelSelect initializes with correct configuration."""
        options = [
            discord.SelectOption(label="Test Model", description="Test description"),
        ]

        select = ImageModelSelect(options)

        assert select.placeholder == "이미지 모델을 선택하세요..."
        assert select.min_values == 1
        assert select.max_values == 1
        assert select.options == options


class TestImageModelSelectCallback:
    """Tests for ImageModelSelect callback."""

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_callback_defers_interaction(self, mock_load_config, mock_send, mock_get_available, mock_interaction, mock_bot, mock_session_manager, sample_image_models):
        """callback defers the interaction."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")
        select = view.children[0]

        with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
            mock_values.return_value = ["Riverflow Pro"]

            await select.callback(mock_interaction)

            mock_interaction.response.defer.assert_called_once()

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.set_channel_image_model')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_callback_updates_channel_model(self, mock_load_config, mock_set_model, mock_send, mock_get_available, mock_interaction, mock_bot, mock_session_manager, sample_image_models):
        """callback updates the channel image model."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")
        select = view.children[0]

        with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
            mock_values.return_value = ["Riverflow Pro"]

            await select.callback(mock_interaction)

            mock_set_model.assert_called_once_with(123, "sourceful/riverflow-v2-pro")

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_callback_sends_confirmation(self, mock_load_config, mock_send, mock_get_available, mock_interaction, mock_bot, mock_session_manager, sample_image_models):
        """callback sends confirmation message."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")
        select = view.children[0]

        with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
            mock_values.return_value = ["Riverflow Pro"]

            await select.callback(mock_interaction)

            mock_send.assert_called_once()
            args, kwargs = mock_send.call_args
            assert "✅ 이미지 모델이 **Riverflow Pro** (`sourceful/riverflow-v2-pro`)로 변경되었습니다." in args[1]

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_callback_deletes_interaction_message(self, mock_load_config, mock_send, mock_get_available, mock_interaction, mock_bot, mock_session_manager, sample_image_models):
        """callback deletes the interaction message."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")
        select = view.children[0]

        with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
            mock_values.return_value = ["Flux 2 Klein"]

            await select.callback(mock_interaction)

            mock_interaction.message.delete.assert_called_once()

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_callback_handles_model_not_found(self, mock_load_config, mock_send, mock_get_available, mock_interaction, mock_bot, mock_session_manager, sample_image_models):
        """callback handles model not found gracefully."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")
        select = view.children[0]

        with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
            # Select a non-existent model
            mock_values.return_value = ["Non-existent Model"]

            await select.callback(mock_interaction)

            args, kwargs = mock_send.call_args
            assert "❌ 선택한 모델을 찾을 수 없습니다." in args[1]

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_callback_sends_to_original_message(self, mock_load_config, mock_send, mock_get_available, mock_interaction, mock_context, mock_bot, mock_session_manager, sample_image_models):
        """callback sends confirmation to original message when available."""
        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b", original_message=mock_context.message)
        select = view.children[0]

        with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
            mock_values.return_value = ["Flux 2 Klein"]

            await select.callback(mock_interaction)

            # Should send to original message
            args, kwargs = mock_send.call_args
            assert args[0] == mock_context.message

    @patch('persbot.bot.cogs.model_selector.get_available_image_models')
    @patch('persbot.bot.cogs.model_selector.send_discord_message')
    @patch('persbot.bot.cogs.model_selector.load_config')
    @pytest.mark.asyncio
    async def test_callback_handles_deletion_gracefully(self, mock_load_config, mock_send, mock_get_available, mock_interaction, mock_bot, mock_session_manager, sample_image_models):
        """callback handles message deletion errors gracefully."""
        # Create a proper Forbidden exception
        forbidden = discord.Forbidden(response=Mock(), message="Forbidden")
        mock_interaction.message.delete = AsyncMock(side_effect=forbidden)

        mock_load_config.return_value = Mock()
        mock_get_available.return_value = sample_image_models
        cog = ModelSelectorCog(mock_bot, mock_session_manager)
        view = ImageModelSelectorView(cog, "black-forest-labs/flux.2-klein-4b")
        select = view.children[0]

        with patch.object(type(select), 'values', new_callable=PropertyMock) as mock_values:
            mock_values.return_value = ["Flux 2 Klein"]

            # Should not raise
            await select.callback(mock_interaction)

            # Confirmation should still be sent
            assert mock_send.called

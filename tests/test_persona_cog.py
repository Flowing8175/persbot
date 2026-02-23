"""Tests for PersonaCog and related UI components.

Tests focus on:
- PersonaCog: initialization, prompt_command, cog_command_error
- PromptManagerView: main manager view functionality
"""

import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch

import pytest
import discord
from discord.ext import commands

from persbot.bot.cogs.persona import (
    PersonaCog,
    PromptManagerView,
)
from persbot.config import AppConfig
from persbot.bot.session import SessionManager
from persbot.services.llm_service import LLMService
from persbot.services.prompt_service import PromptService


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
    config.no_check_permission = False
    config.command_prefix = "!"
    return config


@pytest.fixture
def mock_llm_service():
    """Create a mock LLMService."""
    service = Mock(spec=LLMService)
    service.generate_questions_from_concept = AsyncMock(return_value=None)
    service.generate_prompt_from_concept = AsyncMock(return_value=None)
    service.generate_prompt_from_concept_with_answers = AsyncMock(return_value=None)
    return service


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager."""
    manager = Mock(spec=SessionManager)
    manager.channel_prompts = {}
    manager.sessions = {}
    manager.set_channel_prompt = Mock()
    return manager


@pytest.fixture
def mock_prompt_service(tmp_path):
    """Create a PromptService instance for testing."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    usage_path = tmp_path / "usage.json"

    with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
         patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
        service = PromptService(prompt_dir=str(prompt_dir), usage_path=str(usage_path))

    # Add a test prompt
    asyncio.run(service.add_prompt("test_persona", "Test persona content"))

    return service


@pytest.fixture
def mock_context():
    """Create a mock Discord context."""
    ctx = Mock(spec=commands.Context)
    ctx.author = Mock(spec=discord.Member)
    ctx.author.id = 789
    ctx.author.display_name = "TestUser"
    ctx.author.guild_permissions = Mock()
    ctx.author.guild_permissions.manage_guild = True
    ctx.author.bot = False
    ctx.channel = Mock()
    ctx.channel.id = 123
    ctx.channel.name = "test-channel"
    ctx.message = Mock()
    ctx.message.delete = AsyncMock()
    ctx.reply = AsyncMock()
    ctx.send = AsyncMock()
    return ctx


@pytest.fixture
def mock_interaction():
    """Create a mock Discord interaction."""
    interaction = Mock(spec=discord.Interaction)
    interaction.user = Mock()
    interaction.user.id = 789
    interaction.user.bot = False
    interaction.user.guild_permissions = Mock()
    interaction.user.guild_permissions.manage_guild = True
    interaction.response = Mock()
    interaction.response.is_done = Mock(return_value=False)
    interaction.response.send_modal = Mock()
    interaction.response.defer = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.followup = Mock()
    interaction.followup.send = AsyncMock()
    interaction.delete_original_response = AsyncMock()
    interaction.message = Mock()
    interaction.message.delete = AsyncMock()
    interaction.channel = Mock()
    interaction.channel.id = 123
    interaction.data = {}
    return interaction


@pytest.fixture
def persona_cog(mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service):
    """Create a PersonaCog instance for testing."""
    return PersonaCog(
        mock_bot,
        mock_config,
        mock_llm_service,
        mock_session_manager,
        mock_prompt_service
    )


# =============================================================================
# PersonaCog Tests
# =============================================================================

class TestPersonaCogInit:
    """Tests for PersonaCog initialization."""

    def test_initialization(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service):
        """PersonaCog initializes with all required attributes."""
        cog = PersonaCog(
            mock_bot,
            mock_config,
            mock_llm_service,
            mock_session_manager,
            mock_prompt_service
        )

        assert cog.bot == mock_bot
        assert cog.config == mock_config
        assert cog.llm_service == mock_llm_service
        assert cog.session_manager == mock_session_manager
        assert cog.prompt_service == mock_prompt_service


class TestPersonaCogPromptCommand:
    """Tests for PersonaCog.prompt_command method."""

    @pytest.mark.asyncio
    async def test_prompt_command_sends_view(self, persona_cog, mock_context):
        """prompt_command sends PromptManagerView to channel."""
        # Mock the command invocation properly
        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock(return_value=[Mock()])) as mock_send:
            # Invoke through the command's callback
            await persona_cog.prompt_command.callback(persona_cog, mock_context)

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "view" in call_args[1]
            assert isinstance(call_args[1]["view"], PromptManagerView)


class TestPersonaCogCommandError:
    """Tests for PersonaCog.cog_command_error method."""

    @pytest.mark.asyncio
    async def test_handles_missing_permissions(self, persona_cog, mock_context):
        """cog_command_error handles MissingPermissions error."""
        error = commands.MissingPermissions(["manage_guild"])
        mock_context.command = Mock()
        mock_context.command.has_error_handler = Mock(return_value=False)

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await persona_cog.cog_command_error(mock_context, error)

            assert "권한이 없습니다" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_handles_bad_argument(self, persona_cog, mock_context):
        """cog_command_error handles BadArgument error."""
        error = commands.BadArgument()
        mock_context.command = Mock()
        mock_context.command.has_error_handler = Mock(return_value=False)

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await persona_cog.cog_command_error(mock_context, error)

            assert "잘못된 인자" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_handles_command_on_cooldown(self, persona_cog, mock_context):
        """cog_command_error handles CommandOnCooldown error."""
        from discord.app_commands import Cooldown
        cooldown = Cooldown(rate=1, per=60.0)
        error = commands.CommandOnCooldown(cooldown, retry_after=5.0, type=commands.BucketType.user)
        mock_context.command = Mock()
        mock_context.command.has_error_handler = Mock(return_value=False)

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await persona_cog.cog_command_error(mock_context, error)

            assert "쿨다운" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_handles_generic_errors(self, persona_cog, mock_context):
        """cog_command_error handles generic exceptions."""
        error = Exception("Something went wrong")
        mock_context.command = Mock()
        mock_context.command.has_error_handler = Mock(return_value=False)
        mock_context.command.qualified_name = "test"

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await persona_cog.cog_command_error(mock_context, error)

            assert "오류가 발생했습니다" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_skips_when_error_handler_exists(self, persona_cog, mock_context):
        """cog_command_error skips handling when error handler exists."""
        error = Exception("Test error")
        mock_context.command = Mock()
        mock_context.command.has_error_handler = Mock(return_value=True)

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await persona_cog.cog_command_error(mock_context, error)

            mock_send.assert_not_called()


# =============================================================================
# PromptManagerView Tests
# =============================================================================

class TestPromptManagerViewInit:
    """Tests for PromptManagerView initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self, persona_cog, mock_context):
        """PromptManagerView initializes correctly."""
        view = PromptManagerView(persona_cog, mock_context)

        assert view.cog == persona_cog
        assert view.ctx == mock_context
        assert view.selected_index is None
        assert view.message is None
        assert view.timeout == 600
        assert len(view.children) > 0  # Has UI components

    @pytest.mark.asyncio
    async def test_initialization_builds_components(self, persona_cog, mock_context):
        """PromptManagerView builds UI components on init."""
        view = PromptManagerView(persona_cog, mock_context)

        # Should have select menu and buttons
        item_types = [type(item) for item in view.children]
        assert discord.ui.Select in item_types
        assert discord.ui.Button in item_types


class TestPromptManagerViewUpdateComponents:
    """Tests for PromptManagerView.update_components method."""

    @pytest.mark.asyncio
    async def test_update_components_with_prompts(self, persona_cog, mock_context):
        """update_components builds select options from prompts."""
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = None

        view.update_components()

        # Should have select with options
        select = [item for item in view.children if isinstance(item, discord.ui.Select)]
        assert len(select) == 1
        assert len(select[0].options) > 0

    @pytest.mark.asyncio
    async def test_update_components_button_states(self, persona_cog, mock_context):
        """update_components sets button enabled/disabled states."""
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = None

        view.update_components()

        # Rename, Apply, Delete should be disabled when no selection
        buttons = [item for item in view.children if isinstance(item, discord.ui.Button)]
        rename_btn = [b for b in buttons if b.label == "이름 변경"][0]
        apply_btn = [b for b in buttons if b.label == "채널에 적용"][0]
        delete_btn = [b for b in buttons if b.label == "삭제"][0]

        assert rename_btn.disabled is True
        assert apply_btn.disabled is True
        assert delete_btn.disabled is True

    @pytest.mark.asyncio
    async def test_update_components_enables_buttons_with_selection(self, persona_cog, mock_context):
        """update_components enables action buttons when prompt selected."""
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 0

        view.update_components()

        buttons = [item for item in view.children if isinstance(item, discord.ui.Button)]
        rename_btn = [b for b in buttons if b.label == "이름 변경"][0]
        apply_btn = [b for b in buttons if b.label == "채널에 적용"][0]
        delete_btn = [b for b in buttons if b.label == "삭제"][0]

        assert rename_btn.disabled is False
        assert apply_btn.disabled is False
        assert delete_btn.disabled is False


class TestPromptManagerViewBuildEmbed:
    """Tests for PromptManagerView.build_embed method."""

    @pytest.mark.asyncio
    async def test_build_embed_with_prompts(self, persona_cog, mock_context):
        """build_embed creates embed with prompt list."""
        view = PromptManagerView(persona_cog, mock_context)

        embed = view.build_embed()

        assert isinstance(embed, discord.Embed)
        assert "페르소나 관리자" in embed.title
        assert embed.color == discord.Color.gold()

    @pytest.mark.asyncio
    async def test_build_embed_with_active_marker(self, persona_cog, mock_context):
        """build_embed shows active marker for active prompt."""
        persona_cog.session_manager.channel_prompts = {123: "Test persona content"}
        view = PromptManagerView(persona_cog, mock_context)

        embed = view.build_embed()

        # Should have checkmark for active prompt
        assert "✅" in embed.description

    @pytest.mark.asyncio
    async def test_build_embed_with_selection(self, persona_cog, mock_context):
        """build_embed shows selected prompt in field."""
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 0

        embed = view.build_embed()

        # Should have selection field
        assert any(f.name == "선택된 페르소나" for f in embed.fields)


class TestPromptManagerViewRefreshView:
    """Tests for PromptManagerView.refresh_view method."""

    @pytest.mark.asyncio
    async def test_refresh_view_with_interaction(self, persona_cog, mock_context):
        """refresh_view updates view via interaction."""
        view = PromptManagerView(persona_cog, mock_context)
        mock_interaction = Mock(spec=discord.Interaction)
        mock_interaction.response = Mock()
        mock_interaction.response.is_done = Mock(return_value=False)
        mock_interaction.response.edit_message = AsyncMock()

        await view.refresh_view(mock_interaction)

        mock_interaction.response.edit_message.assert_called_once()
        assert "embed" in mock_interaction.response.edit_message.call_args[1]
        assert "view" in mock_interaction.response.edit_message.call_args[1]

    @pytest.mark.asyncio
    async def test_refresh_view_with_message(self, persona_cog, mock_context):
        """refresh_view updates view via message when interaction done."""
        view = PromptManagerView(persona_cog, mock_context)
        view.message = Mock()
        view.message.edit = AsyncMock()
        mock_interaction = Mock(spec=discord.Interaction)
        mock_interaction.response = Mock()
        mock_interaction.response.is_done = Mock(return_value=True)

        await view.refresh_view(mock_interaction)

        view.message.edit.assert_called_once()
        assert "embed" in view.message.edit.call_args[1]
        assert "view" in view.message.edit.call_args[1]


class TestPromptManagerViewOnSelect:
    """Tests for PromptManagerView.on_select callback."""

    @pytest.mark.asyncio
    async def test_on_select_updates_index(self, persona_cog, mock_context):
        """on_select updates selected_index and refreshes."""
        view = PromptManagerView(persona_cog, mock_context)
        mock_interaction = Mock()
        mock_interaction.data = {"values": ["0"]}
        mock_interaction.response.defer = AsyncMock()

        await view.on_select(mock_interaction)

        assert view.selected_index == 0
        mock_interaction.response.defer.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_select_ignores_invalid_value(self, persona_cog, mock_context):
        """on_select ignores -1 value (no prompts)."""
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 5
        mock_interaction = Mock()
        mock_interaction.data = {"values": ["-1"]}

        await view.on_select(mock_interaction)

        # Should not change
        assert view.selected_index == 5


class TestPromptManagerViewOnNew:
    """Tests for PromptManagerView.on_new callback."""

    @pytest.mark.asyncio
    async def test_on_new_checks_limit(self, persona_cog, mock_context):
        """on_new checks daily limit before proceeding."""
        persona_cog.prompt_service.check_today_limit = AsyncMock(return_value=False)
        view = PromptManagerView(persona_cog, mock_context)
        mock_interaction = Mock()

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_new(mock_interaction)

            # Should show limit message
            mock_send.assert_called_once()
            assert "생성 한도" in mock_send.call_args[0][1]


class TestPromptManagerViewOnFileAdd:
    """Tests for PromptManagerView.on_file_add callback."""

    @pytest.mark.asyncio
    async def test_on_file_add_checks_permission(self, persona_cog, mock_context):
        """on_file_add checks manage_guild permission."""
        persona_cog.config.no_check_permission = False
        mock_interaction = Mock()
        mock_interaction.user.guild_permissions.manage_guild = False

        view = PromptManagerView(persona_cog, mock_context)

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_file_add(mock_interaction)

            # Should show permission error
            mock_send.assert_called_once()
            assert "권한" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_on_file_add_sends_upload_message(self, persona_cog, mock_context):
        """on_file_add sends upload instructions when permission granted."""
        persona_cog.config.no_check_permission = True
        view = PromptManagerView(persona_cog, mock_context)
        mock_interaction = Mock()
        mock_interaction.user.id = 789
        mock_interaction.channel.id = 123

        # Mock wait_for to avoid blocking - simulate timeout
        persona_cog.bot.wait_for = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_file_add(mock_interaction)

            # Check that the initial upload message was sent
            assert mock_send.call_count >= 1
            first_call_args = mock_send.call_args_list[0]
            # The message is the second positional argument
            assert "업로드해 주세요" in first_call_args[0][1]

    @pytest.mark.asyncio
    async def test_on_file_add_waits_for_file_upload(self, persona_cog, mock_context):
        """on_file_add waits for .txt file upload."""
        persona_cog.config.no_check_permission = True
        view = PromptManagerView(persona_cog, mock_context)
        view.refresh_view = AsyncMock()
        mock_interaction = Mock()
        mock_interaction.user.id = 789
        mock_interaction.channel.id = 123

        mock_message = Mock()
        mock_message.author.id = 789
        mock_message.channel.id = 123
        mock_message.attachments = [Mock(filename="test.txt", read=AsyncMock(return_value=b"Test content"))]

        # Make wait_for return the message
        persona_cog.bot.wait_for = AsyncMock(return_value=mock_message)

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()):
            await view.on_file_add(mock_interaction)

            persona_cog.bot.wait_for.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_file_add_rejects_non_txt_files(self, persona_cog, mock_context):
        """on_file_add rejects non-.txt files."""
        persona_cog.config.no_check_permission = True
        view = PromptManagerView(persona_cog, mock_context)
        mock_interaction = Mock()
        mock_interaction.user.id = 789
        mock_interaction.channel.id = 123

        mock_message = Mock()
        mock_message.author.id = 789
        mock_message.channel.id = 123
        mock_message.attachments = [Mock(filename="test.png")]

        persona_cog.bot.wait_for = AsyncMock(return_value=mock_message)

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_file_add(mock_interaction)

            calls = [call for call in mock_send.call_args_list if ".txt" in str(call)]
            assert len(calls) > 0

    @pytest.mark.asyncio
    async def test_on_file_add_handles_timeout(self, persona_cog, mock_context):
        """on_file_add handles timeout waiting for upload."""
        persona_cog.config.no_check_permission = True
        view = PromptManagerView(persona_cog, mock_context)
        mock_interaction = Mock()
        mock_interaction.user.id = 789
        mock_interaction.channel.id = 123

        persona_cog.bot.wait_for = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_file_add(mock_interaction)

            calls = [call for call in mock_send.call_args_list if "시간 초과" in str(call)]
            assert len(calls) > 0


class TestPromptManagerViewOnApply:
    """Tests for PromptManagerView.on_apply callback."""

    @pytest.mark.asyncio
    async def test_on_apply_sets_channel_prompt(self, persona_cog, mock_context):
        """on_apply sets prompt for current channel."""
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 0
        view.refresh_view = AsyncMock()
        mock_interaction = Mock()

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()):
            await view.on_apply(mock_interaction)

            persona_cog.session_manager.set_channel_prompt.assert_called_once()
            call_args = persona_cog.session_manager.set_channel_prompt.call_args
            assert call_args[0][0] == 123  # channel id

    @pytest.mark.asyncio
    async def test_on_apply_shows_success_message(self, persona_cog, mock_context):
        """on_apply sends success message."""
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 0
        view.refresh_view = AsyncMock()
        mock_interaction = Mock()

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_apply(mock_interaction)

            mock_send.assert_called_once()
            assert "적용되었습니다" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_on_apply_handles_missing_prompt(self, persona_cog, mock_context):
        """on_apply handles case where prompt doesn't exist."""
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 999  # Invalid index
        mock_interaction = Mock()

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_apply(mock_interaction)

            assert "찾을 수 없습니다" in mock_send.call_args[0][1]


class TestPromptManagerViewOnDelete:
    """Tests for PromptManagerView.on_delete callback."""

    @pytest.mark.asyncio
    async def test_on_delete_checks_permission(self, persona_cog, mock_context):
        """on_delete checks manage_guild permission."""
        persona_cog.config.no_check_permission = False
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 0
        mock_interaction = Mock()
        mock_interaction.user.guild_permissions.manage_guild = False

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_delete(mock_interaction)

            assert "권한" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_on_delete_deletes_prompt(self, persona_cog, mock_context):
        """on_delete deletes the selected prompt."""
        persona_cog.config.no_check_permission = True
        persona_cog.prompt_service.delete_prompt = AsyncMock(return_value=True)
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 0
        view.refresh_view = AsyncMock()
        mock_interaction = Mock()

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_delete(mock_interaction)

            persona_cog.prompt_service.delete_prompt.assert_called_once_with(0)
            assert view.selected_index is None
            mock_send.assert_called_once()
            assert "삭제 완료" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_on_delete_handles_failure(self, persona_cog, mock_context):
        """on_delete handles delete failure."""
        persona_cog.config.no_check_permission = True
        persona_cog.prompt_service.delete_prompt = AsyncMock(return_value=False)
        view = PromptManagerView(persona_cog, mock_context)
        view.selected_index = 0
        mock_interaction = Mock()

        with patch("persbot.bot.cogs.persona.send_discord_message", AsyncMock()) as mock_send:
            await view.on_delete(mock_interaction)

            assert "삭제 실패" in mock_send.call_args[0][1]


class TestPromptManagerViewOnClose:
    """Tests for PromptManagerView.on_close callback."""

    @pytest.mark.asyncio
    async def test_on_close_deletes_message_and_stops(self, persona_cog, mock_context):
        """on_close deletes message and stops view."""
        view = PromptManagerView(persona_cog, mock_context)
        view.message = Mock()
        view.message.delete = AsyncMock()
        view.stop = Mock()
        mock_interaction = Mock()
        mock_interaction.message.delete = AsyncMock()

        await view.on_close(mock_interaction)

        mock_interaction.message.delete.assert_called_once()
        view.stop.assert_called_once()

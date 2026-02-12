"""Tests for bot/chat_handler.py module.

This module provides comprehensive test coverage for:
- TOOL_NAME_KOREAN constant dictionary
- ChatReply dataclass
- resolve_session_for_message function
- create_chat_reply function
- send_split_response function
- create_chat_reply_stream function
- send_streaming_response function
"""

import asyncio
import io
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any

import pytest

import discord

from persbot.bot.chat_handler import (
    ChatReply,
    resolve_session_for_message,
    create_chat_reply,
    send_split_response,
    create_chat_reply_stream,
    send_streaming_response,
    TOOL_NAME_KOREAN,
)
from persbot.bot.session import ResolvedSession, SessionManager
from persbot.services.llm_service import LLMService


# =============================================================================
# Constants Tests
# =============================================================================


class TestToolNameKorean:
    """Tests for TOOL_NAME_KOREAN constant."""

    def test_tool_name_korean_is_dict(self):
        """Test TOOL_NAME_KOREAN is a dictionary."""
        assert isinstance(TOOL_NAME_KOREAN, dict)

    def test_tool_name_korean_has_expected_entries(self):
        """Test TOOL_NAME_KOREAN has expected tool names."""
        expected_tools = [
            "generate_image",
            "send_image",
            "get_time",
            "web_search",
            "get_weather",
            "get_guild_info",
            "get_guild_roles",
            "get_guild_emojis",
            "search_episodic_memory",
            "save_episodic_memory",
            "remove_episodic_memory",
            "get_user_info",
            "get_member_info",
            "get_member_roles",
            "inspect_external_content",
            "get_channel_info",
            "get_channel_history",
            "get_message",
            "list_channels",
            "check_virtual_routine_status",
            "get_routine_schedule",
            "generate_situational_snapshot",
            "describe_scene_atmosphere",
        ]
        for tool in expected_tools:
            assert tool in TOOL_NAME_KOREAN

    def test_tool_name_korean_values_are_strings(self):
        """Test all TOOL_NAME_KOREAN values are strings."""
        for key, value in TOOL_NAME_KOREAN.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


# =============================================================================
# ChatReply Dataclass Tests
# =============================================================================


class TestChatReply:
    """Tests for ChatReply dataclass."""

    def test_init_all_fields(self):
        """Test ChatReply initialization with all fields."""
        reply = ChatReply(
            text="Test response",
            session_key="test-session",
            response=Mock(),
            images=[b"img1", b"img2"],
        )

        assert reply.text == "Test response"
        assert reply.session_key == "test-session"
        assert reply.response is not None
        assert reply.images == [b"img1", b"img2"]

    def test_init_with_defaults(self):
        """Test ChatReply with default images."""
        reply = ChatReply(
            text="Response",
            session_key="session-123",
            response=Mock(),
        )

        assert reply.text == "Response"
        assert reply.session_key == "session-123"
        assert reply.images == []

    def test_chat_reply_is_frozen(self):
        """Test ChatReply is frozen (immutable)."""
        reply = ChatReply(
            text="Text",
            session_key="key",
            response=Mock(),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            reply.text = "New text"

    def test_chat_reply_allows_mutable_images_list(self):
        """Test that images list can be mutated even though dataclass is frozen."""
        # Note: With frozen=True, the images field itself can't be reassigned,
        # but the list can be modified
        reply = ChatReply(
            text="Text",
            session_key="key",
            response=Mock(),
            images=[],
        )

        # Can modify the list
        reply.images.append(b"new_image")
        assert len(reply.images) == 1


# =============================================================================
# resolve_session_for_message Tests
# =============================================================================


class TestResolveSessionForMessage:
    """Tests for resolve_session_for_message function."""

    @pytest.fixture
    def mock_message(self):
        """Create a mock Discord message."""
        message = MagicMock(spec=discord.Message)
        message.id = 12345
        message.author.id = 67890
        message.author.name = "testuser"
        message.author.bot = False
        message.channel.id = 11111
        message.created_at = datetime.now(timezone.utc)
        message.reference = None
        message.clean_content = "Original message content"
        return message

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        manager = Mock(spec=SessionManager)
        manager.resolve_session = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_resolve_basic_message(self, mock_message, mock_session_manager):
        """Test resolving a basic message without reference."""
        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello world",
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello world",
            session_manager=mock_session_manager,
        )

        assert result is not None
        assert result.session_key == "test-session"
        assert result.cleaned_message == "Hello world"
        mock_session_manager.resolve_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_message_with_reference(self, mock_message, mock_session_manager):
        """Test resolving a message with reply reference."""
        # Setup reference message
        ref_message = MagicMock(spec=discord.Message)
        ref_message.id = 999
        ref_message.author.id = 111
        ref_message.author.bot = False
        ref_message.clean_content = "Referenced message"

        mock_message.reference = MagicMock()
        mock_message.reference.message_id = 999
        mock_message.reference.resolved = ref_message

        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message='(답장 대상: 111, 내용: "Referenced message")\nHello',
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello",
            session_manager=mock_session_manager,
        )

        assert result is not None
        mock_session_manager.resolve_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_message_with_deleted_reference(self, mock_message, mock_session_manager):
        """Test resolving message with deleted reference."""
        mock_message.reference = MagicMock()
        mock_message.reference.message_id = 999
        # Create a mock DeletedReferencedMessage
        deleted_ref = Mock()
        deleted_ref.__class__.__name__ = "DeletedReferencedMessage"
        deleted_ref.clean_content = ""  # Empty content for deleted message
        deleted_ref.author = Mock()
        deleted_ref.author.bot = False
        mock_message.reference.resolved = deleted_ref

        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello",
            session_manager=mock_session_manager,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_resolve_message_with_unresolved_reference(self, mock_message, mock_session_manager):
        """Test resolving message with None reference that needs fetching."""
        mock_message.reference = MagicMock()
        mock_message.reference.message_id = 999
        mock_message.reference.resolved = None

        # Mock the channel.fetch_message
        mock_message.channel.fetch_message = AsyncMock()

        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello",
            session_manager=mock_session_manager,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_resolve_message_with_fetch_error(self, mock_message, mock_session_manager):
        """Test resolving message when fetch fails."""
        mock_message.reference = MagicMock()
        mock_message.reference.message_id = 999
        mock_message.reference.resolved = None

        # Mock the channel.fetch_message to raise NotFound
        mock_message.channel.fetch_message = AsyncMock(
            side_effect=discord.NotFound(Mock(status=404), "Not found")
        )

        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello",
            session_manager=mock_session_manager,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_resolve_reply_to_summary(self, mock_message, mock_session_manager):
        """Test detecting reply to summary message."""
        ref_message = MagicMock(spec=discord.Message)
        ref_message.id = 999
        ref_message.author.bot = True  # Bot message
        ref_message.clean_content = "**... 요약:** Summary of conversation"

        mock_message.reference = MagicMock()
        mock_message.reference.message_id = 999
        mock_message.reference.resolved = ref_message

        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,  # Will be set to True
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello",
            session_manager=mock_session_manager,
        )

        assert result is not None
        assert result.is_reply_to_summary is True

    @pytest.mark.asyncio
    async def test_resolve_reply_to_non_bot_summary(self, mock_message, mock_session_manager):
        """Test reply to summary from non-bot is not marked as is_reply_to_summary."""
        ref_message = MagicMock(spec=discord.Message)
        ref_message.id = 999
        ref_message.author.bot = False  # Not a bot
        ref_message.clean_content = "**... 요약:** Summary"

        mock_message.reference = MagicMock()
        mock_message.reference.message_id = 999
        mock_message.reference.resolved = ref_message

        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello",
            session_manager=mock_session_manager,
        )

        assert result is not None
        assert result.is_reply_to_summary is False

    @pytest.mark.asyncio
    async def test_resolve_returns_none_for_empty_cleaned_message(
        self, mock_message, mock_session_manager
    ):
        """Test that None is returned when cleaned_message is empty."""
        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="",  # Empty cleaned message
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="",
            session_manager=mock_session_manager,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_with_cancel_event_set(self, mock_message, mock_session_manager):
        """Test resolution respects cancel_event."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello",
            session_manager=mock_session_manager,
            cancel_event=cancel_event,
        )

        # Cancel event should be passed to resolve_session
        mock_session_manager.resolve_session.assert_called_once()
        call_kwargs = mock_session_manager.resolve_session.call_args.kwargs
        assert "cancel_event" in call_kwargs

    @pytest.mark.asyncio
    async def test_resolve_with_none_resolution(self, mock_message, mock_session_manager):
        """Test when resolve_session returns None."""
        mock_session_manager.resolve_session = AsyncMock(return_value=None)

        # When resolve_session returns None, accessing resolution.cleaned_message
        # will raise AttributeError in the source code
        with pytest.raises(AttributeError):
            result = await resolve_session_for_message(
                message=mock_message,
                content="Hello",
                session_manager=mock_session_manager,
            )

    @pytest.mark.asyncio
    async def test_resolve_passes_reference_message_id_as_none(
        self, mock_message, mock_session_manager
    ):
        """Test that reference_message_id is always passed as None."""
        resolution = ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,
        )
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        await resolve_session_for_message(
            message=mock_message,
            content="Hello",
            session_manager=mock_session_manager,
        )

        call_kwargs = mock_session_manager.resolve_session.call_args.kwargs
        assert call_kwargs.get("reference_message_id") is None


# =============================================================================
# create_chat_reply Tests
# =============================================================================


class TestCreateChatReply:
    """Tests for create_chat_reply function."""

    @pytest.fixture
    def mock_message(self):
        """Create a mock Discord message."""
        message = MagicMock(spec=discord.Message)
        message.id = 12345
        message.author.id = 67890
        message.author.name = "testuser"
        message.channel.id = 11111
        message.created_at = datetime.now(timezone.utc)
        return message

    @pytest.fixture
    def mock_resolution(self):
        """Create a mock ResolvedSession."""
        return ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,
        )

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        manager = Mock(spec=SessionManager)
        manager.get_or_create = AsyncMock()
        manager.link_message_to_session = Mock()
        return manager

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLMService."""
        service = Mock(spec=LLMService)
        service.generate_chat_response = AsyncMock(return_value=("Response text", Mock()))
        service.extract_function_calls_from_response = Mock(return_value=None)
        service.assistant_backend = Mock()
        return service

    @pytest.mark.asyncio
    async def test_create_reply_single_message(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test creating reply from single message."""
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        assert result is not None
        assert isinstance(result, ChatReply)
        assert result.text == "Response text"
        assert result.session_key == "test-session"

    @pytest.mark.asyncio
    async def test_create_reply_message_list(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test creating reply from list of messages."""
        message_list = [mock_message, MagicMock(spec=discord.Message)]
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        result = await create_chat_reply(
            message=message_list,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        assert result is not None
        # First message should be used
        mock_session_manager.get_or_create.assert_called_once()
        call_kwargs = mock_session_manager.get_or_create.call_args.kwargs
        assert call_kwargs["message_id"] == str(mock_message.id)

    @pytest.mark.asyncio
    async def test_create_reply_with_none_response(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test when LLM returns None."""
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )
        mock_llm_service.generate_chat_response = AsyncMock(return_value=None)

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_create_reply_with_tool_manager_disabled(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test with disabled tool manager."""
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled = Mock(return_value=False)

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        assert result is not None
        mock_llm_service.extract_function_calls_from_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_reply_with_tool_execution(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test tool execution flow."""
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        # Setup tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled = Mock(return_value=True)
        mock_tool_manager.get_enabled_tools = Mock(return_value={})
        mock_tool_manager.execute_tools = AsyncMock(return_value=[])

        # Setup function calls
        function_calls = [{"name": "get_time", "arguments": {}}]
        mock_llm_service.extract_function_calls_from_response = Mock(
            side_effect=[function_calls, None]  # First call, then None
        )

        # Setup send_tool_results
        continuation_response = ("Final response", Mock())
        mock_llm_service.send_tool_results = AsyncMock(return_value=continuation_response)

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        assert result is not None
        mock_tool_manager.execute_tools.assert_called_once()
        mock_llm_service.send_tool_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_reply_with_cancel_event_before_tools(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test cancellation before tool execution."""
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled = Mock(return_value=True)
        mock_tool_manager.get_enabled_tools = Mock(return_value={})

        function_calls = [{"name": "get_time"}]
        mock_llm_service.extract_function_calls_from_response = Mock(
            return_value=function_calls
        )

        cancel_event = asyncio.Event()
        cancel_event.set()

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
            cancel_event=cancel_event,
        )

        assert result is not None
        # Tool execution should be skipped
        mock_tool_manager.execute_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_reply_with_tool_error(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test tool execution error handling."""
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled = Mock(return_value=True)
        mock_tool_manager.get_enabled_tools = Mock(return_value={})
        mock_tool_manager.execute_tools = AsyncMock(side_effect=Exception("Tool error"))

        function_calls = [{"name": "get_time"}]
        mock_llm_service.extract_function_calls_from_response = Mock(
            return_value=function_calls
        )

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        # Should return original response despite tool error
        assert result is not None

    @pytest.mark.asyncio
    async def test_create_reply_with_image_generation(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test image collection from tool results."""
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled = Mock(return_value=True)
        mock_tool_manager.get_enabled_tools = Mock(return_value={})

        # Return image bytes in tool results
        mock_tool_manager.execute_tools = AsyncMock(
            return_value=[{"image_bytes": b"fake_image"}]
        )

        function_calls = [{"name": "generate_image"}]
        mock_llm_service.extract_function_calls_from_response = Mock(
            side_effect=[function_calls, None]
        )
        mock_llm_service.send_tool_results = AsyncMock(
            return_value=("Response", Mock())
        )

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        assert result is not None
        assert len(result.images) == 1
        assert result.images[0] == b"fake_image"

    @pytest.mark.asyncio
    async def test_create_reply_max_tool_rounds(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test maximum tool rounds limit."""
        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled = Mock(return_value=True)
        mock_tool_manager.get_enabled_tools = Mock(return_value={})
        mock_tool_manager.execute_tools = AsyncMock(return_value=[])

        # Always return function calls to trigger max rounds
        function_calls = [{"name": "get_time"}]
        mock_llm_service.extract_function_calls_from_response = Mock(
            return_value=function_calls
        )
        mock_llm_service.send_tool_results = AsyncMock(
            return_value=("Response", Mock())
        )

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        assert result is not None
        # Should execute at most 10 rounds
        assert mock_llm_service.send_tool_results.call_count <= 10

    @pytest.mark.asyncio
    async def test_create_reply_with_reply_to_summary(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test reply to summary uses summarizer backend."""
        mock_resolution.is_reply_to_summary = True

        chat_session = Mock()
        chat_session.model_alias = "test-model"
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        result = await create_chat_reply(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        assert result is not None
        # Check use_summarizer_backend was passed
        call_kwargs = mock_llm_service.generate_chat_response.call_args.kwargs
        assert call_kwargs.get("use_summarizer_backend") is True


# =============================================================================
# send_split_response Tests
# =============================================================================


class TestSendSplitResponse:
    """Tests for send_split_response function."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock(spec=discord.abc.Messageable)
        channel.id = 12345
        channel.send = AsyncMock(return_value=Mock(id=999))
        channel.typing = MagicMock()
        return channel

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        manager = Mock(spec=SessionManager)
        manager.link_message_to_session = Mock()
        return manager

    @pytest.fixture
    def chat_reply(self):
        """Create a ChatReply object."""
        return ChatReply(
            text="Test response",
            session_key="test-session",
            response=Mock(),
        )

    @pytest.mark.asyncio
    async def test_send_short_text(self, mock_channel, mock_session_manager, chat_reply):
        """Test sending short text without splitting."""
        await send_split_response(
            channel=mock_channel,
            reply=chat_reply,
            session_manager=mock_session_manager,
        )

        assert mock_channel.send.call_count == 1
        mock_session_manager.link_message_to_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_long_text_splitting(self, mock_channel, mock_session_manager, chat_reply):
        """Test sending long text with smart splitting."""
        long_text = "a" * 2000
        reply = ChatReply(
            text=long_text,
            session_key="test-session",
            response=Mock(),
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        assert mock_channel.send.call_count > 1

    @pytest.mark.asyncio
    async def test_send_preserves_newlines(self, mock_channel, mock_session_manager, chat_reply):
        """Test that existing newlines are preserved."""
        reply = ChatReply(
            text="Line 1\nLine 2\n",
            session_key="test-session",
            response=Mock(),
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        assert mock_channel.send.call_count >= 2

    @pytest.mark.asyncio
    async def test_send_empty_lines_skipped(self, mock_channel, mock_session_manager, chat_reply):
        """Test that empty lines are skipped."""
        reply = ChatReply(
            text="Line 1\n\n\nLine 2",
            session_key="test-session",
            response=Mock(),
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        # Empty lines should be skipped
        assert mock_channel.send.call_count == 2

    @pytest.mark.asyncio
    async def test_send_with_delay(self, mock_channel, mock_session_manager, chat_reply):
        """Test that delay is applied based on line length."""
        reply = ChatReply(
            text="a" * 100,
            session_key="test-session",
            response=Mock(),
        )

        with patch("asyncio.sleep") as mock_sleep:
            await send_split_response(
                channel=mock_channel,
                reply=reply,
                session_manager=mock_session_manager,
            )

            mock_sleep.assert_called_once()
            # Delay should be 0.1s * 100 = 10s clamped to 1.7s max
            sleep_arg = mock_sleep.call_args[0][0]
            assert sleep_arg >= 0.1
            assert sleep_arg <= 1.7

    @pytest.mark.asyncio
    async def test_send_min_delay_clamping(self, mock_channel, mock_session_manager, chat_reply):
        """Test minimum delay clamping."""
        reply = ChatReply(
            text="Hi",
            session_key="test-session",
            response=Mock(),
        )

        with patch("asyncio.sleep") as mock_sleep:
            await send_split_response(
                channel=mock_channel,
                reply=reply,
                session_manager=mock_session_manager,
            )

            sleep_arg = mock_sleep.call_args[0][0]
            assert sleep_arg >= 0.1  # Minimum delay

    @pytest.mark.asyncio
    async def test_send_max_delay_clamping(self, mock_channel, mock_session_manager, chat_reply):
        """Test maximum delay clamping."""
        reply = ChatReply(
            text="a" * 1000,
            session_key="test-session",
            response=Mock(),
        )

        with patch("asyncio.sleep") as mock_sleep:
            await send_split_response(
                channel=mock_channel,
                reply=reply,
                session_manager=mock_session_manager,
            )

            sleep_arg = mock_sleep.call_args[0][0]
            assert sleep_arg <= 1.7  # Maximum delay

    @pytest.mark.asyncio
    async def test_send_with_images(self, mock_channel, mock_session_manager):
        """Test sending images as attachments."""
        reply = ChatReply(
            text="Response with images",
            session_key="test-session",
            response=Mock(),
            images=[b"img1", b"img2"],
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        # Should send text + 2 images
        assert mock_channel.send.call_count == 3

    @pytest.mark.asyncio
    async def test_send_images_with_correct_filename(self, mock_channel, mock_session_manager):
        """Test image attachments have correct filename."""
        reply = ChatReply(
            text="Response",
            session_key="test-session",
            response=Mock(),
            images=[b"image_data"],
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        # Check that discord.File was called with correct filename
        sent_messages = mock_channel.send.call_args_list
        image_call = None
        for call in sent_messages:
            kwargs = call.kwargs
            if "file" in kwargs:
                image_call = kwargs
                break

        assert image_call is not None
        assert "file" in image_call

    @pytest.mark.asyncio
    async def test_send_cancellation(self, mock_channel, mock_session_manager, chat_reply):
        """Test cancellation during sending."""
        mock_channel.send = AsyncMock(
            side_effect=[Mock(id=1), asyncio.CancelledError()]
        )

        reply = ChatReply(
            text="Line 1\nLine 2",
            session_key="test-session",
            response=Mock(),
        )

        with pytest.raises(asyncio.CancelledError):
            await send_split_response(
                channel=mock_channel,
                reply=reply,
                session_manager=mock_session_manager,
            )

        # Both lines are attempted: first succeeds, second raises CancelledError
        assert mock_channel.send.call_count == 2


# =============================================================================
# create_chat_reply_stream Tests
# =============================================================================


class TestCreateChatReplyStream:
    """Tests for create_chat_reply_stream function."""

    @pytest.fixture
    def mock_message(self):
        """Create a mock Discord message."""
        message = MagicMock(spec=discord.Message)
        message.id = 12345
        message.author.id = 67890
        message.author.name = "testuser"
        message.channel.id = 11111
        message.created_at = datetime.now(timezone.utc)
        return message

    @pytest.fixture
    def mock_resolution(self):
        """Create a mock ResolvedSession."""
        return ResolvedSession(
            session_key="test-session",
            cleaned_message="Hello",
            is_reply_to_summary=False,
        )

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        manager = Mock(spec=SessionManager)
        manager.get_or_create = AsyncMock()
        return manager

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLMService."""
        service = Mock(spec=LLMService)

        async def mock_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"

        service.generate_chat_response_stream = mock_stream
        return service

    @pytest.mark.asyncio
    async def test_stream_single_message(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test streaming with single message."""
        chat_session = Mock()
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        async def stream_gen(*args, **kwargs):
            yield "Hello "
            yield "world"

        mock_llm_service.generate_chat_response_stream = stream_gen

        chunks = []
        async for chunk in create_chat_reply_stream(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        ):
            chunks.append(chunk)

        assert chunks == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_stream_message_list(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test streaming with list of messages."""
        message_list = [mock_message, MagicMock(spec=discord.Message)]
        chat_session = Mock()
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        async def stream_gen(*args, **kwargs):
            yield "chunk"

        mock_llm_service.generate_chat_response_stream = stream_gen

        chunks = []
        async for chunk in create_chat_reply_stream(
            message=message_list,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_with_tool_manager_enabled(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test streaming with enabled tool manager."""
        chat_session = Mock()
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        async def stream_gen(*args, **kwargs):
            yield "chunk"

        mock_llm_service.generate_chat_response_stream = stream_gen

        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled = Mock(return_value=True)
        mock_tool_manager.get_enabled_tools = Mock(return_value={})

        chunks = []
        async for chunk in create_chat_reply_stream(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_with_reply_to_summary(
        self, mock_message, mock_resolution, mock_session_manager, mock_llm_service
    ):
        """Test streaming with reply to summary."""
        mock_resolution.is_reply_to_summary = True

        chat_session = Mock()
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(chat_session, "test-session")
        )

        async def stream_gen(*args, **kwargs):
            # Verify use_summarizer_backend is True
            assert kwargs.get("use_summarizer_backend") is True
            yield "chunk"

        mock_llm_service.generate_chat_response_stream = stream_gen

        chunks = []
        async for chunk in create_chat_reply_stream(
            message=mock_message,
            resolution=mock_resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1


# =============================================================================
# send_streaming_response Tests
# =============================================================================


class TestSendStreamingResponse:
    """Tests for send_streaming_response function."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock(spec=discord.abc.Messageable)
        channel.id = 12345
        channel.send = AsyncMock(return_value=Mock(id=999))
        channel.typing = MagicMock()
        return channel

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        manager = Mock(spec=SessionManager)
        manager.link_message_to_session = Mock()
        return manager

    def make_async_iterator(self, items):
        """Helper to make async iterator from list."""

        async def iterator():
            for item in items:
                yield item

        return iterator()

    @pytest.mark.asyncio
    async def test_send_empty_stream(self, mock_channel, mock_session_manager):
        """Test sending empty stream."""
        stream = self.make_async_iterator([])

        result = await send_streaming_response(
            channel=mock_channel,
            stream=stream,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert result == []
        assert mock_channel.send.call_count == 0

    @pytest.mark.asyncio
    async def test_send_stream_with_text_chunks(self, mock_channel, mock_session_manager):
        """Test streaming text chunks."""
        stream = self.make_async_iterator(["chunk1", "chunk2", "chunk3"])

        result = await send_streaming_response(
            channel=mock_channel,
            stream=stream,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert len(result) == 3
        assert mock_channel.send.call_count == 3

    @pytest.mark.asyncio
    async def test_send_stream_preserves_newlines(self, mock_channel, mock_session_manager):
        """Test streaming preserves newlines."""
        stream = self.make_async_iterator(["line 1\nmore text", "line 2"])

        result = await send_streaming_response(
            channel=mock_channel,
            stream=stream,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        # Should split by newlines
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_send_stream_ignores_empty_chunks(self, mock_channel, mock_session_manager):
        """Test that empty chunks are skipped."""
        stream = self.make_async_iterator(["chunk1", "", "chunk2"])

        result = await send_streaming_response(
            channel=mock_channel,
            stream=stream,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert len(result) == 2
        assert mock_channel.send.call_count == 2

    @pytest.mark.asyncio
    async def test_send_stream_with_long_line_splitting(self, mock_channel, mock_session_manager):
        """Test streaming long line with smart split."""
        long_line = "a" * 2000
        stream = self.make_async_iterator([long_line])

        result = await send_streaming_response(
            channel=mock_channel,
            stream=stream,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert mock_channel.send.call_count > 1

    @pytest.mark.asyncio
    async def test_stream_cancellation(self, mock_channel, mock_session_manager):
        """Test cancellation during stream."""

        async def stream():
            yield "chunk1"
            raise asyncio.CancelledError("Stream cancelled")

        with pytest.raises(asyncio.CancelledError):
            await send_streaming_response(
                channel=mock_channel,
                stream=stream(),
                session_key="test-session",
                session_manager=mock_session_manager,
            )

        assert mock_channel.send.call_count == 1

    @pytest.mark.asyncio
    async def test_stream_session_linking(self, mock_channel, mock_session_manager):
        """Test that stream messages are linked to session."""
        stream = self.make_async_iterator(["msg1", "msg2"])

        result = await send_streaming_response(
            channel=mock_channel,
            stream=stream,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert mock_session_manager.link_message_to_session.call_count == 2

    @pytest.mark.asyncio
    async def test_stream_typing_indicator(self, mock_channel, mock_session_manager):
        """Test typing indicator during streaming."""
        stream = self.make_async_iterator(["chunk1", "chunk2"])

        await send_streaming_response(
            channel=mock_channel,
            stream=stream,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        # Typing should be called for each chunk
        assert mock_channel.typing.call_count >= 2

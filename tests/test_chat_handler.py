"""Feature tests for chat_handler module.

Tests focus on behavior using mocking to avoid complex dependencies:
- ChatReply: container for LLM response
- resolve_session_for_message: session resolution for Discord message
- create_chat_reply: creating LLM reply
- send_split_response: sending response line by line
- send_streaming_response: streaming response
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Mock discord module before importing anything else
@pytest.fixture(autouse=True)
def mock_discord():
    """Mock discord module to avoid import issues."""
    import sys
    mock_discord = MagicMock()
    mock_discord.File = MagicMock
    mock_discord.DeletedReferencedMessage = MagicMock
    mock_discord.NotFound = Exception
    mock_discord.HTTPException = Exception
    sys.modules['discord'] = mock_discord
    yield mock_discord
    if 'discord' in sys.modules:
        del sys.modules['discord']


class MockMessage:
    """Mock Discord message for testing."""

    def __init__(self, content="test", author_id=123, channel_id=456, message_id=789):
        self.content = content
        self.author = MagicMock()
        self.author.id = author_id
        self.author.name = "TestUser"
        self.channel = MagicMock()
        self.channel.id = channel_id
        self.id = message_id
        self.created_at = MagicMock()
        self.reference = None
        self.guild = None
        self.clean_content = content


class MockChatReply:
    """Mock ChatReply for testing."""

    def __init__(self, text, session_key, response=None, images=None):
        self.text = text
        self.session_key = session_key
        self.response = response
        self.images = images or []


class MockResolvedSession:
    """Mock ResolvedSession for testing."""

    def __init__(self, session_key, cleaned_message, is_reply_to_summary=False):
        self.session_key = session_key
        self.cleaned_message = cleaned_message
        self.is_reply_to_summary = is_reply_to_summary


class TestChatReplyDataclass:
    """Tests for ChatReply dataclass."""

    def test_chat_reply_exists(self):
        """ChatReply class can be imported."""
        # Import from chat_handler module
        from persbot.bot.chat_handler import ChatReply
        assert ChatReply is not None

    def test_chat_reply_has_required_fields(self):
        """ChatReply has required fields."""
        from persbot.bot.chat_handler import ChatReply

        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=MagicMock(),
        )
        assert reply.text == "Hello"
        assert reply.session_key == "channel:123"
        assert reply.response is not None

    def test_chat_reply_default_images_is_empty(self):
        """ChatReply images defaults to empty list."""
        from persbot.bot.chat_handler import ChatReply

        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=MagicMock(),
        )
        assert reply.images == []

    def test_chat_reply_can_have_images(self):
        """ChatReply can include images."""
        from persbot.bot.chat_handler import ChatReply

        reply = ChatReply(
            text="Here's an image",
            session_key="channel:123",
            response=MagicMock(),
            images=[b"fake_image_data"],
        )
        assert len(reply.images) == 1


class TestResolveSessionForMessage:
    """Tests for resolve_session_for_message function."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        manager = MagicMock()
        manager.resolve_session = AsyncMock()
        manager.resolve_session.return_value = MockResolvedSession(
            session_key="channel:456",
            cleaned_message="test message",
        )
        return manager

    @pytest.mark.asyncio
    async def test_resolves_session_for_simple_message(self, mock_session_manager):
        """resolve_session_for_message resolves session for simple message."""
        from persbot.bot.chat_handler import resolve_session_for_message

        message = MockMessage(content="Hello")
        resolution = await resolve_session_for_message(
            message,
            "Hello",
            session_manager=mock_session_manager,
        )

        assert resolution is not None
        assert resolution.session_key == "channel:456"

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_message(self, mock_session_manager):
        """resolve_session_for_message returns None for empty message."""
        from persbot.bot.chat_handler import resolve_session_for_message

        # Mock returns empty cleaned_message
        mock_session_manager.resolve_session.return_value = MockResolvedSession(
            session_key="channel:456",
            cleaned_message="",
        )

        message = MockMessage(content="")
        resolution = await resolve_session_for_message(
            message,
            "",
            session_manager=mock_session_manager,
        )

        assert resolution is None

    @pytest.mark.asyncio
    async def test_handles_reply_with_reference(self, mock_session_manager):
        """resolve_session_for_message handles replies with reference."""
        from persbot.bot.chat_handler import resolve_session_for_message

        message = MockMessage(content="Reply content")
        ref_msg = MagicMock()
        ref_msg.author.id = 999
        ref_msg.author.bot = False
        ref_msg.clean_content = "Original message"
        ref_msg.author.name = "OriginalAuthor"

        message.reference = MagicMock()
        message.reference.message_id = 111
        message.reference.resolved = ref_msg

        resolution = await resolve_session_for_message(
            message,
            "Reply content",
            session_manager=mock_session_manager,
        )

        assert resolution is not None


class TestToolNameKoreanMapping:
    """Tests for TOOL_NAME_KOREAN mapping."""

    def test_tool_name_mapping_exists(self):
        """TOOL_NAME_KOREAN mapping exists."""
        from persbot.bot.chat_handler import TOOL_NAME_KOREAN
        assert TOOL_NAME_KOREAN is not None

    def test_has_common_tool_translations(self):
        """TOOL_NAME_KOREAN has common tool translations."""
        from persbot.bot.chat_handler import TOOL_NAME_KOREAN

        # Check some expected translations
        assert "generate_image" in TOOL_NAME_KOREAN
        assert "web_search" in TOOL_NAME_KOREAN
        assert "get_time" in TOOL_NAME_KOREAN
        assert "get_weather" in TOOL_NAME_KOREAN


class TestSendSplitResponse:
    """Tests for send_split_response function."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock()
        channel.typing = MagicMock()
        channel.send = AsyncMock()
        channel.id = 123
        # Create proper async context manager for typing()
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=None)
        channel.typing.return_value = typing_cm
        return channel

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        manager = MagicMock()
        manager.link_message_to_session = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_sends_multiline_text(self, mock_channel, mock_session_manager):
        """send_split_response sends multiline text."""
        from persbot.bot.chat_handler import send_split_response, ChatReply

        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        reply = ChatReply(
            text="Line 1\nLine 2\nLine 3",
            session_key="channel:123",
            response=MagicMock(),
        )

        await send_split_response(mock_channel, reply, mock_session_manager)

        # Should send 3 messages
        assert mock_channel.send.call_count == 3

    @pytest.mark.asyncio
    async def test_sends_images_as_attachments(self, mock_channel, mock_session_manager):
        """send_split_response sends images as attachments."""
        from persbot.bot.chat_handler import send_split_response, ChatReply

        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        reply = ChatReply(
            text="Here's an image",
            session_key="channel:123",
            response=MagicMock(),
            images=[b"fake_image_data"],
        )

        await send_split_response(mock_channel, reply, mock_session_manager)

        # Should send text + 1 image
        assert mock_channel.send.call_count == 2


class TestSendStreamingResponse:
    """Tests for send_streaming_response function."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock()
        channel.typing = MagicMock()
        channel.send = AsyncMock()
        channel.id = 123
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=None)
        channel.typing.return_value = typing_cm
        return channel

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        manager = MagicMock()
        manager.link_message_to_session = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_sends_stream_chunks(self, mock_channel, mock_session_manager):
        """send_streaming_response sends stream chunks."""
        from persbot.bot.chat_handler import send_streaming_response

        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        async def stream():
            yield "Chunk 1"
            yield "Chunk 2"
            yield "Chunk 3"

        result = await send_streaming_response(
            mock_channel, stream(), "channel:123", mock_session_manager
        )

        assert len(result) == 3
        assert mock_channel.send.call_count == 3

    @pytest.mark.asyncio
    async def test_skips_empty_chunks(self, mock_channel, mock_session_manager):
        """send_streaming_response skips empty chunks."""
        from persbot.bot.chat_handler import send_streaming_response

        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        async def stream():
            yield "Chunk 1"
            yield ""
            yield "   "
            yield "Chunk 2"

        result = await send_streaming_response(
            mock_channel, stream(), "channel:123", mock_session_manager
        )

        # Only 2 non-empty chunks
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_propagates_cancelled_error(self, mock_channel, mock_session_manager):
        """send_streaming_response propagates CancelledError."""
        from persbot.bot.chat_handler import send_streaming_response

        async def stream():
            yield "Chunk 1"
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await send_streaming_response(
                mock_channel, stream(), "channel:123", mock_session_manager
            )

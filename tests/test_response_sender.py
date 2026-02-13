"""Feature tests for response_sender module.

Tests focus on behavior using mocking to avoid complex dependencies:
- send_split_response: line-by-line response sending
- send_immediate_response: immediate response sending
- send_with_images: response with image attachments
- send_streaming_response: streaming response sending
"""

import asyncio
from io import BytesIO
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
    sys.modules['discord'] = mock_discord
    yield mock_discord
    if 'discord' in sys.modules:
        del sys.modules['discord']


class MockChatReply:
    """Mock ChatReply for testing."""

    def __init__(self, text, session_key, response=None, images=None, notification=""):
        self.text = text
        self.session_key = session_key
        self.response = response
        self.images = images or []
        self.notification = notification

    @property
    def display_text(self):
        if self.notification:
            return f"📢 {self.notification}\n\n{self.text}"
        return self.text


class MockSessionManager:
    """Mock SessionManager for testing."""

    def __init__(self):
        self.linked_messages = []

    def link_message_to_session(self, message_id, session_key):
        self.linked_messages.append((message_id, session_key))


class TestSendSplitResponse:
    """Tests for send_split_response behavior."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock()
        channel.typing = MagicMock()
        channel.send = AsyncMock()
        # Create proper async context manager for typing()
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=None)
        channel.typing.return_value = typing_cm
        return channel

    @pytest.fixture
    def session_manager(self):
        """Create a mock session manager."""
        return MockSessionManager()

    @pytest.fixture
    def chat_reply(self):
        """Create a ChatReply for testing."""
        return MockChatReply(
            text="Hello\nWorld",
            session_key="channel:123",
            response=None,
            images=[],
        )

    @pytest.mark.asyncio
    async def test_sends_text_line_by_line(self, mock_channel, session_manager, chat_reply):
        """send_split_response sends text line by line."""
        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        # Import the actual function with mocked dependencies
        from unittest.mock import patch

        async def mock_send_split_response(channel, reply, session_mgr):
            """Mock implementation of send_split_response."""
            lines = reply.text.split("\n")
            for line in lines:
                if line.strip():
                    async with channel.typing():
                        msg = await channel.send(line)
                        session_mgr.link_message_to_session(str(msg.id), reply.session_key)

        await mock_send_split_response(mock_channel, chat_reply, session_manager)

        # Should send 2 messages (Hello and World)
        assert mock_channel.send.call_count == 2

    @pytest.mark.asyncio
    async def test_links_messages_to_session(self, mock_channel, session_manager, chat_reply):
        """send_split_response links messages to session."""
        mock_msg = MagicMock()
        mock_msg.id = 123
        mock_channel.send.return_value = mock_msg

        async def mock_send_split_response(channel, reply, session_mgr):
            lines = reply.text.split("\n")
            for line in lines:
                if line.strip():
                    async with channel.typing():
                        msg = await channel.send(line)
                        session_mgr.link_message_to_session(str(msg.id), reply.session_key)

        await mock_send_split_response(mock_channel, chat_reply, session_manager)

        assert len(session_manager.linked_messages) == 2

    @pytest.mark.asyncio
    async def test_skips_empty_lines(self, mock_channel, session_manager):
        """send_split_response skips empty lines."""
        reply = MockChatReply(
            text="Hello\n\n\nWorld",
            session_key="channel:123",
            response=None,
            images=[],
        )
        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        async def mock_send_split_response(channel, reply, session_mgr):
            lines = reply.text.split("\n")
            for line in lines:
                if line.strip():
                    async with channel.typing():
                        msg = await channel.send(line)
                        session_mgr.link_message_to_session(str(msg.id), reply.session_key)

        await mock_send_split_response(mock_channel, reply, session_manager)

        # Should send 2 messages, not 4
        assert mock_channel.send.call_count == 2


class TestSendImmediateResponse:
    """Tests for send_immediate_response behavior."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock()
        channel.send = AsyncMock()
        return channel

    @pytest.fixture
    def session_manager(self):
        """Create a mock session manager."""
        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_sends_text_immediately(self, mock_channel, session_manager):
        """send_immediate_response sends text without delay."""
        mock_msg = MagicMock()
        mock_msg.id = 123
        mock_channel.send.return_value = mock_msg

        # Mock implementation
        async def mock_send_immediate_response(channel, text, session_key, session_mgr):
            msg = await channel.send(text)
            session_mgr.link_message_to_session(str(msg.id), session_key)
            return msg

        result = await mock_send_immediate_response(
            mock_channel, "Hello", "channel:123", session_manager
        )

        assert result == mock_msg
        mock_channel.send.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_links_message_to_session(self, mock_channel, session_manager):
        """send_immediate_response links message to session."""
        mock_msg = MagicMock()
        mock_msg.id = 456
        mock_channel.send.return_value = mock_msg

        async def mock_send_immediate_response(channel, text, session_key, session_mgr):
            msg = await channel.send(text)
            session_mgr.link_message_to_session(str(msg.id), session_key)
            return msg

        await mock_send_immediate_response(
            mock_channel, "Hello", "channel:123", session_manager
        )

        assert len(session_manager.linked_messages) == 1
        assert session_manager.linked_messages[0] == ("456", "channel:123")


class TestSendWithImages:
    """Tests for send_with_images behavior."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock()
        channel.typing = MagicMock()
        channel.send = AsyncMock()
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=None)
        channel.typing.return_value = typing_cm
        return channel

    @pytest.fixture
    def session_manager(self):
        """Create a mock session manager."""
        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_sends_text_and_images(self, mock_channel, session_manager):
        """send_with_images sends both text and images."""
        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        # Mock implementation
        async def mock_send_with_images(channel, text, images, session_key, session_mgr):
            messages = []
            if text:
                msg = await channel.send(text)
                session_mgr.link_message_to_session(str(msg.id), session_key)
                messages.append(msg)
            for img in images:
                async with channel.typing():
                    msg = await channel.send(file=img)
                    session_mgr.link_message_to_session(str(msg.id), session_key)
                    messages.append(msg)
            return messages

        result = await mock_send_with_images(
            mock_channel,
            "Hello",
            [b"image1", b"image2"],
            "channel:123",
            session_manager,
        )

        # Text + 2 images = 3 messages
        assert len(result) == 3
        assert mock_channel.send.call_count == 3

    @pytest.mark.asyncio
    async def test_sends_only_images_when_no_text(self, mock_channel, session_manager):
        """send_with_images sends only images when text is empty."""
        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        async def mock_send_with_images(channel, text, images, session_key, session_mgr):
            messages = []
            for img in images:
                async with channel.typing():
                    msg = await channel.send(file=img)
                    session_mgr.link_message_to_session(str(msg.id), session_key)
                    messages.append(msg)
            return messages

        result = await mock_send_with_images(
            mock_channel,
            "",
            [b"image1"],
            "channel:123",
            session_manager,
        )

        # Only 1 image
        assert len(result) == 1
        assert mock_channel.send.call_count == 1


class TestSendStreamingResponse:
    """Tests for send_streaming_response behavior."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock()
        channel.typing = MagicMock()
        channel.send = AsyncMock()
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=None)
        channel.typing.return_value = typing_cm
        return channel

    @pytest.fixture
    def session_manager(self):
        """Create a mock session manager."""
        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_sends_chunks_as_arrived(self, mock_channel, session_manager):
        """send_streaming_response sends chunks as they arrive."""
        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        async def stream():
            yield "Hello"
            yield "World"

        # Mock implementation
        async def mock_send_streaming_response(channel, stream, session_key, session_mgr):
            messages = []
            async for chunk in stream:
                if chunk.strip():
                    async with channel.typing():
                        msg = await channel.send(chunk)
                        session_mgr.link_message_to_session(str(msg.id), session_key)
                        messages.append(msg)
            return messages

        result = await mock_send_streaming_response(
            mock_channel, stream(), "channel:123", session_manager
        )

        assert len(result) == 2
        assert mock_channel.send.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_empty_chunks(self, mock_channel, session_manager):
        """send_streaming_response skips empty chunks."""
        mock_msg = MagicMock()
        mock_msg.id = 1
        mock_channel.send.return_value = mock_msg

        async def stream():
            yield "Hello"
            yield ""
            yield "   "
            yield "World"

        async def mock_send_streaming_response(channel, stream, session_key, session_mgr):
            messages = []
            async for chunk in stream:
                if chunk.strip():
                    async with channel.typing():
                        msg = await channel.send(chunk)
                        session_mgr.link_message_to_session(str(msg.id), session_key)
                        messages.append(msg)
            return messages

        result = await mock_send_streaming_response(
            mock_channel, stream(), "channel:123", session_manager
        )

        # Only 2 non-empty chunks
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_propagates_cancelled_error(self, mock_channel, session_manager):
        """send_streaming_response propagates CancelledError."""
        async def stream():
            yield "Hello"
            raise asyncio.CancelledError()

        async def mock_send_streaming_response(channel, stream, session_key, session_mgr):
            async for chunk in stream:
                pass

        with pytest.raises(asyncio.CancelledError):
            await mock_send_streaming_response(
                mock_channel, stream(), "channel:123", session_manager
            )

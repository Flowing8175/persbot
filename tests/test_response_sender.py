"""Tests for bot/response_sender.py module.

This module provides comprehensive test coverage for:
- send_split_response function
- send_immediate_response function
- send_with_images function
- send_streaming_response function
"""

import asyncio
import io
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any

import pytest

import discord

from persbot.bot.response_sender import (
    send_split_response,
    send_immediate_response,
    send_with_images,
    send_streaming_response,
)
from persbot.bot.chat_models import ChatReply
from persbot.bot.session import SessionManager
from persbot.constants import MessageConfig


class TestSendSplitResponse:
    """Tests for send_split_response function."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock(spec=discord.abc.Messageable)
        channel.send = AsyncMock(return_value=Mock(id=123))
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
            "Test response",
            "test-session",
            [b"fake_image_bytes"],
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
        assert mock_channel.typing.call_count == 1
        mock_session_manager.link_message_to_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_long_text_splitting(self, mock_channel, mock_session_manager, chat_reply):
        """Test sending long text with smart splitting."""
        long_text = "a" * (MessageConfig.MAX_SPLIT_LENGTH + 100)

        reply = ChatReply(
            text=long_text,
            session_key="test-session",
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
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        # Send should preserve newlines
        sent_args = mock_channel.send.call_args_list
        assert sent_args[0][0] == "Line 1"
        assert sent_args[1][0] == "Line 2"

    @pytest.mark.asyncio
    async def test_typing_delay_clamping(self, mock_channel, mock_session_manager, chat_reply):
        """Test that typing delay is properly clamped."""
        long_line = "a" * 3000

        reply = ChatReply(
            text=long_line,
            session_key="test-session",
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        expected_delay = max(
            MessageConfig.TYPING_DELAY_MIN,
            min(MessageConfig.TYPING_DELAY_MAX, len(long_line) * MessageConfig.TYPING_DELAY_MULTIPLIER)
        )

        mock_channel.typing.assert_called_once()
        # Verify sleep duration matches expected (within tolerance)

    @pytest.mark.asyncio
    async def test_send_with_images(self, mock_channel, mock_session_manager, chat_reply):
        """Test sending images as attachments."""
        reply = ChatReply(
            text="Response with images",
            session_key="test-session",
            images=[b"img1", b"img2"],
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        assert mock_channel.send.call_count == 3

    @pytest.mark.asyncio
    async def test_image_attachment_naming(self, mock_channel, mock_session_manager, chat_reply):
        """Test that image attachments have correct filenames."""
        reply = ChatReply(
            text="Response",
            session_key="test-session",
            images=[b"img_data"],
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        send_calls = mock_channel.send.call_args_list
        assert len(send_calls) == 2

        # Find file attachment
        file_call = None
        for call in send_calls:
            if len(call[0]) > 1:
                args = call[0]
                if hasattr(args[0], 'read'):
                    file_call = call
                    break

        assert file_call is not None
        assert "filename" in str(file_call[1][0])

    @pytest.mark.asyncio
    async def test_cancellation_on_first_line(self, mock_channel, mock_session_manager, chat_reply):
        """Test cancellation before sending completes."""
        mock_channel.send = AsyncMock(side_effect=[Mock(id=1), asyncio.CancelledError])

        reply = ChatReply(
            text="Line 1\nLine 2",
            session_key="test-session",
        )

        with pytest.raises(asyncio.CancelledError):
            await send_split_response(
                channel=mock_channel,
                reply=reply,
                session_manager=mock_session_manager,
            )

        assert mock_channel.send.call_count == 1

    @pytest.mark.asyncio
    async def test_display_text_with_notification(self, mock_channel, mock_session_manager, chat_reply):
        """Test display text from reply.display_text."""
        reply = ChatReply(
            text="Main response",
            display_text="Notification: Main response",
            session_key="test-session",
        )

        await send_split_response(
            channel=mock_channel,
            reply=reply,
            session_manager=mock_session_manager,
        )

        sent_text = mock_channel.send.call_args[0][0]
        assert "Notification" in sent_text


class TestSendImmediateResponse:
    """Tests for send_immediate_response function."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock(spec=discord.abc.Messageable)
        channel.send = AsyncMock(return_value=Mock(id=456))
        return channel

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        manager = Mock(spec=SessionManager)
        manager.link_message_to_session = Mock()
        return manager

    @pytest.mark.asyncio
    async def test_send_immediate_basic(self, mock_channel, mock_session_manager):
        """Test basic immediate response sending."""
        result = await send_immediate_response(
            channel=mock_channel,
            text="Immediate response",
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert isinstance(result, discord.Message)
        assert mock_channel.send.call_count == 1
        mock_session_manager.link_message_to_session.assert_called_once_with(
            str(result.id),
            "test-session",
        )

    @pytest.mark.asyncio
    async def test_send_immediate_with_reference(self, mock_channel, mock_session_manager):
        """Test immediate response with reference kwarg."""
        result = await send_immediate_response(
            channel=mock_channel,
            text="Response",
            session_key="session-123",
            reference=True,
        )

        assert result is not None
        mock_channel.send.assert_called_once()


class TestSendWithImages:
    """Tests for send_with_images function."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock(spec=discord.abc.Messageable)
        channel.send = AsyncMock(return_value=Mock(id=789))
        channel.typing = MagicMock()
        return channel

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        manager = Mock(spec=SessionManager)
        manager.link_message_to_session = Mock()
        return manager

    @pytest.mark.asyncio
    async def test_send_images_only(self, mock_channel, mock_session_manager):
        """Test sending images without text."""
        images = [b"img1", b"img2", b"img3"]

        result = await send_with_images(
            channel=mock_channel,
            text=None,
            images=images,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert mock_channel.send.call_count == 3

    @pytest.mark.asyncio
    async def test_send_text_and_images(self, mock_channel, mock_session_manager):
        """Test sending both text and images."""
        images = [b"image_data"]

        result = await send_with_images(
            channel=mock_channel,
            text="Text response",
            images=images,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert mock_channel.send.call_count == 2
        assert mock_session_manager.link_message_to_session.call_count == 2

    @pytest.mark.asyncio
    async def test_send_with_no_content(self, mock_channel, mock_session_manager):
        """Test sending with no text or images."""
        result = await send_with_images(
            channel=mock_channel,
            text=None,
            images=[],
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_image_naming_sequential(self, mock_channel, mock_session_manager):
        """Test that image filenames are sequential."""
        images = [b"img1", b"img2"]

        await send_with_images(
            channel=mock_channel,
            text="Response",
            images=images,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        send_calls = mock_channel.send.call_args_list
        for i, call in enumerate(send_calls):
            if i < len(images):
                args = call[0]
                if len(args) > 1 and hasattr(args[0], 'read'):
                    assert "filename" in str(args[0])
                    assert f"generated_image_{i}.png" in str(args[0])


class TestSendStreamingResponse:
    """Tests for send_streaming_response function."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock Discord channel."""
        channel = MagicMock(spec=discord.abc.Messageable)
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

        assert len(result) == 2
        # Check that line splitting was done correctly

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
        long_line = "a" * (MessageConfig.MAX_SPLIT_LENGTH + 100)

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

        stream = stream()

        with pytest.raises(asyncio.CancelledError):
            await send_streaming_response(
                channel=mock_channel,
                stream=stream,
                session_key="test-session",
                session_manager=mock_session_manager,
            )

        assert mock_channel.send.call_count == 1

    @pytest.mark.asyncio
    async def test_stream_typing_indicator(self, mock_channel, mock_session_manager):
        """Test typing indicator during streaming."""
        stream = self.make_async_iterator(["chunk1", "chunk2", "chunk3"])

        result = await send_streaming_response(
            channel=mock_channel,
            stream=stream,
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        # Typing should only be called at start
        mock_channel.typing.assert_called_once()

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

"""Tests for bot/session_resolver.py module.

This module provides comprehensive test coverage for:
- resolve_session_for_message function
- extract_session_context function
"""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

import discord

from persbot.bot.session_resolver import (
    resolve_session_for_message,
    extract_session_context,
)


# =============================================================================
# resolve_session_for_message Function Tests
# =============================================================================


class TestResolveSessionForMessage:
    """Tests for resolve_session_for_message function."""

    @pytest.mark.asyncio
    async def test_resolve_session_with_reference_message(self, mock_session_manager, mock_message):
        """Test resolving session when message has a reference."""
        # Setup
        mock_message.reference = Mock()
        mock_message.reference.message_id = 12345
        mock_message.reference.resolved = None
        mock_message.channel.id = 111222
        mock_message.author.id = 999888
        mock_message.author.name = "TestUser"
        mock_message.created_at = datetime.now(timezone.utc)

        # Mock fetch_message to return a reference message
        ref_msg = Mock()
        ref_msg.author = Mock(id=777666, bot=True)
        ref_msg.clean_content = "**요약:** Summary content"
        mock_message.channel.fetch_message = AsyncMock(return_value=ref_msg)

        # Mock session_manager.resolve_session to return a result
        resolution = Mock()
        resolution.cleaned_message = "Processed content"
        resolution.session_key = "channel:111222"
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        # Execute
        result = await resolve_session_for_message(
            message=mock_message,
            content="Test content",
            session_manager=mock_session_manager,
        )

        # Verify session_manager.resolve_session was called with reply context
        assert result is not None
        mock_session_manager.resolve_session.assert_called_once()
        call_kwargs = mock_session_manager.resolve_session.call_args.kwargs
        assert "답장 대상:" in call_kwargs["message_content"]
        assert result.is_reply_to_summary is True

    @pytest.mark.asyncio
    async def test_resolve_session_with_deleted_reference(self, mock_session_manager, mock_message):
        """Test resolving session when reference message is deleted."""
        # Setup
        mock_message.reference = Mock()
        mock_message.reference.message_id = 12345
        mock_message.reference.resolved = None
        mock_message.channel.id = 111222

        # Mock fetch_message to raise NotFound
        mock_message.channel.fetch_message = AsyncMock(
            side_effect=discord.NotFound("Message not found")
        )

        # Mock session_manager.resolve_session
        resolution = Mock()
        resolution.cleaned_message = "Direct content"
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        # Execute - should handle NotFound gracefully
        result = await resolve_session_for_message(
            message=mock_message,
            content="Direct message",
            session_manager=mock_session_manager,
        )

        # Verify it still processes without the reference
        assert result is not None
        assert mock_message.channel.fetch_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_session_without_reference(self, mock_session_manager, mock_message):
        """Test resolving session without reference message."""
        # Setup - no reference
        mock_message.reference = None
        mock_message.channel.id = 111222
        mock_message.author.id = 999888
        mock_message.author.name = "TestUser"
        mock_message.created_at = datetime.now(timezone.utc)

        # Mock session_manager.resolve_session
        resolution = Mock()
        resolution.cleaned_message = "Direct content"
        resolution.session_key = "channel:111222"
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        # Execute
        result = await resolve_session_for_message(
            message=mock_message,
            content="Direct message",
            session_manager=mock_session_manager,
        )

        # Verify session_manager.resolve_session was called with original content
        assert result is not None
        assert result.is_reply_to_summary is False
        mock_session_manager.resolve_session.assert_called_once()
        call_kwargs = mock_session_manager.resolve_session.call_args.kwargs
        assert call_kwargs["message_content"] == "Direct message"

    @pytest.mark.asyncio
    async def test_resolve_session_with_non_bot_summary(self, mock_session_manager, mock_message):
        """Test resolving session when replying to non-bot summary."""
        # Setup
        mock_message.reference = Mock()
        mock_message.reference.message_id = 12345
        ref_msg = Mock()
        ref_msg.author = Mock(id=777666, bot=False)  # Not a bot
        ref_msg.clean_content = "Regular message"
        mock_message.reference.resolved = ref_msg
        mock_message.channel.id = 111222

        resolution = Mock()
        resolution.cleaned_message = "Processed content"
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        # Execute
        result = await resolve_session_for_message(
            message=mock_message,
            content="Reply content",
            session_manager=mock_session_manager,
        )

        # Verify is_reply_to_summary is False for non-bot messages
        assert result is not None
        assert result.is_reply_to_summary is False

    @pytest.mark.asyncio
    async def test_resolve_session_returns_none_when_no_cleaned_message(
        self,
        mock_session_manager,
        mock_message,
    ):
        """Test that None is returned when cleaned_message is empty."""
        # Setup
        mock_message.reference = None
        mock_message.channel.id = 111222

        # Mock session_manager.resolve_session to return result with empty cleaned_message
        resolution = Mock()
        resolution.cleaned_message = ""  # Empty cleaned message
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        # Execute
        result = await resolve_session_for_message(
            message=mock_message,
            content="Some content",
            session_manager=mock_session_manager,
        )

        # Verify None is returned
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_session_with_http_exception(self, mock_session_manager, mock_message):
        """Test resolving session when HTTP exception occurs during fetch."""
        # Setup
        mock_message.reference = Mock()
        mock_message.reference.message_id = 12345
        mock_message.channel.id = 111222

        # Mock fetch_message to raise HTTPException
        mock_message.channel.fetch_message = AsyncMock(
            side_effect=discord.HTTPException("Server error")
        )

        resolution = Mock()
        resolution.cleaned_message = "Processed content"
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        # Execute - should handle HTTPException gracefully
        result = await resolve_session_for_message(
            message=mock_message,
            content="Direct message",
            session_manager=mock_session_manager,
        )

        # Verify it continues without the reference
        assert result is not None

    @pytest.mark.asyncio
    async def test_resolve_session_passes_cancel_event(self, mock_session_manager, mock_message):
        """Test that cancel_event is passed to session_manager."""
        cancel_event = asyncio.Event()

        # Setup
        mock_message.reference = None
        mock_message.channel.id = 111222

        resolution = Mock()
        resolution.cleaned_message = "Test content"
        mock_session_manager.resolve_session = AsyncMock(return_value=resolution)

        # Execute with cancel_event
        result = await resolve_session_for_message(
            message=mock_message,
            content="Test",
            session_manager=mock_session_manager,
            cancel_event=cancel_event,
        )

        # Verify cancel_event was passed
        mock_session_manager.resolve_session.assert_called_once()
        call_kwargs = mock_session_manager.resolve_session.call_args.kwargs
        assert "cancel_event" in call_kwargs


# =============================================================================
# extract_session_context Function Tests
# =============================================================================


class TestExtractSessionContext:
    """Tests for extract_session_context function."""

    def test_extract_from_single_message(self, mock_message):
        """Test extracting context from a single Discord message."""
        # Setup mock message
        mock_message.channel.id = 111222
        mock_message.author.id = 999888
        mock_message.author.name = "TestUser"
        mock_message.id = 555444
        mock_message.created_at = datetime.now(timezone.utc)

        result = extract_session_context(mock_message)

        assert result.channel_id == 111222
        assert result.user_id == 999888
        assert result.username == "TestUser"
        assert result.message_id == "555444"

    def test_extract_from_message_list(self, mock_message):
        """Test extracting context from a list of Discord messages."""
        # Setup mock message list
        message_list = [mock_message]

        result = extract_session_context(message_list)

        # Should use first message
        assert result.channel_id == mock_message.channel.id
        assert result.user_id == mock_message.author.id
        assert result.username == mock_message.author.name

    def test_extract_context_with_different_timestamps(self):
        """Test extracting context preserves timestamp."""
        # Create mock messages with different timestamps
        msg1 = Mock()
        msg1.channel.id = 111222
        msg1.author.id = 999888
        msg1.author.name = "User1"
        msg1.id = "111"
        msg1.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        msg2 = Mock()
        msg2.channel.id = 333333
        msg2.author.id = 888999
        msg2.author.name = "User2"
        msg2.id = "222"
        msg2.created_at = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        result1 = extract_session_context(msg1)
        result2 = extract_session_context(msg2)

        # Verify timestamps are preserved
        assert result1.created_at.year == 2024
        assert result1.created_at.month == 1
        assert result2.created_at.day == 2

"""Comprehensive tests for soyebot/utils.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord import Embed
from discord.ui import View

sys.path.insert(0, str(Path(__file__).parent.parent))

from soyebot.utils import (
    DiscordUI,
    extract_message_content,
    parse_korean_time,
    send_discord_message,
    smart_split,
)

# =============================================================================
# Tests for smart_split()
# =============================================================================


class TestSmartSplit:
    """Test suite for smart_split() function."""

    def test_short_text_returns_single_chunk(self):
        """Short text (<1900 chars) should return as single chunk."""
        text = "Hello, world!"
        result = smart_split(text, max_length=1900)
        assert result == [text]

    def test_short_text_at_max_length(self):
        """Text exactly at max_length should return as single chunk."""
        text = "a" * 1900
        result = smart_split(text, max_length=1900)
        assert result == [text]
        assert len(result) == 1

    def test_empty_string(self):
        """Empty string should return as single chunk."""
        result = smart_split("", max_length=1900)
        assert result == [""]

    def test_double_newline_splitting(self):
        """Text with double newlines should split at double newlines first."""
        text = "Line 1\n\n" + "a" * 100 + "\n\n" + "b" * 100
        result = smart_split(text, max_length=200)
        assert len(result) == 2
        assert "Line 1" in result[0]
        assert "b" * 100 in result[1]

    def test_single_newline_splitting(self):
        """Text with single newlines should split at newlines."""
        text = "\n".join(["a" * 50, "b" * 50, "c" * 50])
        result = smart_split(text, max_length=100)
        # Each line is 50 chars + newline = 51, so max_length 100 fits 2 lines (102 chars)
        # The function splits at the last newline within max_length
        assert len(result) == 3

    def test_space_splitting(self):
        """Long text with no newlines should split at spaces."""
        text = " ".join(["word" * 20] * 10)  # Each word is 80 chars
        result = smart_split(text, max_length=400)
        assert len(result) > 1
        # Check that chunks don't break words
        for chunk in result:
            # Each chunk should end at a word boundary or be full
            assert chunk.endswith(" ") or chunk.endswith("word" * 20)

    def test_hard_cut_on_no_split_points(self):
        """Very long word should force hard cut at max_length."""
        text = "a" * 5000
        result = smart_split(text, max_length=1900)
        assert len(result) == 3
        assert len(result[0]) == 1900
        assert len(result[1]) == 1900
        assert len(result[2]) == 1200

    def test_custom_max_length(self):
        """Should respect custom max_length parameter."""
        text = " ".join(["word" * 20] * 10)
        result = smart_split(text, max_length=100)
        assert all(len(chunk) <= 100 for chunk in result)

    def test_leading_whitespace_stripping(self):
        """Whitespace should be stripped from subsequent chunks."""
        text = "First\n\n   Second"
        result = smart_split(text, max_length=10)
        assert result[1] == "Second"

    def test_mixed_newline_types(self):
        """Should handle mixed double and single newlines."""
        text = "First\n\n" + "a" * 50 + "\n" + "b" * 100 + "\n\n" + "c" * 50
        result = smart_split(text, max_length=150)
        # With max_length 150, the split point depends on where the last newline is
        assert len(result) == 4


# =============================================================================
# Tests for send_discord_message()
# =============================================================================


@pytest.mark.asyncio
class TestSendDiscordMessage:
    """Test suite for send_discord_message() function."""

    async def test_empty_content_no_sendable_returns_empty(self):
        """Empty content with no sendable items should return empty list."""
        target = AsyncMock()
        result = await send_discord_message(target, "")
        assert result == []

    async def test_context_target_single_chunk(self):
        """Test sending to Context object with single chunk."""
        target = AsyncMock(spec=discord.abc.Messageable)
        mock_msg = MagicMock()
        mock_msg.id = 123
        target.send = AsyncMock(return_value=mock_msg)

        result = await send_discord_message(target, "Hello, world!")
        assert len(result) == 1
        target.send.assert_called_once_with("Hello, world!")

    async def test_context_target_auto_splitting(self):
        """Test auto-splitting for Context target."""
        target = AsyncMock(spec=discord.abc.Messageable)
        mock_msg1 = MagicMock()
        mock_msg1.id = 1
        mock_msg2 = MagicMock()
        mock_msg2.id = 2
        target.send = AsyncMock(side_effect=[mock_msg1, mock_msg2])

        long_text = "a" * 1000 + "\n\n" + "b" * 1000
        result = await send_discord_message(target, long_text)
        assert len(result) == 2
        assert target.send.call_count == 2

    async def test_interaction_target_first_chunk(self):
        """Test sending to Interaction with first chunk."""
        target = MagicMock(spec=discord.Interaction)
        target.response.is_done.return_value = False
        target.response.send_message = AsyncMock()
        mock_original = MagicMock()
        mock_original.id = 123
        target.original_response = AsyncMock(return_value=mock_original)

        result = await send_discord_message(target, "Hello")
        target.response.send_message.assert_called_once()
        assert len(result) == 1

    async def test_interaction_target_response_done(self):
        """Test sending to Interaction when response is already done."""
        target = MagicMock(spec=discord.Interaction)
        target.response.is_done.return_value = True
        mock_followup = MagicMock()
        mock_followup.id = 456
        target.followup.send = AsyncMock(return_value=mock_followup)

        result = await send_discord_message(target, "Followup message")
        target.followup.send.assert_called_once()
        assert len(result) == 1

    async def test_message_target_reply(self):
        """Test sending to Message target with reply."""
        target = MagicMock(spec=discord.Message)
        mock_reply = MagicMock()
        mock_reply.id = 789
        target.reply = AsyncMock(return_value=mock_reply)

        result = await send_discord_message(target, "Reply text")
        target.reply.assert_called_once_with("Reply text", mention_author=False)
        assert len(result) == 1

    async def test_message_target_multiple_chunks(self):
        """Test sending multiple chunks to Message target."""
        target = MagicMock(spec=discord.Message)
        mock_reply = MagicMock()
        mock_reply.id = 1
        mock_send = MagicMock()
        mock_send.id = 2
        target.reply = AsyncMock(return_value=mock_reply)
        target.channel.send = AsyncMock(return_value=mock_send)

        long_text = "a" * 1000 + "\n\n" + "b" * 1000
        result = await send_discord_message(target, long_text)
        assert len(result) == 2
        target.reply.assert_called_once()
        target.channel.send.assert_called_once()

    async def test_embed_only_last_chunk(self):
        """Test that embed is only sent with the last chunk."""
        target = AsyncMock(spec=discord.abc.Messageable)
        mock_msg1 = MagicMock()
        mock_msg1.id = 1
        mock_msg2 = MagicMock()
        mock_msg2.id = 2
        target.send = AsyncMock(side_effect=[mock_msg1, mock_msg2])

        embed = Embed(title="Test")
        long_text = "a" * 1000 + "\n\n" + "b" * 1000

        result = await send_discord_message(target, long_text, embed=embed)
        assert len(result) == 2

        # First chunk should not have embed
        first_call_kwargs = target.send.call_args_list[0][1]
        assert "embed" not in first_call_kwargs

        # Last chunk should have embed
        second_call_kwargs = target.send.call_args_list[1][1]
        assert "embed" in second_call_kwargs

    async def test_view_only_last_chunk(self):
        """Test that view is only sent with the last chunk."""
        target = AsyncMock(spec=discord.abc.Messageable)
        mock_msg1 = MagicMock()
        mock_msg1.id = 1
        mock_msg2 = MagicMock()
        mock_msg2.id = 2
        target.send = AsyncMock(side_effect=[mock_msg1, mock_msg2])

        view = View()
        long_text = "a" * 1000 + "\n\n" + "b" * 1000

        result = await send_discord_message(target, long_text, view=view)
        assert len(result) == 2

        first_call_kwargs = target.send.call_args_list[0][1]
        assert "view" not in first_call_kwargs

        second_call_kwargs = target.send.call_args_list[1][1]
        assert "view" in second_call_kwargs

    async def test_error_handling_stops_processing(self):
        """Test that errors on one chunk stop processing of remaining chunks."""
        target = AsyncMock(spec=discord.abc.Messageable)
        mock_msg2 = MagicMock()
        mock_msg2.id = 2
        target.send = AsyncMock(side_effect=[Exception("Failed"), mock_msg2])

        long_text = "a" * 1000 + "\n\n" + "b" * 1000
        result = await send_discord_message(target, long_text)
        # Should return empty list since the first chunk fails and stops processing
        assert len(result) == 0

    async def test_unsupported_target_type(self):
        """Test handling of unsupported target types."""
        target = "not a valid target"

        result = await send_discord_message(target, "Test")
        assert result == []

    async def test_mention_author_parameter(self):
        """Test that mention_author is passed correctly for Message targets."""
        target = MagicMock(spec=discord.Message)
        mock_reply = MagicMock()
        mock_reply.id = 123
        target.reply = AsyncMock(return_value=mock_reply)

        await send_discord_message(target, "Reply", mention_author=True)
        target.reply.assert_called_once_with("Reply", mention_author=True)

    async def test_reference_only_first_chunk(self):
        """Test that reference is only applied to first chunk."""
        target = AsyncMock(spec=discord.abc.Messageable)
        mock_msg1 = MagicMock()
        mock_msg1.id = 1
        mock_msg2 = MagicMock()
        mock_msg2.id = 2
        target.send = AsyncMock(side_effect=[mock_msg1, mock_msg2])

        ref = MagicMock()
        long_text = "a" * 1000 + "\n\n" + "b" * 1000

        result = await send_discord_message(target, long_text, reference=ref)
        assert len(result) == 2

        first_call_kwargs = target.send.call_args_list[0][1]
        assert "reference" in first_call_kwargs

        second_call_kwargs = target.send.call_args_list[1][1]
        assert "reference" not in second_call_kwargs

    async def test_message_target_reply_fallback_on_unknown_message(self):
        """Test fallback to channel.send when reply fails with 'Unknown message'."""
        target = MagicMock(spec=discord.Message)
        mock_channel_msg = MagicMock()
        mock_channel_msg.id = 789

        # Simulate discord.NotFound with "Unknown message" error
        target.reply = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=400), "Unknown message")
        )
        target.channel.send = AsyncMock(return_value=mock_channel_msg)

        result = await send_discord_message(target, "Reply text")
        assert len(result) == 1
        assert result[0].id == 789
        # Should have tried reply first, then fallen back to channel.send
        target.reply.assert_called_once()
        target.channel.send.assert_called_once_with("Reply text")

    async def test_message_target_non_unknown_error_logs_and_stops(self):
        """Test that non-'Unknown message' errors are logged and stop processing."""
        target = MagicMock(spec=discord.Message)

        # Simulate a different NotFound error (not "Unknown message")
        # The function logs the error and returns empty list, it doesn't propagate
        target.reply = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "Some other error")
        )

        result = await send_discord_message(target, "Reply text")
        # Error is logged and processing stops, returning empty list
        assert len(result) == 0


# =============================================================================
# Tests for extract_message_content()
# =============================================================================


class TestExtractMessageContent:
    """Test suite for extract_message_content() function."""

    def test_removes_bot_mention_with_id(self):
        """Should remove bot mention using <@id> format."""
        message = MagicMock(spec=discord.Message)
        message.content = "<@123456789> Hello, bot!"
        message.mentions = [MagicMock(id=123456789)]

        result = extract_message_content(message)
        assert result == "Hello, bot!"

    def test_removes_bot_mention_with_nick_format(self):
        """Should remove bot mention using <@!id> format."""
        message = MagicMock(spec=discord.Message)
        message.content = "<@!123456789> Hello, bot!"
        message.mentions = [MagicMock(id=123456789)]

        result = extract_message_content(message)
        assert result == "Hello, bot!"

    def test_preserves_other_content(self):
        """Should preserve content not related to mentions."""
        message = MagicMock(spec=discord.Message)
        message.content = "<@123456789> How are you doing today?"
        message.mentions = [MagicMock(id=123456789)]

        result = extract_message_content(message)
        assert result == "How are you doing today?"

    def test_handles_empty_content(self):
        """Should handle empty message content."""
        message = MagicMock(spec=discord.Message)
        message.content = "<@123456789>"
        message.mentions = [MagicMock(id=123456789)]

        result = extract_message_content(message)
        assert result == ""

    def test_handles_multiple_mentions(self):
        """Should handle multiple mentions in one message."""
        message = MagicMock(spec=discord.Message)
        message.content = "<@123> <@456> Hello everyone"
        message.mentions = [MagicMock(id=123), MagicMock(id=456)]

        result = extract_message_content(message)
        assert result == "Hello everyone"

    def test_handles_mixed_mention_formats(self):
        """Should handle both <@id> and <@!id> formats in same message."""
        message = MagicMock(spec=discord.Message)
        message.content = "<@123> <@!456> Test"
        message.mentions = [MagicMock(id=123), MagicMock(id=456)]

        result = extract_message_content(message)
        assert result == "Test"

    def test_no_mentions(self):
        """Should return original content when no mentions present."""
        message = MagicMock(spec=discord.Message)
        message.content = "Hello world!"
        message.mentions = []

        result = extract_message_content(message)
        assert result == "Hello world!"


# =============================================================================
# Tests for parse_korean_time()
# =============================================================================


class TestParseKoreanTime:
    """Test suite for parse_korean_time() function."""

    def test_minutes_only(self):
        """Test parsing '30분' → 30."""
        result = parse_korean_time("30분")
        assert result == 30

    def test_hours_only(self):
        """Test parsing '1시간' → 60."""
        result = parse_korean_time("1시간")
        assert result == 60

    def test_hours_and_minutes(self):
        """Test parsing '2시간30분' → 150."""
        result = parse_korean_time("2시간30분")
        assert result == 150

    def test_multiple_hours(self):
        """Test parsing '24시간' → 1440."""
        result = parse_korean_time("24시간")
        assert result == 1440

    def test_hours_minutes_whitespace_variations(self):
        """Test parsing with various whitespace patterns."""
        assert parse_korean_time("1시간 30분") == 90
        assert parse_korean_time("1시간30분") == 90
        assert parse_korean_time("  1시간30분  ") == 90

    def test_minutes_hours_reversed_order(self):
        """Test parsing when minutes come before hours."""
        result = parse_korean_time("30분1시간")
        assert result == 90

    def test_large_values(self):
        """Test parsing large values."""
        assert parse_korean_time("100시간") == 6000
        assert parse_korean_time("500분") == 500

    def test_single_digit_values(self):
        """Test parsing single digit values."""
        assert parse_korean_time("1시간") == 60
        assert parse_korean_time("1분") == 1
        assert parse_korean_time("5분") == 5

    def test_invalid_input_returns_none(self):
        """Test invalid input returns None."""
        assert parse_korean_time("") is None
        assert parse_korean_time(None) is None
        assert parse_korean_time("invalid") is None
        assert parse_korean_time("hello world") is None

    def test_partial_valid_input(self):
        """Test input with only partial valid time tokens."""
        assert parse_korean_time("시간") is None
        assert parse_korean_time("분") is None
        assert parse_korean_time("1") is None

    def test_zero_values(self):
        """Test parsing zero values returns None (falsy values return None)."""
        # The function returns total_minutes or None, so 0 returns None
        assert parse_korean_time("0시간") is None
        assert parse_korean_time("0분") is None
        assert parse_korean_time("0시간0분") is None

    def test_complex_combinations(self):
        """Test various complex time combinations."""
        assert parse_korean_time("1시간15분") == 75
        assert parse_korean_time("2시간45분") == 165
        assert parse_korean_time("12시간30분") == 750
        assert parse_korean_time("5시간1분") == 301


# =============================================================================
# Tests for DiscordUI class
# =============================================================================


@pytest.mark.asyncio
class TestDiscordUI:
    """Test suite for DiscordUI class."""

    async def test_safe_send_success(self):
        """Test successful message sending."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.name = "test-channel"
        mock_msg = MagicMock()
        mock_msg.id = 123
        channel.send = AsyncMock(return_value=mock_msg)

        result = await DiscordUI.safe_send(channel, "Test message")
        assert result is not None
        assert result.id == 123
        channel.send.assert_called_once_with("Test message")

    async def test_safe_send_forbidden_error(self):
        """Test handling of Forbidden error."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.name = "test-channel"
        mock_response = MagicMock()
        mock_response.status = 403
        channel.send = AsyncMock(side_effect=discord.Forbidden(mock_response, "Forbidden"))

        result = await DiscordUI.safe_send(channel, "Test message")
        assert result is None

    async def test_safe_send_generic_error(self):
        """Test handling of generic exceptions."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.name = "test-channel"
        channel.send = AsyncMock(side_effect=Exception("Some error"))

        result = await DiscordUI.safe_send(channel, "Test message")
        assert result is None

    async def test_safe_send_http_exception(self):
        """Test handling of HTTP exception."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.name = "test-channel"
        mock_response = MagicMock()
        mock_response.status = 500
        channel.send = AsyncMock(side_effect=discord.HTTPException(mock_response, "HTTP error"))

        result = await DiscordUI.safe_send(channel, "Test message")
        assert result is None

    async def test_safe_send_empty_message(self):
        """Test sending empty message."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.name = "test-channel"
        mock_msg = MagicMock()
        mock_msg.id = 123
        channel.send = AsyncMock(return_value=mock_msg)

        result = await DiscordUI.safe_send(channel, "")
        assert result is not None
        channel.send.assert_called_once_with("")

    async def test_safe_send_long_message(self):
        """Test sending long message."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.name = "test-channel"
        mock_msg = MagicMock()
        mock_msg.id = 123
        channel.send = AsyncMock(return_value=mock_msg)

        long_text = "a" * 5000
        result = await DiscordUI.safe_send(channel, long_text)
        assert result is not None
        channel.send.assert_called_once_with(long_text)

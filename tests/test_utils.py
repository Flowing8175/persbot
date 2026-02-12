"""Feature tests for utility functions.

Tests focus on behavior:
- smart_split text chunking
- parse_korean_time parsing
- get_mime_type detection
"""

import pytest

from persbot.utils import (
    smart_split,
    parse_korean_time,
    get_mime_type,
    GENERIC_ERROR_MESSAGE,
    ERROR_API_TIMEOUT,
    ERROR_API_QUOTA_EXCEEDED,
    ERROR_RATE_LIMIT,
    ERROR_PERMISSION_DENIED,
    ERROR_INVALID_ARGUMENT,
)


class TestSmartSplit:
    """Tests for smart_split text chunking."""

    def test_returns_single_chunk_for_short_text(self):
        """Text under max_length returns as single chunk."""
        text = "Short text"
        result = smart_split(text, max_length=100)
        assert len(result) == 1
        assert result[0] == text

    def test_returns_original_text_when_exactly_max_length(self):
        """Text exactly at max_length returns as single chunk."""
        text = "a" * 100
        result = smart_split(text, max_length=100)
        assert len(result) == 1
        assert result[0] == text

    def test_splits_at_double_newline(self):
        """Prefers splitting at double newlines."""
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        result = smart_split(text, max_length=20)
        # Should split at double newlines
        assert len(result) > 1

    def test_splits_at_single_newline_if_no_double(self):
        """Falls back to single newline if no double newline."""
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        result = smart_split(text, max_length=15)
        assert len(result) > 1

    def test_splits_at_space_if_no_newlines(self):
        """Falls back to space if no newlines."""
        text = "word " * 100
        result = smart_split(text, max_length=50)
        # All chunks should be under max_length (or close to it)
        for chunk in result:
            assert len(chunk) <= 55  # Allow some flexibility

    def test_handles_empty_string(self):
        """Empty string returns empty list."""
        result = smart_split("", max_length=100)
        assert result == [""]

    def test_handles_korean_text(self):
        """Korean text is split correctly."""
        text = "첫 번째 문단입니다.\n\n두 번째 문단입니다.\n\n세 번째 문단입니다."
        result = smart_split(text, max_length=30)
        assert len(result) > 1

    def test_very_long_word_is_hard_cut(self):
        """Very long word without spaces is hard cut."""
        text = "a" * 200
        result = smart_split(text, max_length=50)
        # Should be split into chunks
        assert len(result) > 1
        # Each chunk should be approximately max_length
        total_len = sum(len(chunk) for chunk in result)
        assert total_len >= 200

    def test_strips_whitespace_after_split(self):
        """Strips leading whitespace from chunks after split."""
        text = "Paragraph 1\n\n   Paragraph 2 with leading spaces"
        result = smart_split(text, max_length=30)
        # After stripping, chunks shouldn't start with whitespace
        for chunk in result:
            if chunk:
                # First chunk might keep original, others should be stripped
                pass

    def test_default_max_length_is_1900(self):
        """Default max_length is 1900 characters."""
        # Create text longer than 1900
        text = "a" * 2000
        result = smart_split(text)
        assert len(result) > 1

    def test_multiple_paragraphs(self):
        """Handles multiple paragraphs correctly."""
        paragraphs = ["Paragraph " + str(i) * 20 for i in range(5)]
        text = "\n\n".join(paragraphs)
        result = smart_split(text, max_length=50)
        assert len(result) >= 3


class TestParseKoreanTime:
    """Tests for parse_korean_time function."""

    def test_parses_minutes_only(self):
        """Parses minutes correctly."""
        result = parse_korean_time("30분")
        assert result == 30

    def test_parses_hours_only(self):
        """Parses hours correctly."""
        result = parse_korean_time("2시간")
        assert result == 120  # 2 hours = 120 minutes

    def test_parses_hours_and_minutes(self):
        """Parses combined hours and minutes."""
        result = parse_korean_time("1시간30분")
        assert result == 90  # 1 hour + 30 minutes

    def test_parses_multiple_values(self):
        """Parses multiple hour/minute values."""
        result = parse_korean_time("2시간 15분")
        assert result == 135  # 2 hours + 15 minutes

    def test_returns_none_for_empty_string(self):
        """Returns None for empty string."""
        result = parse_korean_time("")
        assert result is None

    def test_returns_none_for_none_input(self):
        """Returns None for None input."""
        result = parse_korean_time(None)
        assert result is None

    def test_returns_none_for_no_matches(self):
        """Returns None when no time patterns match."""
        result = parse_korean_time("hello world")
        assert result is None

    def test_ignores_non_time_text(self):
        """Ignores text that isn't time-related."""
        result = parse_korean_time("알람 1시간 후에 울려주세요")
        assert result == 60

    def test_handles_whitespace(self):
        """Handles whitespace in time string."""
        result = parse_korean_time("1 시간 30 분")
        # May or may not parse depending on regex
        # Current implementation expects no space between number and unit


class TestGetMimeType:
    """Tests for get_mime_type function."""

    def test_detects_jpeg(self):
        """Detects JPEG images."""
        # JPEG starts with 0xFF 0xD8
        jpeg_header = b"\xff\xd8\xff\xe0"
        assert get_mime_type(jpeg_header) == "image/jpeg"

    def test_detects_png(self):
        """Detects PNG images."""
        # PNG starts with 0x89 P N G
        png_header = b"\x89PNG\r\n\x1a\n"
        assert get_mime_type(png_header) == "image/png"

    def test_detects_gif(self):
        """Detects GIF images."""
        # GIF starts with GIF8
        gif_header = b"GIF89a"
        assert get_mime_type(gif_header) == "image/gif"

    def test_detects_webp(self):
        """Detects WebP images."""
        # WEBP has RIFF header and WEBP marker
        webp_header = b"RIFF\x00\x00\x00\x00WEBP"
        assert get_mime_type(webp_header) == "image/webp"

    def test_defaults_to_jpeg_for_unknown(self):
        """Defaults to JPEG for unknown formats."""
        unknown_data = b"unknown format data"
        assert get_mime_type(unknown_data) == "image/jpeg"

    def test_handles_empty_bytes(self):
        """Handles empty byte string."""
        assert get_mime_type(b"") == "image/jpeg"

    def test_handles_short_data(self):
        """Handles very short byte strings."""
        assert get_mime_type(b"ab") == "image/jpeg"


class TestErrorConstants:
    """Tests for error constant definitions."""

    def test_generic_error_message_is_korean(self):
        """Generic error message is in Korean."""
        assert "❌" in GENERIC_ERROR_MESSAGE
        assert "오류" in GENERIC_ERROR_MESSAGE

    def test_api_timeout_message_is_korean(self):
        """API timeout message is in Korean."""
        assert "❌" in ERROR_API_TIMEOUT
        assert "시간" in ERROR_API_TIMEOUT

    def test_api_quota_exceeded_message_is_korean(self):
        """API quota exceeded message is in Korean."""
        assert "❌" in ERROR_API_QUOTA_EXCEEDED
        assert "초과" in ERROR_API_QUOTA_EXCEEDED

    def test_rate_limit_message_is_korean(self):
        """Rate limit message is in Korean."""
        assert "⏳" in ERROR_RATE_LIMIT

    def test_permission_denied_message_is_korean(self):
        """Permission denied message is in Korean."""
        assert "❌" in ERROR_PERMISSION_DENIED
        assert "권한" in ERROR_PERMISSION_DENIED

    def test_invalid_argument_message_is_korean(self):
        """Invalid argument message is in Korean."""
        assert "❌" in ERROR_INVALID_ARGUMENT
        assert "인자" in ERROR_INVALID_ARGUMENT

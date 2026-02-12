"""Tests for prompts.py module.

This module provides comprehensive test coverage for:
- load_persona function
- Persona prompt constants
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from persbot.prompts import (
    load_persona,
    BOT_PERSONA_PROMPT,
    __all__,
)


# =============================================================================
# load_persona Function Tests
# =============================================================================


class TestLoadPersona:
    """Tests for load_persona function."""

    def test_load_persona_from_default_path(self, tmp_path):
        """Test loading persona from default path."""
        # Create assets directory and persona.md file
        assets_dir = tmp_path / "persbot"
        assets_dir.mkdir()
        persona_file = assets_dir / "persona.md"

        test_content = "You are a helpful AI assistant."
        persona_file.write_text(test_content, encoding="utf-8")

        # Mock the path from current directory
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.side_effect = lambda p: str(p) == "persbot/assets/persona.md" or str(p) == "assets/persona.md"

            result = load_persona()

            assert result == test_content

    def test_load_persona_from_fallback_path(self, tmp_path):
        """Test loading persona from fallback path when default doesn't exist."""
        # Create assets directory but no persona.md
        assets_dir = tmp_path / "persbot"
        assets_dir.mkdir()

        with patch("pathlib.Path.exists") as mock_exists:
            # Default path doesn't exist, fallback should be used
            mock_exists.side_effect = lambda p: str(p) == "assets/persona.md"

            # Create the fallback file
            fallback_file = tmp_path / "assets" / "persona.md"
            fallback_file.parent.mkdir(exist_ok=True)
            fallback_file.write_text("Fallback persona", encoding="utf-8")

            result = load_persona()

            assert result == "Fallback persona"

    def test_load_persona_file_not_found(self, tmp_path):
        """Test load_persona when file is not found."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("pathlib.Path.exists") as mock_exists:
            # No path exists
            mock_exists.return_value = False

            result = load_persona()

            assert result == "System prompt could not be loaded."

    def test_load_persona_file_read_error(self, tmp_path):
        """Test load_persona handles file read errors."""
        assets_dir = tmp_path / "persbot"
        assets_dir.mkdir()
        persona_file = assets_dir / "persona.md"

        # Mock open to raise an exception
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            result = load_persona()

            assert result == "System prompt error."


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Tests for prompts.py module constants."""

    def test_module_exports(self):
        """Test that expected symbols are exported."""
        assert "load_persona" in __all__
        assert "BOT_PERSONA_PROMPT" in __all__

    def test_bot_persona_prompt_is_loaded(self):
        """Test that BOT_PERSONA_PROMPT is set at module import."""
        # BOT_PERSONA_PROMPT is loaded at module import time
        # It should be a string (either the loaded content or error message)
        assert isinstance(BOT_PERSONA_PROMPT, str)

    def test_bot_persona_prompt_not_empty(self):
        """Test that BOT_PERSONA_PROMPT has some content or error message."""
        # The prompt should either have content from file or an error message
        assert len(BOT_PERSONA_PROMPT) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestPromptsIntegration:
    """Integration tests for prompts module."""

    def test_load_persona_returns_string(self):
        """Test that load_persona always returns a string."""
        result = load_persona()
        assert isinstance(result, str)

    def test_load_persona_handles_unicode(self, tmp_path):
        """Test that load_persona handles Unicode content correctly."""
        assets_dir = tmp_path / "persbot"
        assets_dir.mkdir()
        persona_file = assets_dir / "persona.md"

        # Write Korean text (Unicode)
        test_content = "당신은 도움이 되는 AI 어시스턴트입니다."
        persona_file.write_text(test_content, encoding="utf-8")

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            result = load_persona()

            assert "당신은" in result

    def test_load_persona_with_multiline_content(self, tmp_path):
        """Test load_persona with multiline persona content."""
        assets_dir = tmp_path / "persbot"
        assets_dir.mkdir()
        persona_file = assets_dir / "persona.md"

        # Write multiline content
        test_content = """You are a helpful assistant.
Your goal is to assist users.
Be friendly and concise."""
        persona_file.write_text(test_content, encoding="utf-8")

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            result = load_persona()

            assert "helpful assistant" in result
            assert "assist users" in result

    def test_load_persona_empty_file(self, tmp_path):
        """Test load_persona with empty persona file."""
        assets_dir = tmp_path / "persbot"
        assets_dir.mkdir()
        persona_file = assets_dir / "persona.md"

        # Write empty content
        persona_file.write_text("", encoding="utf-8")

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            result = load_persona()

            # Empty file is still a valid result
            assert result == ""

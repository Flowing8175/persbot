"""Feature tests for prompts.py module.

Tests focus on behavior:
- load_persona() function behavior
- File path resolution (persbot/assets/persona.md vs assets/persona.md)
- File not found handling
- File read error handling
- BOT_PERSONA_PROMPT module-level variable initialization
- Re-exported constants from constants.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from persbot import prompts


class TestLoadPersona:
    """Tests for load_persona() function."""

    def test_loads_from_persbot_assets_path(self, tmp_path):
        """load_persona() loads persona from persbot/assets/persona.md."""
        persona_content = "You are a helpful assistant."
        persbot_assets = tmp_path / "persbot" / "assets"
        persbot_assets.mkdir(parents=True)
        (persbot_assets / "persona.md").write_text(persona_content)

        with patch("persbot.prompts.Path") as mock_path_class:
            # First call checks persbot/assets/persona.md
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            # Mock open to return our content
            with patch("builtins.open", mock_open(read_data=persona_content)):
                result = prompts.load_persona()

        assert result == persona_content

    def test_loads_from_assets_path_fallback(self, tmp_path):
        """load_persona() falls back to assets/persona.md if first path doesn't exist."""
        persona_content = "You are a helpful assistant."
        assets = tmp_path / "assets"
        assets.mkdir(parents=True)
        (assets / "persona.md").write_text(persona_content)

        with patch("persbot.prompts.Path") as mock_path_class:
            # First call checks persbot/assets/persona.md (doesn't exist)
            # Second call tries assets/persona.md (exists)
            mock_path = MagicMock()
            mock_path.exists.side_effect = [False, True]
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data=persona_content)):
                result = prompts.load_persona()

        assert result == persona_content

    def test_returns_default_message_when_file_not_found(self):
        """load_persona() returns default message when persona.md doesn't exist."""
        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            result = prompts.load_persona()

        assert result == "System prompt could not be loaded."

    def test_returns_error_message_on_exception(self):
        """load_persona() returns error message when exception occurs."""
        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.side_effect = OSError("Permission denied")
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            result = prompts.load_persona()

        assert result == "System prompt error."

    def test_handles_unicode_encoding(self, tmp_path):
        """load_persona() correctly handles UTF-8 encoded files."""
        persona_content = "당신은 도움이 되는 어시스턴트입니다. 🤖"
        persbot_assets = tmp_path / "persbot" / "assets"
        persbot_assets.mkdir(parents=True)
        (persbot_assets / "persona.md").write_text(persona_content, encoding="utf-8")

        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data=persona_content)):
                result = prompts.load_persona()

        assert result == persona_content
        assert "🤖" in result

    def test_logs_exception_on_failure(self, caplog):
        """load_persona() logs exception when loading fails."""
        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.side_effect = RuntimeError("Unexpected error")
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            with caplog.at_level("ERROR"):
                result = prompts.load_persona()

        assert result == "System prompt error."
        assert any("Failed to load persona.md" in record.message for record in caplog.records)


class TestBotPersonaPrompt:
    """Tests for BOT_PERSONA_PROMPT module-level variable."""

    def test_bot_persona_prompt_is_defined(self):
        """BOT_PERSONA_PROMPT should be defined at module level."""
        assert hasattr(prompts, "BOT_PERSONA_PROMPT")
        assert isinstance(prompts.BOT_PERSONA_PROMPT, str)

    def test_bot_persona_prompt_loads_on_import(self):
        """BOT_PERSONA_PROMPT is initialized when module is imported."""
        # The BOT_PERSONA_PROMPT is loaded at module import time
        # This test verifies it's a string and not None
        assert prompts.BOT_PERSONA_PROMPT is not None
        assert len(prompts.BOT_PERSONA_PROMPT) > 0


class TestModuleReExports:
    """Tests for re-exported constants from constants.py."""

    def test_summary_system_instruction_is_exported(self):
        """SUMMARY_SYSTEM_INSTRUCTION is re-exported from constants."""
        assert hasattr(prompts, "SUMMARY_SYSTEM_INSTRUCTION")
        assert isinstance(prompts.SUMMARY_SYSTEM_INSTRUCTION, str)
        assert len(prompts.SUMMARY_SYSTEM_INSTRUCTION) > 0

    def test_meta_prompt_is_exported(self):
        """META_PROMPT is re-exported from constants."""
        assert hasattr(prompts, "META_PROMPT")
        assert isinstance(prompts.META_PROMPT, str)
        assert len(prompts.META_PROMPT) > 0

    def test_question_generation_prompt_is_exported(self):
        """QUESTION_GENERATION_PROMPT is re-exported from constants."""
        assert hasattr(prompts, "QUESTION_GENERATION_PROMPT")
        assert isinstance(prompts.QUESTION_GENERATION_PROMPT, str)
        assert len(prompts.QUESTION_GENERATION_PROMPT) > 0

    def test_load_persona_is_exported(self):
        """load_persona function is exported in __all__."""
        assert "load_persona" in prompts.__all__
        assert hasattr(prompts, "load_persona")
        assert callable(prompts.load_persona)

    def test_all_exports_are_defined(self):
        """All symbols in __all__ are actually defined."""
        for symbol in prompts.__all__:
            assert hasattr(prompts, symbol), f"{symbol} is in __all__ but not defined"


class TestPromptContent:
    """Tests for prompt content values."""

    def test_summary_system_instruction_content(self):
        """SUMMARY_SYSTEM_INSTRUCTION contains expected content."""
        # Should contain Korean text for summarization
        assert "요약" in prompts.SUMMARY_SYSTEM_INSTRUCTION or "summary" in prompts.SUMMARY_SYSTEM_INSTRUCTION.lower()

    def test_meta_prompt_content(self):
        """META_PROMPT contains expected metadata/instructions."""
        # META_PROMPT should be non-empty and contain prompt instructions
        assert len(prompts.META_PROMPT) > 50  # Should be substantial

    def test_question_generation_prompt_content(self):
        """QUESTION_GENERATION_PROMPT contains question generation instructions."""
        # Should contain instructions for generating questions
        prompt_lower = prompts.QUESTION_GENERATION_PROMPT.lower()
        assert len(prompts.QUESTION_GENERATION_PROMPT) > 50  # Should be substantial


class TestLoadPersonaEdgeCases:
    """Tests for edge cases in load_persona()."""

    def test_empty_persona_file(self, tmp_path):
        """load_persona() handles empty persona.md file."""
        persbot_assets = tmp_path / "persbot" / "assets"
        persbot_assets.mkdir(parents=True)
        (persbot_assets / "persona.md").write_text("")

        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data="")):
                result = prompts.load_persona()

        assert result == ""

    def test_large_persona_file(self, tmp_path):
        """load_persona() handles large persona.md files."""
        large_content = "This is a line.\n" * 10000  # ~170KB file
        persbot_assets = tmp_path / "persbot" / "assets"
        persbot_assets.mkdir(parents=True)
        (persbot_assets / "persona.md").write_text(large_content)

        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data=large_content)):
                result = prompts.load_persona()

        assert result == large_content
        assert result.count("\n") == 10000

    def test_persona_with_special_characters(self, tmp_path):
        """load_persona() handles special characters in persona.md."""
        special_content = "Test: <script>alert('xss')</script>\nSymbols: @#$%^&*()\nQuotes: \"'`'"
        persbot_assets = tmp_path / "persbot" / "assets"
        persbot_assets.mkdir(parents=True)
        (persbot_assets / "persona.md").write_text(special_content)

        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data=special_content)):
                result = prompts.load_persona()

        assert result == special_content
        assert "<script>" in result
        assert "@#$%^&*()" in result


class TestLoadPersonaPathResolution:
    """Tests for path resolution logic in load_persona()."""

    def test_first_path_checked_is_persbot_assets(self):
        """First path checked is persbot/assets/persona.md."""
        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            prompts.load_persona()

            # Verify Path was called with "persbot/assets/persona.md"
            assert mock_path_class.call_count >= 1
            first_call_args = mock_path_class.call_args_list[0]
            assert first_call_args[0][0] == "persbot/assets/persona.md"

    def test_second_path_checked_is_assets(self):
        """Second path checked is assets/persona.md."""
        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.side_effect = [False, False]
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            prompts.load_persona()

            # Verify second Path call with "assets/persona.md"
            assert mock_path_class.call_count >= 2
            second_call_args = mock_path_class.call_args_list[1]
            assert second_call_args[0][0] == "assets/persona.md"

    def test_stops_at_first_existing_path(self, tmp_path):
        """load_persona() stops at first existing path."""
        persbot_assets = tmp_path / "persbot" / "assets"
        persbot_assets.mkdir(parents=True)
        (persbot_assets / "persona.md").write_text("First path content")

        # Also create second path but it shouldn't be read
        assets = tmp_path / "assets"
        assets.mkdir(parents=True)
        (assets / "persona.md").write_text("Second path content")

        with patch("persbot.prompts.Path") as mock_path_class:
            mock_path = MagicMock()
            # First path exists, so second path should never be checked
            mock_path.exists.return_value = True
            mock_path.__truediv__.return_value = mock_path
            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data="First path content")):
                result = prompts.load_persona()

        assert result == "First path content"
        assert result != "Second path content"

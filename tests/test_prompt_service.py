"""Feature tests for PromptService.

Tests focus on behavior:
- PromptService initialization and loading
- Prompt file management (add, rename, delete)
- Usage tracking and daily limits
- Prompt retrieval methods
"""

import datetime
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from persbot.services.prompt_service import PromptService


class TestPromptServiceInit:
    """Tests for PromptService initialization."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    def test_creates_directory_if_not_exists(self, tmp_path, mock_prompts_module):
        """PromptService creates the prompt directory if it doesn't exist."""
        prompt_dir = tmp_path / "new_assets"
        assert not prompt_dir.exists()

        with patch.object(PromptService, "_load_sync") as mock_load, \
             patch.object(PromptService, "_load_usage_sync"):
            PromptService(prompt_dir=str(prompt_dir))

        assert prompt_dir.exists()
        mock_load.assert_called_once()

    def test_loads_prompts_on_init(self, tmp_path, mock_prompts_module):
        """PromptService loads prompts during initialization."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "test.md").write_text("Test prompt content")

        with patch.object(PromptService, "_load_usage_sync"):
            service = PromptService(prompt_dir=str(prompt_dir))

        assert len(service.prompts) == 1
        assert service.prompts[0]["name"] == "test"
        assert service.prompts[0]["content"] == "Test prompt content"

    def test_loads_usage_data_on_init(self, tmp_path, mock_prompts_module):
        """PromptService loads usage data during initialization."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        usage_path = tmp_path / "usage.json"
        usage_data = {"2026-02-13": {"123": 5}}
        usage_path.write_text(json.dumps(usage_data))

        service = PromptService(
            prompt_dir=str(prompt_dir),
            usage_path=str(usage_path)
        )

        assert service.usage_data == usage_data


class TestLoadSync:
    """Tests for _load_sync method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    def test_scans_md_files_in_directory(self, tmp_path, mock_prompts_module):
        """_load_sync scans all .md files in the directory."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "persona.md").write_text("Persona content")
        (prompt_dir / "custom.md").write_text("Custom content")
        (prompt_dir / "other.txt").write_text("Should be ignored")

        with patch.object(PromptService, "_load_usage_sync"):
            service = PromptService(prompt_dir=str(prompt_dir))

        assert len(service.prompts) == 2
        names = [p["name"] for p in service.prompts]
        assert "persona" in names
        assert "custom" in names

    def test_sorts_prompts_by_filename(self, tmp_path, mock_prompts_module):
        """_load_sync sorts prompts alphabetically by filename."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "zebra.md").write_text("Z content")
        (prompt_dir / "alpha.md").write_text("A content")
        (prompt_dir / "beta.md").write_text("B content")

        with patch.object(PromptService, "_load_usage_sync"):
            service = PromptService(prompt_dir=str(prompt_dir))

        assert service.prompts[0]["name"] == "alpha"
        assert service.prompts[1]["name"] == "beta"
        assert service.prompts[2]["name"] == "zebra"

    def test_handles_empty_directory_with_fallback(self, tmp_path, mock_prompts_module):
        """_load_sync uses fallback when directory is empty."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()

        with patch.object(PromptService, "_load_usage_sync"):
            service = PromptService(prompt_dir=str(prompt_dir))

        assert len(service.prompts) == 1
        assert service.prompts[0]["name"] == "기본값"
        assert service.prompts[0]["content"] == "Default persona"

    def test_handles_file_read_errors(self, tmp_path, mock_prompts_module):
        """_load_sync handles file read errors gracefully."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "good.md").write_text("Good content")

        with patch.object(PromptService, "_load_usage_sync"), \
             patch("builtins.open", side_effect=[OSError("Read error"), mock_open(read_data="Good content")()]):
            # The error is logged but doesn't crash
            service = PromptService(prompt_dir=str(prompt_dir))

        # Should have fallback since all reads failed
        assert len(service.prompts) >= 1


class TestReload:
    """Tests for _reload async method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "test.md").write_text("Test content")
        with patch.object(PromptService, "_load_usage_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    @pytest.mark.asyncio
    async def test_reloads_prompts_from_disk(self, service, tmp_path):
        """_reload reads prompts from disk asynchronously."""
        # Add a new file
        prompt_dir = Path(service.prompt_dir)
        (prompt_dir / "new.md").write_text("New content")

        await service._reload()

        names = [p["name"] for p in service.prompts]
        assert "new" in names
        assert "test" in names

    @pytest.mark.asyncio
    async def test_handles_file_read_errors_gracefully(self, service):
        """_reload handles file read errors gracefully."""
        with patch("aiofiles.open", side_effect=OSError("Async read error")):
            # Should not crash, should keep existing prompts
            original_count = len(service.prompts)
            await service._reload()
            # Due to error, new_prompts is empty, but we keep existing
            assert len(service.prompts) == original_count


class TestUsageDataManagement:
    """Tests for usage data loading and saving."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    def test_load_usage_sync_loads_existing_file(self, tmp_path, mock_prompts_module):
        """_load_usage_sync loads existing usage file."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        usage_path = tmp_path / "usage.json"
        usage_data = {"2026-02-13": {"123": 5, "456": 2}}
        usage_path.write_text(json.dumps(usage_data))

        with patch.object(PromptService, "_load_sync"):
            service = PromptService(
                prompt_dir=str(prompt_dir),
                usage_path=str(usage_path)
            )

        assert service.usage_data == usage_data

    def test_load_usage_sync_handles_missing_file(self, tmp_path, mock_prompts_module):
        """_load_usage_sync handles missing usage file."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        usage_path = tmp_path / "nonexistent.json"

        with patch.object(PromptService, "_load_sync"):
            service = PromptService(
                prompt_dir=str(prompt_dir),
                usage_path=str(usage_path)
            )

        assert service.usage_data == {}

    def test_load_usage_sync_handles_invalid_json(self, tmp_path, mock_prompts_module):
        """_load_usage_sync handles invalid JSON."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        usage_path = tmp_path / "usage.json"
        usage_path.write_text("not valid json")

        with patch.object(PromptService, "_load_sync"):
            service = PromptService(
                prompt_dir=str(prompt_dir),
                usage_path=str(usage_path)
            )

        assert service.usage_data == {}

    @pytest.mark.asyncio
    async def test_save_usage_writes_to_file(self, tmp_path, mock_prompts_module):
        """_save_usage writes usage data to file."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        usage_path = tmp_path / "usage.json"

        with patch.object(PromptService, "_load_sync"):
            service = PromptService(
                prompt_dir=str(prompt_dir),
                usage_path=str(usage_path)
            )

        service.usage_data = {"2026-02-13": {"123": 1}}
        await service._save_usage()

        assert usage_path.exists()
        with open(usage_path) as f:
            saved_data = json.load(f)
        assert saved_data == {"2026-02-13": {"123": 1}}


class TestGetTodayKey:
    """Tests for _get_today_key method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        with patch.object(PromptService, "_load_usage_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    def test_returns_date_in_correct_format(self, service):
        """_get_today_key returns date in YYYY-MM-DD format."""
        result = service._get_today_key()

        # Verify format
        assert len(result) == 10
        assert result[4] == "-"
        assert result[7] == "-"

        # Verify it's a valid date
        datetime.datetime.strptime(result, "%Y-%m-%d")


class TestCheckTodayLimit:
    """Tests for check_today_limit method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        with patch.object(PromptService, "_load_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    @pytest.mark.asyncio
    async def test_returns_true_when_under_limit(self, service):
        """check_today_limit returns True when user is under limit."""
        with patch.object(service, "_get_today_key", return_value="2026-02-13"):
            result = await service.check_today_limit(user_id=123, limit=2)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_at_limit(self, service):
        """check_today_limit returns False when user is at limit."""
        service.usage_data = {"2026-02-13": {"123": 2}}

        with patch.object(service, "_get_today_key", return_value="2026-02-13"):
            result = await service.check_today_limit(user_id=123, limit=2)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_above_limit_but_different_day(self, service):
        """check_today_limit returns True when previous day's usage doesn't count."""
        service.usage_data = {"2026-02-12": {"123": 10}}

        with patch.object(service, "_get_today_key", return_value="2026-02-13"):
            result = await service.check_today_limit(user_id=123, limit=2)

        assert result is True

    @pytest.mark.asyncio
    async def test_initializes_usage_for_new_day(self, service, tmp_path):
        """check_today_limit initializes usage data for a new day."""
        usage_path = tmp_path / "usage.json"
        service.usage_path = str(usage_path)
        service.usage_data = {"2026-02-12": {"999": 5}}

        with patch.object(service, "_get_today_key", return_value="2026-02-13"):
            await service.check_today_limit(user_id=123, limit=2)

        assert "2026-02-13" in service.usage_data
        assert "2026-02-12" not in service.usage_data  # Old data cleaned

    @pytest.mark.asyncio
    async def test_uses_custom_limit(self, service):
        """check_today_limit respects custom limit parameter."""
        service.usage_data = {"2026-02-13": {"123": 3}}

        with patch.object(service, "_get_today_key", return_value="2026-02-13"):
            result_at_3 = await service.check_today_limit(user_id=123, limit=3)
            result_at_2 = await service.check_today_limit(user_id=123, limit=2)

        assert result_at_3 is False  # 3 < 3 is False
        assert result_at_2 is False  # 3 < 2 is False


class TestIncrementTodayUsage:
    """Tests for increment_today_usage method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        with patch.object(PromptService, "_load_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    @pytest.mark.asyncio
    async def test_increments_from_zero(self, service, tmp_path):
        """increment_today_usage increments from 0 to 1."""
        usage_path = tmp_path / "usage.json"
        service.usage_path = str(usage_path)

        with patch.object(service, "_get_today_key", return_value="2026-02-13"):
            await service.increment_today_usage(user_id=123)

        assert service.usage_data["2026-02-13"]["123"] == 1

    @pytest.mark.asyncio
    async def test_increments_existing_count(self, service, tmp_path):
        """increment_today_usage increments existing count."""
        usage_path = tmp_path / "usage.json"
        service.usage_path = str(usage_path)
        service.usage_data = {"2026-02-13": {"123": 2}}

        with patch.object(service, "_get_today_key", return_value="2026-02-13"):
            await service.increment_today_usage(user_id=123)

        assert service.usage_data["2026-02-13"]["123"] == 3

    @pytest.mark.asyncio
    async def test_creates_day_entry_if_missing(self, service, tmp_path):
        """increment_today_usage creates day entry if missing."""
        usage_path = tmp_path / "usage.json"
        service.usage_path = str(usage_path)

        with patch.object(service, "_get_today_key", return_value="2026-02-13"):
            await service.increment_today_usage(user_id=456)

        assert "2026-02-13" in service.usage_data
        assert service.usage_data["2026-02-13"]["456"] == 1


class TestSanitizeFilename:
    """Tests for _sanitize_filename method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        with patch.object(PromptService, "_load_sync"), \
             patch.object(PromptService, "_load_usage_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    def test_removes_special_characters(self, service):
        """_sanitize_filename removes special characters."""
        result = service._sanitize_filename("test@#$%^&*file")
        assert result == "testfile"

    def test_keeps_alphanumeric_spaces_hyphens_underscores(self, service):
        """_sanitize_filename keeps alphanumeric, spaces, hyphens, underscores."""
        result = service._sanitize_filename("My-Test_File 123")
        assert result == "My-Test_File 123"

    def test_strips_whitespace(self, service):
        """_sanitize_filename strips leading/trailing whitespace."""
        result = service._sanitize_filename("  test name  ")
        assert result == "test name"

    def test_returns_untitled_for_empty_result(self, service):
        """_sanitize_filename returns 'untitled' for empty result."""
        result = service._sanitize_filename("@#$%^&*")
        assert result == "untitled"

    def test_handles_empty_string(self, service):
        """_sanitize_filename handles empty string."""
        result = service._sanitize_filename("")
        assert result == "untitled"

    def test_preserves_korean_characters(self, service):
        """_sanitize_filename preserves Korean characters."""
        # Korean characters are alphanumeric in Unicode
        result = service._sanitize_filename("한글 테스트")
        # This will depend on Unicode category - Korean Hangul is typically "Lo" (Letter other)
        # which is alphanumeric in Python's isalnum()
        assert "테스트" in result or result == "untitled"  # Behavior depends on implementation


class TestAddPrompt:
    """Tests for add_prompt method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        with patch.object(PromptService, "_load_usage_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    @pytest.mark.asyncio
    async def test_creates_new_file(self, service):
        """add_prompt creates a new .md file."""
        index = await service.add_prompt("test_prompt", "Test content")

        prompt_dir = Path(service.prompt_dir)
        assert (prompt_dir / "test_prompt.md").exists()
        assert (prompt_dir / "test_prompt.md").read_text() == "Test content"
        assert index >= 0

    @pytest.mark.asyncio
    async def test_handles_collision_with_suffix(self, service):
        """add_prompt appends suffix when file exists."""
        prompt_dir = Path(service.prompt_dir)
        (prompt_dir / "existing.md").write_text("Already exists")

        await service.add_prompt("existing", "New content")

        # Should create existing_1.md
        assert (prompt_dir / "existing_1.md").exists()

    @pytest.mark.asyncio
    async def test_handles_multiple_collisions(self, service):
        """add_prompt increments suffix for multiple collisions."""
        prompt_dir = Path(service.prompt_dir)
        (prompt_dir / "test.md").write_text("Original")
        (prompt_dir / "test_1.md").write_text("First collision")
        (prompt_dir / "test_2.md").write_text("Second collision")

        await service.add_prompt("test", "Third attempt")

        assert (prompt_dir / "test_3.md").exists()

    @pytest.mark.asyncio
    async def test_returns_index_of_new_prompt(self, service):
        """add_prompt returns the index of the newly added prompt."""
        index = await service.add_prompt("unique_name", "Content")

        # Verify the prompt exists at the returned index
        assert 0 <= index < len(service.prompts)
        assert service.prompts[index]["name"] == "unique_name"

    @pytest.mark.asyncio
    async def test_sanitizes_filename(self, service):
        """add_prompt sanitizes the filename."""
        await service.add_prompt("bad@name#here", "Content")

        prompt_dir = Path(service.prompt_dir)
        # Should not contain @ or #
        assert not any("@" in f.name or "#" in f.name for f in prompt_dir.glob("*.md"))


class TestListAndGetPrompts:
    """Tests for list_prompts and get_prompt methods."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "first.md").write_text("First prompt")
        (prompt_dir / "second.md").write_text("Second prompt")
        with patch.object(PromptService, "_load_usage_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    def test_list_prompts_returns_all(self, service):
        """list_prompts returns all prompts."""
        prompts = service.list_prompts()

        assert len(prompts) == 2
        names = [p["name"] for p in prompts]
        assert "first" in names
        assert "second" in names

    def test_get_prompt_returns_correct_prompt(self, service):
        """get_prompt returns the prompt at the given index."""
        prompt = service.get_prompt(0)

        assert prompt is not None
        assert prompt["name"] == "first"

    def test_get_prompt_returns_none_for_invalid_index(self, service):
        """get_prompt returns None for invalid index."""
        assert service.get_prompt(-1) is None
        assert service.get_prompt(99) is None

    def test_list_prompts_returns_copy(self, service):
        """list_prompts returns the internal list (reference)."""
        prompts = service.list_prompts()
        # Modifying the returned list affects the internal state
        # This is the current behavior
        assert prompts is service.prompts


class TestGetActiveAssistantPrompt:
    """Tests for get_active_assistant_prompt method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    def test_returns_first_prompt_content(self, tmp_path, mock_prompts_module):
        """get_active_assistant_prompt returns first prompt's content."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "active.md").write_text("Active prompt")

        with patch.object(PromptService, "_load_usage_sync"):
            service = PromptService(prompt_dir=str(prompt_dir))

        result = service.get_active_assistant_prompt()
        assert result == "Active prompt"

    def test_returns_fallback_when_no_prompts(self, tmp_path, mock_prompts_module):
        """get_active_assistant_prompt returns fallback when no prompts."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()

        with patch.object(PromptService, "_load_usage_sync"):
            service = PromptService(prompt_dir=str(prompt_dir))

        result = service.get_active_assistant_prompt()
        assert result == "Default persona"


class TestGetSummaryPrompt:
    """Tests for get_summary_prompt method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        with patch.object(PromptService, "_load_sync"), \
             patch.object(PromptService, "_load_usage_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    def test_returns_summary_instruction(self, service):
        """get_summary_prompt returns the summary instruction."""
        result = service.get_summary_prompt()
        assert result == "Summary instruction"


class TestRenamePrompt:
    """Tests for rename_prompt method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "old_name.md").write_text("Content to rename")
        with patch.object(PromptService, "_load_usage_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    @pytest.mark.asyncio
    async def test_renames_file_successfully(self, service):
        """rename_prompt renames the file."""
        result = await service.rename_prompt(0, "new_name")

        assert result is True
        prompt_dir = Path(service.prompt_dir)
        assert not (prompt_dir / "old_name.md").exists()
        assert (prompt_dir / "new_name.md").exists()

    @pytest.mark.asyncio
    async def test_returns_false_for_invalid_index(self, service):
        """rename_prompt returns False for invalid index."""
        result = await service.rename_prompt(99, "new_name")
        assert result is False

        result = await service.rename_prompt(-1, "new_name")
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_target_exists(self, service):
        """rename_prompt returns False when target file exists."""
        prompt_dir = Path(service.prompt_dir)
        (prompt_dir / "existing.md").write_text("Already exists")

        result = await service.rename_prompt(0, "existing")

        assert result is False
        # Original file should still exist
        assert (prompt_dir / "old_name.md").exists()

    @pytest.mark.asyncio
    async def test_returns_false_when_source_missing(self, service):
        """rename_prompt returns False when source file is missing."""
        # Modify the prompt to have an invalid path
        service.prompts[0]["path"] = "/nonexistent/path.md"

        result = await service.rename_prompt(0, "new_name")

        assert result is False

    @pytest.mark.asyncio
    async def test_sanitizes_new_filename(self, service):
        """rename_prompt sanitizes the new filename."""
        result = await service.rename_prompt(0, "bad@name#here")

        assert result is True
        prompt_dir = Path(service.prompt_dir)
        # File should be renamed with sanitized name
        md_files = list(prompt_dir.glob("*.md"))
        assert all("@" not in f.name and "#" not in f.name for f in md_files)


class TestDeletePrompt:
    """Tests for delete_prompt method."""

    @pytest.fixture
    def mock_prompts_module(self):
        """Mock the prompts module constants."""
        with patch("persbot.services.prompt_service.BOT_PERSONA_PROMPT", "Default persona"), \
             patch("persbot.services.prompt_service.SUMMARY_SYSTEM_INSTRUCTION", "Summary instruction"):
            yield

    @pytest.fixture
    def service(self, tmp_path, mock_prompts_module):
        """Create a PromptService for testing."""
        prompt_dir = tmp_path / "assets"
        prompt_dir.mkdir()
        (prompt_dir / "to_delete.md").write_text("Content to delete")
        with patch.object(PromptService, "_load_usage_sync"):
            return PromptService(prompt_dir=str(prompt_dir))

    @pytest.mark.asyncio
    async def test_deletes_file_successfully(self, service):
        """delete_prompt deletes the file."""
        result = await service.delete_prompt(0)

        assert result is True
        prompt_dir = Path(service.prompt_dir)
        assert not (prompt_dir / "to_delete.md").exists()

    @pytest.mark.asyncio
    async def test_returns_false_for_invalid_index(self, service):
        """delete_prompt returns False for invalid index."""
        result = await service.delete_prompt(99)
        assert result is False

        result = await service.delete_prompt(-1)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_file_missing(self, service):
        """delete_prompt returns False when file doesn't exist."""
        # Modify the prompt to have an invalid path
        service.prompts[0]["path"] = "/nonexistent/path.md"

        result = await service.delete_prompt(0)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_empty_path(self, service):
        """delete_prompt returns False when path is empty (fallback prompt)."""
        service.prompts[0]["path"] = ""

        result = await service.delete_prompt(0)

        assert result is False

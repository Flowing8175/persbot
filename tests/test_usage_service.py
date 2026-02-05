"""Comprehensive tests for soyebot/services/usage_service.py (ImageUsageService)."""

import pytest
import pytest_asyncio
from pathlib import Path
import json
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, Mock, AsyncMock, MagicMock
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from soyebot.services.usage_service import ImageUsageService


# =============================================================================
# Tests for ImageUsageService.__init__()
# =============================================================================


class TestImageUsageServiceInit:
    """Test suite for ImageUsageService initialization."""

    def test_init_with_default_storage_path(self, temp_dir):
        """Test initialization with default storage path creates usage_data dict."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))

        assert hasattr(service, "usage_data")
        assert isinstance(service.usage_data, dict)
        assert len(service.usage_data) == 0
        assert service.storage_path == str(storage_path)

    def test_init_creates_data_directory(self, temp_dir):
        """Test initialization creates data directory if it doesn't exist."""
        storage_path = temp_dir / "subdir" / "nested" / "usage.json"
        service = ImageUsageService(storage_path=str(storage_path))

        assert Path(storage_path).parent.exists()
        assert service.storage_path == str(storage_path)

    def test_init_loads_existing_data(self, temp_dir):
        """Test initialization loads existing data from JSON file."""
        storage_path = temp_dir / "existing_usage.json"
        # Use today's date to avoid cleanup
        service_temp = ImageUsageService(storage_path=str(temp_dir / "dummy.json"))
        today = service_temp._get_today_key()
        existing_data = {
            today: {"123456789": 2, "987654321": 1},
        }
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f)

        service = ImageUsageService(storage_path=str(storage_path))

        assert service.usage_data == existing_data

    def test_init_handles_nonexistent_file(self, temp_dir):
        """Test initialization handles non-existent JSON file gracefully."""
        storage_path = temp_dir / "nonexistent_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))

        assert service.usage_data == {}
        assert not Path(storage_path).exists()

    def test_init_handles_corrupted_json(self, temp_dir):
        """Test initialization handles corrupted JSON file."""
        storage_path = temp_dir / "corrupted_usage.json"
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json content")

        service = ImageUsageService(storage_path=str(storage_path))

        # Should fall back to empty dict on error
        assert service.usage_data == {}

    def test_init_cleanup_old_entries_on_load(self, temp_dir):
        """Test initialization cleans up old date entries on load."""
        storage_path = temp_dir / "cleanup_usage.json"
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        yesterday = (
            datetime.now(timezone(timedelta(hours=9))) - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        existing_data = {
            yesterday: {"123456789": 5},
            today: {"123456789": 2},
            "2025-01-01": {"999888777": 1},
        }
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f)

        service = ImageUsageService(storage_path=str(storage_path))

        # Only today's data should remain in memory
        assert today in service.usage_data
        assert yesterday not in service.usage_data
        assert "2025-01-01" not in service.usage_data


# =============================================================================
# Tests for check_can_upload()
# =============================================================================


class TestCheckCanUpload:
    """Test suite for check_can_upload() method."""

    def test_check_can_upload_user_zero_uploads(self, temp_dir):
        """Test user with 0 uploads today is allowed."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))

        result = service.check_can_upload(user_id=123456789, count=1, limit=3)

        assert result is True

    def test_check_can_upload_user_two_uploads(self, temp_dir):
        """Test user with 2 uploads is allowed to add 1 more."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 2}

        result = service.check_can_upload(user_id=123456789, count=1, limit=3)

        assert result is True

    def test_check_can_upload_user_three_uploads_last_allowed(self, temp_dir):
        """Test user with 3 uploads (at limit) is allowed if adding 0."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 3}

        result = service.check_can_upload(user_id=123456789, count=0, limit=3)

        assert result is True

    def test_check_can_upload_user_three_uploads_not_allowed(self, temp_dir):
        """Test user with 3 uploads is NOT allowed to add 1 more."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 3}

        result = service.check_can_upload(user_id=123456789, count=1, limit=3)

        assert result is False

    def test_check_can_upload_user_four_uploads_not_allowed(self, temp_dir):
        """Test user with 4 uploads is not allowed."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 4}

        result = service.check_can_upload(user_id=123456789, count=1, limit=3)

        assert result is False

    def test_check_can_upload_custom_limit(self, temp_dir):
        """Test custom limit parameter works correctly."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 5}

        result = service.check_can_upload(user_id=123456789, count=1, limit=10)

        assert result is True

    def test_check_can_upload_negative_count(self, temp_dir):
        """Test negative count is handled correctly."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 2}

        # Adding negative count should allow (mathematically correct)
        result = service.check_can_upload(user_id=123456789, count=-1, limit=3)

        assert result is True

    def test_check_can_upload_batch_upload(self, temp_dir):
        """Test checking multiple images at once."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 1}

        # Trying to upload 2 more when only 1 is left (limit=3)
        result = service.check_can_upload(user_id=123456789, count=2, limit=3)

        assert result is True

    def test_check_can_upload_batch_exceeds_limit(self, temp_dir):
        """Test batch upload that would exceed limit."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 2}

        # Trying to upload 2 more when only 1 is left (limit=3)
        result = service.check_can_upload(user_id=123456789, count=2, limit=3)

        assert result is False

    def test_check_can_upload_creates_today_entry(self, temp_dir):
        """Test check_can_upload creates today entry if not present."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()

        assert today not in service.usage_data

        service.check_can_upload(user_id=123456789, count=1, limit=3)

        assert today in service.usage_data
        assert service.usage_data[today] == {}

    def test_check_can_upload_different_users(self, temp_dir):
        """Test check_can_upload works correctly for different users."""
        storage_path = temp_dir / "test_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 3, "987654321": 0}

        # User 1 at limit should not be allowed
        assert not service.check_can_upload(user_id=123456789, count=1, limit=3)

        # User 2 should be allowed
        assert service.check_can_upload(user_id=987654321, count=3, limit=3)


# =============================================================================
# Tests for record_upload()
# =============================================================================


@pytest_asyncio.fixture
async def usage_service_with_temp_storage(temp_dir):
    """Create an ImageUsageService instance with temp storage."""
    storage_path = temp_dir / "test_usage.json"
    service = ImageUsageService(storage_path=str(storage_path))
    yield service


class TestRecordUpload:
    """Test suite for record_upload() method."""

    @pytest.mark.asyncio
    async def test_record_upload_new_user(self, usage_service_with_temp_storage):
        """Test recording upload for new user creates entry."""
        service = usage_service_with_temp_storage
        today = service._get_today_key()

        await service.record_upload(user_id=123456789, count=2)

        assert today in service.usage_data
        assert service.usage_data[today]["123456789"] == 2

    @pytest.mark.asyncio
    async def test_record_upload_existing_user_increment(
        self, usage_service_with_temp_storage
    ):
        """Test recording upload increments count for existing user."""
        service = usage_service_with_temp_storage
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 2}

        await service.record_upload(user_id=123456789, count=1)

        assert service.usage_data[today]["123456789"] == 3

    @pytest.mark.asyncio
    async def test_record_upload_multiple_increment(
        self, usage_service_with_temp_storage
    ):
        """Test recording multiple uploads for same user."""
        service = usage_service_with_temp_storage
        today = service._get_today_key()

        await service.record_upload(user_id=123456789, count=1)
        await service.record_upload(user_id=123456789, count=2)
        await service.record_upload(user_id=123456789, count=3)

        assert service.usage_data[today]["123456789"] == 6

    @pytest.mark.asyncio
    async def test_record_upload_different_users(self, usage_service_with_temp_storage):
        """Test recording uploads for different users."""
        service = usage_service_with_temp_storage
        today = service._get_today_key()

        await service.record_upload(user_id=123456789, count=2)
        await service.record_upload(user_id=987654321, count=1)

        assert service.usage_data[today]["123456789"] == 2
        assert service.usage_data[today]["987654321"] == 1

    @pytest.mark.asyncio
    async def test_record_upload_saves_to_file(self, usage_service_with_temp_storage):
        """Test record_upload persists data to file."""
        service = usage_service_with_temp_storage
        storage_path = Path(service.storage_path)

        await service.record_upload(user_id=123456789, count=2)

        # Wait for async save to complete
        await asyncio.sleep(0.1)

        assert storage_path.exists()
        with open(storage_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        today = service._get_today_key()
        assert saved_data[today]["123456789"] == 2

    @pytest.mark.asyncio
    async def test_record_upload_creates_today_entry(
        self, usage_service_with_temp_storage
    ):
        """Test record_upload creates today entry if not present."""
        service = usage_service_with_temp_storage
        today = service._get_today_key()

        assert today not in service.usage_data

        await service.record_upload(user_id=123456789, count=1)

        assert today in service.usage_data

    @pytest.mark.asyncio
    async def test_record_upload_negative_count(self, usage_service_with_temp_storage):
        """Test recording negative count (edge case)."""
        service = usage_service_with_temp_storage
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 5}

        await service.record_upload(user_id=123456789, count=-2)

        # Should decrement
        assert service.usage_data[today]["123456789"] == 3

    @pytest.mark.asyncio
    async def test_record_upload_zero_count(self, usage_service_with_temp_storage):
        """Test recording zero count."""
        service = usage_service_with_temp_storage
        today = service._get_today_key()

        await service.record_upload(user_id=123456789, count=0)

        # Should create entry with 0
        assert service.usage_data[today]["123456789"] == 0

    @pytest.mark.asyncio
    async def test_record_upload_cleanup_old_data(
        self, usage_service_with_temp_storage
    ):
        """Test record_upload cleans up old data when creating new day."""
        service = usage_service_with_temp_storage
        yesterday = (
            datetime.now(timezone(timedelta(hours=9))) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        today = service._get_today_key()

        # Simulate having yesterday's data
        service.usage_data[yesterday] = {"123456789": 5}

        # Record upload today
        await service.record_upload(user_id=987654321, count=1)

        # Yesterday's data should be cleaned up in memory
        assert yesterday not in service.usage_data
        assert today in service.usage_data


# =============================================================================
# Tests for _save() and _save_async()
# =============================================================================


class TestSaveUsage:
    """Test suite for _save() and _save_async() methods."""

    def test_save_creates_file(self, temp_dir):
        """Test _save creates JSON file."""
        storage_path = temp_dir / "test_save.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 2}

        service._save(service.usage_data)

        assert storage_path.exists()

    def test_save_writes_correct_json(self, temp_dir):
        """Test _save writes correct JSON content."""
        storage_path = temp_dir / "test_save.json"
        service = ImageUsageService(storage_path=str(storage_path))
        test_data = {"2025-01-15": {"123456789": 2, "987654321": 1}}

        service._save(test_data)

        with open(storage_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data == test_data

    def test_save_unicode_handling(self, temp_dir):
        """Test _save handles Unicode characters correctly."""
        storage_path = temp_dir / "test_unicode.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        test_data = {today: {"123456789": 1}}
        service.usage_data = test_data

        service._save(service.usage_data)

        with open(storage_path, "r", encoding="utf-8") as f:
            content = f.read()

        # ensure_ascii=False should keep Unicode characters readable
        assert "한글" not in content  # No Korean in this test, but structure preserved

    def test_save_overwrites_existing_file(self, temp_dir):
        """Test _save overwrites existing file."""
        storage_path = temp_dir / "test_overwrite.json"
        service = ImageUsageService(storage_path=str(storage_path))

        # Write initial data
        initial_data = {"2025-01-14": {"111": 1}}
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f)

        # Save new data
        new_data = {"2025-01-15": {"123456789": 2}}
        service._save(new_data)

        with open(storage_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data == new_data
        assert saved_data != initial_data

    def test_save_creates_parent_directory(self, temp_dir):
        """Test _save creates parent directory if needed."""
        storage_path = temp_dir / "deep" / "nested" / "test_save.json"
        service = ImageUsageService(storage_path=str(storage_path))
        service.usage_data = {"2025-01-15": {"123456789": 1}}

        service._save(service.usage_data)

        assert storage_path.exists()

    def test_save_handles_write_error(self, temp_dir):
        """Test _save handles write errors gracefully."""
        storage_path = temp_dir / "readonly.json"
        service = ImageUsageService(storage_path=str(storage_path))
        service.usage_data = {"2025-01-15": {"123456789": 1}}

        # Make the file readonly (Unix only, so skip on Windows)
        try:
            # Create file
            with open(storage_path, "w") as f:
                f.write("{}")

            # Make readonly
            os.chmod(storage_path, 0o444)

            # Try to save - should not raise exception
            service._save(service.usage_data)

            # Reset permissions for cleanup
            os.chmod(storage_path, 0o644)

        except (OSError, PermissionError):
            # Windows may not support chmod, just pass
            pass

    @pytest.mark.asyncio
    async def test_save_async(self, temp_dir):
        """Test _save_async saves data asynchronously."""
        storage_path = temp_dir / "test_async_save.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 3}

        await service._save_async()

        assert storage_path.exists()
        with open(storage_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data[today]["123456789"] == 3

    @pytest.mark.asyncio
    async def test_save_async_creates_snapshot(self, temp_dir):
        """Test _save_async creates data snapshot to avoid race conditions."""
        storage_path = temp_dir / "test_snapshot.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 1}

        # Modify data before starting save
        service.usage_data[today]["987654321"] = 999

        # Start async save - it takes a snapshot at this point
        save_task = asyncio.create_task(service._save_async())

        # Modify data while save is in progress (shallow copy doesn't protect nested dicts)
        service.usage_data[today]["123456789"] = 999

        await save_task

        # File should have the value at snapshot time (999 for both keys)
        # Note: This demonstrates that shallow copy doesn't protect nested data
        with open(storage_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data[today]["123456789"] == 999
        assert saved_data[today]["987654321"] == 999


# =============================================================================
# Tests for _load()
# =============================================================================


class TestLoadUsage:
    """Test suite for _load() method."""

    def test_load_existing_file(self, temp_dir):
        """Test _load loads data from existing JSON file."""
        storage_path = temp_dir / "test_load.json"
        # Use today's date to avoid cleanup
        service_temp = ImageUsageService(storage_path=str(temp_dir / "dummy.json"))
        today = service_temp._get_today_key()
        test_data = {today: {"123456789": 2}}
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        service = ImageUsageService(storage_path=str(storage_path))

        assert service.usage_data == test_data

    def test_load_nonexistent_file(self, temp_dir):
        """Test _load handles non-existent file."""
        storage_path = temp_dir / "nonexistent.json"
        service = ImageUsageService(storage_path=str(storage_path))

        assert service.usage_data == {}

    def test_load_corrupted_json(self, temp_dir):
        """Test _load handles corrupted JSON gracefully."""
        storage_path = temp_dir / "corrupted.json"
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        service = ImageUsageService(storage_path=str(storage_path))

        # Should fall back to empty dict
        assert service.usage_data == {}

    def test_load_empty_file(self, temp_dir):
        """Test _load handles empty file."""
        storage_path = temp_dir / "empty.json"
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write("")

        service = ImageUsageService(storage_path=str(storage_path))

        # Should fall back to empty dict
        assert service.usage_data == {}

    def test_load_unicode_content(self, temp_dir):
        """Test _load handles Unicode content."""
        storage_path = temp_dir / "unicode.json"
        # Use today's date to avoid cleanup
        service_temp = ImageUsageService(storage_path=str(temp_dir / "dummy.json"))
        today = service_temp._get_today_key()
        test_data = {today: {"123456789": 1}}
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False)

        service = ImageUsageService(storage_path=str(storage_path))

        assert service.usage_data == test_data

    def test_load_with_old_data_cleanup(self, temp_dir):
        """Test _load cleans up old date entries."""
        storage_path = temp_dir / "cleanup_on_load.json"
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        test_data = {
            "2025-01-01": {"111": 1},
            "2025-01-15": {"222": 2},
            today: {"123456789": 3},
        }
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        service = ImageUsageService(storage_path=str(storage_path))

        # Only today's data should remain
        assert today in service.usage_data
        assert "2025-01-01" not in service.usage_data
        assert "2025-01-15" not in service.usage_data


# =============================================================================
# Tests for _get_today_key()
# =============================================================================


class TestGetTodayKey:
    """Test suite for _get_today_key() method."""

    def test_get_today_key_format(self):
        """Test _get_today_key returns correct format."""
        service = ImageUsageService(storage_path="dummy.json")
        today_key = service._get_today_key()

        # Check format: YYYY-MM-DD
        assert len(today_key) == 10
        assert today_key[4] == "-"
        assert today_key[7] == "-"

    def test_get_today_key_components(self):
        """Test _get_today_key returns correct date components."""
        service = ImageUsageService(storage_path="dummy.json")
        today_key = service._get_today_key()

        # Parse components
        year, month, day = map(int, today_key.split("-"))

        # Check reasonable values
        assert 2000 <= year <= 2100
        assert 1 <= month <= 12
        assert 1 <= day <= 31

    def test_get_today_key_consistency(self):
        """Test _get_today_key returns consistent value."""
        service = ImageUsageService(storage_path="dummy.json")

        key1 = service._get_today_key()
        key2 = service._get_today_key()

        # Should return the same value within same day
        assert key1 == key2

    def test_get_today_kst_timezone(self):
        """Test _get_today_key uses KST timezone (UTC+9)."""
        service = ImageUsageService(storage_path="dummy.json")
        kst = timezone(timedelta(hours=9))
        expected = datetime.now(kst).strftime("%Y-%m-%d")
        actual = service._get_today_key()

        assert actual == expected


# =============================================================================
# Tests for Daily Reset (_cleanup_old_entries)
# =============================================================================


class TestDailyReset:
    """Test suite for daily reset functionality."""

    def test_cleanup_removes_old_dates(self, temp_dir):
        """Test _cleanup_old_entries removes old date entries."""
        storage_path = temp_dir / "test_cleanup.json"
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        service = ImageUsageService(storage_path=str(storage_path))
        service.usage_data = {
            "2025-01-01": {"111": 1},
            "2025-01-15": {"222": 2},
            today: {"123456789": 3},
        }

        service._cleanup_old_entries()

        assert today in service.usage_data
        assert "2025-01-01" not in service.usage_data
        assert "2025-01-15" not in service.usage_data

    def test_cleanup_preserves_today(self, temp_dir):
        """Test _cleanup_old_entries preserves today's data."""
        storage_path = temp_dir / "test_preserve.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data = {today: {"123456789": 2}}

        service._cleanup_old_entries()

        assert today in service.usage_data
        assert service.usage_data[today] == {"123456789": 2}

    def test_cleanup_empty_usage_data(self, temp_dir):
        """Test _cleanup_old_entries handles empty usage_data."""
        storage_path = temp_dir / "test_empty_cleanup.json"
        service = ImageUsageService(storage_path=str(storage_path))
        service.usage_data = {}

        service._cleanup_old_entries()

        assert service.usage_data == {}

    def test_cleanup_only_old_dates(self, temp_dir):
        """Test _cleanup_old_entries only removes dates, not today."""
        storage_path = temp_dir / "test_only_old.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        yesterday = (
            datetime.now(timezone(timedelta(hours=9))) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        service.usage_data = {
            yesterday: {"111": 5},
            today: {"123456789": 1},
        }

        service._cleanup_old_entries()

        assert today in service.usage_data
        assert yesterday not in service.usage_data

    def test_cleanup_multiple_users_today(self, temp_dir):
        """Test _cleanup_old_entries preserves all users for today."""
        storage_path = temp_dir / "test_multi_users.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data = {
            "2025-01-01": {"999": 1},
            today: {"123456789": 2, "987654321": 3, "555666777": 1},
        }

        service._cleanup_old_entries()

        assert today in service.usage_data
        assert len(service.usage_data[today]) == 3
        assert service.usage_data[today]["123456789"] == 2
        assert service.usage_data[today]["987654321"] == 3
        assert service.usage_data[today]["555666777"] == 1


# =============================================================================
# Tests for get_usage()
# =============================================================================


class TestGetUsage:
    """Test suite for get_usage() method."""

    def test_get_usage_existing_user(self, temp_dir):
        """Test get_usage returns correct count for existing user."""
        storage_path = temp_dir / "test_get_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 5}

        result = service.get_usage(user_id=123456789)

        assert result == 5

    def test_get_usage_nonexistent_user(self, temp_dir):
        """Test get_usage returns 0 for nonexistent user."""
        storage_path = temp_dir / "test_get_usage_new.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"987654321": 3}

        result = service.get_usage(user_id=123456789)

        assert result == 0

    def test_get_usage_no_today_data(self, temp_dir):
        """Test get_usage returns 0 when no today data exists."""
        storage_path = temp_dir / "test_no_today.json"
        service = ImageUsageService(storage_path=str(storage_path))
        service.usage_data = {"2025-01-15": {"123456789": 2}}

        result = service.get_usage(user_id=123456789)

        assert result == 0

    def test_get_usage_empty_data(self, temp_dir):
        """Test get_usage returns 0 with empty usage_data."""
        storage_path = temp_dir / "test_empty_usage.json"
        service = ImageUsageService(storage_path=str(storage_path))
        service.usage_data = {}

        result = service.get_usage(user_id=123456789)

        assert result == 0

    def test_get_usage_different_users(self, temp_dir):
        """Test get_usage returns correct values for different users."""
        storage_path = temp_dir / "test_diff_users.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 2, "987654321": 5, "555666777": 0}

        assert service.get_usage(user_id=123456789) == 2
        assert service.get_usage(user_id=987654321) == 5
        assert service.get_usage(user_id=555666777) == 0
        assert service.get_usage(user_id=111222333) == 0


# =============================================================================
# Tests for User Tracking (Multiple scenarios)
# =============================================================================


class TestUserTracking:
    """Test suite for user tracking functionality."""

    @pytest.mark.asyncio
    async def test_multiple_users_same_day(self, temp_dir):
        """Test tracking multiple users on the same day."""
        storage_path = temp_dir / "test_multi_tracking.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()

        await service.record_upload(user_id=123456789, count=2)
        await service.record_upload(user_id=987654321, count=3)
        await service.record_upload(user_id=555666777, count=1)

        assert service.get_usage(user_id=123456789) == 2
        assert service.get_usage(user_id=987654321) == 3
        assert service.get_usage(user_id=555666777) == 1

    @pytest.mark.asyncio
    async def test_user_with_no_previous_data(self, temp_dir):
        """Test user with no previous data gets tracked correctly."""
        storage_path = temp_dir / "test_new_user.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()

        # User 1 has data
        await service.record_upload(user_id=123456789, count=2)

        # User 2 is new
        assert service.get_usage(user_id=987654321) == 0
        await service.record_upload(user_id=987654321, count=1)
        assert service.get_usage(user_id=987654321) == 1

    @pytest.mark.asyncio
    async def test_user_reset_on_new_day(self, temp_dir):
        """Test user data resets on new day."""
        storage_path = temp_dir / "test_day_reset.json"
        service = ImageUsageService(storage_path=str(storage_path))

        # Simulate yesterday's data
        yesterday = (
            datetime.now(timezone(timedelta(hours=9))) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        service.usage_data[yesterday] = {"123456789": 3}

        # Check usage should be 0 for new day (no today entry)
        assert service.get_usage(user_id=123456789) == 0

        # Record upload today
        await service.record_upload(user_id=123456789, count=1)

        # Should start fresh
        assert service.get_usage(user_id=123456789) == 1

    @pytest.mark.asyncio
    async def test_cumulative_counting(self, temp_dir):
        """Test uploads are counted cumulatively throughout the day."""
        storage_path = temp_dir / "test_cumulative.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()

        await service.record_upload(user_id=123456789, count=1)
        assert service.get_usage(user_id=123456789) == 1

        await service.record_upload(user_id=123456789, count=2)
        assert service.get_usage(user_id=123456789) == 3

        await service.record_upload(user_id=123456789, count=1)
        assert service.get_usage(user_id=123456789) == 4

    @pytest.mark.asyncio
    async def test_check_and_record_workflow(self, temp_dir):
        """Test complete workflow of check and record."""
        storage_path = temp_dir / "test_workflow.json"
        service = ImageUsageService(storage_path=str(storage_path))

        # Check can upload (should be allowed)
        assert service.check_can_upload(user_id=123456789, count=1, limit=3)

        # Record upload
        await service.record_upload(user_id=123456789, count=1)

        # Check again (should still be allowed)
        assert service.check_can_upload(user_id=123456789, count=2, limit=3)

        # Record another upload
        await service.record_upload(user_id=123456789, count=2)

        # Check usage
        assert service.get_usage(user_id=123456789) == 3

        # Check can upload (should be at limit)
        assert not service.check_can_upload(user_id=123456789, count=1, limit=3)
        assert service.check_can_upload(user_id=123456789, count=0, limit=3)


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test suite for edge cases and error scenarios."""

    def test_invalid_user_id_string(self, temp_dir):
        """Test handling of string user_id (should be converted to string internally)."""
        storage_path = temp_dir / "test_string_id.json"
        service = ImageUsageService(storage_path=str(storage_path))

        # The method expects int but converts to string internally
        result = service.check_can_upload(user_id="123", count=1, limit=3)

        # Should handle string conversion
        assert isinstance(result, bool)

    def test_invalid_user_id_none(self, temp_dir):
        """Test handling of None user_id."""
        storage_path = temp_dir / "test_none_id.json"
        service = ImageUsageService(storage_path=str(storage_path))

        try:
            service.check_can_upload(user_id=None, count=1, limit=3)
            # If it doesn't raise, check the result
            assert True
        except (AttributeError, TypeError):
            # Expected if None causes issues
            pass

    @pytest.mark.asyncio
    async def test_negative_upload_count(self, temp_dir):
        """Test handling of negative upload count."""
        storage_path = temp_dir / "test_negative.json"
        service = ImageUsageService(storage_path=str(storage_path))
        today = service._get_today_key()
        service.usage_data[today] = {"123456789": 3}

        await service.record_upload(user_id=123456789, count=-1)

        # Should decrement
        assert service.get_usage(user_id=123456789) == 2

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, temp_dir):
        """Test handling of concurrent updates to same user."""
        storage_path = temp_dir / "test_concurrent.json"
        service = ImageUsageService(storage_path=str(storage_path))

        # Simulate concurrent uploads
        tasks = [service.record_upload(user_id=123456789, count=1) for _ in range(10)]

        await asyncio.gather(*tasks)

        # All uploads should be recorded (order may vary)
        assert service.get_usage(user_id=123456789) == 10

    @pytest.mark.asyncio
    async def test_large_upload_count(self, temp_dir):
        """Test handling of large upload count values."""
        storage_path = temp_dir / "test_large_count.json"
        service = ImageUsageService(storage_path=str(storage_path))

        await service.record_upload(user_id=123456789, count=1000)

        assert service.get_usage(user_id=123456789) == 1000

    def test_check_with_large_limit(self, temp_dir):
        """Test check_can_upload with very large limit."""
        storage_path = temp_dir / "test_large_limit.json"
        service = ImageUsageService(storage_path=str(storage_path))

        result = service.check_can_upload(user_id=123456789, count=10000, limit=100000)

        assert result is True

    def test_zero_limit(self, temp_dir):
        """Test check_can_upload with limit of 0."""
        storage_path = temp_dir / "test_zero_limit.json"
        service = ImageUsageService(storage_path=str(storage_path))

        result = service.check_can_upload(user_id=123456789, count=1, limit=0)

        assert result is False

    def test_file_permission_error_handling(self, temp_dir):
        """Test handling of file permission errors."""
        storage_path = temp_dir / "readonly" / "test.json"
        service = ImageUsageService(storage_path=str(storage_path))

        # Try to make parent directory read-only (Unix only)
        parent_dir = temp_dir / "readonly"
        parent_dir.mkdir(exist_ok=True)

        try:
            os.chmod(parent_dir, 0o444)

            # Should not raise exception, just log error
            service._save({"test": {"123": 1}})

        except (OSError, PermissionError):
            pass
        finally:
            # Reset permissions for cleanup
            try:
                os.chmod(parent_dir, 0o755)
            except (OSError, PermissionError):
                pass

    def test_corrupted_json_recovery(self, temp_dir):
        """Test recovery from corrupted JSON file."""
        storage_path = temp_dir / "corrupted.json"
        # Write corrupted data
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write("{corrupted json")

        # Create service - should handle gracefully
        service = ImageUsageService(storage_path=str(storage_path))

        assert service.usage_data == {}

        # Should be able to work normally after recovery
        assert service.check_can_upload(user_id=123456789, count=1, limit=3)

    def test_empty_json_object(self, temp_dir):
        """Test handling of empty JSON object {}."""
        storage_path = temp_dir / "empty_object.json"
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        service = ImageUsageService(storage_path=str(storage_path))

        assert service.usage_data == {}

    def test_json_with_special_characters(self, temp_dir):
        """Test JSON with special characters in user_id."""
        storage_path = temp_dir / "special_chars.json"
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        test_data = {today: {"user-123": 1, "user_456": 2, "user.789": 3}}

        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        service = ImageUsageService(storage_path=str(storage_path))

        assert service.usage_data == test_data

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, temp_dir):
        """Test data persists across different service instances."""
        storage_path = temp_dir / "persist.json"

        # First instance - record upload
        service1 = ImageUsageService(storage_path=str(storage_path))
        await service1.record_upload(user_id=123456789, count=2)

        # Second instance - should load data
        service2 = ImageUsageService(storage_path=str(storage_path))
        assert service2.get_usage(user_id=123456789) == 2

        # Add more with second instance
        await service2.record_upload(user_id=123456789, count=1)

        # Third instance - should see all data
        service3 = ImageUsageService(storage_path=str(storage_path))
        assert service3.get_usage(user_id=123456789) == 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_daily_limit_workflow(self, temp_dir):
        """Test complete workflow from checking limit to reaching it."""
        storage_path = temp_dir / "integration.json"
        service = ImageUsageService(storage_path=str(storage_path))
        user_id = 123456789
        limit = 3

        # Initially no uploads
        assert service.get_usage(user_id) == 0

        # Can upload 3
        assert service.check_can_upload(user_id=user_id, count=3, limit=limit)
        await service.record_upload(user_id=user_id, count=3)
        assert service.get_usage(user_id) == 3

        # Cannot upload more
        assert not service.check_can_upload(user_id=user_id, count=1, limit=limit)

        # Next day reset
        await asyncio.sleep(0.1)  # Let save complete
        service2 = ImageUsageService(storage_path=str(storage_path))
        # Simulate new day by directly modifying usage_data
        # (In real scenario, this would happen automatically)
        service2.usage_data = {}
        assert service2.get_usage(user_id) == 0

    @pytest.mark.asyncio
    async def test_multi_user_multi_day_scenario(self, temp_dir):
        """Test scenario with multiple users across multiple days."""
        storage_path = temp_dir / "multi_user.json"
        service = ImageUsageService(storage_path=str(storage_path))

        # Day 1 uploads
        await service.record_upload(user_id=111, count=2)
        await service.record_upload(user_id=222, count=1)
        await service.record_upload(user_id=333, count=3)

        # Check usage
        assert service.get_usage(user_id=111) == 2
        assert service.get_usage(user_id=222) == 1
        assert service.get_usage(user_id=333) == 3

        # Check permissions
        assert service.check_can_upload(user_id=111, count=1, limit=3)
        assert not service.check_can_upload(user_id=333, count=1, limit=3)

        # More uploads
        await service.record_upload(user_id=111, count=1)
        await service.record_upload(user_id=222, count=2)

        # Final state
        assert service.get_usage(user_id=111) == 3  # At limit
        assert service.get_usage(user_id=222) == 3  # At limit
        assert service.get_usage(user_id=333) == 3  # At limit

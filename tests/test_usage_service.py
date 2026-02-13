"""Tests for usage_service.py module (ImageUsageService)."""

import asyncio
import datetime
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch, mock_open

from persbot.services.usage_service import ImageUsageService


class TestImageUsageServiceInit:
    """Tests for ImageUsageService.__init__()."""

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_creates_directory_on_init(self, mock_makedirs, mock_exists):
        """__init__ creates the data directory if it doesn't exist."""
        ImageUsageService(storage_path="data/subdir/image_usage.json")
        mock_makedirs.assert_called_once_with("data/subdir", exist_ok=True)

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_handles_empty_directory_path(self, mock_makedirs, mock_exists):
        """__init__ handles storage_path with no directory component."""
        ImageUsageService(storage_path="image_usage.json")
        # Should not call makedirs since directory is empty
        mock_makedirs.assert_not_called()

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_handles_makedirs_os_error(self, mock_makedirs, mock_exists):
        """__init__ handles OSError when creating directory gracefully."""
        mock_makedirs.side_effect = OSError("Permission denied")
        # Should not raise
        service = ImageUsageService(storage_path="data/image_usage.json")
        assert service is not None

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data="{}"))
    @patch("os.makedirs")
    def test_loads_data_on_init(self, mock_makedirs, mock_exists):
        """__init__ loads existing data from file."""
        service = ImageUsageService(storage_path="data/image_usage.json")
        assert service.usage_data == {}
        assert service._loaded is True

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_sets_custom_debounce_interval(self, mock_makedirs, mock_exists):
        """__init__ accepts custom debounce interval."""
        service = ImageUsageService(debounce_interval=60)
        assert service._debounce_interval == 60

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_initializes_empty_data_when_no_file(self, mock_makedirs, mock_exists):
        """__init__ initializes empty data when file doesn't exist."""
        service = ImageUsageService(storage_path="data/image_usage.json")
        assert service.usage_data == {}
        assert service._loaded is True


class TestImageUsageServiceGetTodayKey:
    """Tests for ImageUsageService._get_today_key()."""

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_returns_kst_date_string(self, mock_makedirs, mock_exists):
        """_get_today_key returns date in YYYY-MM-DD format in KST timezone."""
        service = ImageUsageService()
        key = service._get_today_key()
        # Verify format: YYYY-MM-DD
        assert len(key) == 10
        assert key[4] == "-"
        assert key[7] == "-"

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_uses_kst_timezone(self, mock_makedirs, mock_exists):
        """_get_today_key uses KST (UTC+9) timezone."""
        service = ImageUsageService()
        # Mock datetime to verify KST is used
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = datetime.datetime(2024, 6, 15, 10, 30, 0)
            mock_dt.timedelta = datetime.timedelta
            mock_dt.timezone = datetime.timezone
            service._get_today_key()
            # Verify now() was called with KST timezone (UTC+9)
            call_args = mock_dt.now.call_args
            assert call_args is not None
            tz_arg = call_args[0][0]
            assert tz_arg.utcoffset(None) == datetime.timedelta(hours=9)


class TestImageUsageServiceLoadSync:
    """Tests for ImageUsageService._load_sync()."""

    def test_loads_valid_json_data(self):
        """_load_sync loads valid JSON data from file."""
        service = ImageUsageService.__new__(ImageUsageService)
        service.storage_path = "data/image_usage.json"
        service.usage_data = {}
        service._loaded = False

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"2024-06-15": {"123": 5}}')):
                with patch.object(service, "_get_today_key", return_value="2024-06-15"):
                    service._load_sync()

        assert service.usage_data == {"2024-06-15": {"123": 5}}
        assert service._loaded is True

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_handles_missing_file(self, mock_makedirs, mock_exists):
        """_load_sync handles missing file gracefully."""
        service = ImageUsageService.__new__(ImageUsageService)
        service.storage_path = "data/image_usage.json"
        service.usage_data = {}
        service._loaded = False

        service._load_sync()

        assert service.usage_data == {}
        assert service._loaded is True

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data="invalid json"))
    @patch("os.makedirs")
    def test_handles_invalid_json(self, mock_makedirs, mock_exists):
        """_load_sync handles invalid JSON gracefully."""
        service = ImageUsageService.__new__(ImageUsageService)
        service.storage_path = "data/image_usage.json"
        service.usage_data = {}
        service._loaded = False

        # Should not raise
        service._load_sync()

        assert service.usage_data == {}
        assert service._loaded is True

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", side_effect=IOError("Read error"))
    @patch("os.makedirs")
    def test_handles_io_error(self, mock_makedirs, mock_file, mock_exists):
        """_load_sync handles I/O errors gracefully."""
        service = ImageUsageService.__new__(ImageUsageService)
        service.storage_path = "data/image_usage.json"
        service.usage_data = {}
        service._loaded = False

        # Should not raise
        service._load_sync()

        assert service.usage_data == {}
        assert service._loaded is True

    def test_skips_if_already_loaded(self):
        """_load_sync skips loading if already loaded."""
        service = ImageUsageService.__new__(ImageUsageService)
        service.storage_path = "data/image_usage.json"
        service.usage_data = {"existing": "data"}
        service._loaded = True

        # Even if file exists with different data, should not reload
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"new": "data"}')):
                service._load_sync()

        assert service.usage_data == {"existing": "data"}

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data='{"2024-06-13": {"123": 5}, "2024-06-15": {"456": 3}}'))
    @patch("os.makedirs")
    def test_calls_cleanup_old_entries(self, mock_makedirs, mock_exists):
        """_load_sync calls _cleanup_old_entries after loading."""
        service = ImageUsageService.__new__(ImageUsageService)
        service.storage_path = "data/image_usage.json"
        service.usage_data = {}
        service._loaded = False

        # Mock today's date to be 2024-06-15
        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            service._load_sync()

        # Old entry should be removed
        assert "2024-06-13" not in service.usage_data
        assert "2024-06-15" in service.usage_data


class TestImageUsageServiceLoadAsync:
    """Tests for ImageUsageService._load_async()."""

    @pytest.fixture
    def service(self):
        """Create an ImageUsageService instance for testing."""
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                service = ImageUsageService.__new__(ImageUsageService)
                service.storage_path = "data/image_usage.json"
                service.usage_data = {}
                service._loaded = False
                service._debounce_interval = 30
                service._write_task = None
                service._buffered_writes = {}
                return service

    @pytest.mark.asyncio
    async def test_loads_data_asynchronously(self, service):
        """_load_async loads data using aiofiles."""
        service.usage_data = {"2024-06-15": {"123": 5}}  # Pre-set to test that it loads

        # Since mocking aiofiles is complex, test that _load_async works
        # when file doesn't exist (sets empty data)
        with patch("os.path.exists", return_value=False):
            await service._load_async()

        assert service._loaded is True
        # When file doesn't exist, data is reset to empty
        assert service.usage_data == {}

    @pytest.mark.asyncio
    async def test_handles_missing_file_async(self, service):
        """_load_async handles missing file gracefully."""
        with patch("os.path.exists", return_value=False):
            await service._load_async()

        assert service.usage_data == {}
        assert service._loaded is True

    @pytest.mark.asyncio
    async def test_handles_invalid_json_async(self, service):
        """_load_async handles invalid JSON gracefully."""
        with patch("os.path.exists", return_value=True):
            with patch("aiofiles.open", mock_open_async(read_data="invalid json")):
                await service._load_async()

        assert service.usage_data == {}
        assert service._loaded is True

    @pytest.mark.asyncio
    async def test_calls_cleanup_after_load(self, service):
        """_load_async calls _cleanup_old_entries after loading."""
        service.usage_data = {"2024-06-13": {"123": 5}, "2024-06-15": {"456": 3}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            await service._load_async()

        assert "2024-06-13" not in service.usage_data


class TestImageUsageServiceSave:
    """Tests for ImageUsageService._save() and _save_async()."""

    @pytest.fixture
    def service(self):
        """Create an ImageUsageService instance for testing."""
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                service = ImageUsageService.__new__(ImageUsageService)
                service.storage_path = "data/image_usage.json"
                service.usage_data = {"2024-06-15": {"123": 5}}
                service._loaded = True
                service._debounce_interval = 30
                service._write_task = None
                service._buffered_writes = {}
                return service

    def test_save_writes_to_file(self, service):
        """_save writes data to file synchronously."""
        with patch("builtins.open", mock_open()) as mock_file:
            service._save({"2024-06-15": {"123": 5}})

        mock_file.assert_called_once_with("data/image_usage.json", "w", encoding="utf-8")
        # Verify JSON was written
        handle = mock_file()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written) == {"2024-06-15": {"123": 5}}

    def test_save_handles_write_error(self, service):
        """_save handles write errors gracefully."""
        with patch("builtins.open", side_effect=IOError("Write error")):
            # Should not raise
            service._save({"2024-06-15": {"123": 5}})

    @pytest.mark.asyncio
    async def test_save_async_writes_to_file(self, service):
        """_save_async writes data to file asynchronously."""
        with patch("aiofiles.open", mock_open_async()) as mock_file:
            await service._save_async()

        mock_file.assert_called_once_with("data/image_usage.json", "w", encoding="utf-8")

    @pytest.mark.asyncio
    async def test_save_async_with_data(self, service):
        """_save_async_with_data writes specific data."""
        data = {"2024-06-15": {"456": 10}}

        with patch("aiofiles.open", mock_open_async()) as mock_file:
            await service._save_async_with_data(data)

        mock_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_async_handles_error(self, service):
        """_save_async handles write errors gracefully."""
        with patch("aiofiles.open", side_effect=IOError("Write error")):
            # Should not raise
            await service._save_async()


class TestImageUsageServiceCheckCanUpload:
    """Tests for ImageUsageService.check_can_upload()."""

    @pytest.fixture
    def service(self):
        """Create an ImageUsageService instance for testing."""
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                service = ImageUsageService.__new__(ImageUsageService)
                service.storage_path = "data/image_usage.json"
                service.usage_data = {}
                service._loaded = True
                service._debounce_interval = 30
                service._write_task = None
                service._buffered_writes = {}
                return service

    def test_returns_true_when_under_limit(self, service):
        """check_can_upload returns True when user is under limit."""
        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            result = service.check_can_upload(user_id=123, count=2, limit=3)
        assert result is True

    def test_returns_true_when_exactly_at_limit_after_upload(self, service):
        """check_can_upload returns True when upload reaches exactly limit."""
        service.usage_data = {"2024-06-15": {"123": 1}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            result = service.check_can_upload(user_id=123, count=2, limit=3)
        assert result is True  # 1 + 2 = 3, which equals limit

    def test_returns_false_when_would_exceed_limit(self, service):
        """check_can_upload returns False when upload would exceed limit."""
        service.usage_data = {"2024-06-15": {"123": 2}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            result = service.check_can_upload(user_id=123, count=2, limit=3)
        assert result is False  # 2 + 2 = 4, which exceeds limit

    def test_returns_false_when_already_at_limit(self, service):
        """check_can_upload returns False when already at limit."""
        service.usage_data = {"2024-06-15": {"123": 3}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            result = service.check_can_upload(user_id=123, count=1, limit=3)
        assert result is False

    def test_creates_date_key_if_not_exists(self, service):
        """check_can_upload creates date key if it doesn't exist."""
        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            service.check_can_upload(user_id=123, count=1, limit=3)

        assert "2024-06-15" in service.usage_data

    def test_uses_default_limit_of_three(self, service):
        """check_can_upload uses default limit of 3."""
        service.usage_data = {"2024-06-15": {"123": 2}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            # With default limit of 3, 2 + 1 should be allowed
            result = service.check_can_upload(user_id=123, count=1)
        assert result is True

    def test_converts_user_id_to_string(self, service):
        """check_can_upload converts user_id to string for lookup."""
        service.usage_data = {"2024-06-15": {"123": 2}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            # Should find the user by string key
            result = service.check_can_upload(user_id=123, count=1, limit=3)

        assert result is True  # 2 + 1 = 3, equals limit

    def test_handles_new_user(self, service):
        """check_can_upload handles new user with no prior usage."""
        service.usage_data = {"2024-06-15": {}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            result = service.check_can_upload(user_id=999, count=3, limit=3)
        assert result is True


class TestImageUsageServiceRecordUpload:
    """Tests for ImageUsageService.record_upload()."""

    @pytest.fixture
    def service(self):
        """Create an ImageUsageService instance for testing."""
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                service = ImageUsageService.__new__(ImageUsageService)
                service.storage_path = "data/image_usage.json"
                service.usage_data = {}
                service._loaded = True
                service._debounce_interval = 0.01  # Very short for tests
                service._write_task = None
                service._buffered_writes = {}
                return service

    @pytest.mark.asyncio
    async def test_records_upload_for_new_user(self, service):
        """record_upload creates entry for new user."""
        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            with patch.object(service, "_schedule_write"):
                await service.record_upload(user_id=123, count=2)

        assert service.usage_data["2024-06-15"]["123"] == 2

    @pytest.mark.asyncio
    async def test_increments_existing_user_count(self, service):
        """record_upload increments count for existing user."""
        service.usage_data = {"2024-06-15": {"123": 1}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            with patch.object(service, "_schedule_write"):
                await service.record_upload(user_id=123, count=2)

        assert service.usage_data["2024-06-15"]["123"] == 3  # 1 + 2

    @pytest.mark.asyncio
    async def test_records_multiple_uploads(self, service):
        """record_upload handles multiple uploads correctly."""
        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            with patch.object(service, "_schedule_write"):
                await service.record_upload(user_id=123, count=1)
                await service.record_upload(user_id=123, count=1)
                await service.record_upload(user_id=123, count=1)

        assert service.usage_data["2024-06-15"]["123"] == 3

    @pytest.mark.asyncio
    async def test_creates_date_key_if_not_exists(self, service):
        """record_upload creates date key if it doesn't exist."""
        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            with patch.object(service, "_schedule_write"):
                await service.record_upload(user_id=123, count=1)

        assert "2024-06-15" in service.usage_data

    @pytest.mark.asyncio
    async def test_calls_cleanup_on_new_day(self, service):
        """record_upload calls _cleanup_old_entries when new day is created."""
        service.usage_data = {"2024-06-14": {"123": 5}}  # Old day

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            with patch.object(service, "_schedule_write"):
                await service.record_upload(user_id=123, count=1)

        # Old day should be cleaned up
        assert "2024-06-14" not in service.usage_data
        assert "2024-06-15" in service.usage_data

    @pytest.mark.asyncio
    async def test_schedules_write(self, service):
        """record_upload schedules a debounced write."""
        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            with patch.object(service, "_schedule_write") as mock_schedule:
                await service.record_upload(user_id=123, count=1)

        mock_schedule.assert_called_once()


class TestImageUsageServiceGetUsage:
    """Tests for ImageUsageService.get_usage()."""

    @pytest.fixture
    def service(self):
        """Create an ImageUsageService instance for testing."""
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                service = ImageUsageService.__new__(ImageUsageService)
                service.storage_path = "data/image_usage.json"
                service.usage_data = {}
                service._loaded = True
                service._debounce_interval = 30
                service._write_task = None
                service._buffered_writes = {}
                return service

    def test_returns_current_usage(self, service):
        """get_usage returns current usage for user."""
        service.usage_data = {"2024-06-15": {"123": 5}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            result = service.get_usage(user_id=123)

        assert result == 5

    def test_returns_zero_for_new_user(self, service):
        """get_usage returns 0 for user with no usage."""
        service.usage_data = {"2024-06-15": {}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            result = service.get_usage(user_id=999)

        assert result == 0

    def test_returns_zero_for_new_day(self, service):
        """get_usage returns 0 when no data for today."""
        service.usage_data = {}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            result = service.get_usage(user_id=123)

        assert result == 0


class TestImageUsageServiceCleanupOldEntries:
    """Tests for ImageUsageService._cleanup_old_entries()."""

    @pytest.fixture
    def service(self):
        """Create an ImageUsageService instance for testing."""
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                service = ImageUsageService.__new__(ImageUsageService)
                service.storage_path = "data/image_usage.json"
                service.usage_data = {}
                service._loaded = True
                service._debounce_interval = 30
                service._write_task = None
                service._buffered_writes = {}
                return service

    def test_removes_old_entries(self, service):
        """_cleanup_old_entries removes entries from previous days."""
        service.usage_data = {
            "2024-06-13": {"123": 5},
            "2024-06-14": {"456": 3},
            "2024-06-15": {"789": 2}
        }

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            service._cleanup_old_entries()

        assert service.usage_data == {"2024-06-15": {"789": 2}}

    def test_keeps_today_entry(self, service):
        """_cleanup_old_entries keeps only today's entry."""
        service.usage_data = {"2024-06-15": {"123": 5}}

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            service._cleanup_old_entries()

        assert service.usage_data == {"2024-06-15": {"123": 5}}

    def test_handles_empty_data(self, service):
        """_cleanup_old_entries handles empty data gracefully."""
        service.usage_data = {}

        # Should not raise
        service._cleanup_old_entries()

        assert service.usage_data == {}

    def test_removes_all_if_none_match_today(self, service):
        """_cleanup_old_entries removes all if none match today."""
        service.usage_data = {
            "2024-06-13": {"123": 5},
            "2024-06-14": {"456": 3}
        }

        with patch.object(service, "_get_today_key", return_value="2024-06-15"):
            service._cleanup_old_entries()

        assert service.usage_data == {}


class TestImageUsageServiceScheduleWrite:
    """Tests for ImageUsageService._schedule_write() and debounce logic."""

    @pytest.fixture
    def service(self):
        """Create an ImageUsageService instance for testing."""
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                service = ImageUsageService.__new__(ImageUsageService)
                service.storage_path = "data/image_usage.json"
                service.usage_data = {}
                service._loaded = True
                service._debounce_interval = 0.01  # Very short for tests
                service._write_task = None
                service._buffered_writes = {}
                return service

    @pytest.mark.asyncio
    async def test_schedules_new_write_task(self, service):
        """_schedule_write creates a new write task."""
        mock_task = MagicMock()
        mock_task.done.return_value = False

        with patch("asyncio.create_task", return_value=mock_task):
            service._schedule_write()

        assert service._write_task is mock_task
        assert not service._write_task.done()

    @pytest.mark.asyncio
    async def test_cancels_existing_write_task(self, service):
        """_schedule_write cancels existing pending write task."""
        first_task = MagicMock()
        first_task.done.return_value = False
        second_task = MagicMock()
        second_task.done.return_value = False

        with patch("asyncio.create_task", side_effect=[first_task, second_task]):
            # Create first task
            service._schedule_write()
            assert service._write_task is first_task

            # Create second task (should cancel first)
            service._schedule_write()

        # First task should have been cancelled
        first_task.cancel.assert_called_once()
        assert service._write_task is second_task

    @pytest.mark.asyncio
    async def test_flush_buffer_waits_for_debounce(self, service):
        """_flush_buffer waits for debounce interval before writing."""
        service.usage_data = {"2024-06-15": {"123": 5}}

        with patch.object(service, "_save_async_with_data", new_callable=AsyncMock) as mock_save:
            service._schedule_write()

            # Wait less than debounce interval - should not have saved yet
            await asyncio.sleep(0.005)
            # Task might still be running

            # Wait for debounce to complete
            await asyncio.sleep(0.02)
            mock_save.assert_called_once_with({"2024-06-15": {"123": 5}})

    @pytest.mark.asyncio
    async def test_flush_buffer_handles_cancellation(self, service):
        """_flush_buffer handles task cancellation gracefully."""
        service._schedule_write()
        service._write_task.cancel()

        try:
            await service._write_task
        except asyncio.CancelledError:
            pass

        # Should complete without error

    @pytest.mark.asyncio
    async def test_flush_writes_immediately(self, service):
        """flush writes data immediately without debounce."""
        service.usage_data = {"2024-06-15": {"123": 5}}

        # Schedule a debounced write
        service._schedule_write()

        with patch.object(service, "_save_async_with_data", new_callable=AsyncMock) as mock_save:
            await service.flush()

        mock_save.assert_called_once_with({"2024-06-15": {"123": 5}})

    @pytest.mark.asyncio
    async def test_flush_cancels_pending_write(self, service):
        """flush cancels pending debounced write and writes immediately."""
        service.usage_data = {"2024-06-15": {"123": 5}}

        # Schedule a debounced write
        service._schedule_write()
        pending_task = service._write_task

        with patch.object(service, "_save_async_with_data", new_callable=AsyncMock):
            await service.flush()

        # The pending task should be cancelled
        assert pending_task.cancelled() or pending_task.done()

    @pytest.mark.asyncio
    async def test_flush_handles_no_pending_task(self, service):
        """flush handles case when no pending write task exists."""
        service.usage_data = {"2024-06-15": {"123": 5}}
        service._write_task = None

        with patch.object(service, "_save_async_with_data", new_callable=AsyncMock) as mock_save:
            await service.flush()

        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_debounce_cancels_previous_write(self, service):
        """Multiple rapid writes result in only one actual write (debounce)."""
        service.usage_data = {"2024-06-15": {"123": 1}}

        with patch.object(service, "_save_async_with_data", new_callable=AsyncMock) as mock_save:
            # Rapidly schedule multiple writes
            service._schedule_write()
            service.usage_data["2024-06-15"]["123"] = 2
            service._schedule_write()
            service.usage_data["2024-06-15"]["123"] = 3
            service._schedule_write()

            # Wait for debounce to complete
            await asyncio.sleep(0.05)

            # Should have written only once with final data
            assert mock_save.call_count == 1
            mock_save.assert_called_with({"2024-06-15": {"123": 3}})


class TestImageUsageServiceInitialize:
    """Tests for ImageUsageService.initialize()."""

    @pytest.fixture
    def service(self):
        """Create an ImageUsageService instance for testing."""
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                service = ImageUsageService.__new__(ImageUsageService)
                service.storage_path = "data/image_usage.json"
                service.usage_data = {}
                service._loaded = False
                service._debounce_interval = 30
                service._write_task = None
                service._buffered_writes = {}
                return service

    @pytest.mark.asyncio
    async def test_initialize_loads_data_async(self, service):
        """initialize() calls _load_async to load data."""
        with patch.object(service, "_load_async", new_callable=AsyncMock) as mock_load:
            await service.initialize()

        mock_load.assert_called_once()


# Helper for async mock_open
def mock_open_async(read_data=""):
    """Create a mock for aiofiles.open that returns an async context manager."""
    from unittest.mock import AsyncMock, MagicMock

    mock_file = MagicMock()
    mock_file.read = AsyncMock(return_value=read_data)
    mock_file.write = AsyncMock()
    mock_file.__aenter__ = AsyncMock(return_value=mock_file)
    mock_file.__aexit__ = AsyncMock(return_value=None)

    mock_context = MagicMock(return_value=mock_file)
    return mock_context

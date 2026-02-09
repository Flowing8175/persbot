import asyncio
import datetime
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ImageUsageService:
    """Tracks daily image upload usage per user."""

    def __init__(self, storage_path: str = "data/image_usage.json"):
        self.storage_path = storage_path
        self._ensure_data_dir()
        self.usage_data: Dict[str, Dict[str, int]] = {}
        self._load()

        # In-memory buffer for debounced writes
        self._buffered_writes: Dict[str, Any] = {}  # (data_snapshot, timestamp)
        self._debounce_interval = 30  # seconds
        self._write_task: Optional[asyncio.Task] = None

    def _ensure_data_dir(self):
        directory = os.path.dirname(self.storage_path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory {directory}: {e}")

    def _get_today_key(self) -> str:
        """Returns the current date string (YYYY-MM-DD) in KST (UTC+9)."""
        kst = datetime.timezone(datetime.timedelta(hours=9))
        return datetime.datetime.now(kst).strftime("%Y-%m-%d")

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.usage_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load image usage data: {e}")
                self.usage_data = {}

        # Cleanup old data (optional, to prevent file growing indefinitely)
        # We can keep only today's data or last few days.
        # For simplicity, let's keep it simple for now, maybe cleanup on save.
        self._cleanup_old_entries()

    def _save(self, data: Dict[str, Dict[str, int]]):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save image usage data: {e}")

    async def _save_async(self):
        # Create a copy of the data in the main thread to avoid race conditions
        data_snapshot = self.usage_data.copy()
        await asyncio.to_thread(self._save, data_snapshot)

    def _schedule_write(self):
        """Schedule a debounced write operation."""
        # Cancel any existing write task
        if self._write_task and not self._write_task.done():
            self._write_task.cancel()

        # Schedule new write with debounce
        self._write_task = asyncio.create_task(self._flush_buffer())

    async def _flush_buffer(self):
        """Flush buffered changes to disk with debounce."""
        try:
            # Wait for debounce interval
            await asyncio.sleep(self._debounce_interval)

            # Write only if no new changes occurred
            if self._buffered_writes:
                data_snapshot = self._buffered_writes.pop("data_snapshot")
                await asyncio.to_thread(self._save, data_snapshot)
        except asyncio.CancelledError:
            # Task cancelled, ignore
            pass
        except Exception as e:
            logger.error(f"Error during flush buffer: {e}", exc_info=True)

    def _cleanup_old_entries(self):
        today = self._get_today_key()
        keys_to_remove = [k for k in self.usage_data.keys() if k != today]
        if keys_to_remove:
            for k in keys_to_remove:
                del self.usage_data[k]
            # We don't save here to avoid I/O in load/init, but it will be saved next time record is called.

    def check_can_upload(self, user_id: int, count: int, limit: int = 3) -> bool:
        """Check if user can upload 'count' more images without exceeding 'limit'."""
        today = self._get_today_key()
        if today not in self.usage_data:
            self.usage_data[today] = {}

        user_key = str(user_id)
        current_usage = self.usage_data[today].get(user_key, 0)

        return (current_usage + count) <= limit

    async def record_upload(self, user_id: int, count: int):
        """Record an upload of 'count' images for the user."""
        today = self._get_today_key()
        if today not in self.usage_data:
            self.usage_data[today] = {}
            # New day, might as well clean up old keys
            self._cleanup_old_entries()

        user_key = str(user_id)
        current_usage = self.usage_data[today].get(user_key, 0)
        self.usage_data[today][user_key] = current_usage + count

        # Schedule debounced write
        self._schedule_write()

    def get_usage(self, user_id: int) -> int:
        """Get current daily usage for user."""
        today = self._get_today_key()
        if today not in self.usage_data:
            return 0
        return self.usage_data[today].get(str(user_id), 0)

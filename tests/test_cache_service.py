"""Feature tests for cache service.

Tests focus on behavior:
- CacheEntry: cache entry dataclass
- CacheResult: cache operation result
- CacheStatus: cache status enumeration
- InMemoryCacheStrategy: in-memory caching strategy
- CacheService: unified cache service
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch

import pytest

from persbot.services.cache_service import (
    CacheStatus,
    CacheEntry,
    CacheResult,
    CacheStrategy,
    InMemoryCacheStrategy,
    CacheService,
)


class TestCacheStatus:
    """Tests for CacheStatus enumeration."""

    def test_has_active_status(self):
        """CacheStatus includes ACTIVE."""
        assert CacheStatus.ACTIVE.value == "active"

    def test_has_expired_status(self):
        """CacheStatus includes EXPIRED."""
        assert CacheStatus.EXPIRED.value == "expired"

    def test_has_refreshing_status(self):
        """CacheStatus includes REFRESHING."""
        assert CacheStatus.REFRESHING.value == "refreshing"

    def test_has_creating_status(self):
        """CacheStatus includes CREATING."""
        assert CacheStatus.CREATING.value == "creating"

    def test_has_error_status(self):
        """CacheStatus includes ERROR."""
        assert CacheStatus.ERROR.value == "error"


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creates_with_required_fields(self):
        """CacheEntry creates with required fields."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test Cache",
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
            created_at=datetime.now(timezone.utc),
        )
        assert entry.name == "cache-123"
        assert entry.display_name == "Test Cache"
        assert entry.model == "gemini-2.5-flash"
        assert entry.system_instruction == "You are helpful"

    def test_default_expires_at_is_none(self):
        """expires_at defaults to None."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
        )
        assert entry.expires_at is None

    def test_default_token_count_is_zero(self):
        """token_count defaults to 0."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
        )
        assert entry.token_count == 0

    def test_default_status_is_active(self):
        """status defaults to ACTIVE."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
        )
        assert entry.status == CacheStatus.ACTIVE

    def test_default_tools_is_empty_list(self):
        """tools defaults to empty list."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
        )
        assert entry.tools == []

    def test_default_metadata_is_empty_dict(self):
        """metadata defaults to empty dict."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
        )
        assert entry.metadata == {}

    def test_is_expired_returns_false_when_no_expiry(self):
        """is_expired returns False when expires_at is None."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
        )
        assert entry.is_expired is False

    def test_is_expired_returns_false_for_future_expiry(self):
        """is_expired returns False when not yet expired."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert entry.is_expired is False

    def test_is_expired_returns_true_for_past_expiry(self):
        """is_expired returns True when expired."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert entry.is_expired is True

    def test_needs_refresh_returns_false_when_no_expiry(self):
        """needs_refresh returns False when expires_at is None."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
        )
        assert entry.needs_refresh is False

    def test_needs_refresh_returns_true_when_close_to_expiry(self):
        """needs_refresh returns True when within buffer period."""
        # Buffer is 5 minutes by default, so 2 minutes should trigger refresh
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
        assert entry.needs_refresh is True

    def test_age_seconds_returns_positive_value(self):
        """age_seconds returns positive value."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        assert entry.age_seconds >= 10


class TestCacheResult:
    """Tests for CacheResult dataclass."""

    def test_creates_successful_result(self):
        """CacheResult creates successful result."""
        result = CacheResult(success=True)
        assert result.success is True
        assert result.cache_name is None
        assert result.entry is None
        assert result.error is None
        assert result.created is False

    def test_creates_result_with_cache_name(self):
        """CacheResult creates with cache name."""
        result = CacheResult(success=True, cache_name="cache-123")
        assert result.cache_name == "cache-123"

    def test_creates_result_with_entry(self):
        """CacheResult creates with entry."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="gemini",
            system_instruction="test",
            created_at=datetime.now(timezone.utc),
        )
        result = CacheResult(success=True, entry=entry)
        assert result.entry == entry

    def test_creates_failed_result(self):
        """CacheResult creates failed result."""
        result = CacheResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_created_flag_defaults_to_false(self):
        """created defaults to False."""
        result = CacheResult(success=True)
        assert result.created is False

    def test_created_flag_can_be_set(self):
        """created can be set to True."""
        result = CacheResult(success=True, created=True)
        assert result.created is True


class TestInMemoryCacheStrategy:
    """Tests for InMemoryCacheStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create an InMemoryCacheStrategy."""
        return InMemoryCacheStrategy()

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new_entry(self, strategy):
        """get_or_create creates new entry when not exists."""
        result = await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        assert result.success is True
        assert result.created is True
        assert result.entry is not None
        assert result.entry.model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing_entry(self, strategy):
        """get_or_create returns existing entry when available."""
        # Create first
        result1 = await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        # Get existing
        result2 = await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        assert result2.created is False
        assert result2.cache_name == result1.cache_name

    @pytest.mark.asyncio
    async def test_get_or_create_different_content_creates_new(self, strategy):
        """get_or_create creates new entry for different content."""
        await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        result = await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are different",
        )

        assert result.created is True

    @pytest.mark.asyncio
    async def test_refresh_updates_ttl(self, strategy):
        """refresh updates the entry's TTL."""
        result = await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        success = await strategy.refresh(result.cache_name, ttl_seconds=3600)

        assert success is True

    @pytest.mark.asyncio
    async def test_refresh_returns_false_for_unknown(self, strategy):
        """refresh returns False for unknown cache."""
        success = await strategy.refresh("unknown-cache", ttl_seconds=3600)
        assert success is False

    @pytest.mark.asyncio
    async def test_delete_removes_entry(self, strategy):
        """delete removes the cache entry."""
        result = await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        success = await strategy.delete(result.cache_name)
        assert success is True

        # Verify deleted
        entries = await strategy.list_all()
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_unknown(self, strategy):
        """delete returns False for unknown cache."""
        success = await strategy.delete("unknown-cache")
        assert success is False

    @pytest.mark.asyncio
    async def test_list_all_returns_entries(self, strategy):
        """list_all returns all cache entries."""
        await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="First",
        )
        await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="Second",
        )

        entries = await strategy.list_all()

        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_list_all_respects_limit(self, strategy):
        """list_all respects the limit parameter."""
        for i in range(5):
            await strategy.get_or_create(
                model="gemini-2.5-flash",
                system_instruction=f"Instruction {i}",
            )

        entries = await strategy.list_all(limit=2)

        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_expired(self, strategy):
        """cleanup_expired removes expired entries."""
        # Create entry and manually expire it
        result = await strategy.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        # Manually expire by setting expires_at to past
        key = strategy._make_key("gemini-2.5-flash", "You are helpful", None)
        strategy._caches[key].expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        removed = await strategy.cleanup_expired()

        assert removed == 1
        assert len(await strategy.list_all()) == 0


class TestCacheService:
    """Tests for CacheService."""

    @pytest.fixture
    def service(self):
        """Create a CacheService with InMemoryCacheStrategy."""
        return CacheService(strategy=InMemoryCacheStrategy())

    @pytest.mark.asyncio
    async def test_get_or_create_delegates_to_strategy(self, service):
        """get_or_create delegates to strategy."""
        result = await service.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        assert result.success is True
        assert result.created is True

    @pytest.mark.asyncio
    async def test_refresh_delegates_to_strategy(self, service):
        """refresh delegates to strategy."""
        result = await service.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        success = await service.refresh(result.cache_name, ttl_seconds=3600)

        assert success is True

    @pytest.mark.asyncio
    async def test_delete_delegates_to_strategy(self, service):
        """delete delegates to strategy."""
        result = await service.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        success = await service.delete(result.cache_name)

        assert success is True

    @pytest.mark.asyncio
    async def test_list_all_delegates_to_strategy(self, service):
        """list_all delegates to strategy."""
        await service.get_or_create(
            model="gemini-2.5-flash",
            system_instruction="You are helpful",
        )

        entries = await service.list_all()

        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_delegates_to_strategy(self, service):
        """cleanup_expired delegates to strategy."""
        removed = await service.cleanup_expired()
        assert removed == 0

    @pytest.mark.asyncio
    async def test_context_manager_starts_cleanup(self, service):
        """Context manager starts periodic cleanup."""
        async with service as s:
            assert s._running is True
            assert s._cleanup_task is not None

    @pytest.mark.asyncio
    async def test_context_manager_stops_cleanup(self, service):
        """Context manager stops periodic cleanup on exit."""
        async with service as s:
            pass

        assert service._running is False

    @pytest.mark.asyncio
    async def test_start_periodic_cleanup_starts_task(self, service):
        """start_periodic_cleanup starts the cleanup task."""
        await service.start_periodic_cleanup(interval_seconds=60)

        assert service._running is True
        assert service._cleanup_task is not None

        await service.stop_periodic_cleanup()

    @pytest.mark.asyncio
    async def test_stop_periodic_cleanup_stops_task(self, service):
        """stop_periodic_cleanup stops the cleanup task."""
        await service.start_periodic_cleanup(interval_seconds=60)
        await service.stop_periodic_cleanup()

        assert service._running is False
        assert service._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self, service):
        """Starting cleanup twice is safe."""
        await service.start_periodic_cleanup(interval_seconds=60)
        await service.start_periodic_cleanup(interval_seconds=60)  # Should not raise

        await service.stop_periodic_cleanup()

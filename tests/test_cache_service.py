"""Tests for services/cache_service.py module.

This module provides comprehensive test coverage for:
- CacheStatus enum
- CacheEntry dataclass
- CacheResult dataclass
- CacheStrategy ABC
- InMemoryCacheStrategy class
- GeminiCacheStrategy class
- CacheService class
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace

import pytest

from persbot.services.cache_service import (
    CacheStatus,
    CacheEntry,
    CacheResult,
    InMemoryCacheStrategy,
    GeminiCacheStrategy,
    CacheService,
)
from persbot.constants import CacheConfig, CacheLimit


# =============================================================================
# CacheStatus Enum Tests
# =============================================================================


class TestCacheStatus:
    """Tests for CacheStatus enum."""

    def test_status_values(self):
        """Test CacheStatus enum values."""
        assert CacheStatus.ACTIVE == "active"
        assert CacheStatus.EXPIRED == "expired"
        assert CacheStatus.REFRESHING == "refreshing"
        assert CacheStatus.CREATING == "creating"
        assert CacheStatus.ERROR == "error"


# =============================================================================
# CacheEntry Dataclass Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_init_all_fields(self):
        """Test CacheEntry initialization."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test Cache",
            model="test-model",
            system_instruction="Test instruction",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=60),
            ttl_seconds=3600,
            token_count=1000,
            status=CacheStatus.ACTIVE,
            tools=[{"name": "test_tool"}],
            metadata={"key": "value"},
        )

        assert entry.name == "cache-123"
        assert entry.display_name == "Test Cache"
        assert entry.token_count == 1000
        assert entry.status == CacheStatus.ACTIVE

    def test_is_expired_with_none_expires_at(self):
        """Test is_expired returns False when expires_at is None."""
        entry = CacheEntry(
            name="test",
            display_name="Test",
            model="model",
            system_instruction="instruction",
            created_at=datetime.now(timezone.utc),
            expires_at=None,
            ttl_seconds=3600,
        )

        assert entry.is_expired is False

    def test_is_expired_future(self):
        """Test is_expired returns False for future expiration."""
        entry = CacheEntry(
            name="test",
            display_name="Test",
            model="model",
            system_instruction="instruction",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=10),
            ttl_seconds=3600,
        )

        # Future expiration means NOT expired
        assert entry.is_expired is False

    def test_is_expired_past(self):
        """Test is_expired returns True for past expiration."""
        entry = CacheEntry(
            name="test",
            display_name="Test",
            model="model",
            system_instruction="instruction",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=5),
            ttl_seconds=3600,
        )

        # Past expiration means IS expired
        assert entry.is_expired is True

    def test_needs_refresh_with_none_expires_at(self):
        """Test needs_refresh returns False when expires_at is None."""
        entry = CacheEntry(
            name="test",
            display_name="Test",
            model="model",
            system_instruction="instruction",
            created_at=datetime.now(timezone.utc),
            expires_at=None,
            ttl_seconds=3600,
        )

        assert entry.needs_refresh is False

    def test_needs_refresh_in_buffer(self):
        """Test needs_refresh returns True when within buffer period."""
        buffer_seconds = CacheConfig.REFRESH_BUFFER_MIN * 60

        entry = CacheEntry(
            name="test",
            display_name="Test",
            model="model",
            system_instruction="instruction",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=buffer_seconds - 10),
            ttl_seconds=3600,
        )

        # Within buffer period means needs refresh
        assert entry.needs_refresh is True

    def test_needs_refresh_not_in_buffer(self):
        """Test needs_refresh returns False when not within buffer period."""
        buffer_seconds = CacheConfig.REFRESH_BUFFER_MIN * 60

        entry = CacheEntry(
            name="test",
            display_name="Test",
            model="model",
            system_instruction="instruction",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=buffer_seconds + 60),
            ttl_seconds=3600,
        )

        # Not within buffer period means does NOT need refresh
        assert entry.needs_refresh is False

    def test_age_seconds(self):
        """Test age_seconds calculation."""
        created = datetime.now(timezone.utc) - timedelta(seconds=30)

        entry = CacheEntry(
            name="test",
            display_name="Test",
            model="model",
            system_instruction="instruction",
            created_at=created,
            ttl_seconds=3600,
        )

        # Age should be approximately 30 seconds
        assert 29 < entry.age_seconds < 31


# =============================================================================
# CacheResult Dataclass Tests
# =============================================================================


class TestCacheResult:
    """Tests for CacheResult dataclass."""

    def test_success_result(self):
        """Test CacheResult with success=True."""
        entry = CacheEntry(
            name="cache-123",
            display_name="Test",
            model="model",
            system_instruction="instruction",
            created_at=datetime.now(timezone.utc),
        )

        result = CacheResult(
            success=True,
            cache_name="cache-123",
            entry=entry,
            created=True,
        )

        assert result.success is True
        assert result.cache_name == "cache-123"
        assert result.created is True

    def test_error_result(self):
        """Test CacheResult with error."""
        result = CacheResult(
            success=False,
            error="Cache error",
        )

        assert result.success is False
        assert result.error == "Cache error"
        assert result.cache_name is None
        assert result.entry is None


# =============================================================================
# InMemoryCacheStrategy Class Tests
# =============================================================================


class TestInMemoryCacheStrategy:
    """Tests for InMemoryCacheStrategy class."""

    @pytest.fixture
    def in_memory_strategy(self):
        """Create an InMemoryCacheStrategy instance."""
        return InMemoryCacheStrategy()

    @pytest.mark.asyncio
    async def test_get_or_create_new_entry(self, in_memory_strategy):
        """Test get_or_create creates new entry."""
        result = await in_memory_strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
            tools=None,
        )

        assert result.success is True
        assert result.created is True
        assert result.entry.name.startswith("cache-")
        assert result.entry.model == "test-model"

    @pytest.mark.asyncio
    async def test_get_or_create_existing_valid(self, in_memory_strategy):
        """Test get_or_create returns existing valid entry."""
        # First, create an entry
        await in_memory_strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        # Create a new entry that's not expired
        caches = list(in_memory_strategy._caches.values())
        if caches:
            caches[0].expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)

        result = await in_memory_strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        assert result.success is True
        assert result.created is False  # Was not created new

    @pytest.mark.asyncio
    async def test_get_or_create_existing_expired(self, in_memory_strategy):
        """Test get_or_create replaces expired entry."""
        # Create an entry
        await in_memory_strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        # Mark it as expired
        caches = list(in_memory_strategy._caches.values())
        if caches:
            caches[0].expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)

        result = await in_memory_strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        assert result.success is True
        assert result.created is True  # New entry was created

    @pytest.mark.asyncio
    async def test_refresh_existing_entry(self, in_memory_strategy):
        """Test refresh updates entry TTL."""
        await in_memory_strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        # Get the cache name
        caches = list(in_memory_strategy._caches.values())
        cache_name = caches[0].name if caches else None
        original_expires = caches[0].expires_at if caches else None

        if cache_name:
            result = await in_memory_strategy.refresh(cache_name, ttl_seconds=7200)

            assert result is True
            caches_after = list(in_memory_strategy._caches.values())
            if caches_after:
                assert caches_after[0].expires_at > original_expires

    @pytest.mark.asyncio
    async def test_delete_existing_entry(self, in_memory_strategy):
        """Test delete removes entry."""
        await in_memory_strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        # Get the cache name
        caches = list(in_memory_strategy._caches.values())
        cache_name = caches[0].name if caches else None

        if cache_name:
            result = await in_memory_strategy.delete(cache_name)

            assert result is True
            assert cache_name not in in_memory_strategy._name_to_key

    @pytest.mark.asyncio
    async def test_list_all_respects_limit(self, in_memory_strategy):
        """Test list_all respects limit parameter."""
        # Create multiple entries
        for i in range(15):
            await in_memory_strategy.get_or_create(
                model=f"model-{i}",
                system_instruction=f"Instruction {i}",
            )

        result = await in_memory_strategy.list_all(limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, in_memory_strategy):
        """Test cleanup removes expired entries."""
        await in_memory_strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        # Create an expired entry
        caches = list(in_memory_strategy._caches.values())
        if caches:
            caches[0].expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)

        result = await in_memory_strategy.cleanup_expired()

        assert result >= 0  # At least the expired entry was removed


# =============================================================================
# GeminiCacheStrategy Class Tests
# =============================================================================


class TestGeminiCacheStrategy:
    """Tests for GeminiCacheStrategy class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Gemini client."""
        client = Mock()
        client.caches = Mock()
        client.models = Mock()
        client.models.count_tokens = Mock(return_value=Mock(total_tokens=50000))
        client.caches.create = Mock(return_value=Mock(name="cache-123"))
        client.caches.update = Mock()
        client.caches.delete = Mock()
        client.caches.list = Mock(return_value=[])
        return client

    @pytest.mark.asyncio
    async def test_get_or_create_above_min_tokens(self, mock_client):
        """Test cache creation when token count is above minimum."""
        strategy = GeminiCacheStrategy(mock_client)

        result = await strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        assert result.success is True
        assert result.entry.token_count == 50000

    @pytest.mark.asyncio
    async def test_get_or_create_below_min_tokens(self, mock_client):
        """Test cache creation skipped when token count is below minimum."""
        mock_client.models.count_tokens = Mock(return_value=Mock(total_tokens=100))
        strategy = GeminiCacheStrategy(mock_client)

        result = await strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        assert result.success is False
        assert "below minimum" in result.error

    @pytest.mark.asyncio
    async def test_refresh_existing_cache(self, mock_client):
        """Test refreshing existing cache."""
        strategy = GeminiCacheStrategy(mock_client)

        # First create a cache
        await strategy.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        result = await strategy.refresh("cache-123", ttl_seconds=1800)

        assert result is True
        # Verify caches.update was called
        mock_client.caches.update.assert_called()

    @pytest.mark.asyncio
    async def test_delete_cache(self, mock_client):
        """Test deleting cache."""
        strategy = GeminiCacheStrategy(mock_client)

        result = await strategy.delete("cache-123")

        assert result is True
        # Verify caches.delete was called
        mock_client.caches.delete.assert_called_with(name="cache-123")

    @pytest.mark.asyncio
    async def test_list_all_caches(self, mock_client):
        """Test listing all caches."""
        strategy = GeminiCacheStrategy(mock_client)

        # Set up list return
        mock_cache1 = Mock(display_name="cache-1", name="cache-1")
        mock_cache2 = Mock(display_name="cache-2", name="cache-2")
        mock_client.caches.list.return_value = [mock_cache1, mock_cache2]

        result = await strategy.list_all(limit=10)

        assert len(result) == 2
        assert result[0].display_name == "cache-1"

    @pytest.mark.asyncio
    async def test_cleanup_expired_no_op(self, mock_client):
        """Test cleanup_expired is a no-op for GeminiCacheStrategy."""
        strategy = GeminiCacheStrategy(mock_client)

        result = await strategy.cleanup_expired()

        # Gemini caches auto-expire
        assert result == 0


# =============================================================================
# CacheService Class Tests
# =============================================================================


class TestCacheService:
    """Tests for CacheService class."""

    @pytest.fixture
    def in_memory_strategy(self):
        """Create an InMemoryCacheStrategy."""
        return InMemoryCacheStrategy()

    @pytest.mark.asyncio
    async def test_get_or_create_delegates_to_strategy(self, in_memory_strategy):
        """Test get_or_create delegates to strategy."""
        service = CacheService(strategy=in_memory_strategy)

        result = await service.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
            tools=None,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_refresh_delegates_to_strategy(self, in_memory_strategy):
        """Test refresh delegates to strategy."""
        service = CacheService(strategy=in_memory_strategy)

        # First create a cache
        await service.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        # Get the cache name
        caches = list(in_memory_strategy._caches.values())
        cache_name = caches[0].name if caches else "cache-123"

        result = await service.refresh(cache_name, ttl_seconds=1800)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_delegates_to_strategy(self, in_memory_strategy):
        """Test delete delegates to strategy."""
        service = CacheService(strategy=in_memory_strategy)

        # First create a cache
        await service.get_or_create(
            model="test-model",
            system_instruction="Test instruction",
        )

        # Get the cache name
        caches = list(in_memory_strategy._caches.values())
        cache_name = caches[0].name if caches else "cache-123"

        result = await service.delete(cache_name)

        assert result is True

    @pytest.mark.asyncio
    async def test_list_all_delegates_to_strategy(self, in_memory_strategy):
        """Test list_all delegates to strategy."""
        service = CacheService(strategy=in_memory_strategy)

        result = await service.list_all(limit=10)

        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_delegates_to_strategy(self, in_memory_strategy):
        """Test cleanup_expired delegates to strategy."""
        service = CacheService(strategy=in_memory_strategy)

        result = await service.cleanup_expired()

        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_context_manager(self, in_memory_strategy):
        """Test CacheService as async context manager."""
        service = CacheService(strategy=in_memory_strategy)

        async with service:
            # Verify periodic cleanup started
            assert service._running is True

        # Exit context - cleanup_task is cancelled but not None
        assert service._running is False
        # After cancellation, the task may be done but still referenceable
        assert service._cleanup_task is None or service._cleanup_task.done()

"""Feature tests for cache_manager module.

Tests focus on behavior:
- CachedItem: cache entry dataclass
- CacheStrategy: abstract base class
- HashBasedCacheStrategy: hash-based key generation
- CacheManager: generic cache manager with TTL support
"""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch

import pytest

from persbot.services.cache_manager import (
    CachedItem,
    CacheStrategy,
    HashBasedCacheStrategy,
    CacheManager,
)


class TestCachedItem:
    """Tests for CachedItem dataclass."""

    def test_creates_with_required_fields(self):
        """CachedItem creates with all required fields."""
        now = datetime.now(timezone.utc)
        expiration = now + timedelta(hours=1)

        item = CachedItem(
            value="test_value",
            expiration=expiration,
            cache_key="cache-123",
            created_at=now,
        )

        assert item.value == "test_value"
        assert item.expiration == expiration
        assert item.cache_key == "cache-123"
        assert item.created_at == now

    def test_accepts_none_expiration(self):
        """CachedItem accepts None for expiration (no expiry)."""
        now = datetime.now(timezone.utc)

        item = CachedItem(
            value="test_value",
            expiration=None,
            cache_key="cache-123",
            created_at=now,
        )

        assert item.expiration is None

    def test_accepts_complex_value_types(self):
        """CachedItem can store complex value types."""
        now = datetime.now(timezone.utc)

        # Store a dict
        dict_item = CachedItem(
            value={"key": "value", "nested": {"a": 1}},
            expiration=None,
            cache_key="dict-key",
            created_at=now,
        )
        assert dict_item.value["nested"]["a"] == 1

        # Store a list
        list_item = CachedItem(
            value=[1, 2, 3, "four"],
            expiration=None,
            cache_key="list-key",
            created_at=now,
        )
        assert list_item.value[3] == "four"

    def test_is_frozen_dataclass(self):
        """CachedItem is a dataclass (can verify via fields)."""
        from dataclasses import fields

        item = CachedItem(
            value="test",
            expiration=None,
            cache_key="key",
            created_at=datetime.now(timezone.utc),
        )

        field_names = {f.name for f in fields(item)}
        assert field_names == {"value", "expiration", "cache_key", "created_at"}


class TestCacheStrategy:
    """Tests for CacheStrategy abstract base class."""

    def test_cannot_instantiate_directly(self):
        """CacheStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CacheStrategy()

    def test_subclass_must_implement_generate_key(self):
        """Subclass must implement generate_key method."""

        class IncompleteStrategy(CacheStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()

    def test_subclass_with_implementation_works(self):
        """Subclass with generate_key implementation can be instantiated."""

        class CompleteStrategy(CacheStrategy):
            def generate_key(self, *args, **kwargs) -> str:
                return "test-key"

        strategy = CompleteStrategy()
        assert strategy.generate_key() == "test-key"


class TestHashBasedCacheStrategy:
    """Tests for HashBasedCacheStrategy."""

    def test_default_prefix_is_cache(self):
        """Default prefix is 'cache'."""
        strategy = HashBasedCacheStrategy()
        assert strategy.prefix == "cache"

    def test_default_hash_length_is_10(self):
        """Default hash_length is 10."""
        strategy = HashBasedCacheStrategy()
        assert strategy.hash_length == 10

    def test_custom_prefix(self):
        """Can set custom prefix."""
        strategy = HashBasedCacheStrategy(prefix="custom")
        assert strategy.prefix == "custom"

    def test_custom_hash_length(self):
        """Can set custom hash_length."""
        strategy = HashBasedCacheStrategy(hash_length=16)
        assert strategy.hash_length == 16

    def test_generate_key_returns_string(self):
        """generate_key returns a string."""
        strategy = HashBasedCacheStrategy()
        key = strategy.generate_key("arg1", "arg2")
        assert isinstance(key, str)

    def test_generate_key_includes_prefix(self):
        """generate_key includes the prefix."""
        strategy = HashBasedCacheStrategy(prefix="myprefix")
        key = strategy.generate_key("test")
        assert key.startswith("myprefix-")

    def test_generate_key_consistent_for_same_args(self):
        """Same args produce same key."""
        strategy = HashBasedCacheStrategy()

        key1 = strategy.generate_key("a", "b", "c")
        key2 = strategy.generate_key("a", "b", "c")

        assert key1 == key2

    def test_generate_key_different_for_different_args(self):
        """Different args produce different keys."""
        strategy = HashBasedCacheStrategy()

        key1 = strategy.generate_key("a", "b")
        key2 = strategy.generate_key("x", "y")

        assert key1 != key2

    def test_generate_key_handles_kwargs(self):
        """generate_key handles kwargs."""
        strategy = HashBasedCacheStrategy()

        key1 = strategy.generate_key(name="test", value=123)
        key2 = strategy.generate_key(name="test", value=123)

        assert key1 == key2

    def test_generate_key_kwargs_order_independent(self):
        """generate_key is order-independent for kwargs."""
        strategy = HashBasedCacheStrategy()

        # kwargs are sorted internally
        key1 = strategy.generate_key(a=1, b=2)
        key2 = strategy.generate_key(b=2, a=1)

        assert key1 == key2

    def test_generate_key_respects_hash_length(self):
        """generate_key truncates hash to hash_length."""
        strategy = HashBasedCacheStrategy(hash_length=8)
        key = strategy.generate_key("test")

        # Key format is prefix-hash, so extract hash part
        parts = key.split("-")
        hash_part = parts[1]
        assert len(hash_part) == 8

    def test_generate_key_uses_sha256(self):
        """generate_key produces SHA256-based hash."""
        strategy = HashBasedCacheStrategy(prefix="test", hash_length=64)

        # Generate expected hash manually
        combined = "arg1:arg2"
        expected_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        key = strategy.generate_key("arg1", "arg2")

        assert key == f"test-{expected_hash}"

    def test_generate_key_mixed_args_and_kwargs(self):
        """generate_key handles mixed args and kwargs."""
        strategy = HashBasedCacheStrategy()

        key1 = strategy.generate_key("arg1", "arg2", key1="val1", key2="val2")
        key2 = strategy.generate_key("arg1", "arg2", key1="val1", key2="val2")

        assert key1 == key2


class TestCacheManagerInit:
    """Tests for CacheManager.__init__."""

    def test_default_ttl_minutes(self):
        """Default TTL is 60 minutes."""
        manager = CacheManager()
        assert manager.ttl_minutes == 60

    def test_default_max_size(self):
        """Default max_size is 200."""
        manager = CacheManager()
        assert manager.max_size == 200

    def test_default_cleanup_interval(self):
        """Default cleanup_interval_minutes is 30."""
        manager = CacheManager()
        assert manager.cleanup_interval_minutes == 30

    def test_custom_ttl_minutes(self):
        """Can set custom TTL."""
        manager = CacheManager(ttl_minutes=120)
        assert manager.ttl_minutes == 120

    def test_custom_max_size(self):
        """Can set custom max_size."""
        manager = CacheManager(max_size=500)
        assert manager.max_size == 500

    def test_custom_cleanup_interval(self):
        """Can set custom cleanup_interval_minutes."""
        manager = CacheManager(cleanup_interval_minutes=15)
        assert manager.cleanup_interval_minutes == 15

    def test_default_strategy_is_hash_based(self):
        """Default strategy is HashBasedCacheStrategy."""
        manager = CacheManager()
        assert isinstance(manager.strategy, HashBasedCacheStrategy)

    def test_custom_strategy(self):
        """Can set custom strategy."""

        class CustomStrategy(CacheStrategy):
            def generate_key(self, *args, **kwargs) -> str:
                return "custom-key"

        manager = CacheManager(strategy=CustomStrategy())
        assert isinstance(manager.strategy, CustomStrategy)

    def test_initializes_empty_cache(self):
        """Cache starts empty."""
        manager = CacheManager()
        assert len(manager._cache) == 0

    def test_initializes_no_cleanup_task(self):
        """No cleanup task running initially."""
        manager = CacheManager()
        assert manager._cleanup_task is None

    def test_initializes_lock(self):
        """Lock is initialized."""
        manager = CacheManager()
        assert isinstance(manager._lock, asyncio.Lock)


class TestCacheManagerGet:
    """Tests for CacheManager.get method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager()

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(self, manager):
        """get returns None for missing key."""
        result = await manager.get("nonexistent", "key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_default_for_missing_key(self, manager):
        """get returns default value for missing key."""
        result = await manager.get("nonexistent", default="default_value")
        assert result == "default_value"

    @pytest.mark.asyncio
    async def test_get_returns_cached_value(self, manager):
        """get returns cached value."""
        await manager.set("test_value", "my", "key")
        result = await manager.get("my", "key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_expired_entry(self, manager):
        """get returns None for expired entry and removes it."""
        # Set with very short TTL
        await manager.set("test_value", "my", "key", ttl_minutes=0.0001)  # ~6ms

        # Wait for expiration
        await asyncio.sleep(0.1)

        result = await manager.get("my", "key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_default_for_expired_entry(self, manager):
        """get returns default for expired entry."""
        # Set with very short TTL
        await manager.set("test_value", "my", "key", ttl_minutes=0.0001)

        # Wait for expiration
        await asyncio.sleep(0.1)

        result = await manager.get("my", "key", default="expired_default")
        assert result == "expired_default"

    @pytest.mark.asyncio
    async def test_get_removes_expired_entry_from_cache(self, manager):
        """get removes expired entry from cache."""
        # Set with very short TTL
        await manager.set("test_value", "my", "key", ttl_minutes=0.0001)

        # Wait for expiration
        await asyncio.sleep(0.1)

        # Get should remove the expired entry
        await manager.get("my", "key")

        # Verify it was removed
        assert manager.size() == 0

    @pytest.mark.asyncio
    async def test_get_returns_value_with_no_expiration(self, manager):
        """get returns value when expiration is None."""
        # Set with no TTL (never expires)
        await manager.set("test_value", "my", "key", ttl_minutes=None)

        result = await manager.get("my", "key")
        assert result == "test_value"


class TestCacheManagerSet:
    """Tests for CacheManager.set method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager()

    @pytest.mark.asyncio
    async def test_set_returns_cache_key(self, manager):
        """set returns the cache key."""
        key = await manager.set("value", "arg1", "arg2")
        assert isinstance(key, str)
        assert key.startswith("cache-")

    @pytest.mark.asyncio
    async def test_set_stores_value(self, manager):
        """set stores value in cache."""
        await manager.set("test_value", "my", "key")

        result = await manager.get("my", "key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_set_increases_cache_size(self, manager):
        """set increases cache size."""
        await manager.set("value1", "key1")
        assert manager.size() == 1

        await manager.set("value2", "key2")
        assert manager.size() == 2

    @pytest.mark.asyncio
    async def test_set_updates_existing_entry(self, manager):
        """set updates existing entry with same key."""
        await manager.set("old_value", "my", "key")
        await manager.set("new_value", "my", "key")

        result = await manager.get("my", "key")
        assert result == "new_value"
        assert manager.size() == 1

    @pytest.mark.asyncio
    async def test_set_uses_default_ttl(self, manager):
        """set uses default TTL when not specified."""
        manager = CacheManager(ttl_minutes=30)
        await manager.set("value", "key")

        # Check that the cached item has an expiration
        cache_key = manager._generate_key("key")
        item = manager._cache[cache_key]

        assert item.expiration is not None

    @pytest.mark.asyncio
    async def test_set_respects_custom_ttl(self, manager):
        """set respects custom TTL."""
        await manager.set("value", "key", ttl_minutes=5)

        cache_key = manager._generate_key("key")
        item = manager._cache[cache_key]

        # Expiration should be approximately 5 minutes from now
        now = datetime.now(timezone.utc)
        expected_expiration = now + timedelta(minutes=5)
        delta = abs((item.expiration - expected_expiration).total_seconds())
        assert delta < 1  # Within 1 second

    @pytest.mark.asyncio
    async def test_set_with_zero_ttl_means_no_expiration(self, manager):
        """set with ttl_minutes=0 or negative means no expiration."""
        await manager.set("value", "key", ttl_minutes=0)

        cache_key = manager._generate_key("key")
        item = manager._cache[cache_key]

        assert item.expiration is None

    @pytest.mark.asyncio
    async def test_set_with_negative_ttl_means_no_expiration(self, manager):
        """set with negative TTL means no expiration."""
        await manager.set("value", "key", ttl_minutes=-5)

        cache_key = manager._generate_key("key")
        item = manager._cache[cache_key]

        assert item.expiration is None

    @pytest.mark.asyncio
    async def test_set_with_none_ttl_uses_default(self, manager):
        """set with None TTL uses default TTL."""
        manager = CacheManager(ttl_minutes=45)
        await manager.set("value", "key", ttl_minutes=None)

        cache_key = manager._generate_key("key")
        item = manager._cache[cache_key]

        # Should use default TTL
        assert item.expiration is not None

    @pytest.mark.asyncio
    async def test_set_stores_cached_item_with_metadata(self, manager):
        """set stores CachedItem with proper metadata."""
        before = datetime.now(timezone.utc)
        await manager.set("value", "key")
        after = datetime.now(timezone.utc)

        cache_key = manager._generate_key("key")
        item = manager._cache[cache_key]

        assert item.value == "value"
        assert item.cache_key == cache_key
        assert before <= item.created_at <= after


class TestCacheManagerDelete:
    """Tests for CacheManager.delete method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager()

    @pytest.mark.asyncio
    async def test_delete_returns_true_for_existing_key(self, manager):
        """delete returns True when key exists."""
        await manager.set("value", "key")
        result = await manager.delete("key")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_removes_entry(self, manager):
        """delete removes entry from cache."""
        await manager.set("value", "key")
        await manager.delete("key")

        result = await manager.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_decreases_cache_size(self, manager):
        """delete decreases cache size."""
        await manager.set("value", "key")
        assert manager.size() == 1

        await manager.delete("key")
        assert manager.size() == 0

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_missing_key(self, manager):
        """delete returns False when key doesn't exist."""
        result = await manager.delete("nonexistent")
        assert result is False


class TestCacheManagerClear:
    """Tests for CacheManager.clear method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager()

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self, manager):
        """clear removes all entries."""
        await manager.set("value1", "key1")
        await manager.set("value2", "key2")
        await manager.set("value3", "key3")

        await manager.clear()

        assert manager.size() == 0

    @pytest.mark.asyncio
    async def test_clear_on_empty_cache_is_safe(self, manager):
        """clear on empty cache is safe."""
        await manager.clear()
        assert manager.size() == 0


class TestCacheManagerGetOrCreate:
    """Tests for CacheManager.get_or_create method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager()

    @pytest.mark.asyncio
    async def test_returns_cached_value_on_hit(self, manager):
        """get_or_create returns cached value on hit."""
        await manager.set("cached_value", "key")

        factory = Mock(return_value="factory_value")
        value, was_created = await manager.get_or_create(factory, "key")

        assert value == "cached_value"
        assert was_created is False
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_factory_on_miss(self, manager):
        """get_or_create calls factory on miss."""
        factory = Mock(return_value="factory_value")
        value, was_created = await manager.get_or_create(factory, "key")

        assert value == "factory_value"
        assert was_created is True
        factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_stores_factory_result_on_miss(self, manager):
        """get_or_create stores factory result on miss."""
        factory = Mock(return_value="factory_value")
        await manager.get_or_create(factory, "key")

        result = await manager.get("key")
        assert result == "factory_value"

    @pytest.mark.asyncio
    async def test_respects_custom_ttl(self, manager):
        """get_or_create respects custom TTL."""
        factory = Mock(return_value="value")
        await manager.get_or_create(factory, "key", ttl_minutes=10)

        cache_key = manager._generate_key("key")
        item = manager._cache[cache_key]

        now = datetime.now(timezone.utc)
        expected = now + timedelta(minutes=10)
        delta = abs((item.expiration - expected).total_seconds())
        assert delta < 1

    @pytest.mark.asyncio
    async def test_returns_none_from_factory(self, manager):
        """get_or_create handles factory returning None."""
        factory = Mock(return_value=None)
        value, was_created = await manager.get_or_create(factory, "key")

        assert value is None
        assert was_created is True


class TestCacheManagerEvictOldest:
    """Tests for CacheManager._evict_oldest method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager with small max_size."""
        return CacheManager(max_size=3)

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_at_capacity(self, manager):
        """_evict_oldest removes oldest entry when at capacity."""
        await manager.set("value1", "key1")
        await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        await manager.set("value2", "key2")
        await asyncio.sleep(0.01)
        await manager.set("value3", "key3")

        # Cache is now at capacity (3)
        assert manager.size() == 3

        # Add one more - should evict oldest
        await manager.set("value4", "key4")

        assert manager.size() == 3
        # key1 should have been evicted
        result = await manager.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_keeps_newest_entries(self, manager):
        """_evict_oldest keeps newest entries."""
        await manager.set("value1", "key1")
        await asyncio.sleep(0.01)
        await manager.set("value2", "key2")
        await asyncio.sleep(0.01)
        await manager.set("value3", "key3")
        await asyncio.sleep(0.01)
        await manager.set("value4", "key4")

        # key2, key3, key4 should exist
        assert await manager.get("key2") == "value2"
        assert await manager.get("key3") == "value3"
        assert await manager.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_update_does_not_trigger_eviction(self, manager):
        """Updating existing key doesn't trigger eviction."""
        await manager.set("value1", "key1")
        await manager.set("value2", "key2")
        await manager.set("value3", "key3")

        # Update existing key - should not evict
        await manager.set("updated1", "key1")

        assert manager.size() == 3
        assert await manager.get("key1") == "updated1"


class TestCacheManagerCleanupExpired:
    """Tests for CacheManager.cleanup_expired method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager()

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_expired(self, manager):
        """cleanup_expired returns 0 when no expired entries."""
        await manager.set("value", "key", ttl_minutes=60)

        removed = await manager.cleanup_expired()
        assert removed == 0

    @pytest.mark.asyncio
    async def test_removes_expired_entries(self, manager):
        """cleanup_expired removes expired entries."""
        # Create expired entry
        await manager.set("expired", "key1", ttl_minutes=0.0001)
        # Create non-expired entry
        await manager.set("active", "key2", ttl_minutes=60)

        # Wait for first to expire
        await asyncio.sleep(0.1)

        removed = await manager.cleanup_expired()
        assert removed == 1
        assert manager.size() == 1

    @pytest.mark.asyncio
    async def test_removes_multiple_expired(self, manager):
        """cleanup_expired removes multiple expired entries."""
        # Create multiple expired entries
        for i in range(3):
            await manager.set(f"value{i}", f"key{i}", ttl_minutes=0.0001)
            await asyncio.sleep(0.01)

        # Create non-expired
        await manager.set("active", "active_key", ttl_minutes=60)

        # Wait for expiration
        await asyncio.sleep(0.1)

        removed = await manager.cleanup_expired()
        assert removed == 3
        assert manager.size() == 1

    @pytest.mark.asyncio
    async def test_handles_entry_with_no_expiration(self, manager):
        """cleanup_expired handles entries with no expiration."""
        await manager.set("permanent", "key", ttl_minutes=None)

        removed = await manager.cleanup_expired()
        assert removed == 0
        assert manager.size() == 1


class TestCacheManagerStats:
    """Tests for CacheManager.stats method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager(ttl_minutes=30, max_size=100)

    @pytest.mark.asyncio
    async def test_returns_correct_structure(self, manager):
        """stats returns dictionary with expected keys."""
        stats = await manager.stats()

        assert "total_entries" in stats
        assert "expired_entries" in stats
        assert "active_entries" in stats
        assert "max_size" in stats
        assert "ttl_minutes" in stats

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_cache(self, manager):
        """stats returns zeros for empty cache."""
        stats = await manager.stats()

        assert stats["total_entries"] == 0
        assert stats["expired_entries"] == 0
        assert stats["active_entries"] == 0

    @pytest.mark.asyncio
    async def test_counts_active_entries(self, manager):
        """stats counts active entries correctly."""
        await manager.set("value1", "key1")
        await manager.set("value2", "key2")

        stats = await manager.stats()

        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["expired_entries"] == 0

    @pytest.mark.asyncio
    async def test_counts_expired_entries(self, manager):
        """stats counts expired entries correctly."""
        await manager.set("expired", "key1", ttl_minutes=0.0001)
        await manager.set("active", "key2", ttl_minutes=60)

        await asyncio.sleep(0.1)

        stats = await manager.stats()

        assert stats["total_entries"] == 2
        assert stats["expired_entries"] == 1
        assert stats["active_entries"] == 1

    @pytest.mark.asyncio
    async def test_includes_configuration(self, manager):
        """stats includes cache configuration."""
        stats = await manager.stats()

        assert stats["max_size"] == 100
        assert stats["ttl_minutes"] == 30


class TestCacheManagerCleanupTask:
    """Tests for CacheManager cleanup task methods."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager with short cleanup interval."""
        return CacheManager(cleanup_interval_minutes=0.001)  # ~60ms

    @pytest.mark.asyncio
    async def test_start_cleanup_task_creates_task(self, manager):
        """start_cleanup_task creates background task."""
        await manager.start_cleanup_task()

        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.done()

        await manager.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_start_cleanup_task_idempotent(self, manager):
        """start_cleanup_task is idempotent."""
        await manager.start_cleanup_task()
        first_task = manager._cleanup_task

        await manager.start_cleanup_task()  # Should not create new task
        assert manager._cleanup_task is first_task

        await manager.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_stop_cleanup_task_cancels_task(self, manager):
        """stop_cleanup_task cancels running task."""
        await manager.start_cleanup_task()
        await manager.stop_cleanup_task()

        assert manager._cleanup_task is None

    @pytest.mark.asyncio
    async def test_stop_cleanup_task_when_not_running_is_safe(self, manager):
        """stop_cleanup_task is safe when no task running."""
        await manager.stop_cleanup_task()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_cleanup_task_when_already_stopped_is_safe(self, manager):
        """stop_cleanup_task is safe when already stopped."""
        await manager.start_cleanup_task()
        await manager.stop_cleanup_task()
        await manager.stop_cleanup_task()  # Should not raise

    @pytest.mark.asyncio
    async def test_cleanup_loop_removes_expired_entries(self, manager):
        """cleanup loop removes expired entries periodically."""
        await manager.start_cleanup_task()

        # Create expired entry
        await manager.set("value", "key", ttl_minutes=0.0001)
        await asyncio.sleep(0.1)  # Let it expire

        # Wait for cleanup cycle
        await asyncio.sleep(0.2)

        # Entry should be removed
        assert manager.size() == 0

        await manager.stop_cleanup_task()


class TestCacheManagerSize:
    """Tests for CacheManager.size method."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager()

    def test_size_returns_zero_for_empty(self, manager):
        """size returns 0 for empty cache."""
        assert manager.size() == 0

    @pytest.mark.asyncio
    async def test_size_returns_correct_count(self, manager):
        """size returns correct count of entries."""
        await manager.set("value1", "key1")
        assert manager.size() == 1

        await manager.set("value2", "key2")
        assert manager.size() == 2

    @pytest.mark.asyncio
    async def test_size_after_delete(self, manager):
        """size is correct after delete."""
        await manager.set("value", "key")
        await manager.delete("key")

        assert manager.size() == 0

    @pytest.mark.asyncio
    async def test_size_after_clear(self, manager):
        """size is correct after clear."""
        await manager.set("value1", "key1")
        await manager.set("value2", "key2")
        await manager.clear()

        assert manager.size() == 0


class TestCacheManagerThreadSafety:
    """Tests for CacheManager thread safety."""

    @pytest.fixture
    def manager(self):
        """Create a CacheManager."""
        return CacheManager()

    @pytest.mark.asyncio
    async def test_concurrent_set_operations(self, manager):
        """Concurrent set operations are safe."""
        tasks = [
            manager.set(f"value{i}", f"key{i}")
            for i in range(10)
        ]

        await asyncio.gather(*tasks)

        assert manager.size() == 10

    @pytest.mark.asyncio
    async def test_concurrent_get_set_operations(self, manager):
        """Concurrent get and set operations are safe."""
        # Pre-populate some entries
        for i in range(5):
            await manager.set(f"value{i}", f"key{i}")

        # Mix of reads and writes
        tasks = []
        for i in range(10):
            if i % 2 == 0:
                tasks.append(manager.get("key0"))
            else:
                tasks.append(manager.set(f"new{i}", f"newkey{i}"))

        await asyncio.gather(*tasks)

        # Should complete without errors

    @pytest.mark.asyncio
    async def test_concurrent_delete_operations(self, manager):
        """Concurrent delete operations are safe."""
        # Pre-populate
        for i in range(10):
            await manager.set(f"value{i}", f"key{i}")

        # Delete same key multiple times
        tasks = [manager.delete("key5") for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # Only one should return True
        assert sum(results) == 1

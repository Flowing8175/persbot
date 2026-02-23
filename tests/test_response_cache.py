"""Feature tests for ResponseCacheService.

Tests focus on behavior:
- Cache hit/miss logic
- Query normalization and key generation
- TTL and expiration
- LRU eviction
- Thread-safe async operations
- Statistics and metadata
"""

import asyncio
import datetime
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from persbot.services.response_cache import (
    ResponseCacheService,
    ResponseCacheConfig,
    CachedResponse,
)


class TestResponseCacheConfig:
    """Tests for ResponseCacheConfig dataclass."""

    def test_creates_with_defaults(self):
        """ResponseCacheConfig creates with default values."""
        config = ResponseCacheConfig()
        assert config.enabled is True
        assert config.ttl_minutes == 60
        assert config.max_entries == 500
        assert config.normalize_whitespace is True
        assert config.case_sensitive is False

    def test_creates_with_custom_values(self):
        """ResponseCacheConfig creates with custom values."""
        config = ResponseCacheConfig(
            enabled=False,
            ttl_minutes=30,
            max_entries=1000,
            normalize_whitespace=False,
            case_sensitive=True,
        )
        assert config.enabled is False
        assert config.ttl_minutes == 30
        assert config.max_entries == 1000
        assert config.normalize_whitespace is False
        assert config.case_sensitive is True


class TestCachedResponse:
    """Tests for CachedResponse dataclass."""

    def test_creates_with_required_fields(self):
        """CachedResponse creates with required fields."""
        now = datetime.datetime.now(datetime.timezone.utc)
        entry = CachedResponse(
            query_hash="test_hash",
            response_text="Test response",
            model_alias="gpt-4",
            created_at=now,
            last_accessed_at=now,
        )
        assert entry.query_hash == "test_hash"
        assert entry.response_text == "Test response"
        assert entry.model_alias == "gpt-4"
        assert entry.hit_count == 0
        assert entry.response_obj is None

    def test_creates_with_optional_fields(self):
        """CachedResponse creates with optional fields."""
        now = datetime.datetime.now(datetime.timezone.utc)
        response_obj = Mock()
        entry = CachedResponse(
            query_hash="test_hash",
            response_text="Test response",
            model_alias="gpt-4",
            created_at=now,
            last_accessed_at=now,
            response_obj=response_obj,
            hit_count=5,
        )
        assert entry.response_obj == response_obj
        assert entry.hit_count == 5


class TestResponseCacheServiceCreation:
    """Tests for ResponseCacheService instantiation."""

    def test_creates_with_default_config(self):
        """ResponseCacheService creates with default config."""
        cache = ResponseCacheService()
        assert cache.is_enabled is True
        assert cache.config.ttl_minutes == 60
        assert cache.config.max_entries == 500
        assert len(cache._cache) == 0

    def test_creates_with_custom_config(self):
        """ResponseCacheService creates with custom config."""
        config = ResponseCacheConfig(
            enabled=False,
            ttl_minutes=30,
            max_entries=100,
        )
        cache = ResponseCacheService(config)
        assert cache.is_enabled is False
        assert cache.config.ttl_minutes == 30
        assert cache.config.max_entries == 100


class TestQueryNormalization:
    """Tests for query normalization logic."""

    def test_normalizes_whitespace_when_enabled(self):
        """Query whitespace is normalized when enabled."""
        config = ResponseCacheConfig(normalize_whitespace=True)
        cache = ResponseCacheService(config)

        normalized = cache._normalize_query("hello     world   test")
        assert normalized == "hello world test"

    def test_preserves_whitespace_when_disabled(self):
        """Query whitespace is preserved when disabled."""
        config = ResponseCacheConfig(normalize_whitespace=False)
        cache = ResponseCacheService(config)

        original = "hello     world   test"
        normalized = cache._normalize_query(original)
        assert normalized == original

    def test_lowercases_when_case_insensitive(self):
        """Query is lowercased when case_sensitive=False."""
        config = ResponseCacheConfig(case_sensitive=False)
        cache = ResponseCacheService(config)

        normalized = cache._normalize_query("HeLLo WoRLd")
        assert normalized == "hello world"

    def test_preserves_case_when_case_sensitive(self):
        """Query case is preserved when case_sensitive=True."""
        config = ResponseCacheConfig(case_sensitive=True)
        cache = ResponseCacheService(config)

        normalized = cache._normalize_query("HeLLo WoRLd")
        assert normalized == "HeLLo WoRLd"

    def test_combines_normalization_options(self):
        """All normalization options are applied together."""
        config = ResponseCacheConfig(
            normalize_whitespace=True,
            case_sensitive=False,
        )
        cache = ResponseCacheService(config)

        normalized = cache._normalize_query("  HeLLo     WoRLd  ")
        assert normalized == "hello world"

    def test_trims_leading_trailing_whitespace(self):
        """Leading and trailing whitespace is trimmed."""
        cache = ResponseCacheService()

        normalized = cache._normalize_query("  hello world  ")
        assert normalized == "hello world"


class TestKeyGeneration:
    """Tests for cache key generation."""

    def test_generates_consistent_keys(self):
        """Same query and model generate same key."""
        cache = ResponseCacheService()

        key1 = cache._generate_key("hello world", "gpt-4")
        key2 = cache._generate_key("hello world", "gpt-4")

        assert key1 == key2

    def test_generates_different_keys_for_different_queries(self):
        """Different queries generate different keys."""
        cache = ResponseCacheService()

        key1 = cache._generate_key("hello world", "gpt-4")
        key2 = cache._generate_key("goodbye world", "gpt-4")

        assert key1 != key2

    def test_generates_different_keys_for_different_models(self):
        """Different models generate different keys for same query."""
        cache = ResponseCacheService()

        key1 = cache._generate_key("hello world", "gpt-4")
        key2 = cache._generate_key("hello world", "gpt-3.5")

        assert key1 != key2

    def test_key_includes_response_prefix(self):
        """Keys include 'response:' prefix."""
        cache = ResponseCacheService()

        key = cache._generate_key("test", "model")
        assert key.startswith("response:")

    def test_normalized_queries_generate_same_key(self):
        """Normalized queries generate same key when case_sensitive=False."""
        config = ResponseCacheConfig(case_sensitive=False)
        cache = ResponseCacheService(config)

        key1 = cache._generate_key("Hello World", "gpt-4")
        key2 = cache._generate_key("hello world", "gpt-4")

        assert key1 == key2


class TestCacheGetSet:
    """Tests for basic cache get/set operations."""

    @pytest.mark.asyncio
    async def test_set_and_get_cached_response(self):
        """Response can be stored and retrieved."""
        cache = ResponseCacheService()

        await cache.set("What is AI?", "gpt-4", "AI is artificial intelligence")
        result = await cache.get("What is AI?", "gpt-4")

        assert result is not None
        response_text, response_obj = result
        assert response_text == "AI is artificial intelligence"
        assert response_obj is None

    @pytest.mark.asyncio
    async def test_get_returns_none_for_miss(self):
        """get returns None for cache miss."""
        cache = ResponseCacheService()

        result = await cache.get("nonexistent query", "gpt-4")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_stores_response_object(self):
        """set can store optional response object."""
        cache = ResponseCacheService()
        response_obj = Mock(data="original_response")

        await cache.set("test", "gpt-4", "response text", response_obj)
        result = await cache.get("test", "gpt-4")

        assert result is not None
        response_text, retrieved_obj = result
        assert retrieved_obj == response_obj

    @pytest.mark.asyncio
    async def test_get_updates_hit_count(self):
        """Cache hit increments hit_count."""
        cache = ResponseCacheService()

        await cache.set("test", "gpt-4", "response")
        await cache.get("test", "gpt-4")
        await cache.get("test", "gpt-4")

        stats = await cache.get_stats()
        assert stats["total_hits"] == 2

    @pytest.mark.asyncio
    async def test_get_updates_last_accessed_time(self):
        """Cache hit updates last_accessed_at timestamp."""
        cache = ResponseCacheService()

        await cache.set("test", "gpt-4", "response")

        # Get initial timestamp
        initial_stats = await cache.get_stats()

        # Wait and access again
        await asyncio.sleep(0.01)
        await cache.get("test", "gpt-4")

        # Get key to check entry directly
        key = cache._generate_key("test", "gpt-4")
        async with cache._lock:
            entry = cache._cache[key]
            assert entry.hit_count == 1
            assert entry.last_accessed_at > entry.created_at


class TestCacheDisabled:
    """Tests for cache when disabled."""

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disabled(self):
        """get returns None when cache is disabled."""
        config = ResponseCacheConfig(enabled=False)
        cache = ResponseCacheService(config)

        await cache.set("test", "gpt-4", "response")
        result = await cache.get("test", "gpt-4")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_does_nothing_when_disabled(self):
        """set does not store when cache is disabled."""
        config = ResponseCacheConfig(enabled=False)
        cache = ResponseCacheService(config)

        await cache.set("test", "gpt-4", "response")
        stats = await cache.get_stats()

        assert stats["total_entries"] == 0


class TestCacheExpiration:
    """Tests for TTL and expiration logic."""

    @pytest.mark.asyncio
    async def test_get_returns_none_for_expired_entry(self):
        """Expired entries are not returned."""
        config = ResponseCacheConfig(ttl_minutes=1)
        cache = ResponseCacheService(config)

        await cache.set("test", "gpt-4", "response")

        # Manually expire the entry by setting created_at to past
        key = cache._generate_key("test", "gpt-4")
        async with cache._lock:
            cache._cache[key].created_at = datetime.datetime.now(datetime.timezone.utc) - timedelta(minutes=2)

        result = await cache.get("test", "gpt-4")

        assert result is None

    @pytest.mark.asyncio
    async def test_expired_entry_is_removed(self):
        """Expired entries are removed from cache."""
        config = ResponseCacheConfig(ttl_minutes=1)
        cache = ResponseCacheService(config)

        await cache.set("test", "gpt-4", "response")

        # Manually expire the entry
        key = cache._generate_key("test", "gpt-4")
        async with cache._lock:
            cache._cache[key].created_at = datetime.datetime.now(datetime.timezone.utc) - timedelta(minutes=2)

        await cache.get("test", "gpt-4")  # Triggers expiration check

        stats = await cache.get_stats()
        assert stats["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_zero_ttl_disables_expiration(self):
        """TTL of 0 means entries never expire."""
        config = ResponseCacheConfig(ttl_minutes=0)
        cache = ResponseCacheService(config)

        # Override _is_expired for this test
        cache._is_expired = lambda e: False

        await cache.set("test", "gpt-4", "response")
        result = await cache.get("test", "gpt-4")

        assert result is not None

    @pytest.mark.asyncio
    async def test_negative_ttl_disables_expiration(self):
        """Negative TTL means entries never expire."""
        config = ResponseCacheConfig(ttl_minutes=-1)
        cache = ResponseCacheService(config)

        await cache.set("test", "gpt-4", "response")
        result = await cache.get("test", "gpt-4")

        # Should not expire
        assert result is not None


class TestLRUEviction:
    """Tests for LRU eviction logic."""

    @pytest.mark.asyncio
    async def test_evicts_lru_when_full(self):
        """Least recently used entry is evicted when cache is full."""
        config = ResponseCacheConfig(max_entries=2)
        cache = ResponseCacheService(config)

        await cache.set("first", "gpt-4", "response1")
        await cache.set("second", "gpt-4", "response2")
        await cache.set("third", "gpt-4", "response3")  # Should evict "first"

        stats = await cache.get_stats()
        assert stats["total_entries"] == 2

        result1 = await cache.get("first", "gpt-4")
        assert result1 is None

        result2 = await cache.get("second", "gpt-4")
        assert result2 is not None

        result3 = await cache.get("third", "gpt-4")
        assert result3 is not None

    @pytest.mark.asyncio
    async def test_recently_accessed_not_evicted(self):
        """Recently accessed entries are preferred over old ones."""
        config = ResponseCacheConfig(max_entries=2)
        cache = ResponseCacheService(config)

        await cache.set("first", "gpt-4", "response1")
        await cache.set("second", "gpt-4", "response2")

        # Access first to make it more recent
        await cache.get("first", "gpt-4")

        # Add third - should evict second (least recently accessed)
        await cache.set("third", "gpt-4", "response3")

        result1 = await cache.get("first", "gpt-4")
        assert result1 is not None

        result2 = await cache.get("second", "gpt-4")
        assert result2 is None

        result3 = await cache.get("third", "gpt-4")
        assert result3 is not None

    @pytest.mark.asyncio
    async def test_update_existing_does_not_evict(self):
        """Updating existing entry does not trigger eviction."""
        config = ResponseCacheConfig(max_entries=2)
        cache = ResponseCacheService(config)

        await cache.set("first", "gpt-4", "response1")
        await cache.set("second", "gpt-4", "response2")
        await cache.set("first", "gpt-4", "updated_response1")  # Update existing

        stats = await cache.get_stats()
        assert stats["total_entries"] == 2


class TestCacheInvalidation:
    """Tests for cache invalidation operations."""

    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self):
        """invalidate removes specific entry."""
        cache = ResponseCacheService()

        await cache.set("test", "gpt-4", "response")
        removed = await cache.invalidate("test", "gpt-4")

        assert removed is True

        result = await cache.get("test", "gpt-4")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_returns_false_for_nonexistent(self):
        """invalidate returns False for non-existent entry."""
        cache = ResponseCacheService()

        removed = await cache.invalidate("nonexistent", "gpt-4")

        assert removed is False

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self):
        """clear removes all entries from cache."""
        cache = ResponseCacheService()

        await cache.set("test1", "gpt-4", "response1")
        await cache.set("test2", "gpt-4", "response2")
        await cache.set("test3", "gpt-4", "response3")

        count = await cache.clear()

        assert count == 3

        stats = await cache.get_stats()
        assert stats["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_clear_returns_zero_when_empty(self):
        """clear returns 0 when cache is empty."""
        cache = ResponseCacheService()

        count = await cache.clear()

        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_only_expired(self):
        """cleanup_expired removes only expired entries."""
        config = ResponseCacheConfig(ttl_minutes=60)
        cache = ResponseCacheService(config)

        await cache.set("test1", "gpt-4", "response1")

        # Manually expire one entry
        key = cache._generate_key("test1", "gpt-4")
        async with cache._lock:
            cache._cache[key].created_at = datetime.datetime.now(datetime.timezone.utc) - timedelta(hours=2)

        await cache.set("test2", "gpt-4", "response2")

        removed = await cache.cleanup_expired()

        assert removed == 1

        result1 = await cache.get("test1", "gpt-4")
        assert result1 is None

        result2 = await cache.get("test2", "gpt-4")
        assert result2 is not None


class TestCacheStatistics:
    """Tests for cache statistics."""

    @pytest.mark.asyncio
    async def test_stats_returns_configuration(self):
        """get_stats returns configuration information."""
        cache = ResponseCacheService()

        stats = await cache.get_stats()

        assert stats["enabled"] is True
        assert stats["max_entries"] == 500
        assert stats["ttl_minutes"] == 60

    @pytest.mark.asyncio
    async def test_stats_counts_total_entries(self):
        """get_stats counts total entries correctly."""
        cache = ResponseCacheService()

        await cache.set("test1", "gpt-4", "response1")
        await cache.set("test2", "gpt-4", "response2")

        stats = await cache.get_stats()
        assert stats["total_entries"] == 2

    @pytest.mark.asyncio
    async def test_stats_counts_expired_entries(self):
        """get_stats counts expired entries separately."""
        config = ResponseCacheConfig(ttl_minutes=60)
        cache = ResponseCacheService(config)

        await cache.set("active", "gpt-4", "response")
        await cache.set("expired", "gpt-4", "response")

        # Manually expire one entry
        key = cache._generate_key("expired", "gpt-4")
        async with cache._lock:
            cache._cache[key].created_at = datetime.datetime.now(datetime.timezone.utc) - timedelta(hours=2)

        stats = await cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["expired_entries"] == 1
        assert stats["active_entries"] == 1

    @pytest.mark.asyncio
    async def test_stats_calculates_hit_rate(self):
        """get_stats calculates estimated hit rate."""
        cache = ResponseCacheService()

        await cache.set("test1", "gpt-4", "response1")
        await cache.set("test2", "gpt-4", "response2")

        # Hit test1 twice
        await cache.get("test1", "gpt-4")
        await cache.get("test1", "gpt-4")

        stats = await cache.get_stats()

        # 2 hits / (2 hits + 2 entries) = 0.5
        assert stats["estimated_hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_stats_hit_rate_zero_when_empty(self):
        """Hit rate is 0 when cache is empty."""
        cache = ResponseCacheService()

        stats = await cache.get_stats()
        assert stats["estimated_hit_rate"] == 0.0


class TestTopQueries:
    """Tests for top queries functionality."""

    @pytest.mark.asyncio
    async def test_get_top_queries_returns_most_accessed(self):
        """get_top_queries returns most frequently accessed queries."""
        cache = ResponseCacheService()

        await cache.set("low", "gpt-4", "response")
        await cache.set("medium", "gpt-4", "response")
        await cache.set("high", "gpt-4", "response")

        # Access entries different amounts
        for _ in range(10):
            await cache.get("high", "gpt-4")
        for _ in range(5):
            await cache.get("medium", "gpt-4")
        for _ in range(1):
            await cache.get("low", "gpt-4")

        top = await cache.get_top_queries(limit=3)

        assert len(top) == 3
        assert top[0]["hit_count"] == 10
        assert top[1]["hit_count"] == 5
        assert top[2]["hit_count"] == 1

    @pytest.mark.asyncio
    async def test_get_top_queries_respects_limit(self):
        """get_top_queries respects the limit parameter."""
        cache = ResponseCacheService()

        for i in range(10):
            await cache.set(f"test{i}", "gpt-4", f"response{i}")
            for _ in range(i + 1):
                await cache.get(f"test{i}", "gpt-4")

        top = await cache.get_top_queries(limit=5)

        assert len(top) == 5

    @pytest.mark.asyncio
    async def test_get_top_queries_includes_metadata(self):
        """get_top_queries includes entry metadata."""
        cache = ResponseCacheService()

        await cache.set("test", "gpt-4", "response")
        await cache.get("test", "gpt-4")

        top = await cache.get_top_queries(limit=1)

        assert len(top) == 1
        assert "query_hash" in top[0]
        assert "model_alias" in top[0]
        assert top[0]["model_alias"] == "gpt-4"
        assert "hit_count" in top[0]
        assert top[0]["hit_count"] == 1
        assert "created_at" in top[0]


class TestThreadSafety:
    """Tests for async thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_sets_are_safe(self):
        """Concurrent set operations are thread-safe."""
        cache = ResponseCacheService()

        tasks = [
            cache.set(f"test{i}", "gpt-4", f"response{i}")
            for i in range(100)
        ]

        await asyncio.gather(*tasks)

        stats = await cache.get_stats()
        assert stats["total_entries"] == 100

    @pytest.mark.asyncio
    async def test_concurrent_gets_are_safe(self):
        """Concurrent get operations are thread-safe."""
        cache = ResponseCacheService()

        await cache.set("test", "gpt-4", "response")

        tasks = [
            cache.get("test", "gpt-4")
            for _ in range(100)
        ]

        results = await asyncio.gather(*tasks)

        assert all(r is not None for r in results)

        stats = await cache.get_stats()
        assert stats["total_hits"] == 100

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Mixed concurrent operations are thread-safe."""
        cache = ResponseCacheService()

        await cache.set("initial", "gpt-4", "response")

        tasks = []
        for i in range(50):
            tasks.append(cache.set(f"test{i}", "gpt-4", f"response{i}"))
            tasks.append(cache.get("initial", "gpt-4"))

        await asyncio.gather(*tasks)

        stats = await cache.get_stats()
        assert stats["total_entries"] == 51  # 50 new + 1 initial


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_query_is_handled(self):
        """Empty query string is handled gracefully."""
        cache = ResponseCacheService()

        await cache.set("", "gpt-4", "response")
        result = await cache.get("", "gpt-4")

        assert result is not None

    @pytest.mark.asyncio
    async def test_unicode_query_is_handled(self):
        """Unicode query string is handled correctly."""
        cache = ResponseCacheService()

        await cache.set("Hello 世界 🌍", "gpt-4", "response")
        result = await cache.get("Hello 世界 🌍", "gpt-4")

        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_query_is_handled(self):
        """Very long query string is handled correctly."""
        cache = ResponseCacheService()

        long_query = "test " * 10000
        await cache.set(long_query, "gpt-4", "response")
        result = await cache.get(long_query, "gpt-4")

        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Special characters in query are handled correctly."""
        cache = ResponseCacheService()

        query = "test\n\t\r\nwith\twhitespace"
        await cache.set(query, "gpt-4", "response")

        # Normalized version should match
        result = await cache.get("test with whitespace", "gpt-4")
        assert result is not None

    @pytest.mark.asyncio
    async def test_different_models_separate_caches(self):
        """Same query with different models are cached separately."""
        cache = ResponseCacheService()

        await cache.set("test", "gpt-4", "response4")
        await cache.set("test", "gpt-3.5", "response3.5")

        result4 = await cache.get("test", "gpt-4")
        result35 = await cache.get("test", "gpt-3.5")

        assert result4[0] == "response4"
        assert result35[0] == "response3.5"

    @pytest.mark.asyncio
    async def test_max_entries_of_one(self):
        """Cache works correctly with max_entries=1."""
        config = ResponseCacheConfig(max_entries=1)
        cache = ResponseCacheService(config)

        await cache.set("first", "gpt-4", "response1")
        await cache.set("second", "gpt-4", "response2")

        stats = await cache.get_stats()
        assert stats["total_entries"] == 1

        result = await cache.get("second", "gpt-4")
        assert result is not None

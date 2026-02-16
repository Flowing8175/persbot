"""Generic cache manager with TTL support and automatic cleanup."""

import asyncio
import datetime
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CachedItem:
    """A cached item with expiration metadata."""

    value: T
    expiration: Optional[datetime.datetime]
    cache_key: str
    created_at: datetime.datetime
    last_accessed_at: datetime.datetime  # For true LRU eviction


class CacheStrategy(ABC):
    """Abstract base class for cache key generation strategies."""

    @abstractmethod
    def generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from the given arguments."""
        pass


class HashBasedCacheStrategy(CacheStrategy):
    """Generate cache keys using hash of arguments."""

    def __init__(self, prefix: str = "cache", hash_length: int = 10):
        self.prefix = prefix
        self.hash_length = hash_length

    def generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key using SHA256 hash."""
        # Combine all arguments into a single string
        parts = [str(arg) for arg in args]
        parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        combined = ":".join(parts)

        # Generate hash
        hash_value = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return f"{self.prefix}-{hash_value[: self.hash_length]}"


class CacheManager:
    """
    Generic cache manager with TTL support and automatic cleanup.

    Features:
    - Thread-safe operations
    - Automatic expiration
    - Configurable TTL
    - Cache size limits
    - Background cleanup task
    """

    def __init__(
        self,
        ttl_minutes: int = 60,
        max_size: int = 200,
        cleanup_interval_minutes: int = 30,
        strategy: Optional[CacheStrategy] = None,
    ):
        """
        Initialize the cache manager.

        Args:
            ttl_minutes: Default time-to-live for cache entries in minutes.
            max_size: Maximum number of entries to store.
            cleanup_interval_minutes: Interval between cleanup cycles.
            strategy: Strategy for generating cache keys.
        """
        self.ttl_minutes = ttl_minutes
        self.max_size = max_size
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.strategy = strategy or HashBasedCacheStrategy()

        self._cache: Dict[str, CachedItem] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key using the configured strategy."""
        return self.strategy.generate_key(*args, **kwargs)

    def _calculate_expiration(
        self, ttl_minutes: Optional[int] = None
    ) -> Optional[datetime.datetime]:
        """Calculate the expiration time for a cache entry."""
        if ttl_minutes is None or ttl_minutes <= 0:
            return None
        return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            minutes=ttl_minutes
        )

    def _is_expired(self, item: CachedItem) -> bool:
        """Check if a cache entry has expired."""
        if item.expiration is None:
            return False
        return datetime.datetime.now(datetime.timezone.utc) >= item.expiration

    async def get(
        self,
        *args,
        default: Optional[T] = None,
    ) -> Optional[T]:
        """
        Get a value from the cache.

        Args:
            *args: Arguments to generate the cache key.
            default: Value to return if key not found or expired.

        Returns:
            The cached value, or default if not found/expired.
        """
        key = self._generate_key(*args)

        async with self._lock:
            item = self._cache.get(key)
            if item is None:
                return default

            if self._is_expired(item):
                del self._cache[key]
                logger.debug("Cache entry expired: %s", key)
                return default

            # Update last accessed time for LRU tracking
            item.last_accessed_at = datetime.datetime.now(datetime.timezone.utc)
            return item.value

    async def set(
        self,
        value: T,
        *args,
        ttl_minutes: Optional[int] = None,
    ) -> str:
        """
        Set a value in the cache.

        Args:
            value: Value to cache.
            *args: Arguments to generate the cache key.
            ttl_minutes: Override default TTL for this entry.

        Returns:
            The cache key used.
        """
        key = self._generate_key(*args)
        ttl = ttl_minutes if ttl_minutes is not None else self.ttl_minutes

        async with self._lock:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()

            now = datetime.datetime.now(datetime.timezone.utc)
            item = CachedItem(
                value=value,
                expiration=self._calculate_expiration(ttl),
                cache_key=key,
                created_at=now,
                last_accessed_at=now,
            )
            self._cache[key] = item
            logger.debug("Cached value with key: %s (TTL: %s min)", key, ttl)

        return key

    async def delete(self, *args) -> bool:
        """
        Delete a value from the cache.

        Args:
            *args: Arguments to generate the cache key.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        key = self._generate_key(*args)

        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug("Deleted cache entry: %s", key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        async with self._lock:
            self._cache.clear()
            logger.debug("Cache cleared")

    async def get_or_create(
        self,
        factory: Callable[[], T],
        *args,
        ttl_minutes: Optional[int] = None,
    ) -> Tuple[T, bool]:
        """
        Get a value from cache or create it using the factory function.

        Args:
            factory: Function to create the value if not cached.
            *args: Arguments to generate the cache key.
            ttl_minutes: Override default TTL for this entry.

        Returns:
            Tuple of (value, was_created) where was_created is True if
            the value was created (not cached).
        """
        # First try to get from cache
        cached = await self.get(*args)
        if cached is not None:
            return cached, False

        # Not in cache, create using factory
        value = factory()
        await self.set(value, *args, ttl_minutes=ttl_minutes)
        return value, True

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry from the cache (true LRU)."""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed_at,
        )
        del self._cache[lru_key]
        logger.debug("Evicted LRU cache entry: %s", lru_key)

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        removed = 0

        async with self._lock:
            expired_keys = [key for key, item in self._cache.items() if self._is_expired(item)]

            for key in expired_keys:
                del self._cache[key]
                removed += 1

        if removed > 0:
            logger.debug("Cleaned up %d expired cache entries", removed)

        return removed

    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            logger.warning("Cleanup task already running")
            return

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.debug("Started cache cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.debug("Cache cleanup task cancelled")
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up expired entries."""
        interval_seconds = self.cleanup_interval_minutes * 60

        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                logger.debug("Cache cleanup loop cancelled")
                break
            except Exception as e:
                logger.error("Error in cache cleanup loop: %s", e, exc_info=True)

    def size(self) -> int:
        """Get the current number of entries in the cache."""
        return len(self._cache)

    async def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics.
        """
        async with self._lock:
            total = len(self._cache)
            expired = sum(1 for item in self._cache.values() if self._is_expired(item))

            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "max_size": self.max_size,
                "ttl_minutes": self.ttl_minutes,
            }

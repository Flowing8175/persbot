"""Cache service for managing LLM context caches.

This module provides a unified interface for managing context caches
across different LLM providers, with special support for Gemini's
CachedContent API.
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional, Union

from persbot.constants import CacheConfig, CacheLimit
from persbot.exceptions import CacheException, ContextCacheException

logger = logging.getLogger(__name__)


class CacheStatus(str, Enum):
    """Status of a cache entry."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REFRESHING = "refreshing"
    CREATING = "creating"
    ERROR = "error"


@dataclass
class CacheEntry:
    """Represents a cached content entry."""

    name: str
    display_name: str
    model: str
    system_instruction: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    ttl_seconds: int = CacheConfig.TTL_MINUTES * 60
    token_count: int = 0
    status: CacheStatus = CacheStatus.ACTIVE
    tools: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def needs_refresh(self) -> bool:
        """Check if this cache entry needs refresh soon."""
        if self.expires_at is None:
            return False
        # Refresh if within buffer period
        buffer = timedelta(minutes=CacheConfig.REFRESH_BUFFER_MIN)
        return datetime.now(timezone.utc) >= (self.expires_at - buffer)

    @property
    def age_seconds(self) -> float:
        """Get the age of this cache entry in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


@dataclass
class CacheResult:
    """Result of a cache operation."""

    success: bool
    cache_name: Optional[str] = None
    entry: Optional[CacheEntry] = None
    error: Optional[str] = None
    created: bool = False  # True if a new cache was created


class CacheStrategy(ABC):
    """Base class for cache strategies."""

    @abstractmethod
    async def get_or_create(
        self,
        model: str,
        system_instruction: str,
        tools: Optional[list] = None,
        min_tokens: int = CacheConfig.MIN_TOKENS,
    ) -> CacheResult:
        """Get existing cache or create new one.

        Args:
            model: The model name.
            system_instruction: The system instruction text.
            tools: Optional list of tools for the cache.
            min_tokens: Minimum token count for caching.

        Returns:
            CacheResult with cache entry or error.
        """
        pass

    @abstractmethod
    async def refresh(self, cache_name: str, ttl_seconds: int) -> bool:
        """Refresh an existing cache entry's TTL.

        Args:
            cache_name: The name of the cache to refresh.
            ttl_seconds: New TTL in seconds.

        Returns:
            True if refresh was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def delete(self, cache_name: str) -> bool:
        """Delete a cache entry.

        Args:
            cache_name: The name of the cache to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def list_all(
        self, limit: int = CacheLimit.MAX_CACHED_ITEMS
    ) -> list[CacheEntry]:
        """List all cache entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of cache entries.
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed.
        """
        pass


class InMemoryCacheStrategy(CacheStrategy):
    """In-memory cache strategy for testing and local caching."""

    def __init__(self):
        self._caches: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, model: str, system_instruction: str, tools: Optional[list]) -> str:
        """Generate a cache key from model and content."""
        content = f"{model}:{system_instruction}"
        if tools:
            import json
            content += ":" + json.dumps([str(t) for t in tools], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get_or_create(
        self,
        model: str,
        system_instruction: str,
        tools: Optional[list] = None,
        min_tokens: int = CacheConfig.MIN_TOKENS,
    ) -> CacheResult:
        """Get or create cache entry."""
        key = self._make_key(model, system_instruction, tools)

        async with self._lock:
            # Check existing
            if key in self._caches:
                entry = self._caches[key]
                if not entry.is_expired:
                    return CacheResult(
                        success=True,
                        cache_name=entry.name,
                        entry=entry,
                        created=False,
                    )
                # Remove expired
                del self._caches[key]

            # Create new entry
            entry = CacheEntry(
                name=f"cache-{key}",
                display_name=f"Cache for {model}",
                model=model,
                system_instruction=system_instruction,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=CacheConfig.TTL_MINUTES),
                ttl_seconds=CacheConfig.TTL_MINUTES * 60,
                tools=tools or [],
            )
            self._caches[key] = entry
            return CacheResult(
                success=True,
                cache_name=entry.name,
                entry=entry,
                created=True,
            )

    async def refresh(self, cache_name: str, ttl_seconds: int) -> bool:
        """Refresh cache TTL."""
        async with self._lock:
            for key, entry in self._caches.items():
                if entry.name == cache_name:
                    entry.expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                    return True
            return False

    async def delete(self, cache_name: str) -> bool:
        """Delete cache entry."""
        async with self._lock:
            for key, entry in list(self._caches.items()):
                if entry.name == cache_name:
                    del self._caches[key]
                    return True
            return False

    async def list_all(self, limit: int = CacheLimit.MAX_CACHED_ITEMS) -> list[CacheEntry]:
        """List all cache entries."""
        async with self._lock:
            return list(self._caches.values())[:limit]

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            to_remove = [k for k, v in self._caches.items() if v.is_expired]
            for key in to_remove:
                del self._caches[key]
            return len(to_remove)


class GeminiCacheStrategy(CacheStrategy):
    """Gemini-specific cache strategy using the CachedContent API."""

    def __init__(self, client: Any):
        """Initialize with Gemini client.

        Args:
            client: The Google GenAI client instance.
        """
        self.client = client
        self._local_cache: dict[str, CacheEntry] = {}

    def _generate_display_name(self, model: str, system_instruction: str, tools: Optional[list]) -> str:
        """Generate a consistent display name for the cache."""
        import re
        safe_model = re.sub(r"[^a-zA-Z0-9-]", "-", model)
        content_hash = hashlib.sha256(system_instruction.encode("utf-8")).hexdigest()
        tool_suffix = "-tools" if tools else ""
        return f"persbot-{safe_model}-{content_hash[:10]}{tool_suffix}"

    async def get_or_create(
        self,
        model: str,
        system_instruction: str,
        tools: Optional[list] = None,
        min_tokens: int = CacheConfig.MIN_TOKENS,
    ) -> CacheResult:
        """Get or create Gemini cache entry."""
        display_name = self._generate_display_name(model, system_instruction, tools)

        try:
            # Count tokens to check if caching is worthwhile
            count_result = await asyncio.to_thread(
                self.client.models.count_tokens,
                model=model,
                contents=[system_instruction],
            )
            token_count = count_result.total_tokens

            if token_count < min_tokens:
                logger.info(
                    f"Cache skipped: tokens ({token_count}) < min_tokens ({min_tokens})"
                )
                return CacheResult(success=False, error="Token count below minimum")

            # Search for existing cache
            for cache in await asyncio.to_thread(self.client.caches.list):
                if cache.display_name == display_name:
                    # Refresh TTL
                    ttl_seconds = CacheConfig.TTL_MINUTES * 60
                    try:
                        await asyncio.to_thread(
                            self.client.caches.update,
                            name=cache.name,
                            config={"ttl": f"{ttl_seconds}s"},
                        )
                        logger.info(f"Refreshed existing cache: {cache.name}")
                        entry = CacheEntry(
                            name=cache.name,
                            display_name=display_name,
                            model=model,
                            system_instruction=system_instruction,
                            created_at=datetime.now(timezone.utc),  # Approximate
                            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
                            ttl_seconds=ttl_seconds,
                            token_count=token_count,
                            tools=tools or [],
                        )
                        return CacheResult(
                            success=True,
                            cache_name=cache.name,
                            entry=entry,
                            created=False,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to refresh cache {cache.name}: {e}")
                        # Continue to create new cache

            # Create new cache
            logger.info(f"Creating new cache '{display_name}'...")
            cache = await asyncio.to_thread(
                self.client.caches.create,
                model=model,
                config={
                    "display_name": display_name,
                    "system_instruction": system_instruction,
                    "tools": tools,
                    "ttl": f"{CacheConfig.TTL_MINUTES * 60}s",
                },
            )

            entry = CacheEntry(
                name=cache.name,
                display_name=display_name,
                model=model,
                system_instruction=system_instruction,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=CacheConfig.TTL_MINUTES),
                ttl_seconds=CacheConfig.TTL_MINUTES * 60,
                token_count=token_count,
                tools=tools or [],
            )

            return CacheResult(
                success=True,
                cache_name=cache.name,
                entry=entry,
                created=True,
            )

        except Exception as e:
            logger.error(f"Cache operation failed: {e}")
            return CacheResult(success=False, error=str(e))

    async def refresh(self, cache_name: str, ttl_seconds: int) -> bool:
        """Refresh Gemini cache TTL."""
        try:
            await asyncio.to_thread(
                self.client.caches.update,
                name=cache_name,
                config={"ttl": f"{ttl_seconds}s"},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to refresh cache {cache_name}: {e}")
            return False

    async def delete(self, cache_name: str) -> bool:
        """Delete Gemini cache entry."""
        try:
            await asyncio.to_thread(self.client.caches.delete, name=cache_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache {cache_name}: {e}")
            return False

    async def list_all(self, limit: int = CacheLimit.MAX_CACHED_ITEMS) -> list[CacheEntry]:
        """List all Gemini caches."""
        try:
            caches = await asyncio.to_thread(self.client.caches.list)
            entries = []
            for cache in list(caches)[:limit]:
                entry = CacheEntry(
                    name=cache.name,
                    display_name=getattr(cache, "display_name", cache.name),
                    model=getattr(cache, "model", "unknown"),
                    system_instruction="",  # Not available from list
                    created_at=datetime.now(timezone.utc),  # Not available
                    status=CacheStatus.ACTIVE,
                )
                entries.append(entry)
            return entries
        except Exception as e:
            logger.error(f"Failed to list caches: {e}")
            return []

    async def cleanup_expired(self) -> int:
        """Gemini caches auto-expire, so this is a no-op."""
        return 0


class CacheService:
    """Unified cache service managing multiple cache strategies."""

    def __init__(self, strategy: Optional[CacheStrategy] = None):
        """Initialize the cache service.

        Args:
            strategy: The cache strategy to use. Defaults to InMemoryCacheStrategy.
        """
        self.strategy = strategy or InMemoryCacheStrategy()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def get_or_create(
        self,
        model: str,
        system_instruction: str,
        tools: Optional[list] = None,
        min_tokens: int = CacheConfig.MIN_TOKENS,
    ) -> CacheResult:
        """Get existing cache or create new one.

        Args:
            model: The model name.
            system_instruction: The system instruction text.
            tools: Optional list of tools for the cache.
            min_tokens: Minimum token count for caching.

        Returns:
            CacheResult with cache entry or error.
        """
        return await self.strategy.get_or_create(model, system_instruction, tools, min_tokens)

    async def refresh(self, cache_name: str, ttl_seconds: Optional[int] = None) -> bool:
        """Refresh an existing cache entry's TTL.

        Args:
            cache_name: The name of the cache to refresh.
            ttl_seconds: New TTL in seconds (defaults to config).

        Returns:
            True if refresh was successful, False otherwise.
        """
        ttl = ttl_seconds or (CacheConfig.TTL_MINUTES * 60)
        return await self.strategy.refresh(cache_name, ttl)

    async def delete(self, cache_name: str) -> bool:
        """Delete a cache entry.

        Args:
            cache_name: The name of the cache to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        return await self.strategy.delete(cache_name)

    async def list_all(
        self, limit: int = CacheLimit.MAX_CACHED_ITEMS
    ) -> list[CacheEntry]:
        """List all cache entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of cache entries.
        """
        return await self.strategy.list_all(limit)

    async def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed.
        """
        return await self.strategy.cleanup_expired()

    async def start_periodic_cleanup(self, interval_seconds: Optional[int] = None) -> None:
        """Start periodic cleanup of expired cache entries.

        Args:
            interval_seconds: Cleanup interval in seconds.
                            Defaults to CacheConfig.CLEANUP_INTERVAL_MULTIPLIER.
        """
        if self._running:
            logger.warning("Periodic cleanup already running")
            return

        self._running = True
        interval = interval_seconds or (CacheConfig.TTL_MINUTES * 30)

        async def _cleanup_loop():
            while self._running:
                try:
                    await asyncio.sleep(interval)
                    if self._running:
                        removed = await self.cleanup_expired()
                        if removed > 0:
                            logger.info(f"Periodic cleanup: removed {removed} expired entries")
                except asyncio.CancelledError:
                    logger.info("Periodic cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in periodic cleanup: {e}", exc_info=True)

        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info(f"Started periodic cache cleanup (interval: {interval}s)")

    async def stop_periodic_cleanup(self) -> None:
        """Stop periodic cleanup."""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped periodic cache cleanup")

    async def __aenter__(self):
        """Context manager entry."""
        await self.start_periodic_cleanup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop_periodic_cleanup()

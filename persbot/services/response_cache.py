"""Response caching service for reducing redundant LLM calls.

This module provides a caching layer for LLM responses to reduce
API costs and latency for repetitive queries.
"""

import asyncio
import datetime
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached LLM response."""

    query_hash: str
    response_text: str
    model_alias: str
    created_at: datetime.datetime
    last_accessed_at: datetime.datetime
    response_obj: Any = None  # Optional full response object
    hit_count: int = 0


@dataclass
class ResponseCacheConfig:
    """Configuration for response caching."""

    enabled: bool = True
    ttl_minutes: int = 60
    max_entries: int = 500
    normalize_whitespace: bool = True
    case_sensitive: bool = False


class ResponseCacheService:
    """Service for caching LLM responses to reduce redundant API calls.

    Features:
    - Hash-based query matching (normalized text + model alias)
    - LRU eviction when cache is full
    - Configurable TTL (default 60 minutes)
    - Enabled by default with opt-out via config
    - Thread-safe async operations

    Usage:
        cache = ResponseCacheService()

        # Check cache before API call
        cached = await cache.get(user_message, model_alias)
        if cached:
            return cached

        # Store response after API call
        await cache.set(user_message, model_alias, response_text)
    """

    def __init__(self, config: Optional[ResponseCacheConfig] = None):
        """Initialize the response cache service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ResponseCacheConfig()
        self._cache: Dict[str, CachedResponse] = {}
        self._lock = asyncio.Lock()

    @property
    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.config.enabled

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing.

        Args:
            query: The original query string.

        Returns:
            Normalized query string.
        """
        text = query

        if not self.config.case_sensitive:
            text = text.lower()

        if self.config.normalize_whitespace:
            # Collapse multiple whitespace to single space
            import re
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        return text

    def _generate_key(self, query: str, model_alias: str) -> str:
        """Generate a cache key from query and model.

        Args:
            query: The user's query.
            model_alias: The model alias used.

        Returns:
            A hash-based cache key.
        """
        normalized = self._normalize_query(query)
        combined = f"{model_alias}:{normalized}"

        # Use SHA256 for consistent hashing
        hash_value = hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
        return f"response:{hash_value}"

    def _is_expired(self, entry: CachedResponse) -> bool:
        """Check if a cache entry has expired.

        Args:
            entry: The cache entry to check.

        Returns:
            True if expired, False otherwise.
        """
        if self.config.ttl_minutes <= 0:
            return False

        age = datetime.datetime.now(datetime.timezone.utc) - entry.created_at
        return age.total_seconds() > (self.config.ttl_minutes * 60)

    async def get(
        self,
        query: str,
        model_alias: str,
    ) -> Optional[Tuple[str, Any]]:
        """Get a cached response if available.

        Args:
            query: The user's query.
            model_alias: The model alias.

        Returns:
            Tuple of (response_text, response_obj) if cached, None otherwise.
        """
        if not self.config.enabled:
            return None

        key = self._generate_key(query, model_alias)

        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if self._is_expired(entry):
                del self._cache[key]
                return None

            # Update access time for LRU
            entry.last_accessed_at = datetime.datetime.now(datetime.timezone.utc)
            entry.hit_count += 1

            return entry.response_text, entry.response_obj

    async def set(
        self,
        query: str,
        model_alias: str,
        response_text: str,
        response_obj: Any = None,
    ) -> None:
        """Store a response in the cache.

        Args:
            query: The user's query.
            model_alias: The model alias.
            response_text: The response text to cache.
            response_obj: Optional full response object.
        """
        if not self.config.enabled:
            return

        key = self._generate_key(query, model_alias)
        now = datetime.datetime.now(datetime.timezone.utc)

        async with self._lock:
            # Evict LRU entry if at capacity
            if len(self._cache) >= self.config.max_entries and key not in self._cache:
                self._evict_lru()

            entry = CachedResponse(
                query_hash=key,
                response_text=response_text,
                model_alias=model_alias,
                created_at=now,
                last_accessed_at=now,
                response_obj=response_obj,
                hit_count=0,
            )
            self._cache[key] = entry


    def _evict_lru(self) -> None:
        """Evict the least recently used entry. Must be called within lock."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed_at,
        )
        del self._cache[lru_key]

    async def invalidate(self, query: str, model_alias: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            query: The user's query.
            model_alias: The model alias.

        Returns:
            True if entry was found and removed, False otherwise.
        """
        key = self._generate_key(query, model_alias)

        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of entries cleared.
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        removed = 0

        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]

            for key in expired_keys:
                del self._cache[key]
                removed += 1

        if removed > 0:
            pass  # Logging removed

        return removed

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        async with self._lock:
            total_entries = len(self._cache)
            total_hits = sum(e.hit_count for e in self._cache.values())
            expired_count = sum(1 for e in self._cache.values() if self._is_expired(e))

            # Calculate hit rate (approximate)
            total_accesses = total_hits + total_entries  # hits + misses (new entries)
            hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0

            return {
                "enabled": self.config.enabled,
                "total_entries": total_entries,
                "expired_entries": expired_count,
                "active_entries": total_entries - expired_count,
                "max_entries": self.config.max_entries,
                "ttl_minutes": self.config.ttl_minutes,
                "total_hits": total_hits,
                "estimated_hit_rate": hit_rate,
            }

    async def get_top_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequently accessed cached queries.

        Args:
            limit: Maximum number of results.

        Returns:
            List of query statistics.
        """
        async with self._lock:
            sorted_entries = sorted(
                self._cache.values(),
                key=lambda e: e.hit_count,
                reverse=True,
            )[:limit]

            return [
                {
                    "query_hash": entry.query_hash[:20],
                    "model_alias": entry.model_alias,
                    "hit_count": entry.hit_count,
                    "created_at": entry.created_at.isoformat(),
                }
                for entry in sorted_entries
            ]

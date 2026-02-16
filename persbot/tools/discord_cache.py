"""Discord API response caching to reduce API calls and improve performance."""

import asyncio
import logging
from datetime import timedelta
from typing import Any, Dict, Optional

from persbot.services.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Global cache instance for Discord API responses
# Using shorter TTL since Discord state changes frequently
_discord_cache: Optional[CacheManager] = None
_cache_lock = asyncio.Lock()


async def get_discord_cache() -> CacheManager:
    """Get or create the global Discord cache instance."""
    global _discord_cache

    async with _cache_lock:
        if _discord_cache is None:
            _discord_cache = CacheManager(
                ttl_minutes=5,  # 5 minute TTL for Discord data
                max_size=500,   # Cache up to 500 items
                cleanup_interval_minutes=5,
            )
            await _discord_cache.start_cleanup_task()
        return _discord_cache


async def get_cached_member(guild_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """Get cached member data if available."""
    cache = await get_discord_cache()
    return await cache.get("member", guild_id, user_id)


async def cache_member(guild_id: int, user_id: int, data: Dict[str, Any]) -> None:
    """Cache member data."""
    cache = await get_discord_cache()
    await cache.set(data, "member", guild_id, user_id)


async def get_cached_message(channel_id: int, message_id: int) -> Optional[Dict[str, Any]]:
    """Get cached message data if available."""
    cache = await get_discord_cache()
    return await cache.get("message", channel_id, message_id)


async def cache_message(channel_id: int, message_id: int, data: Dict[str, Any]) -> None:
    """Cache message data."""
    cache = await get_discord_cache()
    await cache.set(data, "message", channel_id, message_id)


async def get_cached_user(user_id: int) -> Optional[Dict[str, Any]]:
    """Get cached user data if available."""
    cache = await get_discord_cache()
    return await cache.get("user", user_id)


async def cache_user(user_id: int, data: Dict[str, Any]) -> None:
    """Cache user data."""
    cache = await get_discord_cache()
    await cache.set(data, "user", user_id)


async def invalidate_member_cache(guild_id: int, user_id: int) -> None:
    """Invalidate cached member data."""
    cache = await get_discord_cache()
    await cache.delete("member", guild_id, user_id)


async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache = await get_discord_cache()
    return await cache.stats()


async def cleanup_discord_cache() -> None:
    """Cleanup the Discord cache on shutdown."""
    global _discord_cache

    async with _cache_lock:
        if _discord_cache is not None:
            await _discord_cache.stop_cleanup_task()
            await _discord_cache.clear()
            _discord_cache = None

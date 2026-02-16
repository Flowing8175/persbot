"""Rate limiting utilities for API calls and tool usage."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Hashable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_requests: int = 10  # Maximum number of requests allowed
    window_seconds: int = 60  # Time window in seconds
    per_user: bool = True  # If True, limit per user; if False, global limit


class SlidingWindowRateLimiter:
    """
    A thread-safe rate limiter using the sliding window algorithm.

    This implementation tracks request timestamps in a deque, allowing
    for precise rate limiting with a sliding time window. Old timestamps
    are automatically evicted as the window advances.

    Example:
        # Allow 5 requests per 60 seconds per user
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        if await limiter.acquire(user_id="user123"):
            # Process request
        else:
            # Rate limited - limiter.get_retry_after() shows wait time
    """

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: int = 60,
        per_user: bool = True,
    ):
        """Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window.
            window_seconds: Length of the time window in seconds.
            per_user: If True, track limits per unique key. If False, use global limit.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.per_user = per_user

        # Use a single global lock for simplicity
        # For high-traffic scenarios, consider per-key locks
        self._lock = asyncio.Lock()
        self._windows: Dict[Hashable, deque] = {} if per_user else {"": deque()}
        self._global_window = deque() if not per_user else None

    def _get_window(self, key: Hashable) -> deque:
        """Get the request window for a given key."""
        if self.per_user:
            if key not in self._windows:
                self._windows[key] = deque()
            return self._windows[key]
        return self._windows[""]

    def _cleanup_old_requests(self, window: deque, current_time: float) -> None:
        """Remove timestamps outside the current time window."""
        cutoff_time = current_time - self.window_seconds
        while window and window[0] <= cutoff_time:
            window.popleft()

    async def acquire(
        self,
        key: Optional[Hashable] = None,
        wait: bool = False,
    ) -> bool:
        """
        Attempt to acquire a rate limit slot.

        Args:
            key: Unique identifier for the requester (user_id, channel_id, etc.).
                 If per_user is False, this parameter is ignored.
            wait: If True, wait until a slot is available. If False, return False immediately.

        Returns:
            True if the request is allowed, False if rate limited (only when wait=False).
        """
        async with self._lock:
            window = self._get_window(key if self.per_user else "")
            current_time = time.time()

            # Clean up old requests outside the window
            self._cleanup_old_requests(window, current_time)

            if len(window) < self.max_requests:
                # Request allowed
                window.append(current_time)
                return True

            # Rate limited
            if not wait:
                return False

            # Wait for a slot to become available
            oldest_request = window[0]
            wait_time = oldest_request + self.window_seconds - current_time

            if wait_time > 0:
                await asyncio.sleep(wait_time)

                # Re-check after waiting (with lock released during sleep)
                return await self.acquire(key=key, wait=False)

            return True

    def get_usage_count(self, key: Optional[Hashable] = None) -> int:
        """
        Get the current usage count for a key.

        Args:
            key: Unique identifier for the requester.

        Returns:
            Number of requests made within the current window.
        """
        window = self._get_window(key if self.per_user else "")
        current_time = time.time()
        self._cleanup_old_requests(window, current_time)
        return len(window)

    def get_retry_after(self, key: Optional[Hashable] = None) -> float:
        """
        Get the time in seconds until a new request will be allowed.

        Args:
            key: Unique identifier for the requester.

        Returns:
            Seconds until the next request can be made, or 0 if immediately available.
        """
        window = self._get_window(key if self.per_user else "")
        current_time = time.time()
        self._cleanup_old_requests(window, current_time)

        if len(window) < self.max_requests:
            return 0.0

        oldest_request = window[0]
        retry_after = oldest_request + self.window_seconds - current_time
        return max(0.0, retry_after)

    def reset(self, key: Optional[Hashable] = None) -> None:
        """
        Reset the rate limit counter for a specific key or all keys.

        Args:
            key: Unique identifier for the requester. If None and per_user is True,
                 resets all keys.
        """
        if self.per_user:
            if key is None:
                self._windows.clear()
            elif key in self._windows:
                self._windows[key].clear()
        else:
            self._windows[""].clear()


class TokenBucketRateLimiter:
    """
    A thread-safe rate limiter using the token bucket algorithm.

    The token bucket algorithm allows for bursts of requests while maintaining
    a long-term average rate. Tokens are added to the bucket at a fixed rate,
    and requests consume tokens. If the bucket is empty, requests are blocked.

    This is ideal for APIs that allow burst traffic but enforce rate limits.

    Example:
        # Allow 10 requests initially, refilling at 1 request per 6 seconds
        # (effectively 10 requests per minute, but allows bursts)
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1/6)
        if await limiter.acquire(user_id="user123"):
            # Process request
    """

    def __init__(
        self,
        capacity: int = 10,
        refill_rate: float = 1.0,  # tokens per second
        per_user: bool = True,
    ):
        """Initialize the token bucket rate limiter.

        Args:
            capacity: Maximum number of tokens the bucket can hold.
            refill_rate: Rate at which tokens are added (tokens per second).
            per_user: If True, track buckets per unique key.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.per_user = per_user

        self._lock = asyncio.Lock()
        # Each bucket stores (token_count, last_refill_time)
        self._buckets: Dict[Hashable, tuple] = {} if per_user else {"": (capacity, time.time())}

    def _get_bucket(self, key: Hashable) -> tuple:
        """Get or create a token bucket for a given key."""
        if self.per_user:
            if key not in self._buckets:
                self._buckets[key] = (self.capacity, time.time())
            return self._buckets[key]
        return self._buckets[""]

    def _refill_tokens(self, bucket: tuple, current_time: float) -> float:
        """Refill tokens based on elapsed time since last refill."""
        token_count, last_refill_time = bucket
        elapsed = current_time - last_refill_time

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        new_count = min(self.capacity, token_count + new_tokens)

        return new_count

    async def acquire(
        self,
        key: Optional[Hashable] = None,
        tokens: float = 1.0,
        wait: bool = False,
    ) -> bool:
        """
        Attempt to acquire tokens from the bucket.

        Args:
            key: Unique identifier for the requester.
            tokens: Number of tokens to acquire (default: 1.0).
            wait: If True, wait until tokens are available.

        Returns:
            True if tokens were acquired, False if insufficient tokens (only when wait=False).
        """
        async with self._lock:
            bucket_key = key if self.per_user else ""
            bucket = self._get_bucket(bucket_key)
            current_time = time.time()

            token_count = self._refill_tokens(bucket, current_time)

            if token_count >= tokens:
                # Enough tokens available
                self._buckets[bucket_key] = (token_count - tokens, current_time)
                return True

            if not wait:
                return False

            # Calculate wait time and wait
            tokens_needed = tokens - token_count
            wait_time = tokens_needed / self.refill_rate

            await asyncio.sleep(wait_time)

            # Re-check after waiting
            return await self.acquire(key=key, tokens=tokens, wait=False)

    def get_available_tokens(self, key: Optional[Hashable] = None) -> float:
        """
        Get the number of available tokens for a key.

        Args:
            key: Unique identifier for the requester.

        Returns:
            Number of tokens currently available.
        """
        bucket_key = key if self.per_user else ""
        if bucket_key not in self._buckets:
            return float(self.capacity)

        bucket = self._buckets[bucket_key]
        current_time = time.time()
        return self._refill_tokens(bucket, current_time)

    def get_retry_after(self, key: Optional[Hashable] = None, tokens: float = 1.0) -> float:
        """
        Get the time in seconds until the requested tokens will be available.

        Args:
            key: Unique identifier for the requester.
            tokens: Number of tokens needed.

        Returns:
            Seconds until tokens will be available, or 0 if immediately available.
        """
        available = self.get_available_tokens(key)
        if available >= tokens:
            return 0.0

        tokens_needed = tokens - available
        return tokens_needed / self.refill_rate

    def reset(self, key: Optional[Hashable] = None) -> None:
        """
        Reset the token bucket for a specific key or all keys.

        Args:
            key: Unique identifier for the requester. If None and per_user is True,
                 resets all keys.
        """
        if self.per_user:
            if key is None:
                self._buckets.clear()
            elif key in self._buckets:
                self._buckets[key] = (self.capacity, time.time())
        else:
            self._buckets[""] = (self.capacity, time.time())


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    retry_after: float = 0.0
    current_usage: int = 0
    max_requests: int = 0
    message: str = ""


class ImageGenerationRateLimiter:
    """
    Specialized rate limiter for image generation requests.

    Provides per-user rate limiting with configurable limits and
    helpful error messages for users.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 3,
        max_requests_per_hour: int = 15,
    ):
        """Initialize the image generation rate limiter.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute per user.
            max_requests_per_hour: Maximum requests allowed per hour per user.
        """
        self.per_minute = SlidingWindowRateLimiter(
            max_requests=max_requests_per_minute,
            window_seconds=60,
            per_user=True,
        )
        self.per_hour = SlidingWindowRateLimiter(
            max_requests=max_requests_per_hour,
            window_seconds=3600,
            per_user=True,
        )
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour

    async def check_rate_limit(
        self,
        user_id: Hashable,
    ) -> RateLimitResult:
        """
        Check if a user is within rate limits for image generation.

        Args:
            user_id: User identifier (Discord user ID, etc.).

        Returns:
            RateLimitResult indicating if the request is allowed.
        """
        # Check per-minute limit first (more restrictive)
        minute_usage = self.per_minute.get_usage_count(user_id)
        minute_retry = self.per_minute.get_retry_after(user_id)

        if minute_usage >= self.max_requests_per_minute and minute_retry > 0:
            return RateLimitResult(
                allowed=False,
                retry_after=minute_retry,
                current_usage=minute_usage,
                max_requests=self.max_requests_per_minute,
                message=f"The user has reached their image generation limit ({self.max_requests_per_minute} per minute). They can try again in about {int(minute_retry)} seconds.",
            )

        # Check per-hour limit
        hour_usage = self.per_hour.get_usage_count(user_id)
        hour_retry = self.per_hour.get_retry_after(user_id)

        if hour_usage >= self.max_requests_per_hour and hour_retry > 0:
            return RateLimitResult(
                allowed=False,
                retry_after=hour_retry,
                current_usage=hour_usage,
                max_requests=self.max_requests_per_hour,
                message=f"The user has reached their hourly image generation limit ({self.max_requests_per_hour} per hour). They can try again in about {int(hour_retry // 60)} minutes.",
            )

        # Both limits passed, acquire the request slot
        await self.per_minute.acquire(key=user_id)
        await self.per_hour.acquire(key=user_id)

        return RateLimitResult(
            allowed=True,
            current_usage=minute_usage + 1,
            max_requests=self.max_requests_per_minute,
        )

    def get_user_status(self, user_id: Hashable) -> dict:
        """Get current rate limit status for a user."""
        minute_usage = self.per_minute.get_usage_count(user_id)
        hour_usage = self.per_hour.get_usage_count(user_id)
        minute_retry = self.per_minute.get_retry_after(user_id)
        hour_retry = self.per_hour.get_retry_after(user_id)

        return {
            "minute_usage": minute_usage,
            "minute_limit": self.max_requests_per_minute,
            "hour_usage": hour_usage,
            "hour_limit": self.max_requests_per_hour,
            "minute_retry_after": minute_retry,
            "hour_retry_after": hour_retry,
        }


# Global rate limiter instance for image generation
_global_image_limiter: Optional[ImageGenerationRateLimiter] = None


def get_image_rate_limiter() -> ImageGenerationRateLimiter:
    """Get or create the global image generation rate limiter."""
    global _global_image_limiter
    if _global_image_limiter is None:
        from persbot.config import load_config

        config = load_config()
        _global_image_limiter = ImageGenerationRateLimiter(
            max_requests_per_minute=getattr(config, "image_rate_limit_per_minute", 3),
            max_requests_per_hour=getattr(config, "image_rate_limit_per_hour", 15),
        )
    return _global_image_limiter


def reset_image_rate_limiter() -> None:
    """Reset the global image generation rate limiter (for testing)."""
    global _global_image_limiter
    _global_image_limiter = None

"""Tests for rate_limiter.py module.

This module provides comprehensive test coverage for:
- RateLimitConfig dataclass
- SlidingWindowRateLimiter
- TokenBucketRateLimiter
- ImageGenerationRateLimiter
- Helper functions
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from persbot.rate_limiter import (
    RateLimitConfig,
    RateLimitResult,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    ImageGenerationRateLimiter,
    get_image_rate_limiter,
    reset_image_rate_limiter,
)


# =============================================================================
# RateLimitConfig Tests
# =============================================================================


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_init_with_default_values(self):
        """Test RateLimitConfig initialization."""
        config = RateLimitConfig(max_requests=10, window_seconds=60, per_user=True)
        assert config.max_requests == 10
        assert config.window_seconds == 60
        assert config.per_user is True

    def test_init_with_per_user_false(self):
        """Test RateLimitConfig with global limit (per_user=False)."""
        config = RateLimitConfig(max_requests=100, window_seconds=300, per_user=False)
        assert config.max_requests == 100
        assert config.per_user is False


# =============================================================================
# SlidingWindowRateLimiter Tests
# =============================================================================


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter class."""

    @pytest.mark.asyncio
    async def test_init_with_defaults(self):
        """Test initialization with default values."""
        limiter = SlidingWindowRateLimiter()
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60
        assert limiter.per_user is True

    @pytest.mark.asyncio
    async def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        limiter = SlidingWindowRateLimiter(
            max_requests=5, window_seconds=120, per_user=False
        )
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 120
        assert limiter.per_user is False

    @pytest.mark.asyncio
    async def test_acquire_first_request(self):
        """Test that first request is allowed."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        result = await limiter.acquire("user123")
        assert result is True
        assert limiter.get_usage_count("user123") == 1

    @pytest.mark.asyncio
    async def test_acquire_up_to_limit(self):
        """Test acquiring up to the limit."""
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)

        # Acquire 3 requests
        assert await limiter.acquire("user123") is True
        assert await limiter.acquire("user123") is True
        assert await limiter.acquire("user123") is True

        # 4th request should be rate limited
        result = await limiter.acquire("user123")
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limit_after_wait(self):
        """Test that request allowed after wait time passes."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1)

        # Use up the limit
        assert await limiter.acquire("user123") is True
        assert await limiter.acquire("user123") is True

        # Wait for window to pass
        await asyncio.sleep(1.1)

        # Should be allowed again
        result = await limiter.acquire("user123", wait=False)
        assert result is True

    @pytest.mark.asyncio
    async def test_get_usage_count(self):
        """Test getting usage count."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)

        await limiter.acquire("user123")
        await limiter.acquire("user123")

        count = limiter.get_usage_count("user123")
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_retry_after(self):
        """Test getting retry after time."""
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=10)

        await limiter.acquire("user123")

        retry_after = limiter.get_retry_after("user123")
        assert retry_after > 0

    @pytest.mark.asyncio
    async def test_per_user_isolation(self):
        """Test that per-user limits work independently."""
        limiter1 = SlidingWindowRateLimiter(max_requests=2, window_seconds=60, per_user=True)
        limiter2 = SlidingWindowRateLimiter(max_requests=2, window_seconds=60, per_user=True)

        # User1 acquires both slots
        assert await limiter1.acquire("user1") is True
        assert await limiter1.acquire("user1") is True

        # User2 should be blocked (different per-user instance)
        assert await limiter2.acquire("user2") is False

    @pytest.mark.asyncio
    async def test_global_limit_per_user_false(self):
        """Test global limit mode."""
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60, per_user=False)

        # Different users share the same limit
        assert await limiter.acquire("user1") is True
        assert await limiter.acquire("user2") is True
        assert limiter.acquire("user3") is False

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting rate limiter."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60, per_user=True)

        # Use up the limit
        await limiter.acquire("user123")
        await limiter.acquire("user123")

        # Reset
        limiter.reset("user123")

        # Should be allowed again
        result = await limiter.acquire("user123", wait=False)
        assert result is True


# =============================================================================
# TokenBucketRateLimiter Tests
# =============================================================================


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter class."""

    @pytest.mark.asyncio
    async def test_init_with_defaults(self):
        """Test initialization with default values."""
        limiter = TokenBucketRateLimiter()
        assert limiter.capacity == 10
        assert limiter.refill_rate == 1.0

    @pytest.mark.asyncio
    async def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        limiter = TokenBucketRateLimiter(capacity=20, refill_rate=2.0)
        assert limiter.capacity == 20
        assert limiter.refill_rate == 2.0

    @pytest.mark.asyncio
    async def test_acquire_single_token(self):
        """Test acquiring a single token."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0)

        result = await limiter.acquire("user123", tokens=1.0)
        assert result is True
        assert limiter.get_available_tokens("user123") == 9

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0)

        # Should be able to acquire up to capacity
        result = await limiter.acquire("user123", tokens=10.0)
        assert result is True
        assert limiter.get_available_tokens("user123") == 0

    @pytest.mark.asyncio
    async def test_insufficient_tokens(self):
        """Test request when insufficient tokens."""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)

        # Start with no tokens (bucket will be empty at t=0)
        # But at t=1, we get 1 token
        result = await limiter.acquire("user123", tokens=5.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=2.0)

        # Acquire all tokens
        await limiter.acquire("user123", tokens=10.0)
        assert limiter.get_available_tokens("user123") == 0

        # Wait for partial refill (2 seconds for 1 token)
        await asyncio.sleep(1.1)

        # Should have 2 tokens now
        available = limiter.get_available_tokens("user123")
        assert available == 2

    @pytest.mark.asyncio
    async def test_get_available_tokens(self):
        """Test getting available token count."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0)

        available = limiter.get_available_tokens("user123")
        assert available == 10

    @pytest.mark.asyncio
    async def test_get_retry_after(self):
        """Test getting retry after time."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=5.0)

        # Acquire all tokens
        await limiter.acquire("user123", tokens=10.0)

        retry_after = limiter.get_retry_after("user123")
        assert retry_after > 0

    @pytest.mark.asyncio
    async def test_per_user_isolation(self):
        """Test that per-user buckets are isolated."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0, per_user=True)

        # User1 acquires tokens
        assert await limiter.acquire("user1", tokens=5.0) is True

        # User2 should not affect User1's bucket
        assert limiter.get_available_tokens("user2") == 10

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting token bucket."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0, per_user=True)

        # Use some tokens
        await limiter.acquire("user1", tokens=5.0)

        # Reset
        limiter.reset("user1")

        # Should have full capacity again
        assert limiter.get_available_tokens("user1") == 10


# =============================================================================
# ImageGenerationRateLimiter Tests
# =============================================================================


class TestImageGenerationRateLimiter:
    """Tests for ImageGenerationRateLimiter class."""

    @pytest.mark.asyncio
    async def test_init_with_defaults(self):
        """Test initialization with default values."""
        limiter = ImageGenerationRateLimiter()
        assert limiter.max_requests_per_minute == 3
        assert limiter.max_requests_per_hour == 15

    @pytest.mark.asyncio
    async def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        limiter = ImageGenerationRateLimiter(max_requests_per_minute=5, max_requests_per_hour=20)
        assert limiter.max_requests_per_minute == 5
        assert limiter.max_requests_per_hour == 20

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self):
        """Test rate limit check when allowed."""
        limiter = ImageGenerationRateLimiter(max_requests_per_minute=2, max_requests_per_hour=10)

        result = await limiter.check_rate_limit(12345)
        assert result.allowed is True
        assert result.current_usage == 1

    @pytest.mark.asyncio
    async def test_check_rate_limit_minute_exceeded(self):
        """Test rate limit when minute limit exceeded."""
        limiter = ImageGenerationRateLimiter(max_requests_per_minute=2, max_requests_per_hour=10)

        # Use up minute limit
        result1 = await limiter.check_rate_limit(12345)
        assert result1.allowed is False
        assert result1.current_usage == 2

        # Use one more should be denied
        result2 = await limiter.check_rate_limit(12345)
        assert result2.allowed is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_hour_exceeded(self):
        """Test rate limit when hour limit exceeded."""
        limiter = ImageGenerationRateLimiter(max_requests_per_minute=5, max_requests_per_hour=10)

        # Use up hour limit (5 per hour * 12 = 60 minutes...but our test is simpler)
        # Let's use 60 requests to exceed hour limit
        for _ in range(10):
            await limiter.check_rate_limit(12345)

        result = await limiter.check_rate_limit(12345)
        assert result.allowed is False
        assert result.current_usage == 10

    @pytest.mark.asyncio
    async def test_get_user_status(self):
        """Test getting user status."""
        limiter = ImageGenerationRateLimiter(max_requests_per_minute=3, max_requests_per_hour=15)

        # Make one request
        await limiter.check_rate_limit(12345)

        status = limiter.get_user_status(12345)
        assert "minute_usage" in status
        assert "hour_usage" in status
        assert "minute_limit" in status
        assert "hour_limit" in status
        assert status["minute_usage"] == 1
        assert status["minute_limit"] == 3
        assert status["hour_usage"] == 1
        assert status["hour_limit"] == 15


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    @pytest.mark.asyncio
    async def test_get_image_rate_limiter_returns_global(self):
        """Test get_image_rate_limiter returns global instance."""
        limiter = get_image_rate_limiter()
        assert limiter is not None
        assert isinstance(limiter, ImageGenerationRateLimiter)

    @pytest.mark.asyncio
    async def test_reset_image_rate_limiter(self):
        """Test reset_image_rate_limiter function."""
        # Set global limiter
        custom_limiter = ImageGenerationRateLimiter(max_requests_per_minute=10)

        # Reset and verify
        reset_image_rate_limiter(custom_limiter)

        new_limiter = get_image_rate_limiter()
        # Should return a new instance with defaults
        assert new_limiter.max_requests_per_minute == 3


# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""

    def test_allowed_true(self):
        """Test RateLimitResult when request is allowed."""
        result = RateLimitResult(
            allowed=True,
            retry_after=0.0,
            current_usage=1,
            max_requests=10,
            message="Request allowed"
        )
        assert result.allowed is True
        assert result.retry_after == 0.0
        assert result.current_usage == 1
        assert result.max_requests == 10
        assert result.message == "Request allowed"

    def test_allowed_false(self):
        """Test RateLimitResult when request is rate limited."""
        result = RateLimitResult(
            allowed=False,
            retry_after=5.0,
            current_usage=2,
            max_requests=5,
            message="Rate limited, retry after 5.0s"
        )
        assert result.allowed is False
        assert result.retry_after == 5.0
        assert result.current_usage == 2
        assert result.max_requests == 5
        assert "Rate limited" in result.message

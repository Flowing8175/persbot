"""Feature tests for retry service.

Tests focus on behavior:
- RetryStrategy: retry strategy enumeration
- RetryPolicy: retry configuration
- RetryResult: retry operation result
- RetryCondition: retry condition checks
- RetryService: service for managing retries
- TokenBucketRateLimiter: token bucket rate limiting
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from persbot.services.retry_service import (
    RetryStrategy,
    RetryPolicy,
    RetryResult,
    RetryCondition,
    DefaultRetryCondition,
    RateLimitRetryCondition,
    RetryService,
    TokenBucketRateLimiter,
    retry_on_error,
)
from persbot.exceptions import RateLimitException


class TestRetryStrategy:
    """Tests for RetryStrategy enumeration."""

    def test_has_exponential_backoff(self):
        """RetryStrategy includes EXPONENTIAL_BACKOFF."""
        assert RetryStrategy.EXPONENTIAL_BACKOFF.value == "exponential_backoff"

    def test_has_linear_backoff(self):
        """RetryStrategy includes LINEAR_BACKOFF."""
        assert RetryStrategy.LINEAR_BACKOFF.value == "linear_backoff"

    def test_has_token_bucket(self):
        """RetryStrategy includes TOKEN_BUCKET."""
        assert RetryStrategy.TOKEN_BUCKET.value == "token_bucket"

    def test_has_immediate(self):
        """RetryStrategy includes IMMEDIATE."""
        assert RetryStrategy.IMMEDIATE.value == "immediate"


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""

    def test_creates_with_defaults(self):
        """RetryPolicy creates with sensible defaults."""
        policy = RetryPolicy()
        assert policy.max_retries == 2
        assert policy.base_delay == 2.0
        assert policy.max_delay == 32.0
        assert policy.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert policy.jitter is True

    def test_creates_with_custom_values(self):
        """RetryPolicy accepts custom values."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter=False,
        )
        assert policy.max_retries == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 120.0
        assert policy.strategy == RetryStrategy.LINEAR_BACKOFF
        assert policy.jitter is False

    def test_calculate_delay_exponential(self):
        """calculate_delay uses exponential backoff."""
        policy = RetryPolicy(
            base_delay=2.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=False,
        )
        assert policy.calculate_delay(1) == 2.0  # 2^1
        assert policy.calculate_delay(2) == 4.0  # 2^2
        assert policy.calculate_delay(3) == 8.0  # 2^3

    def test_calculate_delay_linear(self):
        """calculate_delay uses linear backoff."""
        policy = RetryPolicy(
            base_delay=2.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter=False,
        )
        assert policy.calculate_delay(1) == 2.0  # 2*1
        assert policy.calculate_delay(2) == 4.0  # 2*2
        assert policy.calculate_delay(3) == 6.0  # 2*3

    def test_calculate_delay_immediate(self):
        """calculate_delay returns 0 for immediate strategy."""
        policy = RetryPolicy(strategy=RetryStrategy.IMMEDIATE, jitter=False)
        assert policy.calculate_delay(1) == 0
        assert policy.calculate_delay(5) == 0

    def test_calculate_delay_respects_max_delay(self):
        """calculate_delay caps at max_delay."""
        policy = RetryPolicy(
            base_delay=2.0,
            max_delay=10.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=False,
        )
        assert policy.calculate_delay(10) == 10.0  # Would be 1024, capped at 10

    def test_calculate_delay_uses_rate_limit_delay(self):
        """calculate_delay uses rate_limit_delay when provided."""
        policy = RetryPolicy(max_delay=120.0, jitter=False)
        result = policy.calculate_delay(1, rate_limit_delay=30.0)
        assert result == 30.0

    def test_calculate_delay_caps_rate_limit_delay(self):
        """calculate_delay caps rate_limit_delay at max_delay."""
        policy = RetryPolicy(max_delay=60.0, jitter=False)
        result = policy.calculate_delay(1, rate_limit_delay=120.0)
        assert result == 60.0

    def test_calculate_delay_with_jitter(self):
        """calculate_delay adds jitter when enabled."""
        policy = RetryPolicy(base_delay=2.0, jitter=True)
        # With jitter, delay should be between 75% and 100% of base
        for _ in range(10):
            delay = policy.calculate_delay(1)
            assert 1.5 <= delay <= 2.0


class TestRetryResult:
    """Tests for RetryResult dataclass."""

    def test_creates_with_defaults(self):
        """RetryResult creates with defaults."""
        result = RetryResult(success=True)
        assert result.success is True
        assert result.value is None
        assert result.error is None
        assert result.attempts == 0
        assert result.total_time == 0.0

    def test_creates_with_all_fields(self):
        """RetryResult accepts all fields."""
        exc = ValueError("test")
        result = RetryResult(
            success=False,
            value=None,
            error=exc,
            attempts=3,
            total_time=5.5,
        )
        assert result.success is False
        assert result.error == exc
        assert result.attempts == 3
        assert result.total_time == 5.5


class TestDefaultRetryCondition:
    """Tests for DefaultRetryCondition."""

    def test_creates_with_defaults(self):
        """DefaultRetryCondition creates with defaults."""
        condition = DefaultRetryCondition()
        assert condition.max_retries == 2

    def test_should_retry_returns_false_when_max_reached(self):
        """should_retry returns False when max retries reached."""
        condition = DefaultRetryCondition(max_retries=3)
        assert condition.should_retry(ValueError("test"), 3) is False

    def test_should_retry_returns_true_for_rate_limit(self):
        """should_retry returns True for RateLimitException."""
        condition = DefaultRetryCondition()
        exc = RateLimitException("Rate limited")
        assert condition.should_retry(exc, 1) is True

    def test_should_retry_returns_true_for_timeout(self):
        """should_retry returns True for asyncio.TimeoutError."""
        condition = DefaultRetryCondition()
        assert condition.should_retry(asyncio.TimeoutError(), 1) is True

    def test_should_retry_returns_true_for_connection_error(self):
        """should_retry returns True for ConnectionError."""
        condition = DefaultRetryCondition()
        assert condition.should_retry(ConnectionError(), 1) is True

    def test_should_retry_returns_true_for_429_in_message(self):
        """should_retry returns True for 429 in error message."""
        condition = DefaultRetryCondition()
        exc = ValueError("Error 429: Too many requests")
        assert condition.should_retry(exc, 1) is True

    def test_should_retry_returns_false_for_generic_error(self):
        """should_retry returns False for generic errors."""
        condition = DefaultRetryCondition()
        exc = ValueError("Some other error")
        assert condition.should_retry(exc, 1) is False

    def test_should_retry_returns_false_for_400_without_quota(self):
        """should_retry returns False for 400 without quota keyword."""
        condition = DefaultRetryCondition()
        exc = ValueError("Error 400: Bad request")
        assert condition.should_retry(exc, 1) is False


class TestRateLimitRetryCondition:
    """Tests for RateLimitRetryCondition."""

    def test_creates_with_defaults(self):
        """RateLimitRetryCondition creates with defaults."""
        condition = RateLimitRetryCondition()
        assert condition.max_retries == 5

    def test_should_retry_returns_true_for_rate_limit_patterns(self):
        """should_retry returns True for rate limit error patterns."""
        condition = RateLimitRetryCondition()

        assert condition.should_retry(ValueError("429 Too many requests"), 1) is True
        assert condition.should_retry(ValueError("Rate limit exceeded"), 1) is True
        assert condition.should_retry(ValueError("Quota exceeded"), 1) is True

    def test_should_retry_returns_false_for_other_errors(self):
        """should_retry returns False for non-rate-limit errors."""
        condition = RateLimitRetryCondition()
        exc = ValueError("Internal server error")
        assert condition.should_retry(exc, 1) is False

    def test_should_retry_returns_false_when_max_reached(self):
        """should_retry returns False when max retries reached."""
        condition = RateLimitRetryCondition(max_retries=3)
        exc = ValueError("Rate limit exceeded")
        assert condition.should_retry(exc, 3) is False


class TestRetryService:
    """Tests for RetryService."""

    @pytest.fixture
    def service(self):
        """Create a RetryService with fast policy for testing."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
        return RetryService(default_policy=policy)

    @pytest.mark.asyncio
    async def test_execute_returns_result_on_success(self, service):
        """execute returns result when function succeeds."""
        async def success_func():
            return "success"

        result = await service.execute(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_retries_on_retryable_error(self, service):
        """execute retries on retryable errors."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitException("Rate limited")
            return "success"

        result = await service.execute(flaky_func)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_raises_after_max_retries(self, service):
        """execute raises after max retries exhausted."""
        async def always_fail():
            raise RateLimitException("Always fails")

        with pytest.raises(RateLimitException):
            await service.execute(always_fail)

    @pytest.mark.asyncio
    async def test_execute_respects_cancel_event(self, service):
        """execute respects cancellation event."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        async def func():
            return "should not be called"

        with pytest.raises(asyncio.CancelledError):
            await service.execute(func, cancel_event=cancel_event)

    @pytest.mark.asyncio
    async def test_execute_propagates_cancelled_error(self, service):
        """execute propagates CancelledError from function."""
        async def cancelled_func():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await service.execute(cancelled_func)

    @pytest.mark.asyncio
    async def test_execute_calls_on_retry_callback(self, service):
        """execute calls on_retry callback before each retry."""
        retry_calls = []

        async def on_retry(error, attempt, delay):
            retry_calls.append((str(error), attempt, delay))

        async def flaky_func():
            if len(retry_calls) < 2:
                raise RateLimitException("Retry me")
            return "success"

        await service.execute(flaky_func, on_retry=on_retry)

        assert len(retry_calls) == 2

    @pytest.mark.asyncio
    async def test_set_condition_changes_behavior(self, service):
        """set_condition changes retry behavior."""
        # Use RateLimitRetryCondition which only retries rate limits
        service.set_condition(RateLimitRetryCondition())

        async def fail_with_generic_error():
            raise ValueError("Generic error")

        with pytest.raises(ValueError, match="Generic error"):
            await service.execute(fail_with_generic_error)


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_creates_with_defaults(self):
        """TokenBucketRateLimiter creates with defaults."""
        limiter = TokenBucketRateLimiter(rate=1.0)
        assert limiter.rate == 1.0
        assert limiter.capacity == 10

    def test_creates_with_custom_values(self):
        """TokenBucketRateLimiter accepts custom values."""
        limiter = TokenBucketRateLimiter(rate=5.0, capacity=20)
        assert limiter.rate == 5.0
        assert limiter.capacity == 20

    @pytest.mark.asyncio
    async def test_acquire_returns_zero_when_tokens_available(self):
        """acquire returns 0 when tokens are available."""
        limiter = TokenBucketRateLimiter(rate=1.0, capacity=10)
        wait_time = await limiter.acquire(1)
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_waits_when_tokens_depleted(self):
        """acquire waits when tokens are depleted."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=2)

        # Use up tokens
        await limiter.acquire(2)

        # Next acquire should wait
        start = time.monotonic()
        wait_time = await limiter.acquire(1)
        elapsed = time.monotonic() - start

        assert wait_time > 0
        assert elapsed >= wait_time * 0.9  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_acquire_raises_for_excessive_tokens(self):
        """acquire raises when requesting more than capacity."""
        limiter = TokenBucketRateLimiter(rate=1.0, capacity=5)

        with pytest.raises(ValueError, match="exceeds capacity"):
            await limiter.acquire(10)

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        """Tokens refill based on elapsed time."""
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=10)  # 100 tokens/sec

        # Use all tokens
        await limiter.acquire(10)

        # Wait a bit
        await asyncio.sleep(0.05)  # Should refill ~5 tokens

        # Should be able to acquire some tokens immediately
        wait_time = await limiter.acquire(3)
        assert wait_time == 0.0


class TestRetryOnError:
    """Tests for retry_on_error convenience function."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """retry_on_error retries on failure."""
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitException("Retry")
            return "done"

        result = await retry_on_error(flaky, max_retries=3)
        assert result == "done"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_respects_max_retries(self):
        """retry_on_error respects max_retries."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise RateLimitException("Always")

        with pytest.raises(RateLimitException):
            await retry_on_error(always_fail, max_retries=2)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_respects_cancel_event(self):
        """retry_on_error respects cancel_event."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        async def func():
            return "should not be called"

        with pytest.raises(asyncio.CancelledError):
            await retry_on_error(func, cancel_event=cancel_event)

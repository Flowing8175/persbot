"""Tests for services/retry_service.py module.

This module provides comprehensive test coverage for:
- RetryStrategy enum
- RetryPolicy dataclass
- RetryResult dataclass
- RetryCondition ABC and implementations
- RateLimitRetryCondition
- DefaultRetryCondition
- TokenBucketRateLimiter
- RetryService class
"""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace

import pytest

from persbot.services.retry_service import (
    RetryStrategy,
    RetryPolicy,
    RetryResult,
    RetryCondition,
    DefaultRetryCondition,
    RateLimitRetryCondition,
    TokenBucketRateLimiter,
    RetryService,
)
from persbot.constants import RetryConfig
from persbot.exceptions import RateLimitException


# =============================================================================
# RetryStrategy Enum Tests
# =============================================================================


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_enum_values(self):
        """Test RetryStrategy enum values."""
        assert RetryStrategy.EXPONENTIAL_BACKOFF == "exponential_backoff"
        assert RetryStrategy.LINEAR_BACKOFF == "linear_backoff"
        assert RetryStrategy.TOKEN_BUCKET == "token_bucket"
        assert RetryStrategy.IMMEDIATE == "immediate"


# =============================================================================
# RetryPolicy Dataclass Tests
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""

    def test_init_default_values(self):
        """Test RetryPolicy with defaults."""
        policy = RetryPolicy()

        assert policy.max_retries == RetryConfig.MAX_RETRIES
        assert policy.base_delay == RetryConfig.BACKOFF_BASE
        assert policy.max_delay == RetryConfig.BACKOFF_MAX
        assert policy.rate_limit_delay == RetryConfig.RATE_LIMIT_RETRY_AFTER
        assert policy.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert policy.jitter is True

    def test_calculate_delay_exponential_no_jitter(self):
        """Test calculate_delay with exponential strategy without jitter."""
        policy = RetryPolicy(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, jitter=False)

        # No rate limit delay
        assert policy.calculate_delay(attempt=1, rate_limit_delay=None) == policy.base_delay
        assert policy.calculate_delay(attempt=2, rate_limit_delay=None) == policy.base_delay ** 2
        assert policy.calculate_delay(attempt=3, rate_limit_delay=None) == policy.base_delay ** 3

        # Verify max delay cap
        assert policy.calculate_delay(attempt=10, rate_limit_delay=None) == policy.max_delay

    def test_calculate_delay_linear_no_jitter(self):
        """Test calculate_delay with linear strategy without jitter."""
        policy = RetryPolicy(strategy=RetryStrategy.LINEAR_BACKOFF, jitter=False)

        assert policy.calculate_delay(attempt=1, rate_limit_delay=None) == policy.base_delay
        assert policy.calculate_delay(attempt=2, rate_limit_delay=None) == policy.base_delay * 2

    def test_calculate_delay_immediate(self):
        """Test calculate_delay with immediate strategy."""
        policy = RetryPolicy(strategy=RetryStrategy.IMMEDIATE)

        assert policy.calculate_delay(attempt=1, rate_limit_delay=None) == 0
        assert policy.calculate_delay(attempt=5, rate_limit_delay=None) == 0

    def test_calculate_delay_with_rate_limit(self):
        """Test calculate_delay respects rate_limit_delay parameter."""
        policy = RetryPolicy(strategy=RetryStrategy.EXPONENTIAL_BACKOFF)

        rate_limit_delay = 5.0
        assert policy.calculate_delay(attempt=1, rate_limit_delay=rate_limit_delay) == rate_limit_delay
        assert policy.calculate_delay(attempt=10, rate_limit_delay=rate_limit_delay) == rate_limit_delay

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness to delay."""
        policy = RetryPolicy(jitter=True, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)

        delay1 = policy.calculate_delay(attempt=1, rate_limit_delay=None)
        delay2 = policy.calculate_delay(attempt=1, rate_limit_delay=None)

        # With jitter, delays should vary
        assert delay1 != delay2


# =============================================================================
# RetryResult Dataclass Tests
# =============================================================================


class TestRetryResult:
    """Tests for RetryResult dataclass."""

    def test_success_result(self):
        """Test RetryResult with success."""
        result = RetryResult(
            success=True,
            value="test_value",
            attempts=3,
            total_time=1.5,
        )

        assert result.success is True
        assert result.value == "test_value"
        assert result.attempts == 3

    def test_failure_result(self):
        """Test RetryResult with failure."""
        result = RetryResult(
            success=False,
            error=Exception("Test error"),
            attempts=2,
            total_time=1.0,
        )

        assert result.success is False
        assert isinstance(result.error, Exception)
        assert result.error.args[0] == "Test error"

    def test_result_with_error_and_value(self):
        """Test RetryResult with both error and value."""
        result = RetryResult(
            success=True,
            value="result_value",
            error=Exception("Warning"),
            attempts=1,
            total_time=0.5,
        )

        assert result.success is True
        assert result.value == "result_value"


# =============================================================================
# RetryCondition Implementation Tests
# =============================================================================


class TestDefaultRetryCondition:
    """Tests for DefaultRetryCondition class."""

    def test_should_retry_on_rate_limit(self):
        """Test retry on rate limit errors."""
        condition = DefaultRetryCondition(max_retries=3)

        assert condition.should_retry(RateLimitException("Rate limited"), attempt=1) is True
        assert condition.should_retry(RateLimitException("Rate limited"), attempt=3) is False

    def test_should_retry_on_timeout(self):
        """Test retry on timeout errors."""
        condition = DefaultRetryCondition()

        assert condition.should_retry(asyncio.TimeoutError(), attempt=1) is True

    def test_should_retry_on_connection_error(self):
        """Test retry on connection errors."""
        condition = DefaultRetryCondition()

        assert condition.should_retry(ConnectionError(), attempt=1) is True

    def test_should_not_retry_on_value_error(self):
        """Test that ValueError is not retryable by default."""
        condition = DefaultRetryCondition()

        assert condition.should_retry(ValueError("Bad input"), attempt=1) is False

    def test_should_retry_on_429_pattern(self):
        """Test retry on 429 error pattern."""
        condition = DefaultRetryCondition()

        assert condition.should_retry(Exception("429 Too Many Requests"), attempt=1) is True

    def test_should_not_retry_on_400_without_quota(self):
        """Test that 400 without quota is not retryable."""
        condition = DefaultRetryCondition()

        assert condition.should_retry(Exception("400 Bad Request"), attempt=1) is False

    def test_should_retry_on_400_with_quota(self):
        """Test that 400 with quota keyword is retryable."""
        condition = DefaultRetryCondition()

        assert condition.should_retry(Exception("400 quota exceeded"), attempt=1) is True


class TestRateLimitRetryCondition:
    """Tests for RateLimitRetryCondition class."""

    def test_should_retry_on_429(self):
        """Test retry on 429 errors."""
        condition = RateLimitRetryCondition(max_retries=3)

        assert condition.should_retry(Exception("429 Too Many Requests"), attempt=1) is True

    def test_should_retry_on_rate_limit_text(self):
        """Test retry on rate limit text."""
        condition = RateLimitRetryCondition()

        assert condition.should_retry(Exception("Rate limit exceeded"), attempt=1) is True

    def test_should_retry_on_quota_text(self):
        """Test retry on quota errors."""
        condition = RateLimitRetryCondition()

        assert condition.should_retry(Exception("Quota exceeded"), attempt=1) is True

    def test_should_not_retry_on_other_errors(self):
        """Test that other errors are not retried."""
        condition = RateLimitRetryCondition()

        assert condition.should_retry(ValueError("Bad input"), attempt=1) is False

    def test_max_retries_limit(self):
        """Test that max_retries is respected."""
        condition = RateLimitRetryCondition(max_retries=3)

        # With max_retries=3, attempts < 3 retry, attempts >= 3 do not
        assert condition.should_retry(Exception("429"), attempt=1) is True
        assert condition.should_retry(Exception("429"), attempt=2) is True
        assert condition.should_retry(Exception("429"), attempt=3) is False
        assert condition.should_retry(Exception("429"), attempt=4) is False


# =============================================================================
# TokenBucketRateLimiter Class Tests
# =============================================================================


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter class."""

    @pytest.mark.asyncio
    async def test_acquire_tokens(self):
        """Test token acquisition."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)

        # Acquire 5 tokens
        wait_time = await limiter.acquire(5)

        # Should have 5 tokens remaining
        assert limiter.tokens == 5
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_requires_wait(self):
        """Test token acquisition with wait."""
        limiter = TokenBucketRateLimiter(rate=1.0, capacity=10)

        # Use all 10 tokens
        await limiter.acquire(10)
        assert limiter.tokens == 0

        # Next acquisition should wait
        wait_time = await limiter.acquire(1)

        assert wait_time > 0
        assert limiter.tokens == 0

    @pytest.mark.asyncio
    async def test_acquire_exceeds_capacity(self):
        """Test error when request exceeds capacity."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)

        with pytest.raises(ValueError) as exc_info:
            await limiter.acquire(15)

        assert "exceeds capacity" in str(exc_info.value).lower()


# =============================================================================
# RetryService Class Tests
# =============================================================================


class TestRetryService:
    """Tests for RetryService class."""

    @pytest.fixture
    def policy(self):
        """Create a RetryPolicy for testing."""
        return RetryPolicy(max_retries=3, base_delay=1.0, max_delay=10.0, jitter=False)

    @pytest.mark.asyncio
    async def test_execute_success_on_first_try(self, policy):
        """Test successful execution on first attempt."""
        service = RetryService(default_policy=policy)

        async def test_func():
            return "success"

        result = await service.execute(test_func, operation_name="test operation")

        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_retry_then_success(self, policy):
        """Test retry then success."""
        service = RetryService(default_policy=policy)

        call_count = [0]

        async def failing_func():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] < 3:
                raise RateLimitException("Temporary failure")
            return "success"

        result = await service.execute(failing_func, operation_name="test operation")

        assert result == "success"
        # Should have retried up to success
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_execute_non_retryable_error(self, policy):
        """Test that non-retryable errors are not retried."""
        service = RetryService(default_policy=policy)

        async def value_error_func():
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError) as exc_info:
            await service.execute(
                value_error_func,
                operation_name="test operation",
            )

        assert "non-retryable error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_with_cancel_event_before_attempt(self, policy):
        """Test cancellation before attempt starts."""
        service = RetryService(default_policy=policy)
        cancel_event = asyncio.Event()
        cancel_event.set()  # Set before execution

        async def cancelable_func():
            raise Exception("Should not execute")

        with pytest.raises(asyncio.CancelledError):
            await service.execute(
                cancelable_func,
                operation_name="test operation",
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_execute_with_custom_condition(self, policy):
        """Test with custom retry condition."""
        service = RetryService(default_policy=policy)

        class CustomCondition(RetryCondition):
            def __init__(self):
                self.max_retries = 1

            def should_retry(self, error, attempt):
                return attempt < self.max_retries

        custom_condition = CustomCondition()
        service.set_condition(custom_condition)

        attempt = [0]

        async def failing_func():
            attempt[0] += 1
            raise Exception("Failure")

        with pytest.raises(Exception):
            await service.execute(failing_func, operation_name="test operation")

        # Should only retry once due to custom condition
        assert attempt[0] == 1

    @pytest.mark.asyncio
    async def test_retry_on_error_callback(self, policy):
        """Test on_retry callback is invoked."""
        service = RetryService(default_policy=policy)

        callback_calls = []

        async def failing_func_with_callback():
            nonlocal callback_calls
            callback_calls.append(1)
            raise RateLimitException("Error")

        async def on_retry(error, attempt, delay):
            callback_calls.append(2)

        with pytest.raises(RateLimitException):
            await service.execute(
                failing_func_with_callback,
                operation_name="test",
                on_retry=on_retry,
            )

        # Verify callback was called
        assert len(callback_calls) > 1
        assert 2 in callback_calls

    @pytest.mark.asyncio
    async def test_rate_limit_delay_extraction(self, policy):
        """Test extraction of rate limit delay from error."""
        from persbot.exceptions import RateLimitException

        service = RetryService(default_policy=policy)

        call_count = [0]

        async def rate_limit_func():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                # Use RateLimitException which is retryable
                raise RateLimitException("Please retry in 0.1s")
            return "success"

        result = await service.execute(rate_limit_func, operation_name="test")

        assert result == "success"
        assert call_count[0] == 2

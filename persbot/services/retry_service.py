"""Retry service for handling API call retries with various strategies.

This module provides a unified retry mechanism that can be used across
all LLM providers, reducing code duplication and providing consistent
retry behavior.
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

from persbot.constants import RetryConfig
from persbot.exceptions import RateLimitException, RetryableError

logger = logging.getLogger(__name__)

# Type variables for generic return types
T = TypeVar("T")


class RetryStrategy(str, Enum):
    """Retry strategy types."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    TOKEN_BUCKET = "token_bucket"
    IMMEDIATE = "immediate"


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behavior."""

    max_retries: int = RetryConfig.MAX_RETRIES
    base_delay: float = RetryConfig.BACKOFF_BASE
    max_delay: float = RetryConfig.BACKOFF_MAX
    rate_limit_delay: int = RetryConfig.RATE_LIMIT_RETRY_AFTER
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True  # Add randomness to prevent thundering herd

    def calculate_delay(self, attempt: int, rate_limit_delay: Optional[float] = None) -> float:
        """Calculate delay for a given retry attempt.

        Args:
            attempt: The attempt number (1-indexed).
            rate_limit_delay: Override delay from rate limit error.

        Returns:
            Delay in seconds before next retry.
        """
        if rate_limit_delay is not None:
            return min(rate_limit_delay, self.max_delay)

        if self.strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt
        else:  # EXPONENTIAL_BACKOFF (default)
            delay = self.base_delay**attempt

        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% randomness
            import random

            delay = delay * (0.75 + random.random() * 0.25)

        return delay


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    value: Optional[Any] = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_time: float = 0.0


class RetryCondition(ABC):
    """Base class for retry condition checks."""

    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried.

        Args:
            error: The exception that was raised.
            attempt: The attempt number (1-indexed).

        Returns:
            True if the operation should be retried, False otherwise.
        """
        pass


class DefaultRetryCondition(RetryCondition):
    """Default retry condition for common API errors."""

    def __init__(
        self,
        max_retries: int = RetryConfig.MAX_RETRIES,
        retryable_errors: tuple[type[Exception], ...] = (
            RateLimitException,
            asyncio.TimeoutError,
            ConnectionError,
        ),
    ):
        self.max_retries = max_retries
        self.retryable_errors = retryable_errors

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if error is retryable and we haven't exceeded max retries."""
        if attempt >= self.max_retries:
            return False

        # Check for specific retryable error types
        for error_type in self.retryable_errors:
            if isinstance(error, error_type):
                return True

        # Check for retryable base exception
        if isinstance(error, RetryableError):
            return True

        # Check for common HTTP error patterns
        error_str = str(error).lower()
        if any(pattern in error_str for pattern in ("429", "rate limit", "quota", "timeout")):
            # But not 400 errors without quota keyword
            if "400" in error_str and "quota" not in error_str:
                return False
            return True

        return False


class RateLimitRetryCondition(RetryCondition):
    """Retry condition specifically for rate limiting."""

    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Only retry rate limit errors."""
        if attempt >= self.max_retries:
            return False

        error_str = str(error).lower()
        return any(
            pattern in error_str for pattern in ("429", "rate limit", "quota", "too many requests")
        )


class RetryService:
    """Service for managing retry logic with configurable strategies."""

    def __init__(self, default_policy: Optional[RetryPolicy] = None):
        """Initialize the retry service.

        Args:
            default_policy: Default retry policy to use.
        """
        self.default_policy = default_policy or RetryPolicy()
        self._condition = DefaultRetryCondition(self.default_policy.max_retries)

    def set_condition(self, condition: RetryCondition) -> None:
        """Set a custom retry condition.

        Args:
            condition: The retry condition to use.
        """
        self._condition = condition

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        policy: Optional[RetryPolicy] = None,
        operation_name: str = "operation",
        cancel_event: Optional[asyncio.Event] = None,
        on_retry: Optional[Callable[[Exception, int, float], Awaitable[None]]] = None,
    ) -> T:
        """Execute an async function with retry logic.

        Args:
            func: The async function to execute.
            policy: Override retry policy for this execution.
            operation_name: Name of the operation for logging.
            cancel_event: Event to check for cancellation.
            on_retry: Optional callback before each retry.

        Returns:
            The result of the function execution.

        Raises:
            The last exception if all retries are exhausted.
            asyncio.CancelledError if cancel_event is set.
        """
        effective_policy = policy or self.default_policy
        last_error: Optional[Exception] = None
        start_time = time.monotonic()

        for attempt in range(1, effective_policy.max_retries + 1):
            # Check for cancellation before attempt
            if cancel_event and cancel_event.is_set():
                logger.info(f"{operation_name} cancelled before attempt {attempt}")
                raise asyncio.CancelledError(f"{operation_name} cancelled")

            try:
                result = await func()
                if attempt > 1:
                    logger.info(
                        f"{operation_name} succeeded on attempt {attempt}/{effective_policy.max_retries}"
                    )
                return result

            except asyncio.CancelledError:
                # Always propagate cancellation
                logger.info(f"{operation_name} cancelled on attempt {attempt}")
                raise

            except Exception as e:
                last_error = e
                should_retry = self._condition.should_retry(e, attempt)

                if not should_retry:
                    logger.error(
                        f"{operation_name} failed on attempt {attempt} with non-retryable error: {e}"
                    )
                    raise

                # Calculate delay
                rate_limit_delay = self._extract_rate_limit_delay(e)
                delay = effective_policy.calculate_delay(attempt, rate_limit_delay)

                logger.warning(
                    f"{operation_name} failed on attempt {attempt}/{effective_policy.max_retries}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                # Call on_retry callback if provided
                if on_retry:
                    try:
                        await on_retry(e, attempt, delay)
                    except Exception as callback_error:
                        logger.error(f"on_retry callback failed: {callback_error}")

                # Wait before retry (with cancellation check)
                try:
                    await asyncio.wait_for(
                        self._sleep_with_cancel_check(delay, cancel_event),
                        timeout=delay + 1,
                    )
                except asyncio.CancelledError:
                    logger.info(f"{operation_name} cancelled during backoff")
                    raise

        # All retries exhausted
        total_time = time.monotonic() - start_time
        logger.error(f"{operation_name} failed after {attempt} attempts ({total_time:.1f}s total)")
        raise last_error

    def _extract_rate_limit_delay(self, error: Exception) -> Optional[float]:
        """Extract retry delay from rate limit error message.

        Args:
            error: The exception to parse.

        Returns:
            Delay in seconds if found, None otherwise.
        """
        error_str = str(error)

        # Match "Please retry in Xs" pattern
        match = re.search(r"Please retry in ([0-9.]+)s", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Match "seconds: X" pattern
        match = re.search(r"seconds:\s*(\d+)", error_str)
        if match:
            return float(match.group(1))

        return None

    async def _sleep_with_cancel_check(
        self, delay: float, cancel_event: Optional[asyncio.Event]
    ) -> None:
        """Sleep with periodic cancellation checks.

        Args:
            delay: Total delay time in seconds.
            cancel_event: Event to check for cancellation.

        Raises:
            asyncio.CancelledError if cancel_event is set during sleep.
        """
        if cancel_event is None:
            await asyncio.sleep(delay)
            return

        # Sleep in small chunks to check for cancellation
        check_interval = min(1.0, delay / 10)
        elapsed = 0.0

        while elapsed < delay:
            if cancel_event.is_set():
                raise asyncio.CancelledError("Cancelled during retry backoff")

            chunk = min(check_interval, delay - elapsed)
            await asyncio.sleep(chunk)
            elapsed += chunk


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling request rates.

    This is useful for APIs that have strict rate limits.
    """

    def __init__(self, rate: float, capacity: int = 10):
        """Initialize the token bucket.

        Args:
            rate: Tokens per second (refill rate).
            capacity: Maximum number of tokens (burst capacity).
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.

        Raises:
            ValueError: If requested tokens exceed capacity.
        """
        if tokens > self.capacity:
            raise ValueError(f"Requested {tokens} tokens exceeds capacity {self.capacity}")

        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = needed / self.rate

            # Wait and refill
            await asyncio.sleep(wait_time)
            self.last_update = time.monotonic()
            self.tokens = 0  # All tokens used

            return wait_time


# Convenience functions for common retry patterns


async def retry_on_error(
    func: Callable[..., Awaitable[T]],
    max_retries: int = RetryConfig.MAX_RETRIES,
    operation_name: str = "operation",
    cancel_event: Optional[asyncio.Event] = None,
) -> T:
    """Simple retry function with default exponential backoff.

    Args:
        func: The async function to execute.
        max_retries: Maximum number of retry attempts.
        operation_name: Name for logging.
        cancel_event: Optional cancellation event.

    Returns:
        The result of the function execution.

    Example:
        result = await retry_on_error(
            lambda: api_call(),
            max_retries=3,
            operation_name="API Call"
        )
    """
    service = RetryService(RetryPolicy(max_retries=max_retries))
    return await service.execute(func, operation_name=operation_name, cancel_event=cancel_event)

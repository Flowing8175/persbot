"""Circuit breaker pattern for API resiliency.

This module implements the circuit breaker pattern to prevent cascading failures
when an external service (LLM provider) becomes unavailable.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from persbot.exceptions import CircuitBreakerOpenException

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state: CircuitState = CircuitState.CLOSED
    state_changed_at: float = field(default_factory=time.monotonic)


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation. Requests pass through.
    - OPEN: Provider is failing. All requests are rejected immediately.
    - HALF_OPEN: Testing recovery. Limited requests allowed to probe health.

    Configuration:
    - failure_threshold: Number of consecutive failures before opening (default: 5)
    - recovery_timeout: Seconds to wait before attempting recovery (default: 60)
    - half_open_max_calls: Max calls allowed in half-open state (default: 3)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        """Initialize the circuit breaker.

        Args:
            name: Identifier for this circuit breaker (e.g., provider name).
            failure_threshold: Consecutive failures before opening circuit.
            recovery_timeout: Seconds before attempting recovery.
            half_open_max_calls: Max probe calls in half-open state.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._stats.state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._stats.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self._stats.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if self._should_attempt_recovery():
                return False
            return True
        return False

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        elapsed = time.monotonic() - self._stats.state_changed_at
        return elapsed >= self.recovery_timeout

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._stats.state
        if old_state == new_state:
            return

        self._stats.state = new_state
        self._stats.state_changed_at = time.monotonic()

        if new_state == CircuitState.OPEN:
            self._stats.failure_count = 0
        elif new_state == CircuitState.CLOSED:
            self._stats.failure_count = 0
            self._stats.success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(
            "Circuit breaker '%s' transitioned: %s -> %s",
            self.name,
            old_state.value,
            new_state.value,
        )

    def _get_recovery_time(self) -> float:
        """Calculate time until recovery attempt."""
        if self._stats.state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self._stats.state_changed_at
        return max(0.0, self.recovery_timeout - elapsed)

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: Async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function.

        Raises:
            CircuitBreakerOpenException: If circuit is open.
            Exception: Re-raises any exception from the function.
        """
        async with self._lock:
            # Check if circuit is open
            if self._stats.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    await self._transition_to(CircuitState.HALF_OPEN)
                else:
                    recovery_time = self._get_recovery_time()
                    raise CircuitBreakerOpenException(self.name, recovery_time)

            # Handle half-open state
            if self._stats.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    # Too many probe calls, treat as open
                    recovery_time = self._get_recovery_time()
                    raise CircuitBreakerOpenException(self.name, recovery_time)
                self._half_open_calls += 1

        # Execute the function (outside lock to allow concurrent calls)
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.success_count += 1
            self._stats.last_success_time = time.monotonic()

            if self._stats.state == CircuitState.HALF_OPEN:
                # Success in half-open -> close the circuit
                await self._transition_to(CircuitState.CLOSED)
            elif self._stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._stats.failure_count = 0

    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.failure_count += 1
            self._stats.last_failure_time = time.monotonic()

            if self._stats.state == CircuitState.HALF_OPEN:
                # Failure in half-open -> open the circuit
                await self._transition_to(CircuitState.OPEN)
            elif self._stats.state == CircuitState.CLOSED:
                if self._stats.failure_count >= self.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)

    async def trip(self) -> None:
        """Manually trip (open) the circuit breaker."""
        async with self._lock:
            await self._transition_to(CircuitState.OPEN)

    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._stats.state.value,
            "failure_count": self._stats.failure_count,
            "success_count": self._stats.success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self._stats.last_failure_time,
            "last_success_time": self._stats.last_success_time,
            "recovery_time_remaining": self._get_recovery_time(),
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self, default_failure_threshold: int = 5, default_recovery_timeout: float = 60.0):
        """Initialize the registry.

        Args:
            default_failure_threshold: Default threshold for new circuit breakers.
            default_recovery_timeout: Default recovery timeout for new circuit breakers.
        """
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_failure_threshold = default_failure_threshold
        self._default_recovery_timeout = default_recovery_timeout
        self._lock = asyncio.Lock()

    def get_or_create(self, name: str) -> CircuitBreaker:
        """Get existing circuit breaker or create new one.

        Args:
            name: Name of the circuit breaker.

        Returns:
            The circuit breaker instance.
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self._default_failure_threshold,
                recovery_timeout=self._default_recovery_timeout,
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name.

        Args:
            name: Name of the circuit breaker.

        Returns:
            The circuit breaker or None if not found.
        """
        return self._breakers.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in self._breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._breakers.values():
            await cb.reset()

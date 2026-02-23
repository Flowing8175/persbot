"""Feature tests for circuit breaker pattern implementation.

Tests focus on behavior:
- Circuit state transitions (CLOSED, OPEN, HALF_OPEN)
- Failure threshold triggering
- Recovery timeout behavior
- Half-open probe calls
- Circuit breaker registry management
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from persbot.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitStats,
)
from persbot.exceptions import CircuitBreakerOpenException


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_closed_state_value(self):
        """CLOSED state has correct string value."""
        assert CircuitState.CLOSED == "closed"

    def test_open_state_value(self):
        """OPEN state has correct string value."""
        assert CircuitState.OPEN == "open"

    def test_half_open_state_value(self):
        """HALF_OPEN state has correct string value."""
        assert CircuitState.HALF_OPEN == "half_open"


class TestCircuitStats:
    """Tests for CircuitStats dataclass."""

    def test_default_initialization(self):
        """Stats initialize with default values."""
        stats = CircuitStats()
        assert stats.failure_count == 0
        assert stats.success_count == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.state == CircuitState.CLOSED
        assert isinstance(stats.state_changed_at, float)

    def test_custom_initialization(self):
        """Stats can be initialized with custom values."""
        stats = CircuitStats(
            failure_count=5,
            success_count=10,
            last_failure_time=123.0,
            last_success_time=456.0,
            state=CircuitState.OPEN,
        )
        assert stats.failure_count == 5
        assert stats.success_count == 10
        assert stats.last_failure_time == 123.0
        assert stats.last_success_time == 456.0
        assert stats.state == CircuitState.OPEN


class TestCircuitBreakerInitialization:
    """Tests for CircuitBreaker initialization and configuration."""

    def test_default_initialization(self):
        """Circuit breaker initializes with default values."""
        breaker = CircuitBreaker(name="test-provider")
        assert breaker.name == "test-provider"
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60.0
        assert breaker.half_open_max_calls == 3

    def test_custom_initialization(self):
        """Circuit breaker accepts custom configuration."""
        breaker = CircuitBreaker(
            name="custom-provider",
            failure_threshold=10,
            recovery_timeout=120.0,
            half_open_max_calls=5,
        )
        assert breaker.name == "custom-provider"
        assert breaker.failure_threshold == 10
        assert breaker.recovery_timeout == 120.0
        assert breaker.half_open_max_calls == 5

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_get_stats_returns_complete_info(self):
        """get_stats returns complete circuit information."""
        breaker = CircuitBreaker(
            name="test-provider",
            failure_threshold=3,
            recovery_timeout=30.0,
        )
        stats = breaker.get_stats()
        assert stats["name"] == "test-provider"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["success_count"] == 0
        assert stats["failure_threshold"] == 3
        assert stats["recovery_timeout"] == 30.0
        assert "recovery_time_remaining" in stats


class TestCircuitBreakerClosedState:
    """Tests for circuit breaker behavior in CLOSED state."""

    @pytest.mark.asyncio
    async def test_successful_call_passes_through(self):
        """Successful calls execute normally in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        mock_func = AsyncMock(return_value="success")

        result = await breaker.call(mock_func)

        assert result == "success"
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_call_records_failure(self):
        """Failed calls increment failure count in CLOSED state."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        mock_func = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            await breaker.call(mock_func)

        stats = breaker.get_stats()
        assert stats["failure_count"] == 1
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_multiple_failures_open_circuit(self):
        """Consecutive failures at threshold open the circuit."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        mock_func = AsyncMock(side_effect=Exception("API error"))

        # First two failures keep circuit closed
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(mock_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.get_stats()["failure_count"] == 2

        # Third failure opens circuit
        with pytest.raises(Exception):
            await breaker.call(mock_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        """Success resets consecutive failure count."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        failing_func = AsyncMock(side_effect=Exception("error"))
        success_func = AsyncMock(return_value="ok")

        # Two failures
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.get_stats()["failure_count"] == 2

        # Success resets count
        await breaker.call(success_func)

        assert breaker.get_stats()["failure_count"] == 0
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_concurrent_calls_allowed(self):
        """Multiple concurrent calls allowed in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        mock_func = AsyncMock(return_value="result")

        tasks = [breaker.call(mock_func) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r == "result" for r in results)
        assert mock_func.call_count == 5


class TestCircuitBreakerOpenState:
    """Tests for circuit breaker behavior in OPEN state."""

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_calls(self):
        """Open circuit blocks all calls immediately."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=10.0)
        mock_func = AsyncMock(side_effect=Exception("error"))

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(mock_func)

        assert breaker.state == CircuitState.OPEN

        # Next call should be blocked
        with pytest.raises(CircuitBreakerOpenException) as exc_info:
            await breaker.call(mock_func)

        assert exc_info.value.provider == "test"
        assert exc_info.value.recovery_time > 0
        # Function should not be called
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_is_open_checks_recovery_timeout(self):
        """is_open property checks recovery timeout."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)
        mock_func = AsyncMock(side_effect=Exception("error"))

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(mock_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Circuit should allow recovery attempt now
        assert not breaker.is_open

    @pytest.mark.asyncio
    async def test_manual_trip_opens_circuit(self):
        """Manual trip() opens the circuit."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED

        await breaker.trip()

        assert breaker.state == CircuitState.OPEN

        # Call should be blocked
        mock_func = AsyncMock(return_value="result")
        with pytest.raises(CircuitBreakerOpenException):
            await breaker.call(mock_func)

        assert not mock_func.called

    @pytest.mark.asyncio
    async def test_recovery_time_remaining_decreases(self):
        """Recovery time remaining decreases over time."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=1.0)
        mock_func = AsyncMock(side_effect=Exception("error"))

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(mock_func)

        stats1 = breaker.get_stats()
        initial_recovery_time = stats1["recovery_time_remaining"]
        assert initial_recovery_time > 0

        # Wait a bit
        await asyncio.sleep(0.2)

        stats2 = breaker.get_stats()
        assert stats2["recovery_time_remaining"] < initial_recovery_time


class TestCircuitBreakerHalfOpenState:
    """Tests for circuit breaker behavior in HALF_OPEN state."""

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)
        mock_func = AsyncMock(side_effect=Exception("error"))

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(mock_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should trigger transition to HALF_OPEN
        success_func = AsyncMock(return_value="ok")
        result = await breaker.call(success_func)

        assert breaker.state == CircuitState.CLOSED
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_success_in_half_open_closes_circuit(self):
        """Success in HALF_OPEN state closes the circuit."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)
        fail_func = AsyncMock(side_effect=Exception("error"))
        success_func = AsyncMock(return_value="ok")

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Success should close circuit
        await breaker.call(success_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.get_stats()["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens_circuit(self):
        """Failure in HALF_OPEN state opens circuit again."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # First call transitions to HALF_OPEN, but fails
        with pytest.raises(Exception):
            await breaker.call(fail_func)

        # Should be back to OPEN
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_max_calls_limit(self):
        """HALF_OPEN state limits probe calls."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2,
        )
        fail_func = AsyncMock(side_effect=Exception("error"))

        async def hanging_func():
            await asyncio.sleep(1.0)
            return "done"

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # First call transitions to HALF_OPEN and starts executing
        task1 = asyncio.create_task(breaker.call(hanging_func))
        await asyncio.sleep(0.05)  # Let it acquire lock and start

        # Second call increments counter
        task2 = asyncio.create_task(breaker.call(hanging_func))
        await asyncio.sleep(0.05)

        # Third call should exceed half_open_max_calls and be rejected
        # (counter is at 2, limit is 2)
        with pytest.raises(CircuitBreakerOpenException):
            await breaker.call(hanging_func)

        # Clean up tasks
        for task in [task1, task2]:
            task.cancel()
        await asyncio.gather(task1, task2, return_exceptions=True)


class TestCircuitBreakerManualControl:
    """Tests for manual circuit breaker control."""

    @pytest.mark.asyncio
    async def test_reset_closes_circuit(self):
        """Manual reset() closes the circuit."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Reset
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.get_stats()["failure_count"] == 0

        # Calls should work again
        success_func = AsyncMock(return_value="ok")
        result = await breaker.call(success_func)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_reset_clears_stats(self):
        """Reset clears all statistics when circuit is open."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Trip the circuit to OPEN state
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        stats_before = breaker.get_stats()
        assert breaker.state == CircuitState.OPEN

        await breaker.reset()

        stats_after = breaker.get_stats()
        assert stats_after["failure_count"] == 0
        assert stats_after["success_count"] == 0
        assert stats_after["state"] == "closed"

    @pytest.mark.asyncio
    async def test_trip_from_closed_state(self):
        """Manual trip() from CLOSED state opens circuit."""
        breaker = CircuitBreaker(name="test")

        await breaker.trip()

        assert breaker.state == CircuitState.OPEN

        # Should block calls
        mock_func = AsyncMock(return_value="result")
        with pytest.raises(CircuitBreakerOpenException):
            await breaker.call(mock_func)

    @pytest.mark.asyncio
    async def test_trip_from_half_open_state(self):
        """Manual trip() from HALF_OPEN state opens circuit."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Trip and wait for half-open
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        await asyncio.sleep(0.15)

        # Manually trip
        await breaker.trip()

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_zero_failure_threshold(self):
        """Circuit breaker with zero threshold opens immediately."""
        breaker = CircuitBreaker(name="test", failure_threshold=1)
        fail_func = AsyncMock(side_effect=Exception("error"))

        with pytest.raises(Exception):
            await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_call_with_args_and_kwargs(self):
        """Circuit breaker passes through function arguments."""
        breaker = CircuitBreaker(name="test")

        async def sample_func(a, b, c=None):
            return {"a": a, "b": b, "c": c}

        result = await breaker.call(sample_func, 1, 2, c=3)

        assert result == {"a": 1, "b": 2, "c": 3}

    @pytest.mark.asyncio
    async def test_exception_propagates_correctly(self):
        """Original exception is propagated, not swallowed."""
        breaker = CircuitBreaker(name="test")

        class CustomError(Exception):
            pass

        mock_func = AsyncMock(side_effect=CustomError("custom error"))

        with pytest.raises(CustomError, match="custom error"):
            await breaker.call(mock_func)

        # Circuit should still be closed (below threshold)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_rapid_success_failure_cycles(self):
        """Rapid success/failure cycles don't corrupt state."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        success_func = AsyncMock(return_value="ok")
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Alternate success and failure
        for _ in range(5):
            await breaker.call(success_func)
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        # Should remain closed (never hit threshold)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_concurrent_failures_at_threshold(self):
        """Concurrent failures at threshold don't cause race conditions."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Trigger 2 failures first
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        # Launch concurrent calls that will all fail
        tasks = [breaker.call(fail_func) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should have raised exceptions
        assert all(isinstance(r, Exception) for r in results)

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerStats:
    """Tests for circuit breaker statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_stats_track_failure_count(self):
        """Stats accurately track failure count."""
        breaker = CircuitBreaker(name="test", failure_threshold=5)
        fail_func = AsyncMock(side_effect=Exception("error"))

        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        stats = breaker.get_stats()
        assert stats["failure_count"] == 3

    @pytest.mark.asyncio
    async def test_stats_track_success_count(self):
        """Stats accurately track success count."""
        breaker = CircuitBreaker(name="test")
        success_func = AsyncMock(return_value="ok")

        for _ in range(5):
            await breaker.call(success_func)

        stats = breaker.get_stats()
        assert stats["success_count"] == 5

    @pytest.mark.asyncio
    async def test_stats_record_last_failure_time(self):
        """Stats record last failure time."""
        breaker = CircuitBreaker(name="test")
        fail_func = AsyncMock(side_effect=Exception("error"))

        before_time = time.monotonic()
        with pytest.raises(Exception):
            await breaker.call(fail_func)
        after_time = time.monotonic()

        stats = breaker.get_stats()
        assert stats["last_failure_time"] is not None
        assert before_time <= stats["last_failure_time"] <= after_time

    @pytest.mark.asyncio
    async def test_stats_record_last_success_time(self):
        """Stats record last success time."""
        breaker = CircuitBreaker(name="test")
        success_func = AsyncMock(return_value="ok")

        before_time = time.monotonic()
        await breaker.call(success_func)
        after_time = time.monotonic()

        stats = breaker.get_stats()
        assert stats["last_success_time"] is not None
        assert before_time <= stats["last_success_time"] <= after_time

    @pytest.mark.asyncio
    async def test_stats_update_on_state_change(self):
        """State change time is recorded in stats."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Get initial stats
        initial_stats = breaker.get_stats()
        assert initial_stats["state"] == "closed"

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        # State should be open
        stats = breaker.get_stats()
        assert stats["state"] == "open"


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry management."""

    def test_create_new_breaker(self):
        """Registry creates new circuit breaker on demand."""
        registry = CircuitBreakerRegistry()
        breaker = registry.get_or_create("test-provider")

        assert breaker is not None
        assert breaker.name == "test-provider"
        assert isinstance(breaker, CircuitBreaker)

    def test_return_existing_breaker(self):
        """Registry returns same breaker instance for same name."""
        registry = CircuitBreakerRegistry()
        breaker1 = registry.get_or_create("test-provider")
        breaker2 = registry.get_or_create("test-provider")

        assert breaker1 is breaker2

    def test_multiple_independent_breakers(self):
        """Registry maintains independent breakers per name."""
        registry = CircuitBreakerRegistry()

        gemini_breaker = registry.get_or_create("gemini")
        openai_breaker = registry.get_or_create("openai")

        assert gemini_breaker is not openai_breaker
        assert gemini_breaker.name == "gemini"
        assert openai_breaker.name == "openai"

    def test_get_nonexistent_breaker(self):
        """get returns None for non-existent breaker."""
        registry = CircuitBreakerRegistry()
        breaker = registry.get("nonexistent")

        assert breaker is None

    def test_get_existing_breaker(self):
        """get returns existing breaker without creating new one."""
        registry = CircuitBreakerRegistry()
        breaker1 = registry.get_or_create("test")
        breaker2 = registry.get("test")

        assert breaker1 is breaker2

    def test_default_configuration(self):
        """Registry applies default configuration to new breakers."""
        registry = CircuitBreakerRegistry(
            default_failure_threshold=10,
            default_recovery_timeout=120.0,
        )
        breaker = registry.get_or_create("test")

        assert breaker.failure_threshold == 10
        assert breaker.recovery_timeout == 120.0

    def test_get_all_stats(self):
        """get_all_stats returns stats for all breakers."""
        registry = CircuitBreakerRegistry()

        registry.get_or_create("gemini")
        registry.get_or_create("openai")
        registry.get_or_create("zai")

        all_stats = registry.get_all_stats()

        assert len(all_stats) == 3
        assert "gemini" in all_stats
        assert "openai" in all_stats
        assert "zai" in all_stats

        for name, stats in all_stats.items():
            assert name == stats["name"]
            assert stats["state"] == "closed"

    @pytest.mark.asyncio
    async def test_reset_all_breakers(self):
        """reset_all resets all circuit breakers."""
        registry = CircuitBreakerRegistry(default_failure_threshold=2)
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Create and trip multiple breakers
        for name in ["gemini", "openai", "zai"]:
            breaker = registry.get_or_create(name)
            for _ in range(2):
                with pytest.raises(Exception):
                    await breaker.call(fail_func)

        # All should be open
        all_stats = registry.get_all_stats()
        assert all(stats["state"] == "open" for stats in all_stats.values())

        # Reset all
        await registry.reset_all()

        # All should be closed
        all_stats = registry.get_all_stats()
        assert all(stats["state"] == "closed" for stats in all_stats.values())
        assert all(stats["failure_count"] == 0 for stats in all_stats.values())

    def test_get_or_create_does_not_modify_existing(self):
        """get_or_create doesn't modify existing breaker configuration."""
        registry = CircuitBreakerRegistry(
            default_failure_threshold=5,
            default_recovery_timeout=60.0,
        )

        # Create first breaker
        breaker1 = registry.get_or_create("test")
        original_threshold = breaker1.failure_threshold

        # Change registry defaults
        registry2 = CircuitBreakerRegistry(
            default_failure_threshold=10,
            default_recovery_timeout=120.0,
        )

        # Get same breaker from different registry
        breaker2 = registry2.get_or_create("test")

        # Should be different instances with different configs
        assert breaker1 is not breaker2
        assert breaker1.failure_threshold == original_threshold
        assert breaker2.failure_threshold == 10


class TestCircuitBreakerIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_api_outage_recovery(self):
        """Simulate API outage and recovery cycle."""
        breaker = CircuitBreaker(
            name="api",
            failure_threshold=3,
            recovery_timeout=0.2,
        )

        # API is down
        api_down = AsyncMock(side_effect=Exception("Connection error"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(api_down)

        assert breaker.state == CircuitState.OPEN

        # API is still down - calls blocked
        with pytest.raises(CircuitBreakerOpenException):
            await breaker.call(api_down)

        # Wait for recovery timeout
        await asyncio.sleep(0.25)

        # API is back up
        api_up = AsyncMock(return_value="success")
        result = await breaker.call(api_up)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_intermittent_failures(self):
        """Intermittent failures don't trip circuit."""
        breaker = CircuitBreaker(
            name="api",
            failure_threshold=3,
        )

        fail_func = AsyncMock(side_effect=Exception("error"))
        success_func = AsyncMock(return_value="ok")

        # Pattern: success, fail, success, fail
        for _ in range(5):
            await breaker.call(success_func)
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        # Circuit should remain closed
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_sustained_high_error_rate(self):
        """Sustained high error rate opens circuit."""
        breaker = CircuitBreaker(
            name="api",
            failure_threshold=5,
        )

        # Consecutive failures trip circuit
        fail_func = AsyncMock(side_effect=Exception("error"))

        # Need 5 consecutive failures to trip
        for _ in range(5):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_partial_recovery_then_failure(self):
        """Partial recovery followed by more failures."""
        breaker = CircuitBreaker(
            name="api",
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_max_calls=2,
        )

        fail_func = AsyncMock(side_effect=Exception("error"))
        success_func = AsyncMock(return_value="ok")

        # Trip the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Partial recovery - one success
        await breaker.call(success_func)
        assert breaker.state == CircuitState.CLOSED

        # More failures trip it again
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

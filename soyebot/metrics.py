"""Performance metrics collection utilities for SoyeBot."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from threading import Lock
from typing import Dict, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single metric measurement snapshot."""
    timestamp: datetime
    value: float
    label: Optional[str] = None


class MetricsCollector:
    """Thread-safe metrics collection with bounded memory usage.

    Designed for low-memory environments (1GB RAM):
    - Fixed-size deques to prevent unbounded growth
    - Minimal overhead per measurement
    - Thread-safe for concurrent access
    """

    def __init__(self, max_history: int = 100):
        """Initialize metrics collector.

        Args:
            max_history: Maximum number of historical data points to keep
                        (default: 100, ~10KB memory per metric)
        """
        self._lock = Lock()
        self.max_history = max_history

        # Latency tracking (in milliseconds)
        self.latencies: Dict[str, deque] = {
            'llm_api': deque(maxlen=max_history),
            'message_processing': deque(maxlen=max_history),
            'database': deque(maxlen=max_history),
            'session_creation': deque(maxlen=max_history),
        }

        # Counter tracking
        self.counters: Dict[str, int] = {
            'messages_processed': 0,
            'api_requests_total': 0,
            'api_requests_success': 0,
            'api_requests_error': 0,
            'sessions_created': 0,
            'sessions_cleaned': 0,
        }

        # System metrics cache (updated periodically, not on every access)
        self._system_metrics_cache = {
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'last_updated': time.time()
        }
        self._cache_ttl = 2.0  # Cache system metrics for 2 seconds

        # Active LLM provider (for display only)
        self.llm_provider: str = 'Gemini'

        # Process handle for psutil
        self._process = psutil.Process()

        logger.info(f"MetricsCollector initialized (max_history={max_history})")

    def record_latency(self, operation: str, duration_ms: float, label: Optional[str] = None):
        """Record operation latency.

        Args:
            operation: Operation name (e.g., 'llm_api', 'database')
            duration_ms: Duration in milliseconds
            label: Optional label for categorization
        """
        with self._lock:
            if operation in self.latencies:
                snapshot = MetricSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    value=duration_ms,
                    label=label
                )
                self.latencies[operation].append(snapshot)
            else:
                logger.warning(f"Unknown latency operation: {operation}")

    def increment_counter(self, counter: str, amount: int = 1):
        """Increment a counter.

        Args:
            counter: Counter name
            amount: Amount to increment (default: 1)
        """
        with self._lock:
            if counter in self.counters:
                self.counters[counter] += amount
            else:
                logger.warning(f"Unknown counter: {counter}")

    def get_latency_stats(self, operation: str) -> Dict[str, float]:
        """Get latency statistics for an operation.

        Returns:
            Dict with 'avg', 'min', 'max', 'p50', 'p95', 'p99' in milliseconds
        """
        with self._lock:
            if operation not in self.latencies or not self.latencies[operation]:
                return {'avg': 0, 'min': 0, 'max': 0, 'p50': 0, 'p95': 0, 'p99': 0}

            values = [s.value for s in self.latencies[operation]]
            values_sorted = sorted(values)
            n = len(values_sorted)

            return {
                'avg': sum(values) / n,
                'min': values_sorted[0],
                'max': values_sorted[-1],
                'p50': values_sorted[int(n * 0.50)] if n > 0 else 0,
                'p95': values_sorted[int(n * 0.95)] if n > 1 else values_sorted[-1],
                'p99': values_sorted[int(n * 0.99)] if n > 1 else values_sorted[-1],
                'count': n,
            }

    def get_counter(self, counter: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self.counters.get(counter, 0)

    def get_all_counters(self) -> Dict[str, int]:
        """Get all counter values."""
        with self._lock:
            return self.counters.copy()

    def _update_system_metrics_cache(self):
        """Update cached system metrics if TTL expired."""
        now = time.time()
        if now - self._system_metrics_cache['last_updated'] > self._cache_ttl:
            try:
                # Get memory in MB
                mem_info = self._process.memory_info()
                self._system_metrics_cache['memory_mb'] = mem_info.rss / 1024 / 1024

                # Get CPU percent (non-blocking)
                self._system_metrics_cache['cpu_percent'] = self._process.cpu_percent(interval=0)

                self._system_metrics_cache['last_updated'] = now
            except Exception as e:
                logger.error(f"Error updating system metrics: {e}")

    def get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics (memory, CPU).

        Cached for 2 seconds to avoid excessive psutil calls.

        Returns:
            Dict with 'memory_mb' and 'cpu_percent'
        """
        with self._lock:
            self._update_system_metrics_cache()
            return {
                'memory_mb': self._system_metrics_cache['memory_mb'],
                'cpu_percent': self._system_metrics_cache['cpu_percent'],
            }

    def get_summary(self) -> Dict:
        """Get complete metrics summary.

        Returns:
            Dict with all metrics (latencies, counters, system)
        """
        with self._lock:
            self._update_system_metrics_cache()

            summary = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'provider': self.llm_provider,
                'latencies': {
                    op: self.get_latency_stats(op)
                    for op in self.latencies.keys()
                },
                'counters': self.counters.copy(),
                'system': {
                    'memory_mb': self._system_metrics_cache['memory_mb'],
                    'cpu_percent': self._system_metrics_cache['cpu_percent'],
                },
            }

            return summary

    def set_llm_provider(self, provider_name: str) -> None:
        """Set the active LLM provider name for display/metrics purposes."""

        with self._lock:
            self.llm_provider = provider_name


# Global metrics instance
_metrics_instance = None
_metrics_lock = Lock()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance (singleton)."""
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = MetricsCollector()
    return _metrics_instance


def measure_latency(operation: str, label: Optional[str] = None):
    """Decorator to measure function execution latency.

    Usage:
        @measure_latency('llm_api')
        async def call_api():
            ...

    Args:
        operation: Operation name for categorization
        label: Optional label for further categorization
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                get_metrics().record_latency(operation, duration_ms, label)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                get_metrics().record_latency(operation, duration_ms, label)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

"""Resource optimization service for SoyeBot - manages caching, cleanup, and memory efficiency."""

import logging
import asyncio
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import OrderedDict

from services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class MemoryCache:
    """LRU cache for frequently accessed memories."""

    def __init__(self, maxsize: int = 50):
        """Initialize memory cache.

        Args:
            maxsize: Maximum cache size
        """
        self.maxsize = maxsize
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.maxsize:
                # Remove least recently used item
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }


class OptimizationService:
    """Service for managing resource optimization and cleanup tasks."""

    def __init__(
        self,
        db_service: DatabaseService,
        cache_size: int = 50,
        cleanup_interval_minutes: int = 30,
    ):
        """Initialize optimization service.

        Args:
            db_service: DatabaseService instance
            cache_size: LRU cache size
            cleanup_interval_minutes: Cleanup task interval
        """
        self.db_service = db_service
        self.cache = MemoryCache(maxsize=cache_size)
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.last_cleanup = datetime.utcnow()
        self._cleanup_task = None

    async def start_cleanup_loop(self) -> None:
        """Start background cleanup loop."""
        try:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Optimization service cleanup loop started")
        except Exception as e:
            logger.error(f"Failed to start cleanup loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                await self.perform_cleanup()
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                # Continue despite errors
                continue

    async def perform_cleanup(self, conversation_retention_days: int = 7) -> Dict[str, Any]:
        """Perform cleanup of old data.

        Args:
            conversation_retention_days: Days to keep conversation history

        Returns:
            Cleanup statistics
        """
        try:
            logger.info("Starting database cleanup...")

            # Run cleanup in thread pool to avoid blocking
            stats = await asyncio.to_thread(
                self.db_service.cleanup_expired_data,
                conversation_retention_days,
            )

            # Clear cache to force reload of fresh data
            self.cache.clear()

            logger.info(f"Cleanup completed: {stats}")
            self.last_cleanup = datetime.utcnow()

            return stats

        except Exception as e:
            logger.error(f"Failed to perform cleanup: {e}")
            return {'error': str(e)}

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics.

        Returns:
            Dictionary with optimization stats
        """
        db_stats = self.db_service.get_database_stats()
        cache_stats = self.cache.get_stats()

        return {
            'database': db_stats,
            'cache': cache_stats,
            'last_cleanup': self.last_cleanup.isoformat(),
            'cleanup_interval_minutes': self.cleanup_interval_minutes,
        }

    async def stop(self) -> None:
        """Stop optimization service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class MemoryCompressionService:
    """Service for compressing and summarizing old conversation histories."""

    def __init__(self, db_service: DatabaseService):
        """Initialize compression service.

        Args:
            db_service: DatabaseService instance
        """
        self.db_service = db_service

    async def compress_old_conversations(
        self,
        days_threshold: int = 7,
    ) -> Dict[str, Any]:
        """Compress old conversation histories.

        Args:
            days_threshold: Age threshold for compression in days

        Returns:
            Compression statistics
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)

            # In a real implementation, you would:
            # 1. Query conversations older than cutoff_date
            # 2. Summarize them using Gemini
            # 3. Replace with compressed summary
            # 4. Delete original conversations

            logger.info(f"Compression threshold: {cutoff_date}")

            return {
                'status': 'not_implemented',
                'message': 'Compression requires Gemini API integration',
            }

        except Exception as e:
            logger.error(f"Failed to compress conversations: {e}")
            return {'error': str(e)}

    def get_compression_candidates(
        self,
        days_threshold: int = 7,
    ) -> Dict[str, int]:
        """Get conversations that could be compressed.

        Args:
            days_threshold: Age threshold in days

        Returns:
            Dictionary with compression candidates
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)

            # Query database for old conversations
            # This is a placeholder - actual implementation would count conversations
            # older than cutoff_date

            return {
                'candidates': 0,
                'estimated_savings_bytes': 0,
            }

        except Exception as e:
            logger.error(f"Failed to get compression candidates: {e}")
            return {'error': str(e)}


class ResourceMonitor:
    """Monitor resource usage and provide alerts."""

    def __init__(self):
        """Initialize resource monitor."""
        self.warnings = []

    def check_memory_usage(self, threshold_percent: float = 80.0) -> bool:
        """Check memory usage.

        Args:
            threshold_percent: Alert threshold percentage

        Returns:
            True if usage is above threshold
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            if usage_percent > threshold_percent:
                warning = f"High memory usage: {usage_percent:.1f}%"
                self.warnings.append(warning)
                logger.warning(warning)
                return True

            return False

        except ImportError:
            logger.debug("psutil not available for memory monitoring")
            return False
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return False

    def check_database_size(self, max_size_mb: float = 100.0) -> bool:
        """Check database file size.

        Args:
            max_size_mb: Maximum size in MB

        Returns:
            True if database is above size limit
        """
        try:
            import os
            db_size_bytes = os.path.getsize('soyebot.db')
            db_size_mb = db_size_bytes / (1024 * 1024)

            if db_size_mb > max_size_mb:
                warning = f"Large database: {db_size_mb:.1f}MB"
                self.warnings.append(warning)
                logger.warning(warning)
                return True

            return False

        except FileNotFoundError:
            logger.debug("Database file not found")
            return False
        except Exception as e:
            logger.warning(f"Failed to check database size: {e}")
            return False

    def get_resource_report(self) -> Dict[str, Any]:
        """Get resource usage report.

        Returns:
            Dictionary with resource stats
        """
        try:
            import psutil
            import os

            memory = psutil.virtual_memory()
            db_size_bytes = os.path.getsize('soyebot.db') if os.path.exists('soyebot.db') else 0

            return {
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_available_mb': memory.available / (1024 * 1024),
                'database_size_mb': db_size_bytes / (1024 * 1024),
                'warnings': self.warnings,
            }

        except ImportError:
            logger.debug("psutil not available")
            return {'status': 'monitoring_unavailable'}
        except Exception as e:
            logger.error(f"Failed to get resource report: {e}")
            return {'error': str(e)}

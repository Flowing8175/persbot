"""State management for bot operations.

This module provides classes for managing active tasks, cancellations,
and message buffers across Discord channels.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import discord

logger = logging.getLogger(__name__)


@dataclass
class ActiveAPICall:
    """Tracks an active API call for cancellation.

    This class ensures that when a new message arrives during batch processing:
    1. The ongoing LLM API-side generation is cancelled (like pressing STOP)
    2. The stacked messages are included in the new request
    """

    task: asyncio.Task
    cancel_event: asyncio.Event
    messages: List[discord.Message] = field(default_factory=list)

    def cancel(self) -> None:
        """Cancel both the task and set the cancel event immediately."""
        if self.cancel_event:
            self.cancel_event.set()
        if self.task and not self.task.done():
            self.task.cancel()


class ChannelStateManager:
    """Manages state for individual Discord channels.

    This includes tracking active tasks, batches, and cancellation signals.
    """

    def __init__(self, channel_id: int):
        """Initialize the channel state manager.

        Args:
            channel_id: The Discord channel ID.
        """
        self.channel_id = channel_id
        self.processing_task: Optional[asyncio.Task] = None
        self.sending_task: Optional[asyncio.Task] = None
        self.active_batch: List[discord.Message] = field(default_factory=list)
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.active_api_call: Optional[ActiveAPICall] = None

    def has_active_processing(self) -> bool:
        """Check if there's an active processing task."""
        return self.processing_task is not None and not self.processing_task.done()

    def has_active_sending(self) -> bool:
        """Check if there's an active sending task."""
        return self.sending_task is not None and not self.sending_task.done()

    def cancel_all(self) -> List[discord.Message]:
        """Cancel all active tasks and return pending messages.

        Returns:
            List of messages from the cancelled batch.
        """
        messages_to_prepend = []

        # Cancel active API call FIRST
        if self.active_api_call:
            logger.debug(f"Cancelling API call for channel {self.channel_id}")
            messages_to_prepend = self.active_api_call.messages
            self.active_api_call.cancel()
            self.active_api_call = None

        # Trigger cancellation signal
        self.cancel_event.set()

        # Cancel tasks
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()

        if self.sending_task and not self.sending_task.done():
            self.sending_task.cancel()

        return messages_to_prepend

    def reset(self) -> None:
        """Reset all state."""
        self.processing_task = None
        self.sending_task = None
        self.active_batch = []
        self.cancel_event = asyncio.Event()
        self.active_api_call = None


class BotStateManager:
    """Manages state across all channels.

    Provides a centralized way to access and manage per-channel state.
    """

    def __init__(self):
        """Initialize the bot state manager."""
        self._channels: Dict[int, ChannelStateManager] = {}

    def get_channel(self, channel_id: int) -> ChannelStateManager:
        """Get or create state for a channel.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            The channel state manager.
        """
        if channel_id not in self._channels:
            self._channels[channel_id] = ChannelStateManager(channel_id)
        return self._channels[channel_id]

    def has_active_processing(self, channel_id: int) -> bool:
        """Check if channel has active processing.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            True if active processing exists.
        """
        channel_state = self._channels.get(channel_id)
        return channel_state is not None and channel_state.has_active_processing()

    def cancel_channel(self, channel_id: int) -> List[discord.Message]:
        """Cancel all active tasks for a channel.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            List of messages from the cancelled batch.
        """
        channel_state = self._channels.get(channel_id)
        if channel_state:
            return channel_state.cancel_all()
        return []

    def cleanup(self, channel_id: int) -> None:
        """Clean up state for a channel.

        Args:
            channel_id: The Discord channel ID.
        """
        self._channels.pop(channel_id, None)


class TaskTracker:
    """Utility for tracking and managing asyncio tasks."""

    def __init__(self, max_concurrent: int = 10):
        """Initialize the task tracker.

        Args:
            max_concurrent: Maximum number of concurrent tasks.
        """
        self.max_concurrent = max_concurrent
        self._tasks: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def create_task(
        self,
        key: str,
        coro: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> asyncio.Task:
        """Create a tracked task.

        Args:
            key: Unique key for the task.
            coro: The coroutine to run.
            *args: Arguments for the coroutine.
            **kwargs: Keyword arguments for the coroutine.

        Returns:
            The created task.
        """
        # Cancel existing task with same key
        if key in self._tasks:
            existing = self._tasks[key]
            if not existing.done():
                existing.cancel()

        async def _wrapped():
            async with self._semaphore:
                try:
                    return await coro(*args, **kwargs)
                finally:
                    self._tasks.pop(key, None)

        task = asyncio.create_task(_wrapped())
        self._tasks[key] = task
        return task

    def cancel(self, key: str) -> bool:
        """Cancel a tracked task.

        Args:
            key: The task key.

        Returns:
            True if task was found and cancelled.
        """
        task = self._tasks.get(key)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def get(self, key: str) -> Optional[asyncio.Task]:
        """Get a tracked task.

        Args:
            key: The task key.

        Returns:
            The task, or None if not found.
        """
        return self._tasks.get(key)

    def cancel_all(self) -> int:
        """Cancel all tracked tasks.

        Returns:
            Number of tasks cancelled.
        """
        cancelled = 0
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

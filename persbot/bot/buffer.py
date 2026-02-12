import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import discord

logger = logging.getLogger(__name__)


class MessageBuffer:
    """
    Buffers messages by channel ID to process them in batches.

    New Logic:
    1. Default Wait: Waits for `delay` seconds (default 2.0s) after the last message.
    2. Typing Extension: If a user starts typing while waiting, the wait is extended
       to `typing_timeout` seconds (default 5.0s) from the typing event.
       This allows the user to finish their thought, but prevents infinite waiting
       if they stop typing without sending.
    """

    def __init__(
        self,
        delay: float = 2.0,
        typing_timeout: float = 5.0,
        max_buffer_size: int = 100,
    ):
        self.default_delay = delay
        self.typing_timeout = typing_timeout
        self.buffers: Dict[int, List[discord.Message]] = {}
        self.tasks: Dict[int, asyncio.Task] = {}
        self.max_buffer_size = max_buffer_size  # Prevent unbounded growth

    def _cleanup_task(self, channel_id: int, task: asyncio.Task) -> None:
        """Remove task from tracking when it completes (whether cancelled or finished)."""
        if self.tasks.get(channel_id) == task:
            self.tasks.pop(channel_id, None)

    async def add_message(
        self,
        channel_id: int,
        message: discord.Message,
        callback: Callable[[List[discord.Message]], Any],
    ):
        """
        Adds a message to the buffer for the given channel.
        Resets the processing timer to the default delay (e.g., 2.0s).
        """
        if channel_id not in self.buffers:
            self.buffers[channel_id] = []

        # Limit buffer size to prevent unbounded growth
        if len(self.buffers[channel_id]) >= self.max_buffer_size:
            logger.warning(
                f"Buffer limit reached for channel {channel_id}. Removing oldest message."
            )
            self.buffers[channel_id].pop(0)

        self.buffers[channel_id].append(message)
        logger.debug(
            f"Message added to buffer for channel {channel_id}. Current count: {len(self.buffers[channel_id])}"
        )

        # If a task is already running, cancel it to reset the timer
        if channel_id in self.tasks:
            self.tasks[channel_id].cancel()

        # Start a new timer task with the default delay
        task = asyncio.create_task(self._process_buffer(channel_id, self.default_delay, callback))
        task.add_done_callback(lambda t: self._cleanup_task(channel_id, t))
        self.tasks[channel_id] = task

    def handle_typing(self, channel_id: int, callback: Callable[[List[discord.Message]], Any]):
        """
        Called when a typing event is detected in the channel.
        If a buffer exists and we are waiting, extend the wait time to `typing_timeout`.
        """
        # We only care if there are messages pending.
        # If the buffer is empty, there's nothing to hold back.
        if channel_id not in self.buffers or not self.buffers[channel_id]:
            return

        # If we are already waiting, cancel the current timer
        if channel_id in self.tasks:
            self.tasks[channel_id].cancel()
            logger.debug(
                f"Typing detected in channel {channel_id}. Extending wait to {self.typing_timeout}s."
            )

        # Restart the timer with the extended timeout
        task = asyncio.create_task(self._process_buffer(channel_id, self.typing_timeout, callback))
        task.add_done_callback(lambda t: self._cleanup_task(channel_id, t))
        self.tasks[channel_id] = task

    def update_delay(self, delay: float):
        """
        Update the default delay for message buffering.
        """
        if delay < 0:
            raise ValueError("Delay must be non-negative")
        self.default_delay = delay
        logger.info(f"Buffer delay updated to {delay}s")

    async def _process_buffer(
        self,
        channel_id: int,
        delay: float,
        callback: Callable[[List[discord.Message]], Any],
    ):
        """
        Waits for the given delay, then processes the buffered messages.
        """
        try:
            await asyncio.sleep(delay)

            # Pop the buffer and task *before* calling the callback
            messages = self.buffers.pop(channel_id, [])
            self.tasks.pop(channel_id, None)

            if messages:
                logger.info(
                    f"Processing batch of {len(messages)} messages for channel {channel_id} (waited {delay:.1f}s)"
                )
                await callback(messages)
        except asyncio.CancelledError:
            # Task was cancelled (new message arrived or typing detected).
            pass
        except Exception as e:
            logger.error(
                f"Error processing message buffer for channel {channel_id}: {e}",
                exc_info=True,
            )
            self.buffers.pop(channel_id, None)
            self.tasks.pop(channel_id, None)

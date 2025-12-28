import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
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
    def __init__(self, delay: float = 2.0, typing_timeout: float = 5.0):
        self.default_delay = delay
        self.typing_timeout = typing_timeout
        self.buffers: Dict[int, List[discord.Message]] = {}
        self.tasks: Dict[int, asyncio.Task] = {}

    async def add_message(self, channel_id: int, message: discord.Message, callback: Callable[[List[discord.Message]], Any]):
        """
        Adds a message to the buffer for the given channel.
        Resets the processing timer to the default delay (e.g., 2.0s).
        """
        if channel_id not in self.buffers:
            self.buffers[channel_id] = []

        self.buffers[channel_id].append(message)
        
        # If a task is already running, cancel it to reset the timer
        if channel_id in self.tasks:
            self.tasks[channel_id].cancel()

        # Start a new timer task with the default delay
        task = asyncio.create_task(
            self._process_buffer(channel_id, self.default_delay, callback)
        )
        self.tasks[channel_id] = task

    def handle_typing(self, channel_id: int, callback: Callable[[List[discord.Message]], Any]):
        """
        Called when a typing event is detected in the channel.
        If a buffer exists and we are waiting, extend the wait time to `typing_timeout`.
        """
        # Only extend wait if we have messages pending
        if channel_id in self.buffers and self.buffers[channel_id]:
             # If a task is running, cancel it and restart with typing timeout
            if channel_id in self.tasks:
                self.tasks[channel_id].cancel()
            
            task = asyncio.create_task(
                self._process_buffer(channel_id, self.typing_timeout, callback)
            )
            self.tasks[channel_id] = task

    async def _process_buffer(self, channel_id: int, delay: float, callback: Callable[[List[discord.Message]], Any]):
        """
        Waits for the given delay, then processes the buffered messages.
        """
        try:
            await asyncio.sleep(delay)

            # Pop the buffer, but NOT the task.
            messages = self.buffers.pop(channel_id, [])
            
            if messages:
                await callback(messages)

        except asyncio.CancelledError:
            # Task was cancelled (new message arrived or typing detected).
            raise  # Re-raise to ensure proper cancellation propagation if needed

        except Exception as e:
            logger.error(f"Error processing message buffer for channel {channel_id}: {e}", exc_info=True)
            
        finally:
            # Cleanup the task from the dictionary ONLY if it is THIS task.
            current_task = asyncio.current_task()
            if self.tasks.get(channel_id) == current_task:
                self.tasks.pop(channel_id, None)


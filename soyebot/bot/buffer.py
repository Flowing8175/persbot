import asyncio
import logging
from typing import Dict, List, Callable, Any
import discord

logger = logging.getLogger(__name__)

class MessageBuffer:
    """
    Buffers messages by channel ID to process them in batches.
    Waits for 2.5 seconds after the first message arrives before processing.
    """
    def __init__(self, delay: float = 2.5):
        self.delay = delay
        self.buffers: Dict[int, List[discord.Message]] = {}
        self.tasks: Dict[int, asyncio.Task] = {}

    async def add_message(self, channel_id: int, message: discord.Message, callback: Callable[[List[discord.Message]], Any]):
        """
        Adds a message to the buffer for the given channel.
        If no timer is running, starts one.
        """
        if channel_id not in self.buffers:
            self.buffers[channel_id] = []

        self.buffers[channel_id].append(message)
        logger.debug(f"Message added to buffer for channel {channel_id}. Current count: {len(self.buffers[channel_id])}")

        if channel_id not in self.tasks:
            self.tasks[channel_id] = asyncio.create_task(self._process_buffer(channel_id, callback))

    async def _process_buffer(self, channel_id: int, callback: Callable[[List[discord.Message]], Any]):
        """
        Waits for the delay, then processes the buffered messages.
        """
        try:
            await asyncio.sleep(self.delay)

            messages = self.buffers.pop(channel_id, [])
            self.tasks.pop(channel_id, None)

            if messages:
                logger.info(f"Processing batch of {len(messages)} messages for channel {channel_id}")
                await callback(messages)

        except Exception as e:
            logger.error(f"Error processing message buffer for channel {channel_id}: {e}", exc_info=True)
            # Cleanup in case of error
            self.buffers.pop(channel_id, None)
            self.tasks.pop(channel_id, None)

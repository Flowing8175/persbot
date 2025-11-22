import asyncio
import logging
from typing import Dict, List, Callable, Any
import discord

logger = logging.getLogger(__name__)

class MessageBuffer:
    """
    Buffers messages by channel ID to process them in batches.
    Waits for a dynamic duration after the *last* message arrives before processing.

    Delay Logic:
    - Short messages (< 5 chars) -> Shorter delay
    - Stacked messages -> Shorter delay (Quadratic curve)
    - Range: 0.8s ~ 4.0s
    """
    def __init__(self, delay: float = 2.5):
        # delay parameter is kept for backward compatibility but might be ignored or used as a fallback if needed.
        # However, the new logic completely replaces the fixed delay.
        self.default_delay = delay
        self.buffers: Dict[int, List[discord.Message]] = {}
        self.tasks: Dict[int, asyncio.Task] = {}

    def _calculate_delay(self, channel_id: int) -> float:
        """
        Calculates the dynamic delay based on stack size and last message length.
        """
        messages = self.buffers.get(channel_id, [])
        if not messages:
            return self.default_delay

        stack_size = len(messages)
        last_msg = messages[-1]
        # Ensure content is treated as string even if empty/None (though discord messages usually have content or attachments)
        msg_content = last_msg.content or ""
        msg_len = len(msg_content)

        MIN_DELAY = 0.8
        MAX_DELAY = 4.0
        STACK_THRESHOLD = 4
        SHORT_MSG_LEN = 5
        SHORT_MSG_MULTIPLIER = 0.4
        LONG_MSG_MULTIPLIER = 0.8

        # Base Delay Calculation (Quadratic)
        if stack_size >= STACK_THRESHOLD:
            base_delay = MIN_DELAY
        else:
            # Normalize stack 1..4 to 0..1
            t = (stack_size - 1) / (STACK_THRESHOLD - 1)
            # Quadratic drop: Delay = Max - (Diff * t^2)
            base_delay = MAX_DELAY - (MAX_DELAY - MIN_DELAY) * (t ** 2)

        # Apply Length Multiplier
        # "Short message (len < 5) -> Shorter term (Multiplier 0.4)"
        # "Long message (len >= 5) -> Longer term (Multiplier 0.8)"
        multiplier = SHORT_MSG_MULTIPLIER if msg_len < SHORT_MSG_LEN else LONG_MSG_MULTIPLIER

        calculated_delay = base_delay * multiplier

        # Clamp
        final_delay = max(MIN_DELAY, min(MAX_DELAY, calculated_delay))

        logger.debug(f"Calculated delay for channel {channel_id}: {final_delay:.2f}s (Stack: {stack_size}, Len: {msg_len})")
        return final_delay

    async def add_message(self, channel_id: int, message: discord.Message, callback: Callable[[List[discord.Message]], Any]):
        """
        Adds a message to the buffer for the given channel.
        Resets the processing timer with a dynamically calculated delay.
        """
        if channel_id not in self.buffers:
            self.buffers[channel_id] = []

        self.buffers[channel_id].append(message)
        logger.debug(f"Message added to buffer for channel {channel_id}. Current count: {len(self.buffers[channel_id])}")

        # If a task is already running for this channel, cancel it to reset the timer
        if channel_id in self.tasks:
            self.tasks[channel_id].cancel()

        # Calculate dynamic delay
        delay = self._calculate_delay(channel_id)

        # Start a new timer task with the calculated delay
        self.tasks[channel_id] = asyncio.create_task(self._process_buffer(channel_id, delay, callback))

    async def _process_buffer(self, channel_id: int, delay: float, callback: Callable[[List[discord.Message]], Any]):
        """
        Waits for the given delay, then processes the buffered messages.
        """
        try:
            await asyncio.sleep(delay)

            messages = self.buffers.pop(channel_id, [])
            self.tasks.pop(channel_id, None)

            if messages:
                logger.info(f"Processing batch of {len(messages)} messages for channel {channel_id}")
                await callback(messages)
        except asyncio.CancelledError:
            # Task was cancelled, likely because a new message arrived.
            # Do nothing, as the buffer is preserved and a new task is started.
            pass
        except Exception as e:
            logger.error(f"Error processing message buffer for channel {channel_id}: {e}", exc_info=True)
            # Cleanup in case of error
            self.buffers.pop(channel_id, None)
            self.tasks.pop(channel_id, None)

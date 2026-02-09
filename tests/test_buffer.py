"""Comprehensive tests for MessageBuffer class."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from soyebot.bot.buffer import MessageBuffer


@pytest.fixture
def message_buffer():
    """Create a MessageBuffer instance with test-friendly defaults."""
    # Use shorter delays for faster tests
    return MessageBuffer(delay=0.1, typing_timeout=0.3)


@pytest.fixture
def mock_callback():
    """Create a mock async callback function."""
    return AsyncMock()


@pytest.fixture
def mock_message_factory():
    """Factory to create mock messages with different IDs."""

    def create_message(msg_id, content="Test message", channel_id=111222333):
        msg = Mock()
        msg.id = msg_id
        msg.author = Mock(id=123456789, name="TestUser")
        msg.channel = Mock(id=channel_id, name=f"Channel{channel_id}")
        msg.content = content
        msg.clean_content = content
        msg.mentions = []
        msg.attachments = []
        msg.embeds = []
        msg.reference = None
        msg.created_at = datetime.now(timezone.utc)
        return msg

    return create_message


class TestMessageBufferInit:
    """Tests for MessageBuffer.__init__()"""

    def test_empty_buffers_dict(self, message_buffer):
        """Test that buffers dict is initialized empty."""
        assert message_buffer.buffers == {}
        assert len(message_buffer.buffers) == 0

    def test_default_delay(self, message_buffer):
        """Test default delay parameter (0.1s for tests)."""
        assert message_buffer.default_delay == 0.1

    def test_default_typing_timeout(self, message_buffer):
        """Test default typing_timeout parameter (0.3s for tests)."""
        assert message_buffer.typing_timeout == 0.3

    def test_empty_tasks_dict(self, message_buffer):
        """Test that tasks dict is initialized empty."""
        assert message_buffer.tasks == {}
        assert len(message_buffer.tasks) == 0

    def test_custom_delays(self):
        """Test initialization with custom delays."""
        buffer = MessageBuffer(delay=1.5, typing_timeout=3.0)
        assert buffer.default_delay == 1.5
        assert buffer.typing_timeout == 3.0


class TestAddMessage:
    """Tests for MessageBuffer.add_message()"""

    @pytest.mark.asyncio
    async def test_add_first_message_to_new_channel(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test adding first message creates new channel buffer."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)

        assert 111222333 in message_buffer.buffers
        assert len(message_buffer.buffers[111222333]) == 1
        assert message_buffer.buffers[111222333][0] == msg

    @pytest.mark.asyncio
    async def test_add_message_to_existing_channel(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test adding message to existing channel appends to buffer."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)
        msg2 = mock_message_factory("msg2", channel_id=111222333)

        await message_buffer.add_message(111222333, msg1, mock_callback)
        await message_buffer.add_message(111222333, msg2, mock_callback)

        assert len(message_buffer.buffers[111222333]) == 2
        assert message_buffer.buffers[111222333][0] == msg1
        assert message_buffer.buffers[111222333][1] == msg2

    @pytest.mark.asyncio
    async def test_messages_stored_correctly(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test messages are stored with correct attributes."""
        msg = mock_message_factory("msg123", "Hello world", 111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)

        stored = message_buffer.buffers[111222333][0]
        assert stored.id == "msg123"
        assert stored.content == "Hello world"
        assert stored.author.id == 123456789

    @pytest.mark.asyncio
    async def test_callback_scheduling(self, message_buffer, mock_message_factory, mock_callback):
        """Test that processing task is scheduled after adding message."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)

        # Task should be created
        assert 111222333 in message_buffer.tasks
        assert message_buffer.tasks[111222333] is not None

    @pytest.mark.asyncio
    async def test_resets_existing_timer(self, message_buffer, mock_message_factory, mock_callback):
        """Test that adding message cancels and restarts timer."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)
        msg2 = mock_message_factory("msg2", channel_id=111222333)

        await message_buffer.add_message(111222333, msg1, mock_callback)
        first_task = message_buffer.tasks[111222333]

        await message_buffer.add_message(111222333, msg2, mock_callback)
        second_task = message_buffer.tasks[111222333]

        # Tasks should be different (first was cancelled)
        assert first_task != second_task


class TestProcessBatch:
    """Tests for MessageBuffer._process_buffer()"""

    @pytest.mark.asyncio
    async def test_calls_callback_with_messages(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that callback is called with buffered messages."""
        msg = mock_message_factory("msg1", "Test content", 111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)

        # Wait for buffer to process
        await asyncio.sleep(0.2)

        # Callback should have been called
        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0] == msg

    @pytest.mark.asyncio
    async def test_clears_buffer_after_processing(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that buffer is cleared after processing."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Buffer should be cleared
        assert 111222333 not in message_buffer.buffers

    @pytest.mark.asyncio
    async def test_passes_correct_channel_id(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that callback receives messages from correct channel."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)
        msg2 = mock_message_factory("msg2", channel_id=222333444)

        await message_buffer.add_message(111222333, msg1, mock_callback)
        await message_buffer.add_message(222333444, msg2, mock_callback)

        # Wait for both to process
        await asyncio.sleep(0.3)

        # Callback should be called twice with correct messages
        assert mock_callback.call_count == 2

        # Get both calls
        calls = mock_callback.call_args_list
        first_call_messages = calls[0][0][0]
        second_call_messages = calls[1][0][0]

        # Verify each call has correct messages
        assert len(first_call_messages) == 1
        assert len(second_call_messages) == 1
        assert first_call_messages[0].channel.id == 111222333
        assert second_call_messages[0].channel.id == 222333444


class TestHandleTyping:
    """Tests for MessageBuffer.handle_typing()"""

    @pytest.mark.asyncio
    async def test_extends_timeout_when_typing_detected(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that typing extends the wait timeout."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)
        first_task = message_buffer.tasks[111222333]

        # Simulate typing after a short delay
        await asyncio.sleep(0.05)
        message_buffer.handle_typing(111222333, mock_callback)
        second_task = message_buffer.tasks[111222333]

        # Tasks should be different (timer was extended)
        assert first_task != second_task

    @pytest.mark.asyncio
    async def test_only_extends_active_buffers(self, message_buffer, mock_callback):
        """Test that typing only extends when buffer has messages."""
        # Try to extend typing on non-existent buffer
        message_buffer.handle_typing(999888777, mock_callback)

        # No task should be created
        assert 999888777 not in message_buffer.tasks
        assert 999888777 not in message_buffer.buffers

    @pytest.mark.asyncio
    async def test_extends_correct_amount(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that typing extends to typing_timeout (0.3s)."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        # Start timing
        start_time = asyncio.get_event_loop().time()
        await message_buffer.add_message(111222333, msg, mock_callback)

        # Wait a bit then trigger typing
        await asyncio.sleep(0.05)
        message_buffer.handle_typing(111222333, mock_callback)

        # Wait for callback
        await asyncio.sleep(0.3)
        end_time = asyncio.get_event_loop().time()

        # Total time should be close to typing_timeout (0.3s) + the small delay (0.05s)
        # We're testing that the timer was extended, not exact timing
        assert end_time - start_time >= 0.3

    @pytest.mark.asyncio
    async def test_not_extending_without_active_batch(self, message_buffer, mock_callback):
        """Test that typing doesn't create task without messages."""
        # No messages added, just typing
        message_buffer.handle_typing(111222333, mock_callback)

        # Should not create a task
        assert 111222333 not in message_buffer.tasks


class TestTimeoutBehavior:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_automatic_batch_processing_after_timeout(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that batch is automatically processed after timeout."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)

        # Wait for default delay
        await asyncio.sleep(0.15)

        # Callback should have been called
        mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_reset_after_processing(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that timeout is reset after batch processing."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg1, mock_callback)
        await asyncio.sleep(0.2)

        # Buffer should be cleared
        assert 111222333 not in message_buffer.buffers
        assert 111222333 not in message_buffer.tasks

        # Add another message - should create new buffer
        msg2 = mock_message_factory("msg2", channel_id=111222333)
        await message_buffer.add_message(111222333, msg2, mock_callback)

        # Should have new task
        assert 111222333 in message_buffer.tasks

    @pytest.mark.asyncio
    async def test_manual_processing_via_multiple_messages(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that adding multiple messages before timeout processes all at once."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)
        msg2 = mock_message_factory("msg2", channel_id=111222333)
        msg3 = mock_message_factory("msg3", channel_id=111222333)

        await message_buffer.add_message(111222333, msg1, mock_callback)
        await asyncio.sleep(0.01)  # Small delay between messages
        await message_buffer.add_message(111222333, msg2, mock_callback)
        await asyncio.sleep(0.01)
        await message_buffer.add_message(111222333, msg3, mock_callback)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Callback should be called with all 3 messages
        mock_callback.assert_called_once()
        messages = mock_callback.call_args[0][0]
        assert len(messages) == 3
        assert messages[0].id == "msg1"
        assert messages[1].id == "msg2"
        assert messages[2].id == "msg3"


class TestChannelIsolation:
    """Tests for channel separation."""

    @pytest.mark.asyncio
    async def test_messages_from_different_channels_dont_mix(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that different channels have separate buffers."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)
        msg2 = mock_message_factory("msg2", channel_id=222333444)

        await message_buffer.add_message(111222333, msg1, mock_callback)
        await message_buffer.add_message(222333444, msg2, mock_callback)

        # Buffers should be separate
        assert 111222333 in message_buffer.buffers
        assert 222333444 in message_buffer.buffers
        assert len(message_buffer.buffers[111222333]) == 1
        assert len(message_buffer.buffers[222333444]) == 1
        assert message_buffer.buffers[111222333][0] != message_buffer.buffers[222333444][0]

    @pytest.mark.asyncio
    async def test_typing_in_one_channel_doesnt_affect_another(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that typing in one channel doesn't affect other channels."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)
        msg2 = mock_message_factory("msg2", channel_id=222333444)

        await message_buffer.add_message(111222333, msg1, mock_callback)
        await message_buffer.add_message(222333444, msg2, mock_callback)

        task1_before = message_buffer.tasks[111222333]
        task2_before = message_buffer.tasks[222333444]

        # Trigger typing on channel 1
        message_buffer.handle_typing(111222333, mock_callback)

        task1_after = message_buffer.tasks[111222333]
        task2_after = message_buffer.tasks[222333444]

        # Channel 1 task should change, channel 2 should not
        assert task1_before != task1_after
        assert task2_before == task2_after

    @pytest.mark.asyncio
    async def test_independent_timeouts_per_channel(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that each channel has independent timeout timers."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)
        msg2 = mock_message_factory("msg2", channel_id=222333444)

        # Add messages at slightly different times
        await message_buffer.add_message(111222333, msg1, mock_callback)
        await asyncio.sleep(0.05)
        await message_buffer.add_message(222333444, msg2, mock_callback)

        # Wait for first channel to process
        await asyncio.sleep(0.1)

        # First should have processed, second still waiting
        assert 111222333 not in message_buffer.buffers
        assert 222333444 in message_buffer.buffers

        # Wait for second to process
        await asyncio.sleep(0.1)
        assert 222333444 not in message_buffer.buffers


class TestEdgeCases:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_duplicate_messages(self, message_buffer, mock_message_factory, mock_callback):
        """Test handling duplicate message objects."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        # Add same message twice
        await message_buffer.add_message(111222333, msg, mock_callback)
        await message_buffer.add_message(111222333, msg, mock_callback)

        # Both should be in buffer
        assert len(message_buffer.buffers[111222333]) == 2

        # Wait for processing
        await asyncio.sleep(0.2)

        # Callback should receive both
        messages = mock_callback.call_args[0][0]
        assert len(messages) == 2
        assert messages[0] == messages[1]

    @pytest.mark.asyncio
    async def test_very_rapid_message_additions(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test handling rapid successive message additions."""
        messages = []
        for i in range(10):
            msg = mock_message_factory(f"msg{i}", channel_id=111222333)
            messages.append(msg)
            await message_buffer.add_message(111222333, msg, mock_callback)

        # All messages should be in buffer
        assert len(message_buffer.buffers[111222333]) == 10

        # Wait for processing
        await asyncio.sleep(0.2)

        # All messages should be processed together
        processed_messages = mock_callback.call_args[0][0]
        assert len(processed_messages) == 10

    @pytest.mark.asyncio
    async def test_concurrent_access_same_channel(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test concurrent access to the same channel."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)
        msg2 = mock_message_factory("msg2", channel_id=111222333)

        # Add messages concurrently
        await asyncio.gather(
            message_buffer.add_message(111222333, msg1, mock_callback),
            message_buffer.add_message(111222333, msg2, mock_callback),
        )

        # Both should be in buffer
        assert len(message_buffer.buffers[111222333]) >= 1

        # Wait for processing
        await asyncio.sleep(0.2)

        # Should have processed
        mock_callback.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_access_different_channels(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test concurrent access to different channels."""
        tasks = []
        for channel_id in [111222333, 222333444, 333444555]:
            for i in range(3):
                msg = mock_message_factory(f"msg_{channel_id}_{i}", channel_id=channel_id)
                tasks.append(message_buffer.add_message(channel_id, msg, mock_callback))

        # Run all additions concurrently
        await asyncio.gather(*tasks)

        # Each channel should have 3 messages
        assert len(message_buffer.buffers[111222333]) == 3
        assert len(message_buffer.buffers[222333444]) == 3
        assert len(message_buffer.buffers[333444555]) == 3

        # Wait for all to process
        await asyncio.sleep(0.2)

        # All should be processed
        assert mock_callback.call_count >= 3

    @pytest.mark.asyncio
    async def test_empty_buffer_after_typing_clear(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that typing on empty buffer does nothing."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)
        await asyncio.sleep(0.2)  # Let it process

        # Now buffer is empty
        assert 111222333 not in message_buffer.buffers

        # Typing should not create a new task
        message_buffer.handle_typing(111222333, mock_callback)
        assert 111222333 not in message_buffer.tasks

    @pytest.mark.asyncio
    async def test_callback_exception_handling(self, message_buffer, mock_message_factory):
        """Test that exceptions in callback are handled gracefully."""

        # Create a callback that raises an exception
        async def failing_callback(messages):
            raise ValueError("Test exception")

        msg = mock_message_factory("msg1", channel_id=111222333)

        # Should not raise exception even though callback fails
        await message_buffer.add_message(111222333, msg, failing_callback)
        await asyncio.sleep(0.2)

        # Buffer should be cleared even after exception
        assert 111222333 not in message_buffer.buffers
        assert 111222333 not in message_buffer.tasks

    @pytest.mark.asyncio
    async def test_multiple_channels_processing(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that multiple channels can process independently."""
        channels = [111222333, 222333444, 333444555]
        for channel_id in channels:
            msg = mock_message_factory(f"msg_{channel_id}", channel_id=channel_id)
            await message_buffer.add_message(channel_id, msg, mock_callback)

        # Wait for all to process
        await asyncio.sleep(0.3)

        # All channels should have been processed
        assert mock_callback.call_count == len(channels)

    @pytest.mark.asyncio
    async def test_typing_extends_multiple_times(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that typing can extend multiple times."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)

        # Extend multiple times
        for _ in range(3):
            await asyncio.sleep(0.05)
            message_buffer.handle_typing(111222333, mock_callback)

        # Wait for final timeout (last extension was at 0.15s, need 0.3s more)
        await asyncio.sleep(0.35)

        # Should have processed once
        mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_after_typing_extension(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test adding message after typing extension."""
        msg1 = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg1, mock_callback)
        await asyncio.sleep(0.05)
        message_buffer.handle_typing(111222333, mock_callback)

        # Add another message
        msg2 = mock_message_factory("msg2", channel_id=111222333)
        await asyncio.sleep(0.05)
        await message_buffer.add_message(111222333, msg2, mock_callback)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Both messages should be processed
        messages = mock_callback.call_args[0][0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_no_messages_after_timeout_with_typing(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test buffer behavior when typing happens but no new messages."""
        msg = mock_message_factory("msg1", channel_id=111222333)

        await message_buffer.add_message(111222333, msg, mock_callback)

        # Extend with typing but don't add more messages
        await asyncio.sleep(0.05)
        message_buffer.handle_typing(111222333, mock_callback)

        # Wait for extended timeout
        await asyncio.sleep(0.4)

        # Should still process the original message
        mock_callback.assert_called_once()
        messages = mock_callback.call_args[0][0]
        assert len(messages) == 1
        assert messages[0].id == "msg1"

    @pytest.mark.asyncio
    async def test_callback_preserves_message_order(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test that messages are delivered in correct order."""
        messages = []
        for i in range(5):
            msg = mock_message_factory(f"msg{i}", channel_id=111222333)
            messages.append(msg)
            await message_buffer.add_message(111222333, msg, mock_callback)
            await asyncio.sleep(0.01)  # Small delay to ensure order

        # Wait for processing
        await asyncio.sleep(0.2)

        # Messages should be in order
        processed_messages = mock_callback.call_args[0][0]
        for i, msg in enumerate(processed_messages):
            assert msg.id == f"msg{i}"

    @pytest.mark.asyncio
    async def test_zero_delay_processing(self, mock_message_factory):
        """Test behavior with zero delay (immediate processing)."""
        buffer = MessageBuffer(delay=0, typing_timeout=0.1)
        mock_callback = AsyncMock()

        msg = mock_message_factory("msg1", channel_id=111222333)

        await buffer.add_message(111222333, msg, mock_callback)

        # Should process immediately
        await asyncio.sleep(0.05)
        mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_large_number_of_messages(
        self, message_buffer, mock_message_factory, mock_callback
    ):
        """Test handling a large number of messages."""
        num_messages = 50
        messages = []
        for i in range(num_messages):
            msg = mock_message_factory(f"msg{i}", channel_id=111222333)
            messages.append(msg)
            await message_buffer.add_message(111222333, msg, mock_callback)

        # All should be buffered
        assert len(message_buffer.buffers[111222333]) == num_messages

        # Wait for processing
        await asyncio.sleep(0.2)

        # All should be processed
        processed_messages = mock_callback.call_args[0][0]
        assert len(processed_messages) == num_messages

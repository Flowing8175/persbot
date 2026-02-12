"""Tests for Bot Infrastructure Components."""

import asyncio
import io
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from persbot.bot.buffer import MessageBuffer
from persbot.bot.chat_handler import (
    ChatReply,
    create_chat_reply,
    create_chat_reply_stream,
    resolve_session_for_message,
    send_split_response,
    send_streaming_response,
)
from persbot.bot.response_sender import (
    send_immediate_response,
    send_split_response as response_sender_send_split_response,
    send_streaming_response as response_sender_send_streaming_response,
    send_with_images,
)
from persbot.bot.session import ResolvedSession, SessionManager
from persbot.bot.session_resolver import (
    extract_session_context,
    resolve_session_for_message as session_resolver_resolve_session,
)
from persbot.bot.state_manager import (
    ActiveAPICall,
    BotStateManager,
    ChannelStateManager,
    TaskTracker,
)


# Helper function to create async iterator
async def async_iterable(items):
    """Helper to create an async iterator for mocking."""
    for item in items:
        yield item


# Helper to create an async context manager for mocking typing
@asynccontextmanager
async def mock_typing_context():
    """Mock async context manager for typing."""
    yield


class TestChatHandler:
    """Test ChatHandler core chat processing logic."""

    @pytest.mark.asyncio
    async def test_resolve_session_for_message_no_reference(
        self, mock_message, mock_session_manager
    ):
        """Test resolving session for a message without reference."""
        mock_session_manager.resolve_session = AsyncMock(
            return_value=Mock(session_key="channel:123", cleaned_message="test")
        )

        result = await resolve_session_for_message(
            mock_message, "test content", session_manager=mock_session_manager
        )

        assert result is not None
        assert result.session_key == "channel:123"
        mock_session_manager.resolve_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_session_for_message_with_reference(
        self, mock_message, mock_session_manager, mock_user
    ):
        """Test resolving session for a message with reply reference."""
        # Create mock reference message
        ref_message = Mock()
        ref_message.author = mock_user
        ref_message.clean_content = "Original message"
        ref_message.author.bot = False

        mock_message.reference = Mock()
        mock_message.reference.message_id = "ref123"
        mock_message.reference.resolved = ref_message

        mock_session_manager.resolve_session = AsyncMock(
            return_value=Mock(session_key="channel:123", cleaned_message="test")
        )

        result = await resolve_session_for_message(
            mock_message, "test content", session_manager=mock_session_manager
        )

        assert result is not None
        # Check that reply context was added
        call_args = mock_session_manager.resolve_session.call_args
        assert "답장 대상" in call_args[1]["message_content"]

    @pytest.mark.asyncio
    async def test_resolve_session_for_message_reply_to_summary(
        self, mock_message, mock_session_manager, mock_user, mock_bot
    ):
        """Test detecting reply to summary message."""
        # Create mock reference message from bot with summary content
        ref_message = Mock()
        ref_message.author = mock_bot.user
        ref_message.clean_content = "**... 요약:** Test summary"
        ref_message.author.bot = True

        mock_message.reference = Mock()
        mock_message.reference.message_id = "ref123"
        mock_message.reference.resolved = ref_message

        mock_session_manager.resolve_session = AsyncMock(
            return_value=Mock(session_key="channel:123", cleaned_message="test")
        )

        result = await resolve_session_for_message(
            mock_message, "test content", session_manager=mock_session_manager
        )

        assert result is not None
        assert result.is_reply_to_summary is True

    @pytest.mark.asyncio
    async def test_resolve_session_for_message_no_cleaned_message(
        self, mock_message, mock_session_manager
    ):
        """Test returning None when there's no cleaned message."""
        mock_session_manager.resolve_session = AsyncMock(
            return_value=Mock(session_key="channel:123", cleaned_message=None)
        )

        result = await resolve_session_for_message(
            mock_message, "test content", session_manager=mock_session_manager
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_create_chat_reply_success(
        self,
        mock_message,
        mock_llm_service,
        mock_session_manager,
    ):
        """Test successful creation of chat reply."""
        mock_llm_service.generate_chat_response = AsyncMock(return_value=("Test response", None))
        mock_session_manager.get_or_create = AsyncMock(
            return_value=(Mock(model_alias="gemini-2.5-flash"), "channel:123")
        )

        resolution = Mock(session_key="channel:123", cleaned_message="test")
        resolution.is_reply_to_summary = False

        result = await create_chat_reply(
            mock_message,
            resolution=resolution,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        assert result is not None
        assert result.text == "Test response"
        assert result.session_key == "channel:123"

    @pytest.mark.asyncio
    async def test_send_split_response_success(self, mock_channel, mock_session_manager):
        """Test successful split response sending."""
        # Mocks typing context manager
        mock_channel.typing = mock_typing_context

        reply = ChatReply(text="Line 1\nLine 2\nLine 3", session_key="channel:123", response=None)

        await send_split_response(mock_channel, reply, mock_session_manager)

        # Verify multiple messages were sent
        assert mock_channel.send.call_count == 3


class TestStateManager:
    """Test StateManager bot state tracking."""

    def test_channel_state_manager_initialization(self):
        """Test ChannelStateManager initialization."""
        channel_state = ChannelStateManager(channel_id=111222333)

        assert channel_state.channel_id == 111222333
        assert channel_state.processing_task is None
        assert channel_state.sending_task is None
        # active_batch is a dataclass field - check if it's defined
        assert hasattr(channel_state, "active_batch")
        assert channel_state.active_api_call is None
        assert not channel_state.cancel_event.is_set()

    def test_channel_state_manager_has_active_processing_true(self):
        """Test has_active_processing returns True when active."""
        channel_state = ChannelStateManager(channel_id=111222333)
        channel_state.processing_task = Mock(done=Mock(return_value=False))

        assert channel_state.has_active_processing() is True

    def test_channel_state_manager_has_active_processing_false(self):
        """Test has_active_processing returns False when inactive."""
        channel_state = ChannelStateManager(channel_id=111222333)

        assert channel_state.has_active_processing() is False

    def test_channel_state_manager_has_active_sending_true(self):
        """Test has_active_sending returns True when active."""
        channel_state = ChannelStateManager(channel_id=111222333)
        channel_state.sending_task = Mock(done=Mock(return_value=False))

        assert channel_state.has_active_sending() is True

    def test_channel_state_manager_cancel_all(self):
        """Test cancel_all cancels all tasks."""
        channel_state = ChannelStateManager(channel_id=111222333)

        # Mock tasks
        processing_task = Mock()
        processing_task.done = Mock(return_value=False)
        sending_task = Mock()
        sending_task.done = Mock(return_value=False)

        channel_state.processing_task = processing_task
        channel_state.sending_task = sending_task
        channel_state.active_api_call = None

        result = channel_state.cancel_all()

        assert isinstance(result, list)
        assert channel_state.cancel_event.is_set()
        processing_task.cancel.assert_called_once()
        sending_task.cancel.assert_called_once()

    def test_channel_state_manager_cancel_all_with_api_call(self):
        """Test cancel_all with active API call."""
        channel_state = ChannelStateManager(channel_id=111222333)

        # Mock API call with messages
        mock_messages = [Mock(id=1), Mock(id=2)]
        api_call = ActiveAPICall(
            task=Mock(),
            cancel_event=asyncio.Event(),
            messages=mock_messages,
        )

        channel_state.active_api_call = api_call

        result = channel_state.cancel_all()

        assert result == mock_messages
        assert api_call.cancel_event.is_set()

    def test_channel_state_manager_reset(self):
        """Test reset clears all state."""
        channel_state = ChannelStateManager(channel_id=111222333)
        channel_state.processing_task = Mock()
        channel_state.sending_task = Mock()
        channel_state.active_api_call = Mock()

        channel_state.reset()

        assert channel_state.processing_task is None
        assert channel_state.sending_task is None
        assert channel_state.active_api_call is None
        assert not channel_state.cancel_event.is_set()

    def test_bot_state_manager_initialization(self):
        """Test BotStateManager initialization."""
        bot_state = BotStateManager()

        assert bot_state._channels == {}

    def test_bot_state_manager_get_channel(self):
        """Test get_channel creates or returns channel state."""
        bot_state = BotStateManager()

        state1 = bot_state.get_channel(111222333)
        state2 = bot_state.get_channel(111222333)

        assert state1 is state2
        assert isinstance(state1, ChannelStateManager)
        assert state1.channel_id == 111222333

    def test_bot_state_manager_has_active_processing(self):
        """Test has_active_processing checks channel state."""
        bot_state = BotStateManager()

        # No channel state
        assert bot_state.has_active_processing(111222333) is False

        # With active processing
        channel_state = bot_state.get_channel(111222333)
        channel_state.processing_task = Mock(done=Mock(return_value=False))

        assert bot_state.has_active_processing(111222333) is True

    def test_bot_state_manager_cancel_channel(self):
        """Test cancel_channel cancels all tasks for channel."""
        bot_state = BotStateManager()
        channel_state = bot_state.get_channel(111222333)

        processing_task = Mock()
        processing_task.done = Mock(return_value=False)
        channel_state.processing_task = processing_task

        result = bot_state.cancel_channel(111222333)

        assert isinstance(result, list)
        processing_task.cancel.assert_called_once()

    def test_bot_state_manager_cleanup(self):
        """Test cleanup removes channel state."""
        bot_state = BotStateManager()
        bot_state.get_channel(111222333)

        assert 111222333 in bot_state._channels

        bot_state.cleanup(111222333)

        assert 111222333 not in bot_state._channels

    @pytest.mark.asyncio
    async def test_task_tracker_initialization(self):
        """Test TaskTracker initialization."""
        tracker = TaskTracker()

        assert tracker.max_concurrent == 10
        assert tracker._tasks == {}
        assert tracker._semaphore is not None

    @pytest.mark.asyncio
    async def test_task_tracker_create_task(self):
        """Test creating a tracked task."""
        tracker = TaskTracker()

        async def mock_coro():
            return "result"

        task = await tracker.create_task("test_key", mock_coro)

        assert "test_key" in tracker._tasks
        assert task is not None

    @pytest.mark.asyncio
    async def test_task_tracker_cancel_existing_task(self):
        """Test cancelling existing task when creating new one."""
        tracker = TaskTracker()

        existing_task = Mock()
        existing_task.done = Mock(return_value=False)
        tracker._tasks["test_key"] = existing_task

        async def mock_coro():
            return "result"

        await tracker.create_task("test_key", mock_coro)

        existing_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_tracker_get(self):
        """Test getting a tracked task."""
        tracker = TaskTracker()
        mock_task = Mock()
        tracker._tasks["test_key"] = mock_task

        result = tracker.get("test_key")

        assert result is mock_task

    @pytest.mark.asyncio
    async def test_task_tracker_cancel(self):
        """Test cancelling a tracked task."""
        tracker = TaskTracker()
        mock_task = Mock()
        mock_task.done = Mock(return_value=False)
        tracker._tasks["test_key"] = mock_task

        result = tracker.cancel("test_key")

        assert result is True
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_tracker_cancel_all(self):
        """Test cancelling all tracked tasks."""
        tracker = TaskTracker()

        for i in range(3):
            mock_task = Mock()
            mock_task.done = Mock(return_value=False)
            tracker._tasks[f"key_{i}"] = mock_task

        result = tracker.cancel_all()

        assert result == 3


class TestSessionResolver:
    """Test SessionResolver session resolution logic."""

    @pytest.mark.asyncio
    async def test_resolve_session_for_message_with_content(
        self, mock_message, mock_session_manager
    ):
        """Test resolving session with message content."""
        mock_session_manager.resolve_session = AsyncMock(
            return_value=Mock(session_key="channel:123", cleaned_message="test")
        )

        result = await session_resolver_resolve_session(
            mock_message, "test content", session_manager=mock_session_manager
        )

        assert result is not None
        assert result.session_key == "channel:123"
        mock_session_manager.resolve_session.assert_called_once_with(
            channel_id=mock_message.channel.id,
            author_id=mock_message.author.id,
            username=mock_message.author.name,
            message_id=str(mock_message.id),
            message_content="test content",  # Uses content parameter
            reference_message_id=None,
            created_at=mock_message.created_at,
            cancel_event=None,
        )

    @pytest.mark.asyncio
    async def test_resolve_session_no_cleaned_message(self, mock_message, mock_session_manager):
        """Test returning None when cleaned_message is None."""
        mock_session_manager.resolve_session = AsyncMock(
            return_value=Mock(session_key="channel:123", cleaned_message=None)
        )

        result = await session_resolver_resolve_session(
            mock_message, "test content", session_manager=mock_session_manager
        )

        assert result is None

    def test_extract_session_context_single_message(self, mock_message):
        """Test extracting context from single message."""
        from persbot.bot.chat_models import SessionContext

        result = extract_session_context(mock_message)

        assert isinstance(result, SessionContext)
        assert result.channel_id == mock_message.channel.id
        assert result.user_id == mock_message.author.id
        assert result.username == mock_message.author.name
        assert result.message_id == str(mock_message.id)
        assert result.created_at == mock_message.created_at

    def test_extract_session_context_message_list(self, mock_message):
        """Test extracting context from message list."""
        from persbot.bot.chat_models import SessionContext

        mock_message2 = Mock()
        mock_message2.channel = mock_message.channel
        mock_message2.author = mock_message.author
        mock_message2.id = "msg2"
        mock_message2.author.name = mock_message.author.name
        mock_message2.created_at = datetime.now(timezone.utc)

        result = extract_session_context([mock_message, mock_message2])

        assert isinstance(result, SessionContext)
        assert result.message_id == str(mock_message.id)  # Uses first message


class TestResponseSender:
    """Test ResponseSender response handling."""

    @pytest.mark.asyncio
    async def test_send_split_response_with_text(self, mock_channel, mock_session_manager):
        """Test sending split response with text."""
        from persbot.bot.chat_models import ChatReply

        # Mocks typing context manager
        mock_channel.typing = mock_typing_context

        reply = ChatReply(text="Line 1\nLine 2\nLine 3", session_key="channel:123", response=None)

        await response_sender_send_split_response(mock_channel, reply, mock_session_manager)

        assert mock_channel.send.call_count == 3
        mock_session_manager.link_message_to_session.assert_called()

    @pytest.mark.asyncio
    async def test_send_split_response_with_images(self, mock_channel, mock_session_manager):
        """Test sending split response with images."""
        from persbot.bot.chat_models import ChatReply

        # Mocks typing context manager
        mock_channel.typing = mock_typing_context

        reply = ChatReply(
            text="Test response",
            session_key="channel:123",
            response=None,
            images=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
        )

        await response_sender_send_split_response(mock_channel, reply, mock_session_manager)

        # 1 text message + 2 image messages
        assert mock_channel.send.call_count == 3

    @pytest.mark.asyncio
    async def test_send_immediate_response(self, mock_channel, mock_session_manager):
        """Test sending immediate response."""
        result = await send_immediate_response(
            mock_channel, "Test message", "channel:123", mock_session_manager
        )

        assert result is not None
        mock_channel.send.assert_called_once_with("Test message")
        mock_session_manager.link_message_to_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_with_images_text_only(self, mock_channel, mock_session_manager):
        """Test sending with images when only text provided."""
        images = []
        result = await send_with_images(
            mock_channel, "Test message", images, "channel:123", mock_session_manager
        )

        assert len(result) == 1
        mock_channel.send.assert_called_once_with("Test message")

    @pytest.mark.asyncio
    async def test_send_with_images_with_image_bytes(self, mock_channel, mock_session_manager):
        """Test sending with actual image bytes."""
        # Mocks typing context manager
        mock_channel.typing = mock_typing_context

        images = [b"fake_image_bytes"]

        result = await send_with_images(
            mock_channel, "Test message", images, "channel:123", mock_session_manager
        )

        assert len(result) == 2  # 1 text + 1 image

    @pytest.mark.asyncio
    async def test_send_streaming_response_success(self, mock_channel, mock_session_manager):
        """Test successful streaming response."""
        # Mocks typing context manager
        mock_channel.typing = mock_typing_context

        async def mock_stream():
            yield "Chunk 1\n"
            yield "Chunk 2\n"
            yield "Chunk 3"

        result = await response_sender_send_streaming_response(
            mock_channel, mock_stream(), "channel:123", mock_session_manager
        )

        assert len(result) == 3
        mock_session_manager.link_message_to_session.assert_called()


class TestMessageBuffer:
    """Test MessageBuffer message buffering."""

    @pytest.mark.asyncio
    async def test_add_message_to_buffer(self):
        """Test adding message to buffer."""
        buffer = MessageBuffer(delay=2.0)
        mock_message = Mock()

        callback = AsyncMock()

        await buffer.add_message(111222333, mock_message, callback)

        assert 111222333 in buffer.buffers
        assert len(buffer.buffers[111222333]) == 1

    @pytest.mark.asyncio
    async def test_add_message_resets_timer(self):
        """Test that adding message resets timer."""
        buffer = MessageBuffer(delay=2.0)
        mock_message = Mock()

        callback = AsyncMock()

        # Add first message
        await buffer.add_message(111222333, mock_message, callback)
        first_task = buffer.tasks.get(111222333)

        # Add second message immediately
        mock_message2 = Mock()
        await buffer.add_message(111222333, mock_message2, callback)
        second_task = buffer.tasks.get(111222333)

        # Tasks should be different (timer was reset)
        assert first_task is not second_task
        assert len(buffer.buffers[111222333]) == 2

    @pytest.mark.asyncio
    async def test_add_message_max_buffer_size(self):
        """Test buffer size limit enforcement."""
        buffer = MessageBuffer(delay=2.0, max_buffer_size=3)
        callback = AsyncMock()

        # Add more messages than max_buffer_size
        for i in range(5):
            mock_message = Mock()
            await buffer.add_message(111222333, mock_message, callback)

        # Should only keep max_buffer_size messages
        assert len(buffer.buffers[111222333]) == 3

    @pytest.mark.asyncio
    async def test_handle_typing_extends_wait(self):
        """Test that typing event extends wait time."""
        buffer = MessageBuffer(delay=2.0, typing_timeout=5.0)
        mock_message = Mock()

        callback = AsyncMock()

        # Add message to buffer
        await buffer.add_message(111222333, mock_message, callback)
        first_task = buffer.tasks.get(111222333)

        # Trigger typing
        buffer.handle_typing(111222333, callback)
        second_task = buffer.tasks.get(111222333)

        # Timer should be reset
        assert first_task is not second_task

    @pytest.mark.asyncio
    async def test_handle_typing_empty_buffer(self):
        """Test typing with empty buffer is ignored."""
        buffer = MessageBuffer(delay=2.0, typing_timeout=5.0)
        callback = AsyncMock()

        # Should not raise error
        buffer.handle_typing(111222333, callback)
        assert 111222333 not in buffer.buffers

    @pytest.mark.asyncio
    async def test_update_delay(self):
        """Test updating buffer delay."""
        buffer = MessageBuffer(delay=2.0)

        buffer.update_delay(5.0)
        assert buffer.default_delay == 5.0

    @pytest.mark.asyncio
    async def test_update_delay_negative_raises_error(self):
        """Test that negative delay raises error."""
        buffer = MessageBuffer(delay=2.0)

        with pytest.raises(ValueError, match="Delay must be non-negative"):
            buffer.update_delay(-1.0)

    @pytest.mark.asyncio
    async def test_process_buffer_calls_callback(self):
        """Test that process_buffer calls callback."""
        buffer = MessageBuffer(delay=0.1)  # Short delay for testing

        mock_message = Mock()
        messages_to_process = []

        async def callback(messages):
            messages_to_process.extend(messages)

        await buffer.add_message(111222333, mock_message, callback)

        # Wait for buffer to process
        await asyncio.sleep(0.3)

        assert len(messages_to_process) == 1
        assert messages_to_process[0] is mock_message

    @pytest.mark.asyncio
    async def test_process_buffer_cancellation(self):
        """Test buffer processing can be cancelled."""
        buffer = MessageBuffer(delay=5.0)  # Long delay

        mock_message = Mock()
        callback = AsyncMock()

        await buffer.add_message(111222333, mock_message, callback)

        # Cancel task
        task = buffer.tasks.get(111222333)
        if task:
            task.cancel()

        # Should not raise error
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_process_buffer_error_handling(self):
        """Test buffer processing error handling."""
        buffer = MessageBuffer(delay=0.1)

        mock_message = Mock()

        async def failing_callback(messages):
            raise Exception("Test error")

        await buffer.add_message(111222333, mock_message, failing_callback)

        # Wait for buffer to process
        await asyncio.sleep(0.3)

        # Buffer should be cleaned up despite error
        assert 111222333 not in buffer.buffers


class TestActiveAPICall:
    """Test ActiveAPICall class."""

    def test_active_api_call_initialization(self):
        """Test ActiveAPICall initialization."""
        mock_task = Mock()
        cancel_event = asyncio.Event()

        api_call = ActiveAPICall(task=mock_task, cancel_event=cancel_event, messages=[])

        assert api_call.task is mock_task
        assert api_call.cancel_event is cancel_event
        assert api_call.messages == []

    def test_active_api_call_cancel(self):
        """Test ActiveAPICall cancel method."""
        mock_task = Mock()
        mock_task.done = Mock(return_value=False)
        cancel_event = asyncio.Event()

        api_call = ActiveAPICall(task=mock_task, cancel_event=cancel_event, messages=[])

        api_call.cancel()

        assert cancel_event.is_set()
        mock_task.cancel.assert_called_once()

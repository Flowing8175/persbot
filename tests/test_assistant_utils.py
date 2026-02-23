"""Feature tests for assistant cog utilities.

Tests focus on behavior:
- should_ignore_message: Message filtering logic
- handle_error: Error handling
- cancel_channel_tasks: Task cancellation
- send_abort_success/no_tasks: Abort command responses
"""

import asyncio
import sys
from unittest.mock import Mock, MagicMock, AsyncMock

import discord
import pytest

# Mock ddgs module before any imports that depend on it
_mock_ddgs = MagicMock()
_mock_ddgs.DDGS = MagicMock
_mock_ddgs.exceptions = MagicMock()
_mock_ddgs.exceptions.RatelimitException = Exception
_mock_ddgs.exceptions.DDGSException = Exception
sys.modules['ddgs'] = _mock_ddgs
_sys_modules = {'ddgs': _mock_ddgs, 'ddgs.exceptions': _mock_ddgs.exceptions}

# Mock bs4 module before any imports that depend on it
_mock_bs4 = MagicMock()
_sys_modules['bs4'] = _mock_bs4
for k, v in _sys_modules.items():
    sys.modules[k] = v


class TestShouldIgnoreMessage:
    """Tests for should_ignore_message utility function."""

    def test_function_exists(self):
        """should_ignore_message function exists."""
        from persbot.bot.cogs.assistant.utils import should_ignore_message
        assert should_ignore_message is not None

    def test_ignores_bot_messages(self):
        """Bot messages are ignored."""
        from persbot.bot.cogs.assistant.utils import should_ignore_message

        message = Mock()
        message.author.bot = True
        message.mention_everyone = False

        bot_user = Mock()
        bot_user.mentioned_in = Mock(return_value=True)

        config = Mock()
        config.auto_reply_channel_ids = []

        result = should_ignore_message(message, bot_user, config)
        assert result is True

    def test_ignores_auto_reply_channels(self):
        """Messages in auto-reply channels are ignored."""
        from persbot.bot.cogs.assistant.utils import should_ignore_message

        message = Mock()
        message.author.bot = False
        message.channel.id = 123
        message.mention_everyone = False

        bot_user = Mock()
        bot_user.mentioned_in = Mock(return_value=True)

        config = Mock()
        config.auto_reply_channel_ids = [123]

        result = should_ignore_message(message, bot_user, config)
        assert result is True

    def test_ignores_everyone_mentions(self):
        """@everyone/@here mentions are ignored."""
        from persbot.bot.cogs.assistant.utils import should_ignore_message

        message = Mock()
        message.author.bot = False
        message.channel.id = 456
        message.mention_everyone = True

        bot_user = Mock()
        bot_user.mentioned_in = Mock(return_value=True)

        config = Mock()
        config.auto_reply_channel_ids = []

        result = should_ignore_message(message, bot_user, config)
        assert result is True

    def test_ignores_messages_without_bot_mention(self):
        """Messages without bot mention are ignored."""
        from persbot.bot.cogs.assistant.utils import should_ignore_message

        message = Mock()
        message.author.bot = False
        message.channel.id = 456
        message.mention_everyone = False

        bot_user = Mock()
        bot_user.mentioned_in = Mock(return_value=False)

        config = Mock()
        config.auto_reply_channel_ids = []

        result = should_ignore_message(message, bot_user, config)
        assert result is True

    def test_processes_valid_messages(self):
        """Valid messages with bot mention are processed."""
        from persbot.bot.cogs.assistant.utils import should_ignore_message

        message = Mock()
        message.author.bot = False
        message.channel.id = 456
        message.mention_everyone = False

        bot_user = Mock()
        bot_user.mentioned_in = Mock(return_value=True)

        config = Mock()
        config.auto_reply_channel_ids = []

        result = should_ignore_message(message, bot_user, config)
        assert result is False

    def test_handles_none_bot_user(self):
        """Handles None bot_user gracefully."""
        from persbot.bot.cogs.assistant.utils import should_ignore_message

        message = Mock()
        message.author.bot = False
        message.channel.id = 456
        message.mention_everyone = False

        config = Mock()
        config.auto_reply_channel_ids = []

        result = should_ignore_message(message, None, config)
        assert result is True


class TestPrepareBatchContext:
    """Tests for prepare_batch_context utility function."""

    def test_function_exists(self):
        """prepare_batch_context function exists."""
        from persbot.bot.cogs.assistant.utils import prepare_batch_context
        assert prepare_batch_context is not None


class TestSendResponse:
    """Tests for send_response utility function."""

    def test_function_exists(self):
        """send_response function exists."""
        from persbot.bot.cogs.assistant.utils import send_response
        assert send_response is not None


class TestHandleError:
    """Tests for handle_error utility function."""

    def test_function_exists(self):
        """handle_error function exists."""
        from persbot.bot.cogs.assistant.utils import handle_error
        assert handle_error is not None

    @pytest.mark.asyncio
    async def test_sends_generic_error_message(self):
        """Sends generic error message to channel."""
        from persbot.bot.cogs.assistant.utils import handle_error
        from persbot.utils import GENERIC_ERROR_MESSAGE

        message = Mock()
        message.reply = AsyncMock()

        error = Exception("Test error")

        await handle_error(message, error)
        message.reply.assert_awaited_once_with(GENERIC_ERROR_MESSAGE, mention_author=False)


class TestProcessRemovedMessages:
    """Tests for process_removed_messages utility function."""

    def test_function_exists(self):
        """process_removed_messages function exists."""
        from persbot.bot.cogs.assistant.utils import process_removed_messages
        assert process_removed_messages is not None


class TestDeleteAssistantMessages:
    """Tests for delete_assistant_messages utility function."""

    @pytest.mark.asyncio
    async def test_deletes_messages_by_ids(self):
        """Deletes messages from Discord."""
        from persbot.bot.cogs.assistant.utils import delete_assistant_messages

        channel = Mock()
        old_msg = Mock()
        channel.fetch_message = AsyncMock(return_value=old_msg)
        old_msg.delete = AsyncMock()

        msg = Mock()
        msg.message_ids = ["123", "456"]

        await delete_assistant_messages(channel, msg)
        assert channel.fetch_message.call_count == 2
        assert old_msg.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_missing_message_ids(self):
        """Handles messages without message_ids attribute."""
        from persbot.bot.cogs.assistant.utils import delete_assistant_messages

        channel = Mock()
        msg = Mock(spec=[])  # Empty spec - no message_ids attribute

        # Should not raise
        await delete_assistant_messages(channel, msg)
        channel.fetch_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_not_found_error(self):
        """Handles NotFound errors during message deletion gracefully."""
        from persbot.bot.cogs.assistant.utils import delete_assistant_messages

        channel = Mock()
        # Create a proper mock NotFound exception
        mock_response = Mock()
        mock_response.status = 404
        not_found = discord.NotFound(mock_response, "Not found")
        channel.fetch_message = AsyncMock(side_effect=not_found)

        msg = Mock()
        msg.message_ids = ["123"]

        # Should not raise
        await delete_assistant_messages(channel, msg)

    @pytest.mark.asyncio
    async def test_handles_forbidden_error(self):
        """Handles Forbidden errors during message deletion gracefully."""
        from persbot.bot.cogs.assistant.utils import delete_assistant_messages

        channel = Mock()
        # Create a proper mock Forbidden exception
        mock_response = Mock()
        mock_response.status = 403
        forbidden = discord.Forbidden(mock_response, "Forbidden")
        channel.fetch_message = AsyncMock(side_effect=forbidden)

        msg = Mock()
        msg.message_ids = ["123"]

        # Should not raise
        await delete_assistant_messages(channel, msg)


class TestCancelChannelTasks:
    """Tests for cancel_channel_tasks utility function."""

    def test_function_exists(self):
        """cancel_channel_tasks function exists."""
        from persbot.bot.cogs.assistant.utils import cancel_channel_tasks
        assert cancel_channel_tasks is not None

    def test_sets_cancellation_signal(self):
        """Sets cancellation signal event."""
        from persbot.bot.cogs.assistant.utils import cancel_channel_tasks

        event = asyncio.Event()
        processing_tasks = {}
        sending_tasks = {}
        cancellation_signals = {123: event}

        cancel_channel_tasks(123, processing_tasks, sending_tasks, cancellation_signals=cancellation_signals)
        assert event.is_set()

    def test_returns_false_when_no_tasks(self):
        """Returns False when no tasks to cancel."""
        from persbot.bot.cogs.assistant.utils import cancel_channel_tasks

        processing_tasks = {}
        sending_tasks = {}
        cancellation_signals = {}

        result = cancel_channel_tasks(123, processing_tasks, sending_tasks, cancellation_signals=cancellation_signals)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancels_processing_task(self):
        """Cancels active processing task."""
        from persbot.bot.cogs.assistant.utils import cancel_channel_tasks

        async def long_sleep():
            await asyncio.sleep(100)

        task = asyncio.create_task(long_sleep())
        processing_tasks = {123: task}
        sending_tasks = {}
        cancellation_signals = {}

        result = cancel_channel_tasks(123, processing_tasks, sending_tasks, cancellation_signals=cancellation_signals)
        assert result is True
        # Give the task a moment to process cancellation
        await asyncio.sleep(0.01)
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_cancels_sending_task(self):
        """Cancels active sending task."""
        from persbot.bot.cogs.assistant.utils import cancel_channel_tasks

        async def long_sleep():
            await asyncio.sleep(100)

        task = asyncio.create_task(long_sleep())
        processing_tasks = {}
        sending_tasks = {123: task}
        cancellation_signals = {}

        result = cancel_channel_tasks(123, processing_tasks, sending_tasks, cancellation_signals=cancellation_signals)
        assert result is True
        # Give the task a moment to process cancellation
        await asyncio.sleep(0.01)
        assert task.cancelled() or task.done()


class TestSendAbortSuccess:
    """Tests for send_abort_success utility function."""

    def test_function_exists(self):
        """send_abort_success function exists."""
        from persbot.bot.cogs.assistant.utils import send_abort_success
        assert send_abort_success is not None

    @pytest.mark.asyncio
    async def test_sends_reply_for_interaction(self):
        """Sends reply for interaction context."""
        from persbot.bot.cogs.assistant.utils import send_abort_success

        ctx = Mock()
        ctx.interaction = Mock()
        ctx.reply = AsyncMock()

        await send_abort_success(ctx)
        ctx.reply.assert_awaited_once_with("🛑 중단되었습니다.", ephemeral=False)

    @pytest.mark.asyncio
    async def test_adds_reaction_for_non_interaction(self):
        """Adds reaction for non-interaction context."""
        from persbot.bot.cogs.assistant.utils import send_abort_success

        ctx = Mock()
        ctx.interaction = None
        ctx.message.add_reaction = AsyncMock()

        await send_abort_success(ctx)
        ctx.message.add_reaction.assert_awaited_once_with("🛑")


class TestSendAbortNoTasks:
    """Tests for send_abort_no_tasks utility function."""

    def test_function_exists(self):
        """send_abort_no_tasks function exists."""
        from persbot.bot.cogs.assistant.utils import send_abort_no_tasks
        assert send_abort_no_tasks is not None

    @pytest.mark.asyncio
    async def test_sends_ephemeral_reply_for_interaction(self):
        """Sends ephemeral reply for interaction context."""
        from persbot.bot.cogs.assistant.utils import send_abort_no_tasks

        ctx = Mock()
        ctx.interaction = Mock()
        ctx.reply = AsyncMock()

        await send_abort_no_tasks(ctx)
        ctx.reply.assert_awaited_once_with("❓ 중단할 작업이 없습니다.", ephemeral=True)

    @pytest.mark.asyncio
    async def test_adds_reaction_for_non_interaction(self):
        """Adds reaction for non-interaction context."""
        from persbot.bot.cogs.assistant.utils import send_abort_no_tasks

        ctx = Mock()
        ctx.interaction = None
        ctx.message.add_reaction = AsyncMock()

        await send_abort_no_tasks(ctx)
        ctx.message.add_reaction.assert_awaited_once_with("❓")

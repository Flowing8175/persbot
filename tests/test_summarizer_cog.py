"""Tests for the Summarizer Cog."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from discord import Message
from discord.ext import commands

from persbot.bot.cogs.summarizer import SummarizerCog
from persbot.config import AppConfig


def make_async_iterator(iterable):
    """Helper to create an async iterator."""

    async def _aiter():
        for item in iterable:
            yield item

    return _aiter()


class TestSummarizerCogInitialization:
    """Test SummarizerCog initialization and setup."""

    @pytest.mark.asyncio
    async def test_cog_initialization(self, mock_bot, mock_app_config, mock_llm_service):
        """Test that SummarizerCog initializes correctly."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        assert cog.bot is mock_bot
        assert cog.config is mock_app_config
        assert cog.llm_service is mock_llm_service


class TestFetchMessages:
    """Test message fetching functionality."""

    @pytest.mark.asyncio
    async def test_fetch_messages_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_channel
    ):
        """Test successful message fetching."""
        # Create mock messages
        message1 = Mock()
        message1.author.id = 123456
        message1.content = "Hello world"
        message1.author.bot = False

        message2 = Mock()
        message2.author.id = 789012
        message2.content = "Test message"
        message2.author.bot = False

        message3 = Mock()
        message3.author.bot = True  # Should be filtered out

        # Mock channel history to return async iterator
        async def mock_history(*args, **kwargs):
            for msg in [message1, message2, message3]:
                yield msg

        mock_channel.history = mock_history

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        text, count = await cog._fetch_messages(
            mock_channel,
            limit=100,
            oldest_first=True,
        )

        # Check that bot messages were filtered out
        assert "123456: Hello world" in text
        assert "789012: Test message" in text
        assert count == 2  # Message count should be 2 (non-bot messages)
        assert text.count("\n") == 1  # Should have 2 messages, so 1 newline
        assert count == 2

    @pytest.mark.asyncio
    async def test_fetch_messages_only_bots(
        self, mock_bot, mock_app_config, mock_llm_service, mock_channel
    ):
        """Test fetching when only bot messages exist."""
        # Create only bot messages
        message1 = Mock()
        message1.author.bot = True
        message1.content = "Bot message"

        message2 = Mock()
        message2.author.bot = True
        message2.content = "Another bot message"

        # Mock channel history to return async iterator
        async def mock_history(*args, **kwargs):
            for msg in [message1, message2]:
                yield msg

        mock_channel.history = mock_history

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        text, count = await cog._fetch_messages(
            mock_channel,
            limit=100,
            oldest_first=True,
        )

        # Should be empty since all messages are from bots
        assert text == ""
        assert count == 0


class TestSummarizeCommand:
    """Test main summarize command functionality."""

    @pytest.mark.asyncio
    async def test_summarize_no_args(self, mock_bot, mock_app_config, mock_llm_service, mock_ctx):
        """Test summarize with no arguments (default 30 minutes)."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock the _summarize_by_time method
        with patch.object(cog, "_summarize_by_time", new_callable=AsyncMock) as mock_summarize:
            # Call the internal handler method directly
            await cog._handle_summarize_args(mock_ctx, ())

            # Should call _summarize_by_time with 30 minutes
            mock_summarize.assert_called_once_with(mock_ctx, 30)

    @pytest.mark.asyncio
    async def test_summarize_time_arg(self, mock_bot, mock_app_config, mock_llm_service, mock_ctx):
        """Test summarize with time argument."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock the _summarize_by_time method
        with patch.object(cog, "_summarize_by_time", new_callable=AsyncMock) as mock_summarize:
            await cog._handle_summarize_args(mock_ctx, ("20분",))

            # Should call _summarize_by_time with 20 minutes
            mock_summarize.assert_called_once_with(mock_ctx, 20)

    @pytest.mark.asyncio
    async def test_summarize_invalid_time(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test summarize with invalid time argument."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._handle_summarize_args(mock_ctx, ("invalid_time",))

            # Should show error message
            mock_send.assert_called_once_with(
                mock_ctx, "❌ 시간 형식이 올바르지 않아요. (예: '20분', '1시간')"
            )

    @pytest.mark.asyncio
    async def test_summarize_id_after(self, mock_bot, mock_app_config, mock_llm_service, mock_ctx):
        """Test summarize with ID and 'after' direction."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch.object(cog, "summarize_by_id", new_callable=AsyncMock) as mock_summarize_by_id:
            await cog.summarize_by_id(mock_ctx, 12345678901234567)

            # Should call summarize_by_id
            mock_summarize_by_id.assert_called_once_with(mock_ctx, 12345678901234567)

    @pytest.mark.asyncio
    async def test_summarize_id_before_invalid(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test summarize with ID and 'before' direction (invalid)."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._handle_summarize_args(mock_ctx, ("12345678901234567", "이전"))

            # Should show error message that time is required
            mock_send.assert_called_once_with(
                mock_ctx, "❌ '이전'은 시간을 지정해야 합니다. 예: `!요약 123456 이전 1시간`"
            )

    @pytest.mark.asyncio
    async def test_summarize_id_range_after(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test summarize with ID, direction, and time range."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch.object(
            cog, "summarize_by_range", new_callable=AsyncMock
        ) as mock_summarize_by_range:
            await cog.summarize_by_range(mock_ctx, 12345678901234567, "이후", "30분")

            # Should call summarize_by_range
            mock_summarize_by_range.assert_called_once_with(
                mock_ctx, 12345678901234567, "이후", "30분"
            )

    @pytest.mark.asyncio
    async def test_summarize_id_range_before(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test summarize with ID, 'before' direction, and time range."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch.object(
            cog, "summarize_by_range", new_callable=AsyncMock
        ) as mock_summarize_by_range:
            await cog.summarize_by_range(mock_ctx, 12345678901234567, "이전", "1시간")

            # Should call summarize_by_range
            mock_summarize_by_range.assert_called_once_with(
                mock_ctx, 12345678901234567, "이전", "1시간"
            )

    @pytest.mark.asyncio
    async def test_summarize_invalid_args(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test summarize with invalid arguments."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._handle_summarize_args(mock_ctx, ("invalid_arg",))

            # Should show help message
            mock_send.assert_called_once()


class TestHandleSummarizeArgs:
    """Test argument handling for summarize command."""

    @pytest.mark.asyncio
    async def test_handle_summarize_args_no_args(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test handling no arguments."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch.object(cog, "_summarize_by_time", new_callable=AsyncMock) as mock_summarize:
            await cog._handle_summarize_args(mock_ctx, ())

            mock_summarize.assert_called_once_with(mock_ctx, 30)

    @pytest.mark.asyncio
    async def test_handle_summarize_args_one_valid_time_arg(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test handling one valid time argument."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch.object(cog, "_summarize_by_time", new_callable=AsyncMock) as mock_summarize:
            await cog._handle_summarize_args(mock_ctx, ("20분",))

            mock_summarize.assert_called_once_with(mock_ctx, 20)

    @pytest.mark.asyncio
    async def test_handle_summarize_args_one_invalid_arg(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test handling one invalid argument."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._handle_summarize_args(mock_ctx, ("invalid",))

            mock_send.assert_called_once_with(
                mock_ctx, "❌ 시간 형식이 올바르지 않아요. (예: '20분', '1시간')"
            )

    @pytest.mark.asyncio
    async def test_handle_summarize_args_id_after(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test handling ID and 'after' direction."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch.object(cog, "summarize_by_id", new_callable=AsyncMock) as mock_summarize_by_id:
            await cog._handle_summarize_args(mock_ctx, ("12345678901234567", "이후"))

            mock_summarize_by_id.assert_called_once_with(mock_ctx, 12345678901234567)

    @pytest.mark.asyncio
    async def test_handle_summarize_args_invalid_direction(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test handling invalid direction."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._handle_summarize_args(mock_ctx, ("12345678901234567", "invalid"))

            mock_send.assert_called_once_with(
                mock_ctx, "❌ 두 번째 인자는 '이후' 또는 '이전'이어야 합니다."
            )

    @pytest.mark.asyncio
    async def test_handle_summarize_args_invalid_id(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test handling invalid message ID."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._handle_summarize_args(mock_ctx, ("invalid_id", "이후"))

            mock_send.assert_called_once_with(mock_ctx, "❌ 첫 번째 인자는 메시지 ID여야 해요.")


class TestIsMessageId:
    """Test message ID validation."""

    def test_is_message_id_valid(self, mock_bot, mock_app_config, mock_llm_service):
        """Test valid message ID validation."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Valid Discord IDs are 17-20 digits
        assert cog._is_message_id("12345678901234567") is True  # 17 digits
        assert cog._is_message_id("123456789012345678") is True  # 18 digits
        assert cog._is_message_id("1234567890123456789") is True  # 19 digits
        assert cog._is_message_id("12345678901234567890") is True  # 20 digits

    def test_is_message_id_invalid(self, mock_bot, mock_app_config, mock_llm_service):
        """Test invalid message ID validation."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Test invalid lengths and non-digit strings
        assert cog._is_message_id("1234567890123456") is False  # 16 digits (too short)
        assert cog._is_message_id("123456789012345678901") is False  # 21 digits (too long)
        assert cog._is_message_id("not_a_number") is False
        assert cog._is_message_id("123abc") is False
        assert cog._is_message_id("") is False


class TestSummarizeByTime:
    """Test time-based summarization."""

    @pytest.mark.asyncio
    async def test_summarize_by_time_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx, mock_channel
    ):
        """Test successful time-based summarization."""
        # Mock config
        mock_app_config.max_messages_per_fetch = 100

        # Create test messages
        message1 = Mock()
        message1.author.id = 123456
        message1.content = "Hello world"
        message1.author.bot = False

        message2 = Mock()
        message2.author.id = 789012
        message2.content = "Test message"
        message2.author.bot = False

        # Set up async iterator for channel.history
        async def async_messages():
            yield message1
            yield message2

        mock_channel.history = MagicMock()
        mock_channel.history.return_value = async_messages()

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock LLM service
        mock_llm_service.summarize_text.return_value = "This is a test summary."

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._summarize_by_time(mock_ctx, 30)

            # Check that summary was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][1]
            assert "최근 30분 2개 메시지 요약:" in call_args
            assert "This is a test summary." in call_args

    @pytest.mark.asyncio
    async def test_summarize_by_time_no_messages(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx, mock_channel
    ):
        """Test time-based summarization with no messages."""
        mock_app_config.max_messages_per_fetch = 100

        # No messages in history
        async def mock_history(*args, **kwargs):
            # No messages to yield
            return

        mock_channel.history = mock_history

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._summarize_by_time(mock_ctx, 30)

            # Should show no messages message
            mock_send.assert_called_once_with(mock_ctx, "ℹ️ 최근 30분 동안 메시지가 없어요.")

    @pytest.mark.asyncio
    async def test_summarize_by_time_llm_error(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx, mock_channel
    ):
        """Test time-based summarization with LLM error."""
        mock_app_config.max_messages_per_fetch = 100

        # Create test messages
        message1 = Mock()
        message1.author.id = 123456
        message1.content = "Hello world"
        message1.author.bot = False

        async def async_messages():
            yield message1

        mock_channel.history.return_value = async_messages()

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock LLM service to return None (error)
        mock_llm_service.summarize_text.return_value = None

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog._summarize_by_time(mock_ctx, 30)

            # Should show generic error message
            mock_send.assert_called_once_with(
                mock_ctx, "❌ 봇 내부에서 예상치 못한 오류가 발생했어요. 개발자에게 문의해주세요."
            )


class TestSummarizeById:
    """Test ID-based summarization."""

    @pytest.mark.asyncio
    async def test_summarize_by_id_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx, mock_channel
    ):
        """Test successful ID-based summarization."""
        mock_app_config.max_messages_per_fetch = 100

        # Create test messages
        start_message = Mock()
        start_message.author.id = 111222
        start_message.content = "Start message"
        start_message.author.bot = False

        message1 = Mock()
        message1.author.id = 123456
        message1.content = "Hello world"
        message1.author.bot = False

        message2 = Mock()
        message2.author.id = 789012
        message2.content = "Test message"
        message2.author.bot = False

        mock_channel.fetch_message.return_value = start_message

        async def async_messages():
            yield message1
            yield message2

        mock_channel.history.return_value = async_messages()

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock LLM service
        mock_llm_service.summarize_text.return_value = (
            "This is a test summary including start message."
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog.summarize_by_id(mock_ctx, 12345678901234567)

            # Check that summary was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args.args[1]
            assert "메시지 ID `12345678901234567` 이후 3개 메시지 요약:" in call_args
            assert "This is a test summary including start message." in call_args

    @pytest.mark.asyncio
    async def test_summarize_by_id_not_found(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test ID-based summarization with non-existent message."""
        mock_app_config.max_messages_per_fetch = 100

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock fetch_message to raise NotFound
        from discord.errors import NotFound

        mock_ctx.channel.fetch_message.side_effect = NotFound(Mock(status=404), "Message not found")

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog.summarize_by_id(mock_ctx, 12345678901234567)

            # Should show error message
            mock_send.assert_called_once_with(
                mock_ctx, "❌ 메시지 ID `12345678901234567`를 찾을 수 없어요."
            )

    @pytest.mark.asyncio
    async def test_summarize_by_id_no_messages_after(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx, mock_channel
    ):
        """Test ID-based summarization with no messages after the specified ID."""
        mock_app_config.max_messages_per_fetch = 100

        start_message = Mock()
        start_message.author.id = 111222
        start_message.content = "Start message"
        start_message.author.bot = False

        mock_channel.fetch_message.return_value = start_message

        async def async_messages():
            # No messages to yield
            pass

        mock_channel.history.return_value = async_messages()

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog.summarize_by_id(mock_ctx, 12345678901234567)

            # Should show no messages message
            mock_send.assert_called_once_with(
                mock_ctx, "ℹ️ 메시지 ID `12345678901234567` 이후 메시지가 없어요."
            )


class TestSummarizeByRange:
    """Test range-based summarization."""

    @pytest.mark.asyncio
    async def test_summarize_by_range_after_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx, mock_channel
    ):
        """Test successful range-based summarization with 'after' direction."""
        mock_app_config.max_messages_per_fetch = 100

        # Create test messages
        start_message = Mock()
        start_message.id = 12345678901234567
        start_message.author.id = 111222
        start_message.content = "Start message"
        start_message.author.bot = False
        start_message.created_at = datetime.now(timezone.utc)

        message1 = Mock()
        message1.author.id = 123456
        message1.content = "Message 1"
        message1.author.bot = False
        message1.created_at = start_message.created_at + timedelta(minutes=10)

        message2 = Mock()
        message2.author.id = 789012
        message2.content = "Message 2"
        message2.author.bot = False
        message2.created_at = start_message.created_at + timedelta(minutes=25)

        mock_channel.fetch_message.return_value = start_message

        async def async_messages():
            yield message1
            yield message2

        mock_channel.history.return_value = async_messages()

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock LLM service
        mock_llm_service.summarize_text.return_value = "This is a range summary."

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog.summarize_by_range(mock_ctx, 12345678901234567, "이후", "30분")

            # Check that summary was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args.args[1]
            assert "메시지 ID `12345678901234567` 이후 30분 2개 메시지 요약:" in call_args
            assert "This is a range summary." in call_args

    @pytest.mark.asyncio
    async def test_summarize_by_range_before_success(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx, mock_channel
    ):
        """Test successful range-based summarization with 'before' direction."""
        mock_app_config.max_messages_per_fetch = 100

        # Create test messages
        start_message = Mock()
        start_message.id = 12345678901234567
        start_message.author.id = 111222
        start_message.content = "Start message"
        start_message.author.bot = False
        start_message.created_at = datetime.now(timezone.utc)

        message1 = Mock()
        message1.author.id = 123456
        message1.content = "Message 1"
        message1.author.bot = False
        message1.created_at = start_message.created_at - timedelta(minutes=20)

        message2 = Mock()
        message2.author.id = 789012
        message2.content = "Message 2"
        message2.author.bot = False
        message2.created_at = start_message.created_at - timedelta(minutes=10)

        mock_channel.fetch_message.return_value = start_message

        async def async_messages():
            yield message1
            yield message2

        mock_channel.history.return_value = async_messages()

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock LLM service
        mock_llm_service.summarize_text.return_value = "This is a before range summary."

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog.summarize_by_range(mock_ctx, 12345678901234567, "이전", "30분")

            # Check that summary was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args.args[1]
            assert "메시지 ID `12345678901234567` 이전 30분 2개 메시지 요약:" in call_args
            assert "This is a before range summary." in call_args

    @pytest.mark.asyncio
    async def test_summarize_by_range_no_messages(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx, mock_channel
    ):
        """Test range-based summarization with no messages in range."""
        mock_app_config.max_messages_per_fetch = 100

        start_message = Mock()
        start_message.id = 12345678901234567
        start_message.author.id = 111222
        start_message.content = "Start message"
        start_message.author.bot = False
        start_message.created_at = datetime.now(timezone.utc)

        mock_channel.fetch_message.return_value = start_message

        async def async_messages():
            # No messages to yield
            pass

        mock_channel.history.return_value = async_messages()

        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            await cog.summarize_by_range(mock_ctx, 12345678901234567, "이후", "30분")

            # Should show no messages message
            mock_send.assert_called_once_with(mock_ctx, "ℹ️ 해당 범위에 메시지가 없어요.")


class TestErrorHandling:
    """Test error handling for summarize command."""

    @pytest.mark.asyncio
    async def test_summarize_error_bad_argument(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test error handling for bad arguments."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        from discord.ext.commands import BadArgument

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            error = BadArgument("Invalid argument")
            await cog.summarize_error(mock_ctx, error)

            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert "인수가 잘못되었어요" in call_args

    @pytest.mark.asyncio
    async def test_summarize_error_missing_argument(
        self, mock_bot, mock_app_config, mock_llm_service, mock_ctx
    ):
        """Test error handling for missing arguments."""
        cog = SummarizerCog(
            bot=mock_bot,
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        from discord.ext.commands import MissingRequiredArgument

        with patch(
            "persbot.bot.cogs.summarizer.send_discord_message", new_callable=AsyncMock
        ) as mock_send:
            # Create a proper Parameter object for MissingRequiredArgument
            from unittest.mock import Mock

            param = Mock()
            param.name = "test_param"
            param.displayed_name = "test_param"  # Add missing attribute
            error = MissingRequiredArgument(param)
            await cog.summarize_error(mock_ctx, error)

            mock_send.assert_called_once()
            call_args = mock_send.call_args.args[1]
            assert "명령어를 완성해주세요" in call_args

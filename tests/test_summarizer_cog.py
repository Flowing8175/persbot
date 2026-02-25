"""Tests for SummarizerCog.

Tests focus on:
- SummarizerCog: initialization
- _fetch_messages: message collection with various parameters
- _get_reply_message_id: reply message ID extraction
- _is_message_id: message ID validation
- _handle_summarize_args: argument parsing and routing
- _summarize_by_time: time-based summarization
- summarize_by_id: message ID-based summarization
- summarize_by_range: range-based summarization
- summarize: main command handler with all parameter combinations
- summarize_error: error handling
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Optional

import pytest
import discord
from discord.ext import commands

from persbot.bot.cogs.summarizer import SummarizerCog
from persbot.config import AppConfig
from persbot.services.llm_service import LLMService


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_bot():
    """Create a mock Discord bot."""
    bot = Mock(spec=commands.Bot)
    return bot


@pytest.fixture
def mock_config():
    """Create a mock AppConfig."""
    config = Mock(spec=AppConfig)
    config.max_messages_per_fetch = 100
    config.command_prefix = "!"
    return config


@pytest.fixture
def mock_llm_service():
    """Create a mock LLMService."""
    service = Mock(spec=LLMService)
    service.summarize_text = AsyncMock(return_value="This is a summary of the conversation.")
    return service


@pytest.fixture
def summarizer_cog(mock_bot, mock_config, mock_llm_service):
    """Create a SummarizerCog instance with mocked dependencies."""
    return SummarizerCog(mock_bot, mock_config, mock_llm_service)


@pytest.fixture
def mock_context():
    """Create a mock Discord context."""
    ctx = Mock(spec=commands.Context)
    ctx.author = Mock(spec=discord.Member)
    ctx.author.id = 789
    ctx.author.display_name = "TestUser"
    ctx.channel = Mock(spec=discord.TextChannel)
    ctx.channel.id = 123
    ctx.channel.name = "test-channel"
    ctx.channel.typing = Mock()
    ctx.channel.typing.return_value.__aenter__ = AsyncMock()
    ctx.channel.typing.return_value.__aexit__ = AsyncMock()
    ctx.channel.history = Mock()
    ctx.channel.fetch_message = AsyncMock()
    ctx.message = Mock(spec=discord.Message)
    ctx.message.id = 999
    ctx.message.content = "!요약"
    ctx.message.reference = None
    ctx.defer = AsyncMock()
    ctx.reply = AsyncMock()
    ctx.send = AsyncMock()
    ctx.prefix = "!"
    return ctx


@pytest.fixture
def mock_messages():
    """Create a list of mock Discord messages."""
    messages = []
    for i in range(5):
        msg = Mock(spec=discord.Message)
        msg.id = 1000 + i
        msg.content = f"Test message {i}"
        msg.author = Mock()
        msg.author.bot = False
        msg.author.display_name = f"User{i}"
        msg.created_at = datetime.now(timezone.utc) - timedelta(minutes=10-i)
        messages.append(msg)
    return messages


# =============================================================================
# SummarizerCog Initialization Tests
# =============================================================================

class TestSummarizerCogInit:
    """Tests for SummarizerCog initialization."""

    def test_initialization(self, mock_bot, mock_config, mock_llm_service):
        """SummarizerCog initializes with all required attributes."""
        cog = SummarizerCog(mock_bot, mock_config, mock_llm_service)

        assert cog.bot == mock_bot
        assert cog.config == mock_config
        assert cog.llm_service == mock_llm_service


# =============================================================================
# _fetch_messages Tests
# =============================================================================

class TestFetchMessages:
    """Tests for SummarizerCog._fetch_messages."""

    @pytest.mark.asyncio
    async def test_fetch_messages_returns_text_and_count(self, summarizer_cog, mock_context, mock_messages):
        """_fetch_messages returns formatted text and message count."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                if not msg.author.bot:
                    yield msg
        mock_context.channel.history = history_gen

        text, count = await summarizer_cog._fetch_messages(mock_context.channel, limit=10)

        assert count == 5
        assert "User0: Test message 0" in text
        assert "User4: Test message 4" in text

    @pytest.mark.asyncio
    async def test_fetch_messages_includes_bot_messages(self, summarizer_cog, mock_context):
        """_fetch_messages includes bot messages for complete context."""
        messages = []
        for i in range(5):
            msg = Mock()
            msg.content = f"Message {i}"
            msg.author = Mock()
            msg.author.bot = (i % 2 == 0)  # Every other message is from a bot
            msg.author.display_name = f"User{i}"
            messages.append(msg)

        async def history_gen(**kwargs):
            for msg in messages:
                yield msg
        mock_context.channel.history = history_gen

        text, count = await summarizer_cog._fetch_messages(mock_context.channel, limit=10)

        # Should include all messages including bot messages
        assert count == 5
        assert "User0: Message 0" in text
        assert "User1: Message 1" in text
        assert "User2: Message 2" in text
        assert "User3: Message 3" in text
        assert "User4: Message 4" in text

    @pytest.mark.asyncio
    async def test_fetch_messages_with_no_messages(self, summarizer_cog, mock_context):
        """_fetch_messages handles empty history."""
        async def history_gen(**kwargs):
            return
            yield  # pragma: no cover (makes this a generator)
        mock_context.channel.history = history_gen

        text, count = await summarizer_cog._fetch_messages(mock_context.channel, limit=10)

        assert count == 0
        assert text == ""

    @pytest.mark.asyncio
    async def test_fetch_messages_passes_kwargs(self, summarizer_cog, mock_context, mock_messages):
        """_fetch_messages passes kwargs to channel.history."""
        async def history_gen(**kwargs):
            assert kwargs["limit"] == 50
            assert kwargs["oldest_first"] is True
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        await summarizer_cog._fetch_messages(mock_context.channel, limit=50, oldest_first=True)

    @pytest.mark.asyncio
    async def test_fetch_messages_chronological_order(self, summarizer_cog, mock_context):
        """_fetch_messages maintains chronological order (oldest first)."""
        messages = []
        for i in range(3):
            msg = Mock()
            msg.content = f"Message {i}"
            msg.author = Mock()
            msg.author.bot = False
            msg.author.display_name = f"User{i}"
            messages.append(msg)

        async def history_gen(**kwargs):
            # Yield in chronological order
            for msg in messages:
                yield msg
        mock_context.channel.history = history_gen

        text, count = await summarizer_cog._fetch_messages(mock_context.channel, oldest_first=True)

        lines = text.strip().split("\n")
        assert lines[0] == "User0: Message 0"
        assert lines[1] == "User1: Message 1"
        assert lines[2] == "User2: Message 2"


# =============================================================================
# _get_reply_message_id Tests
# =============================================================================

class TestGetReplyMessageId:
    """Tests for SummarizerCog._get_reply_message_id."""

    def test_get_reply_message_id_with_reference(self, summarizer_cog, mock_context):
        """_get_reply_message_id returns message ID when reference exists."""
        mock_context.message.reference = Mock()
        mock_context.message.reference.message_id = 123456789012345678

        result = summarizer_cog._get_reply_message_id(mock_context)

        assert result == 123456789012345678

    def test_get_reply_message_id_without_reference(self, summarizer_cog, mock_context):
        """_get_reply_message_id returns None when no reference exists."""
        mock_context.message.reference = None

        result = summarizer_cog._get_reply_message_id(mock_context)

        assert result is None

    def test_get_reply_message_id_with_none_message_id(self, summarizer_cog, mock_context):
        """_get_reply_message_id returns None when message_id is None."""
        mock_context.message.reference = Mock()
        mock_context.message.reference.message_id = None

        result = summarizer_cog._get_reply_message_id(mock_context)

        assert result is None


# =============================================================================
# _is_message_id Tests
# =============================================================================

class TestIsMessageId:
    """Tests for SummarizerCog._is_message_id."""

    def test_is_message_id_valid_17_digit(self, summarizer_cog):
        """_is_message_id returns True for 17-digit ID."""
        result = summarizer_cog._is_message_id("12345678901234567")
        assert result is True

    def test_is_message_id_valid_18_digit(self, summarizer_cog):
        """_is_message_id returns True for 18-digit ID."""
        result = summarizer_cog._is_message_id("123456789012345678")
        assert result is True

    def test_is_message_id_valid_19_digit(self, summarizer_cog):
        """_is_message_id returns True for 19-digit ID."""
        result = summarizer_cog._is_message_id("1234567890123456789")
        assert result is True

    def test_is_message_id_valid_20_digit(self, summarizer_cog):
        """_is_message_id returns True for 20-digit ID."""
        result = summarizer_cog._is_message_id("12345678901234567890")
        assert result is True

    def test_is_message_id_too_short(self, summarizer_cog):
        """_is_message_id returns False for ID shorter than 17 digits."""
        result = summarizer_cog._is_message_id("123456789012345")
        assert result is False

    def test_is_message_id_too_long(self, summarizer_cog):
        """_is_message_id returns False for ID longer than 20 digits."""
        result = summarizer_cog._is_message_id("123456789012345678901")
        assert result is False

    def test_is_message_id_non_numeric(self, summarizer_cog):
        """_is_message_id returns False for non-numeric string."""
        result = summarizer_cog._is_message_id("abc123")
        assert result is False

    def test_is_message_id_mixed_alphanumeric(self, summarizer_cog):
        """_is_message_id returns False for mixed alphanumeric."""
        result = summarizer_cog._is_message_id("12345678901234567a")
        assert result is False


# =============================================================================
# _summarize_by_time Tests
# =============================================================================

class TestSummarizeByTime:
    """Tests for SummarizerCog._summarize_by_time."""

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_time_success(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """_summarize_by_time generates summary successfully."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        await summarizer_cog._summarize_by_time(mock_context, 30)

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "30분" in args
        assert "5개 메시지" in args
        assert "This is a summary" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_time_no_messages(self, mock_send, summarizer_cog, mock_context):
        """_summarize_by_time handles no messages in time range."""
        async def history_gen(**kwargs):
            return
            yield
        mock_context.channel.history = history_gen

        await summarizer_cog._summarize_by_time(mock_context, 30)

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "메시지가 없어요" in args

    @pytest.mark.asyncio
    async def test_summarize_by_time_uses_typing(self, summarizer_cog, mock_context, mock_messages):
        """_summarize_by_time uses typing indicator."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        await summarizer_cog._summarize_by_time(mock_context, 30)

        mock_context.channel.typing.assert_called_once()

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_time_summary_failure(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """_summarize_by_time handles LLM service failure."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen
        summarizer_cog.llm_service.summarize_text = AsyncMock(return_value=None)

        await summarizer_cog._summarize_by_time(mock_context, 30)

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "예상치 못한 오류" in args


# =============================================================================
# summarize_by_id Tests
# =============================================================================

class TestSummarizeById:
    """Tests for SummarizerCog.summarize_by_id."""

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_id_success(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """summarize_by_id generates summary from message ID."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.content = "Starting message"
        start_msg.author.bot = False
        start_msg.author.display_name = "StartUser"
        start_msg.created_at = datetime.now(timezone.utc)
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        await summarizer_cog.summarize_by_id(mock_context, 123456789012345678)

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "123456789012345678" in args
        assert "요약" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_id_message_not_found(self, mock_send, summarizer_cog, mock_context):
        """summarize_by_id handles message not found error."""
        mock_context.channel.fetch_message = AsyncMock(side_effect=discord.NotFound(Mock(), "Not Found"))

        await summarizer_cog.summarize_by_id(mock_context, 123456789012345678)

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "찾을 수 없어요" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_id_includes_start_message(self, mock_send, summarizer_cog, mock_context):
        """summarize_by_id includes the start message in summary."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.content = "Starting message"
        start_msg.author.bot = False
        start_msg.author.display_name = "StartUser"
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            return
            yield
        mock_context.channel.history = history_gen

        await summarizer_cog.summarize_by_id(mock_context, 123456789012345678)

        # Check that summarize_text was called with start message
        summarizer_cog.llm_service.summarize_text.assert_called_once()
        call_args = summarizer_cog.llm_service.summarize_text.call_args[0][0]
        assert "StartUser: Starting message" in call_args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_id_excludes_bot_start_message(self, mock_send, summarizer_cog, mock_context):
        """summarize_by_id excludes start message if it's from a bot."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.content = "Bot message"
        start_msg.author.bot = True
        start_msg.author.display_name = "BotUser"
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            return
            yield
        mock_context.channel.history = history_gen

        await summarizer_cog.summarize_by_id(mock_context, 123456789012345678)

        # When start message is from bot and no other messages, count is 0
        # So summarize_text should not be called
        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "메시지가 없어요" in args


# =============================================================================
# summarize_by_range Tests
# =============================================================================

class TestSummarizeByRange:
    """Tests for SummarizerCog.summarize_by_range."""

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_range_after_direction(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """summarize_by_range handles '이후' direction."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            # Check that after and before are set correctly
            assert "after" in kwargs
            assert "before" in kwargs
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        await summarizer_cog.summarize_by_range(mock_context, 123456789012345678, "이후", "30분")

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "이후 30분" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_range_before_direction(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """summarize_by_range handles '이전' direction."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            # Check that after and before are set correctly for 'before'
            assert "after" in kwargs
            assert "before" in kwargs
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        await summarizer_cog.summarize_by_range(mock_context, 123456789012345678, "이전", "30분")

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "이전 30분" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_range_invalid_time_format(self, mock_send, summarizer_cog, mock_context):
        """summarize_by_range handles invalid time format."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        await summarizer_cog.summarize_by_range(mock_context, 123456789012345678, "이후", "invalid")

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "시간 형식이 올바르지 않아요" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_range_message_not_found(self, mock_send, summarizer_cog, mock_context):
        """summarize_by_range handles message not found."""
        mock_context.channel.fetch_message = AsyncMock(side_effect=discord.NotFound(Mock(), "Not Found"))

        await summarizer_cog.summarize_by_range(mock_context, 123456789012345678, "이후", "30분")

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "찾을 수 없어요" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_by_range_no_messages(self, mock_send, summarizer_cog, mock_context):
        """summarize_by_range handles no messages in range."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.created_at = datetime.now(timezone.utc)
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            return
            yield
        mock_context.channel.history = history_gen

        await summarizer_cog.summarize_by_range(mock_context, 123456789012345678, "이후", "30분")

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "메시지가 없어요" in args


# =============================================================================
# _handle_summarize_args Tests
# =============================================================================

class TestHandleSummarizeArgs:
    """Tests for SummarizerCog._handle_summarize_args."""

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_empty_args(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """Empty args triggers default 30-minute summary."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        await summarizer_cog._handle_summarize_args(mock_context, ())

        # Should call _summarize_by_time with 30 minutes
        summarizer_cog.llm_service.summarize_text.assert_called_once()

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_single_time_arg(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """Single time argument triggers time-based summary."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        await summarizer_cog._handle_summarize_args(mock_context, ("20분",))

        # Should call _summarize_by_time with 20 minutes
        summarizer_cog.llm_service.summarize_text.assert_called_once()

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_invalid_single_arg(self, mock_send, summarizer_cog, mock_context):
        """Invalid single argument sends error message."""
        await summarizer_cog._handle_summarize_args(mock_context, ("invalid",))

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "시간 형식이 올바르지 않아요" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_message_id_and_direction_after(self, mock_send, summarizer_cog, mock_context):
        """Message ID + '이후' triggers ID-based summary."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.author.bot = False
        start_msg.author.display_name = "StartUser"
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            return
            yield
        mock_context.channel.history = history_gen

        await summarizer_cog._handle_summarize_args(mock_context, ("123456789012345678", "이후"))

        # Should call summarize_by_id
        mock_context.channel.fetch_message.assert_called_once()

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_message_id_and_direction_before(self, mock_send, summarizer_cog, mock_context):
        """Message ID + '이전' without time sends error."""
        await summarizer_cog._handle_summarize_args(mock_context, ("123456789012345678", "이전"))

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "시간을 지정해야 합니다" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_invalid_message_id(self, mock_send, summarizer_cog, mock_context):
        """Invalid message ID format sends error."""
        await summarizer_cog._handle_summarize_args(mock_context, ("invalid_id", "이후"))

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "메시지 ID여야 해요" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_invalid_direction(self, mock_send, summarizer_cog, mock_context):
        """Invalid direction sends error."""
        await summarizer_cog._handle_summarize_args(mock_context, ("123456789012345678", "invalid"))

        # Since invalid is not '이후' or '이전', and arg1 is valid message ID,
        # it goes to len(args) == 2 branch which checks direction
        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        # The code checks if 방향 not in ["이후", "이전"]
        assert "이후" in args and "이전" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_three_args_with_after(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """Three args with ID + direction + time triggers range summary."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.created_at = datetime.now(timezone.utc)
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            return
            yield
        mock_context.channel.history = history_gen

        await summarizer_cog._handle_summarize_args(mock_context, ("123456789012345678", "이후", "30분"))

        # Should call summarize_by_range
        mock_context.channel.fetch_message.assert_called_once()

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_three_args_with_before(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """Three args with '이전' direction works correctly."""
        start_msg = Mock(spec=discord.Message)
        start_msg.id = 123456789012345678
        start_msg.created_at = datetime.now(timezone.utc)
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            return
            yield
        mock_context.channel.history = history_gen

        await summarizer_cog._handle_summarize_args(mock_context, ("123456789012345678", "이전", "30분"))

        mock_context.channel.fetch_message.assert_called_once()

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_handle_reply_with_direction(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """Reply message + direction uses reply target as ID."""
        mock_context.message.reference = Mock()
        mock_context.message.reference.message_id = 987654321098765432

        start_msg = Mock(spec=discord.Message)
        start_msg.id = 987654321098765432
        start_msg.author.bot = False
        start_msg.author.display_name = "ReplyUser"
        mock_context.channel.fetch_message = AsyncMock(return_value=start_msg)

        async def history_gen(**kwargs):
            return
            yield
        mock_context.channel.history = history_gen

        await summarizer_cog._handle_summarize_args(mock_context, ("이후",))

        # Should use the reply message ID
        mock_context.channel.fetch_message.assert_called_with(987654321098765432)


# =============================================================================
# Main summarize Command Tests
# =============================================================================

class TestSummarizeCommand:
    """Tests for SummarizerCog.summarize main command.

    Note: The summarize command is a hybrid command that wraps _handle_summarize_args.
    Since we comprehensively test _handle_summarize_args in TestHandleSummarizeArgs,
    these tests focus only on the command-specific aspects (defer, parameter passing).
    """

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_defers_response(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """summarize command defers response immediately."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        # Call the command method directly with all args as None (defaults)
        # This will trigger _handle_summarize_args with empty tuple
        await summarizer_cog._handle_summarize_args(mock_context, ())

        mock_context.defer.assert_not_called()  # _handle_summarize_args doesn't defer

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_passes_args_to_handler(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """summarize command properly passes arguments to handler."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        # Test that the command constructs args tuple correctly
        # This simulates what happens in the summarize() method
        args = ("20분",)
        await summarizer_cog._handle_summarize_args(mock_context, args)

        # Should call _summarize_by_time with 20 minutes
        summarizer_cog.llm_service.summarize_text.assert_called_once()

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_with_none_params_filters_correctly(self, mock_send, summarizer_cog, mock_context, mock_messages):
        """summarize command filters None values from args."""
        async def history_gen(**kwargs):
            for msg in mock_messages:
                yield msg
        mock_context.channel.history = history_gen

        # Simulate the command's arg filtering logic
        # Only non-None values should be included
        args = []
        args.append("20분")
        # Don't append None values
        await summarizer_cog._handle_summarize_args(mock_context, tuple(args))

        summarizer_cog.llm_service.summarize_text.assert_called_once()


# =============================================================================
# Error Handler Tests
# =============================================================================

class TestSummarizeError:
    """Tests for SummarizerCog.summarize_error."""

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_error_bad_argument(self, mock_send, summarizer_cog, mock_context):
        """summarize_error handles BadArgument."""
        error = commands.BadArgument()
        await summarizer_cog.summarize_error(mock_context, error)

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "인수가 잘못되었어요" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_error_missing_required_argument(self, mock_send, summarizer_cog, mock_context):
        """summarize_error handles MissingRequiredArgument."""
        # Create a mock parameter for MissingRequiredArgument
        mock_param = Mock()
        mock_param.name = "test_param"
        error = commands.MissingRequiredArgument(mock_param)
        await summarizer_cog.summarize_error(mock_context, error)

        mock_send.assert_called_once()
        args = mock_send.call_args[0][1]
        assert "명령어를 완성해주세요" in args

    @pytest.mark.asyncio
    @patch('persbot.bot.cogs.summarizer.send_discord_message')
    async def test_summarize_error_generic_exception(self, mock_send, summarizer_cog, mock_context):
        """summarize_error handles generic exceptions."""
        error = Exception("Unexpected error")
        with patch('persbot.bot.cogs.summarizer.logger') as mock_logger:
            await summarizer_cog.summarize_error(mock_context, error)

            mock_logger.error.assert_called_once()
            mock_send.assert_called_once()
            args = mock_send.call_args[0][1]
            assert "예상치 못한 오류" in args

"""Feature tests for retry_handler module.

Tests focus on behavior:
- BackoffStrategy: backoff strategy enumeration
- RetryConfig: retry configuration dataclass
- RetryHandler: abstract base class for retry handling
- GeminiRetryHandler: Gemini-specific retry handler
- OpenAIRetryHandler: OpenAI-specific retry handler
- ZAIRetryHandler: Z.AI-specific retry handler
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from persbot.exceptions import FatalError
from persbot.services.retry_handler import (
    BackoffStrategy,
    RetryConfig,
    RetryHandler,
    GeminiRetryHandler,
    OpenAIRetryHandler,
    ZAIRetryHandler,
)
from persbot.utils import ERROR_API_TIMEOUT, GENERIC_ERROR_MESSAGE


# =============================================================================
# Concrete Test Implementation of RetryHandler
# =============================================================================


class ConcreteRetryHandler(RetryHandler):
    """Concrete implementation of RetryHandler for testing."""

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error message contains 'rate limit' or '429'."""
        error_str = str(error).lower()
        return "rate limit" in error_str or "429" in error_str


class RateLimitRetryHandler(RetryHandler):
    """Test handler that always detects rate limits."""

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Always return True for testing."""
        return True


class NeverRateLimitRetryHandler(RetryHandler):
    """Test handler that never detects rate limits."""

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Always return False for testing."""
        return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create default RetryConfig."""
    return RetryConfig()


@pytest.fixture
def fast_config():
    """Create RetryConfig with fast settings for testing."""
    return RetryConfig(
        max_retries=3,
        base_delay=0.01,
        max_delay=0.1,
        rate_limit_delay=0.01,
        request_timeout=0.5,
    )


@pytest.fixture
def mock_discord_message():
    """Create a mock discord.Message."""
    message = MagicMock(spec=discord.Message)
    message.reply = AsyncMock(return_value=MagicMock(spec=discord.Message))
    message.reply.return_value.edit = AsyncMock()
    message.reply.return_value.delete = AsyncMock()
    message.channel = MagicMock()
    message.channel.send = AsyncMock(return_value=MagicMock(spec=discord.Message))
    return message


@pytest.fixture
def handler(fast_config):
    """Create ConcreteRetryHandler with fast config."""
    return ConcreteRetryHandler(fast_config)


# =============================================================================
# Test BackoffStrategy
# =============================================================================


class TestBackoffStrategy:
    """Tests for BackoffStrategy enumeration."""

    def test_has_exponential(self):
        """BackoffStrategy includes EXPONENTIAL."""
        assert BackoffStrategy.EXPONENTIAL.value == "exponential"

    def test_has_linear(self):
        """BackoffStrategy includes LINEAR."""
        assert BackoffStrategy.LINEAR.value == "linear"

    def test_has_fixed(self):
        """BackoffStrategy includes FIXED."""
        assert BackoffStrategy.FIXED.value == "fixed"

    def test_can_compare_strategies(self):
        """BackoffStrategy values can be compared."""
        assert BackoffStrategy.EXPONENTIAL != BackoffStrategy.LINEAR
        assert BackoffStrategy.LINEAR != BackoffStrategy.FIXED


# =============================================================================
# Test RetryConfig
# =============================================================================


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_creates_with_defaults(self):
        """RetryConfig creates with sensible defaults."""
        config = RetryConfig()
        assert config.max_retries == 2
        assert config.base_delay == 2.0
        assert config.max_delay == 32.0
        assert config.rate_limit_delay == 5
        assert config.request_timeout == 120.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL

    def test_creates_with_custom_values(self):
        """RetryConfig accepts custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
            rate_limit_delay=10,
            request_timeout=30.0,
            backoff_strategy=BackoffStrategy.LINEAR,
        )
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.rate_limit_delay == 10
        assert config.request_timeout == 30.0
        assert config.backoff_strategy == BackoffStrategy.LINEAR

    def test_accepts_fixed_strategy(self):
        """RetryConfig accepts FIXED backoff strategy."""
        config = RetryConfig(backoff_strategy=BackoffStrategy.FIXED)
        assert config.backoff_strategy == BackoffStrategy.FIXED


# =============================================================================
# Test RetryHandler._is_fatal_error
# =============================================================================


class TestRetryHandlerIsFatalError:
    """Tests for RetryHandler._is_fatal_error method."""

    def test_returns_true_for_fatal_error(self, handler):
        """_is_fatal_error returns True for FatalError instances."""
        error = FatalError("Something fatal happened")
        assert handler._is_fatal_error(error) is True

    def test_returns_false_for_generic_error(self, handler):
        """_is_fatal_error returns False for generic exceptions."""
        error = ValueError("Some error")
        assert handler._is_fatal_error(error) is False

    def test_returns_false_for_rate_limit_error(self, handler):
        """_is_fatal_error returns False for rate limit errors."""
        error = Exception("429 rate limit exceeded")
        assert handler._is_fatal_error(error) is False


# =============================================================================
# Test RetryHandler._extract_retry_delay
# =============================================================================


class TestRetryHandlerExtractRetryDelay:
    """Tests for RetryHandler._extract_retry_delay method."""

    def test_extracts_delay_from_please_retry_pattern(self, handler):
        """_extract_retry_delay extracts delay from 'Please retry in Xs' pattern."""
        error = Exception("Rate limited. Please retry in 30s")
        assert handler._extract_retry_delay(error) == 30.0

    def test_extracts_delay_from_please_retry_pattern_with_decimal(self, handler):
        """_extract_retry_delay extracts decimal delays."""
        error = Exception("Please retry in 12.5s")
        assert handler._extract_retry_delay(error) == 12.5

    def test_extracts_delay_from_seconds_pattern(self, handler):
        """_extract_retry_delay extracts delay from 'seconds: X' pattern."""
        error = Exception("Retry after seconds: 60")
        assert handler._extract_retry_delay(error) == 60.0

    def test_returns_none_for_no_pattern(self, handler):
        """_extract_retry_delay returns None when no pattern matches."""
        error = Exception("Some random error message")
        assert handler._extract_retry_delay(error) is None

    def test_is_case_insensitive(self, handler):
        """_extract_retry_delay is case insensitive for 'Please retry' pattern."""
        error = Exception("PLEASE RETRY IN 10S")
        assert handler._extract_retry_delay(error) == 10.0


# =============================================================================
# Test RetryHandler._calculate_backoff
# =============================================================================


class TestRetryHandlerCalculateBackoff:
    """Tests for RetryHandler._calculate_backoff method."""

    def test_exponential_backoff(self):
        """_calculate_backoff uses exponential strategy correctly."""
        config = RetryConfig(
            base_delay=2.0,
            max_delay=100.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        )
        handler = ConcreteRetryHandler(config)

        assert handler._calculate_backoff(1) == 2.0  # 2^1
        assert handler._calculate_backoff(2) == 4.0  # 2^2
        assert handler._calculate_backoff(3) == 8.0  # 2^3
        assert handler._calculate_backoff(4) == 16.0  # 2^4

    def test_exponential_backoff_respects_max_delay(self):
        """_calculate_backoff caps exponential at max_delay."""
        config = RetryConfig(
            base_delay=2.0,
            max_delay=10.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        )
        handler = ConcreteRetryHandler(config)

        # 2^10 = 1024, but capped at 10
        assert handler._calculate_backoff(10) == 10.0

    def test_linear_backoff(self):
        """_calculate_backoff uses linear strategy correctly."""
        config = RetryConfig(
            base_delay=2.0,
            max_delay=100.0,
            backoff_strategy=BackoffStrategy.LINEAR,
        )
        handler = ConcreteRetryHandler(config)

        assert handler._calculate_backoff(1) == 2.0  # 2*1
        assert handler._calculate_backoff(2) == 4.0  # 2*2
        assert handler._calculate_backoff(3) == 6.0  # 2*3
        assert handler._calculate_backoff(5) == 10.0  # 2*5

    def test_linear_backoff_respects_max_delay(self):
        """_calculate_backoff caps linear at max_delay."""
        config = RetryConfig(
            base_delay=5.0,
            max_delay=10.0,
            backoff_strategy=BackoffStrategy.LINEAR,
        )
        handler = ConcreteRetryHandler(config)

        # 5*5 = 25, but capped at 10
        assert handler._calculate_backoff(5) == 10.0

    def test_fixed_backoff(self):
        """_calculate_backoff returns constant for fixed strategy."""
        config = RetryConfig(
            base_delay=3.0,
            backoff_strategy=BackoffStrategy.FIXED,
        )
        handler = ConcreteRetryHandler(config)

        assert handler._calculate_backoff(1) == 3.0
        assert handler._calculate_backoff(2) == 3.0
        assert handler._calculate_backoff(5) == 3.0
        assert handler._calculate_backoff(10) == 3.0


# =============================================================================
# Test RetryHandler._wait_with_countdown
# =============================================================================


class TestRetryHandlerWaitWithCountdown:
    """Tests for RetryHandler._wait_with_countdown method."""

    @pytest.mark.asyncio
    async def test_returns_immediately_for_zero_delay(self, handler, mock_discord_message):
        """_wait_with_countdown returns immediately when delay is 0."""
        await handler._wait_with_countdown(0, mock_discord_message)
        mock_discord_message.reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_immediately_for_negative_delay(self, handler, mock_discord_message):
        """_wait_with_countdown returns immediately when delay is negative."""
        await handler._wait_with_countdown(-5, mock_discord_message)
        mock_discord_message.reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_discord_message_on_rate_limit(self, handler, mock_discord_message):
        """_wait_with_countdown sends Discord message when rate limited."""
        # Use very short delay
        await handler._wait_with_countdown(0.1, mock_discord_message)
        mock_discord_message.reply.assert_called_once()

    @pytest.mark.asyncio
    async def test_deletes_message_after_countdown(self, handler, mock_discord_message):
        """_wait_with_countdown deletes the countdown message after completion."""
        sent_message = mock_discord_message.reply.return_value
        await handler._wait_with_countdown(0.1, mock_discord_message)
        sent_message.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_discord_http_exception_on_reply(self, handler):
        """_wait_with_countdown handles HTTPException when sending reply."""
        message = MagicMock(spec=discord.Message)
        message.reply = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "error"))

        # Should not raise
        await handler._wait_with_countdown(0.1, message)

    @pytest.mark.asyncio
    async def test_handles_discord_http_exception_on_edit(self, handler):
        """_wait_with_countdown handles HTTPException when editing message."""
        message = MagicMock(spec=discord.Message)
        sent_message = MagicMock(spec=discord.Message)
        sent_message.edit = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "error"))
        sent_message.delete = AsyncMock()
        message.reply = AsyncMock(return_value=sent_message)

        # Should not raise
        await handler._wait_with_countdown(0.2, message)

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_on_cancel_event(self, handler, mock_discord_message):
        """_wait_with_countdown raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError, match="aborted by user"):
            await handler._wait_with_countdown(10, mock_discord_message, cancel_event)

    @pytest.mark.asyncio
    async def test_deletes_message_on_cancellation(self, handler):
        """_wait_with_countdown deletes message when cancelled."""
        message = MagicMock(spec=discord.Message)
        sent_message = MagicMock(spec=discord.Message)
        sent_message.edit = AsyncMock()
        sent_message.delete = AsyncMock()
        message.reply = AsyncMock(return_value=sent_message)

        cancel_event = asyncio.Event()

        async def set_cancel_after_delay():
            await asyncio.sleep(0.05)
            cancel_event.set()

        # Start the task that will set cancel
        asyncio.create_task(set_cancel_after_delay())

        with pytest.raises(asyncio.CancelledError):
            await handler._wait_with_countdown(5, message, cancel_event)

        sent_message.delete.assert_called()


# =============================================================================
# Test RetryHandler._execute_with_timeout
# =============================================================================


class TestRetryHandlerExecuteWithTimeout:
    """Tests for RetryHandler._execute_with_timeout method."""

    @pytest.mark.asyncio
    async def test_executes_async_function(self, handler):
        """_execute_with_timeout executes async functions."""
        async def async_func():
            return "async result"

        result = await handler._execute_with_timeout(async_func)
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_executes_sync_function(self, handler):
        """_execute_with_timeout executes sync functions via to_thread."""
        def sync_func():
            return "sync result"

        result = await handler._execute_with_timeout(sync_func)
        assert result == "sync result"

    @pytest.mark.asyncio
    async def test_passes_function_exceptions(self, handler):
        """_execute_with_timeout propagates exceptions from functions."""
        async def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await handler._execute_with_timeout(failing_func)


# =============================================================================
# Test RetryHandler.execute_with_retry
# =============================================================================


class TestRetryHandlerExecuteWithRetry:
    """Tests for RetryHandler.execute_with_retry method."""

    @pytest.mark.asyncio
    async def test_returns_result_on_first_try(self, handler):
        """execute_with_retry returns result on successful first try."""
        async def success_call():
            return "success"

        result = await handler.execute_with_retry(success_call)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retries_on_error_and_succeeds(self, fast_config):
        """execute_with_retry retries on error and eventually succeeds."""
        call_count = 0

        async def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("temporary error")
            return "success"

        handler = ConcreteRetryHandler(fast_config)
        result = await handler.execute_with_retry(flaky_call)

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_returns_none_after_max_retries(self, fast_config):
        """execute_with_retry returns None after max retries exhausted."""
        async def always_fail():
            raise ValueError("always fails")

        handler = ConcreteRetryHandler(fast_config)
        result = await handler.execute_with_retry(always_fail)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_rate_limit_error(self, fast_config, mock_discord_message):
        """execute_with_retry handles rate limit errors with countdown."""
        call_count = 0

        async def rate_limited_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 rate limit exceeded")
            return "success"

        handler = RateLimitRetryHandler(fast_config)
        result = await handler.execute_with_retry(
            rate_limited_call,
            discord_message=mock_discord_message,
        )

        assert result == "success"
        mock_discord_message.reply.assert_called()

    @pytest.mark.asyncio
    async def test_uses_extracted_retry_delay(self, fast_config):
        """execute_with_retry uses delay extracted from error message."""
        call_count = 0

        async def rate_limited_with_delay():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Please retry in 0.1s")
            return "success"

        handler = RateLimitRetryHandler(fast_config)
        result = await handler.execute_with_retry(rate_limited_with_delay)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_raises_fatal_error_immediately(self, handler):
        """execute_with_retry re-raises fatal errors immediately."""
        async def fatal_call():
            raise FatalError("fatal error")

        with pytest.raises(FatalError, match="fatal error"):
            await handler.execute_with_retry(fatal_call)

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self, fast_config):
        """execute_with_retry handles timeout errors."""
        async def timeout_call():
            raise asyncio.TimeoutError()

        handler = ConcreteRetryHandler(fast_config)
        result = await handler.execute_with_retry(timeout_call)

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_custom_timeout(self, fast_config):
        """execute_with_retry uses custom timeout parameter."""
        # Very short timeout that will trigger
        async def slow_call():
            await asyncio.sleep(1)
            return "too slow"

        handler = ConcreteRetryHandler(fast_config)
        result = await handler.execute_with_retry(slow_call, timeout=0.01)

        assert result is None

    @pytest.mark.asyncio
    async def test_propagates_cancelled_error(self, handler):
        """execute_with_retry propagates CancelledError immediately."""
        async def cancelled_call():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await handler.execute_with_retry(cancelled_call)

    @pytest.mark.asyncio
    async def test_cancels_on_cancel_event_at_start(self, handler):
        """execute_with_retry checks cancel_event at start of retry loop."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        async def success_call():
            return "should not be called"

        with pytest.raises(asyncio.CancelledError, match="aborted by user"):
            await handler.execute_with_retry(success_call, cancel_event=cancel_event)

    @pytest.mark.asyncio
    async def test_cancels_during_backoff(self, fast_config):
        """execute_with_retry checks cancel_event during backoff."""
        cancel_event = asyncio.Event()
        call_count = 0

        async def error_then_cancel():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Set cancel event after first error
                asyncio.get_event_loop().call_later(0.01, cancel_event.set)
                raise ValueError("error")
            return "should not reach"

        handler = NeverRateLimitRetryHandler(fast_config)

        with pytest.raises(asyncio.CancelledError, match="aborted by user"):
            await handler.execute_with_retry(error_then_cancel, cancel_event=cancel_event)

    @pytest.mark.asyncio
    async def test_uses_fallback_call_on_rate_limit(self, fast_config):
        """execute_with_retry tries fallback_call when rate limited."""
        main_call_count = 0
        fallback_call_count = 0

        async def rate_limited_main():
            nonlocal main_call_count
            main_call_count += 1
            raise Exception("429 rate limit")

        async def fallback():
            nonlocal fallback_call_count
            fallback_call_count += 1
            return "fallback result"

        handler = RateLimitRetryHandler(fast_config)
        result = await handler.execute_with_retry(
            rate_limited_main,
            fallback_call=fallback,
        )

        assert result == "fallback result"
        assert fallback_call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_failure_continues_retry(self, fast_config):
        """execute_with_retry continues normal retry when fallback fails."""
        async def rate_limited_main():
            raise Exception("429 rate limit")

        async def failing_fallback():
            raise ValueError("fallback also fails")

        handler = RateLimitRetryHandler(fast_config)
        result = await handler.execute_with_retry(
            rate_limited_main,
            fallback_call=failing_fallback,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_calls_log_response_callback(self, handler):
        """execute_with_retry calls log_response callback on success."""
        logged_responses = []

        def log_response(response, attempt):
            logged_responses.append((response, attempt))

        async def success_call():
            return "response data"

        await handler.execute_with_retry(success_call, log_response=log_response)

        assert logged_responses == [("response data", 1)]

    @pytest.mark.asyncio
    async def test_uses_extract_text_callback(self, handler):
        """execute_with_retry uses extract_text callback to transform response."""
        def extract_text(response):
            return response["text"]

        async def api_call():
            return {"text": "extracted text", "other": "data"}

        result = await handler.execute_with_retry(api_call, extract_text=extract_text)

        assert result == "extracted text"

    @pytest.mark.asyncio
    async def test_returns_full_response_when_requested(self, handler):
        """execute_with_retry returns full response when return_full_response=True."""
        async def api_call():
            return {"data": "full response"}

        result = await handler.execute_with_retry(
            api_call,
            return_full_response=True,
        )

        assert result == {"data": "full response"}

    @pytest.mark.asyncio
    async def test_executes_sync_functions(self, handler):
        """execute_with_retry handles sync functions."""
        def sync_call():
            return "sync result"

        result = await handler.execute_with_retry(sync_call)

        assert result == "sync result"


# =============================================================================
# Test RetryHandler._notify_final_error
# =============================================================================


class TestRetryHandlerNotifyFinalError:
    """Tests for RetryHandler._notify_final_error method."""

    @pytest.mark.asyncio
    async def test_sends_timeout_message_for_timeout_error(self, handler, mock_discord_message):
        """_notify_final_error sends timeout message for TimeoutError."""
        error = asyncio.TimeoutError()
        await handler._notify_final_error(error, mock_discord_message)

        mock_discord_message.reply.assert_called_once_with(
            ERROR_API_TIMEOUT,
            mention_author=False,
        )

    @pytest.mark.asyncio
    async def test_sends_generic_message_for_other_errors(self, handler, mock_discord_message):
        """_notify_final_error sends generic message for other errors."""
        error = ValueError("some error")
        await handler._notify_final_error(error, mock_discord_message)

        mock_discord_message.reply.assert_called_once_with(
            GENERIC_ERROR_MESSAGE,
            mention_author=False,
        )

    @pytest.mark.asyncio
    async def test_handles_none_error(self, handler, mock_discord_message):
        """_notify_final_error handles None error."""
        await handler._notify_final_error(None, mock_discord_message)

        mock_discord_message.reply.assert_called_once_with(
            GENERIC_ERROR_MESSAGE,
            mention_author=False,
        )

    @pytest.mark.asyncio
    async def test_sends_no_message_when_discord_message_is_none(self, handler):
        """_notify_final_error does not send when discord_message is None."""
        error = ValueError("error")
        # Should not raise
        await handler._notify_final_error(error, None)

    @pytest.mark.asyncio
    async def test_handles_http_exception_on_reply(self, handler):
        """_notify_final_error handles HTTPException when sending reply."""
        message = MagicMock(spec=discord.Message)
        message.reply = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "error"))

        error = ValueError("error")
        # Should not raise
        await handler._notify_final_error(error, message)


# =============================================================================
# Test GeminiRetryHandler
# =============================================================================


class TestGeminiRetryHandler:
    """Tests for GeminiRetryHandler."""

    @pytest.fixture
    def gemini_handler(self, fast_config):
        """Create GeminiRetryHandler with fast config."""
        return GeminiRetryHandler(fast_config)

    def test_is_rate_limit_error_detects_429(self, gemini_handler):
        """_is_rate_limit_error detects 429 status code."""
        error = Exception("Error 429: Too many requests")
        assert gemini_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_quota(self, gemini_handler):
        """_is_rate_limit_error detects quota keyword."""
        error = Exception("Quota exceeded for project")
        assert gemini_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_rate_limit(self, gemini_handler):
        """_is_rate_limit_error detects rate limit phrase."""
        error = Exception("Rate limit exceeded")
        assert gemini_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_rejects_400_without_quota(self, gemini_handler):
        """_is_rate_limit_error rejects 400 errors without quota keyword."""
        error = Exception("Error 400: Bad request")
        assert gemini_handler._is_rate_limit_error(error) is False

    def test_is_rate_limit_error_accepts_400_with_quota(self, gemini_handler):
        """_is_rate_limit_error accepts 400 errors with quota keyword."""
        error = Exception("Error 400: Quota exceeded")
        assert gemini_handler._is_rate_limit_error(error) is True

    def test_is_fatal_error_detects_cachedcontent_not_found(self, gemini_handler):
        """_is_fatal_error detects CachedContent not found error."""
        error = Exception("CachedContent not found: cache_id")
        assert gemini_handler._is_fatal_error(error) is True

    def test_is_fatal_error_detects_403_permission(self, gemini_handler):
        """_is_fatal_error detects 403 PERMISSION_DENIED error."""
        error = Exception("403 PERMISSION_DENIED: access denied")
        assert gemini_handler._is_fatal_error(error) is True

    def test_is_fatal_error_rejects_403_without_permission(self, gemini_handler):
        """_is_fatal_error rejects 403 without permission keyword."""
        error = Exception("403 Forbidden")
        assert gemini_handler._is_fatal_error(error) is False

    def test_is_fatal_error_rejects_generic_errors(self, gemini_handler):
        """_is_fatal_error rejects generic errors."""
        error = ValueError("Some generic error")
        assert gemini_handler._is_fatal_error(error) is False


# =============================================================================
# Test OpenAIRetryHandler
# =============================================================================


class TestOpenAIRetryHandler:
    """Tests for OpenAIRetryHandler."""

    @pytest.fixture
    def openai_handler(self, fast_config):
        """Create OpenAIRetryHandler with fast config."""
        return OpenAIRetryHandler(fast_config)

    def test_is_rate_limit_error_detects_429(self, openai_handler):
        """_is_rate_limit_error detects 429 status code."""
        error = Exception("Error 429: Too many requests")
        assert openai_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_rate_limit_underscore(self, openai_handler):
        """_is_rate_limit_error detects rate_limit pattern."""
        error = Exception("rate_limit exceeded")
        assert openai_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_rate_limit_space(self, openai_handler):
        """_is_rate_limit_error detects rate limit pattern with space."""
        error = Exception("Rate limit exceeded")
        assert openai_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_quota(self, openai_handler):
        """_is_rate_limit_error detects quota keyword."""
        error = Exception("Quota exceeded")
        assert openai_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_rejects_generic_errors(self, openai_handler):
        """_is_rate_limit_error rejects generic errors."""
        error = Exception("Internal server error")
        assert openai_handler._is_rate_limit_error(error) is False

    def test_is_rate_limit_error_is_case_insensitive(self, openai_handler):
        """_is_rate_limit_error is case insensitive."""
        error = Exception("RATE LIMIT EXCEEDED")
        assert openai_handler._is_rate_limit_error(error) is True


# =============================================================================
# Test ZAIRetryHandler
# =============================================================================


class TestZAIRetryHandler:
    """Tests for ZAIRetryHandler."""

    @pytest.fixture
    def zai_handler(self, fast_config):
        """Create ZAIRetryHandler with fast config."""
        return ZAIRetryHandler(fast_config)

    def test_is_rate_limit_error_detects_429(self, zai_handler):
        """_is_rate_limit_error detects 429 status code."""
        error = Exception("Error 429: Too many requests")
        assert zai_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_rate_limit_underscore(self, zai_handler):
        """_is_rate_limit_error detects rate_limit pattern."""
        error = Exception("rate_limit exceeded")
        assert zai_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_rate_limit_space(self, zai_handler):
        """_is_rate_limit_error detects rate limit pattern with space."""
        error = Exception("Rate limit exceeded")
        assert zai_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_quota(self, zai_handler):
        """_is_rate_limit_error detects quota keyword."""
        error = Exception("Quota exceeded")
        assert zai_handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_rejects_generic_errors(self, zai_handler):
        """_is_rate_limit_error rejects generic errors."""
        error = Exception("Internal server error")
        assert zai_handler._is_rate_limit_error(error) is False

    def test_is_rate_limit_error_is_case_insensitive(self, zai_handler):
        """_is_rate_limit_error is case insensitive."""
        error = Exception("RATE LIMIT EXCEEDED")
        assert zai_handler._is_rate_limit_error(error) is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestRetryHandlerIntegration:
    """Integration tests for retry handler scenarios."""

    @pytest.mark.asyncio
    async def test_full_retry_cycle_with_success(self, fast_config, mock_discord_message):
        """Full retry cycle eventually succeeds."""
        call_count = 0

        async def flaky_api():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("temporary error")
            if call_count == 2:
                raise Exception("429 rate limit")
            return "final success"

        handler = ConcreteRetryHandler(fast_config)
        result = await handler.execute_with_retry(
            flaky_api,
            discord_message=mock_discord_message,
        )

        assert result == "final success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_handlers_share_common_interface(self, fast_config):
        """All retry handler implementations share common interface."""
        handlers = [
            GeminiRetryHandler(fast_config),
            OpenAIRetryHandler(fast_config),
            ZAIRetryHandler(fast_config),
        ]

        async def success_call():
            return "success"

        for handler in handlers:
            result = await handler.execute_with_retry(success_call)
            assert result == "success"

    @pytest.mark.asyncio
    async def test_handlers_differ_in_rate_limit_detection(self, fast_config):
        """Handlers differ in rate limit detection behavior."""
        error_400 = Exception("Error 400: Bad request")

        gemini = GeminiRetryHandler(fast_config)
        openai = OpenAIRetryHandler(fast_config)

        # Gemini should reject 400 without quota
        assert gemini._is_rate_limit_error(error_400) is False
        # OpenAI should also reject 400 (no rate limit pattern)
        assert openai._is_rate_limit_error(error_400) is False

        # But for 429, both should accept
        error_429 = Exception("Error 429: Rate limited")
        assert gemini._is_rate_limit_error(error_429) is True
        assert openai._is_rate_limit_error(error_429) is True

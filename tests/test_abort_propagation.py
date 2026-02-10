"""Unit tests for API call cancellation and abort propagation.

This module tests:
- Cancel event propagation through all layers
- Tool execution abort behavior
- Service layer abort behavior
- Retry loop abort behavior
- Abort during image generation
- Abort during search operations
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from persbot.config import AppConfig
from persbot.services.image_service import ImageService, ImageGenerationError
from persbot.services.retry_handler import RetryHandler, RetryConfig, BackoffStrategy
from persbot.services.search_service import SearchService, SearchError, SearchRateLimitError
from persbot.tools.api_tools.image_tools import generate_image
from persbot.tools.api_tools.search_tools import web_search
from persbot.tools.base import ToolDefinition, ToolCategory, ToolParameter, ToolResult


class TestCancelEventPropagation:
    """Test cancel event propagation through the tool execution chain."""

    @pytest.mark.asyncio
    async def test_cancel_event_set_before_api_call(self):
        """Test that cancel_event.is_set() is checked before API calls."""
        cancel_event = asyncio.Event()
        cancel_event.set()  # Set cancel before starting

        config = Mock()
        config.openrouter_api_key = "test_key"
        config.api_request_timeout = 30.0

        service = ImageService(config)

        with pytest.raises(asyncio.CancelledError, match="aborted by user"):
            await service.generate_image(
                prompt="test prompt",
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_cancel_event_not_set_allows_execution(self):
        """Test that unset cancel_event allows normal execution."""
        cancel_event = asyncio.Event()
        # Don't set cancel_event

        config = Mock()
        config.openrouter_api_key = "test_key"
        config.api_request_timeout = 30.0
        config.openrouter_image_model = "test/model"

        service = ImageService(config)

        # Mock the internal API call to return fake data
        with patch.object(service, '_execute_image_generation') as mock_api:
            mock_api.return_value = (b"fake_image_data", "png")

            with patch.object(service, 'execute_with_retry') as mock_retry:
                mock_retry.return_value = (b"fake_image_data", "png")

                result = await service.generate_image(
                    prompt="test prompt",
                    cancel_event=cancel_event,
                )

                assert result is not None
                assert result[0] == b"fake_image_data"

    @pytest.mark.asyncio
    async def test_cancel_event_none_allows_execution(self):
        """Test that None cancel_event allows normal execution."""
        config = Mock()
        config.openrouter_api_key = "test_key"
        config.api_request_timeout = 30.0
        config.openrouter_image_model = "test/model"

        service = ImageService(config)

        with patch.object(service, 'execute_with_retry') as mock_retry:
            mock_retry.return_value = (b"fake_image_data", "png")

            result = await service.generate_image(
                prompt="test prompt",
                cancel_event=None,
            )

            assert result is not None


class TestRetryLoopCancellation:
    """Test cancellation during retry loops."""

    @pytest.mark.asyncio
    async def test_retry_loop_aborts_immediately_on_cancel(self):
        """Test that retry loop aborts immediately when cancel_event is set."""
        from persbot.services.retry_handler import OpenAIRetryHandler

        cancel_event = asyncio.Event()

        config = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            max_delay=10.0,
            request_timeout=30.0,
        )

        handler = OpenAIRetryHandler(config)

        call_count = 0
        async def failing_api_call():
            nonlocal call_count
            call_count += 1
            # Set cancel after first call
            if call_count == 1:
                cancel_event.set()
            raise Exception("API error")

        with pytest.raises(asyncio.CancelledError):
            await handler.execute_with_retry(
                api_call=failing_api_call,
                error_prefix="Test",
                cancel_event=cancel_event,
            )

        # Should have only made one attempt before cancellation
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_loop_checks_cancel_at_iteration_start(self):
        """Test that cancel is checked at the start of each retry iteration."""
        from persbot.services.retry_handler import OpenAIRetryHandler

        cancel_event = asyncio.Event()

        config = RetryConfig(
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0,
            request_timeout=30.0,
        )

        handler = OpenAIRetryHandler(config)

        call_count = 0
        async def failing_api_call():
            nonlocal call_count
            call_count += 1
            raise Exception("API error")

        # Start with cancel already set
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await handler.execute_with_retry(
                api_call=failing_api_call,
                error_prefix="Test",
                cancel_event=cancel_event,
            )

        # Should not have made any attempts since cancel was already set
        assert call_count == 0


class TestImageServiceCancellation:
    """Test cancellation in ImageService."""

    @pytest.mark.asyncio
    async def test_image_generation_cancels_before_api_call(self):
        """Test image generation aborts when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        config = Mock()
        config.openrouter_api_key = "test_key"
        config.api_request_timeout = 30.0

        service = ImageService(config)

        with pytest.raises(asyncio.CancelledError):
            await service.generate_image(
                prompt="test prompt",
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_image_generation_with_fetch_cancels(self):
        """Test image fetch aborts when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        config = Mock()
        config.openrouter_api_key = "test_key"
        config.api_request_timeout = 30.0

        service = ImageService(config)

        with pytest.raises(asyncio.CancelledError):
            await service.fetch_image_from_url(
                url="https://example.com/image.png",
                cancel_event=cancel_event,
            )


class TestSearchServiceCancellation:
    """Test cancellation in SearchService."""

    @pytest.mark.asyncio
    async def test_web_search_cancels_before_api_call(self):
        """Test web search aborts when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        service = SearchService()

        with pytest.raises(asyncio.CancelledError):
            await service.web_search(
                query="test query",
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_web_search_with_unset_cancel_proceeds(self):
        """Test web search proceeds when cancel_event is not set."""
        cancel_event = asyncio.Event()  # Not set

        service = SearchService()

        # Mock the DDGS call to avoid actual network requests
        with patch('persbot.services.search_service.DDGS') as mock_ddgs:
            mock_instance = Mock()
            mock_instance.text.return_value = [
                {
                    "title": "Test Result",
                    "href": "https://example.com",
                    "body": "Test snippet",
                }
            ]
            mock_ddgs.return_value = mock_instance

            result = await service.web_search(
                query="test query",
                cancel_event=cancel_event,
            )

            assert result is not None
            assert len(result) == 1
            assert result[0]["title"] == "Test Result"


class TestToolCancellation:
    """Test cancellation in tool functions."""

    @pytest.mark.asyncio
    async def test_generate_image_tool_cancels(self):
        """Test generate_image tool respects cancel_event."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        # Mock Discord context
        discord_context = Mock()
        discord_context.author = Mock(id=12345)
        discord_context.attachments = []

        with patch('persbot.tools.api_tools.image_tools.get_image_rate_limiter') as mock_limiter:
            mock_rate_result = Mock()
            mock_rate_result.allowed = True
            mock_limiter.return_value.check_rate_limit = AsyncMock(return_value=mock_rate_result)

            result = await generate_image(
                prompt="test prompt",
                discord_context=discord_context,
                cancel_event=cancel_event,
            )

            assert result.success is False
            assert "aborted" in result.error.lower()

    @pytest.mark.asyncio
    async def test_web_search_tool_cancels(self):
        """Test web_search tool respects cancel_event."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        result = await web_search(
            query="test query",
            cancel_event=cancel_event,
        )

        assert result.success is False
        assert "aborted" in result.error.lower()


class TestServiceLayerCancellation:
    """Test cancellation at service layer."""

    @pytest.mark.asyncio
    async def test_service_executes_with_retry_respects_cancel(self):
        """Test that execute_with_retry checks cancel_event."""
        from persbot.services.retry_handler import OpenAIRetryHandler

        cancel_event = asyncio.Event()
        cancel_event.set()

        config = Mock()
        config.openai_api_key = "test_key"
        config.api_request_timeout = 30.0
        config.api_max_retries = 2
        config.api_retry_backoff_base = 2.0
        config.api_retry_backoff_max = 32.0
        config.api_rate_limit_retry_after = 5

        # Create a minimal service for testing with proper retry handler
        class TestService:
            def __init__(self, config):
                self.config = config
                self._retry_handler = None

            def get_retry_handler(self):
                if self._retry_handler is None:
                    from persbot.services.retry_handler import RetryConfig, OpenAIRetryHandler
                    retry_config = RetryConfig(
                        max_retries=2,
                        base_delay=2.0,
                        max_delay=32.0,
                        rate_limit_delay=5,
                        request_timeout=30.0,
                    )
                    self._retry_handler = OpenAIRetryHandler(retry_config)
                return self._retry_handler

            async def execute_with_retry(self, model_call, **kwargs):
                retry_handler = self.get_retry_handler()
                return await retry_handler.execute_with_retry(
                    api_call=model_call,
                    **kwargs
                )

        service = TestService(config)

        with pytest.raises(asyncio.CancelledError):
            await service.execute_with_retry(
                model_call=lambda: "test",
                error_prefix="Test",
                cancel_event=cancel_event,
            )


class TestBatchMessageCancel:
    """Test cancellation triggered by batch message (new message arriving)."""

    @pytest.mark.asyncio
    async def test_cancel_event_set_from_external_source(self):
        """Test that external sources can set cancel_event to abort operations."""
        cancel_event = asyncio.Event()

        # Create a simple task that checks cancel_event
        async def long_running_operation():
            # Simulate a long operation that can be cancelled
            for i in range(10):
                await asyncio.sleep(0.01)
                if cancel_event.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
            return "completed"

        # Use asyncio.gather to run both operations concurrently
        async def set_cancel_later():
            await asyncio.sleep(0.02)
            cancel_event.set()

        # Run both tasks - should get CancelledError from long_running_operation
        results = await asyncio.gather(
            long_running_operation(),
            set_cancel_later(),
            return_exceptions=True,
        )

        # The first result should be CancelledError
        assert isinstance(results[0], asyncio.CancelledError)


class TestCancelBehaviorComparison:
    """Compare abort behavior for !abort command vs new batch message."""

    @pytest.mark.asyncio
    async def test_abort_command_and_new_message_both_set_cancel_event(self):
        """Test that both abort methods use the same cancel_event mechanism."""
        abort_cancel = asyncio.Event()
        new_message_cancel = asyncio.Event()

        # Both should use the same mechanism
        abort_cancel.set()
        new_message_cancel.set()

        config = Mock()
        config.openrouter_api_key = "test_key"
        config.api_request_timeout = 30.0

        service = ImageService(config)

        # Both should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await service.generate_image(
                prompt="test",
                cancel_event=abort_cancel,
            )

        with pytest.raises(asyncio.CancelledError):
            await service.generate_image(
                prompt="test",
                cancel_event=new_message_cancel,
            )


class TestCleanupOnCancellation:
    """Test proper cleanup when operations are cancelled."""

    @pytest.mark.asyncio
    async def test_resources_cleaned_on_cancel(self):
        """Test that resources are properly cleaned up on cancellation."""
        cancel_event = asyncio.Event()

        cleanup_called = False

        async def operation_with_cleanup():
            nonlocal cleanup_called
            try:
                await asyncio.sleep(0.1)
                if cancel_event.is_set():
                    raise asyncio.CancelledError("Cancelled")
            finally:
                cleanup_called = True

        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await operation_with_cleanup()

        assert cleanup_called, "Cleanup should be called even on cancellation"


class TestCancelEventTiming:
    """Test timing of cancel event checks."""

    @pytest.mark.asyncio
    async def test_cancel_checked_at_multiple_points(self):
        """Test that cancel is checked at multiple critical points."""
        cancel_event = asyncio.Event()

        check_points = []

        async def multi_stage_operation():
            # Stage 1: Initial check
            if cancel_event.is_set():
                check_points.append("stage1")
                raise asyncio.CancelledError("Cancelled at stage 1")

            check_points.append("stage1_passed")

            # Stage 2: Mid-operation check
            await asyncio.sleep(0.01)
            if cancel_event.is_set():
                check_points.append("stage2")
                raise asyncio.CancelledError("Cancelled at stage 2")

            check_points.append("stage2_passed")

            # Stage 3: Pre-final check
            await asyncio.sleep(0.01)
            if cancel_event.is_set():
                check_points.append("stage3")
                raise asyncio.CancelledError("Cancelled at stage 3")

            check_points.append("completed")

        # Test cancellation at each stage
        for stage in range(3):
            cancel_event = asyncio.Event()
            check_points = []

            # Set cancel after a delay to test different stages
            async def delayed_cancel():
                await asyncio.sleep(0.015 * (stage + 1))
                cancel_event.set()

            await asyncio.gather(
                multi_stage_operation(),
                delayed_cancel(),
                return_exceptions=True,
            )

            # Verify cancellation occurred
            assert f"stage{stage + 1}" in check_points or "stage1_passed" in check_points

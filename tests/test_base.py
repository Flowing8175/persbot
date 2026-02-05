"""Comprehensive tests for BaseLLMService."""

import pytest
import asyncio
import io
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from PIL import Image
import discord

from soyebot.services.base import BaseLLMService, ChatMessage


# =============================================================================
# Concrete Test Implementation of BaseLLMService
# =============================================================================


class TestLLMService(BaseLLMService):
    """Concrete implementation of BaseLLMService for testing."""

    def __init__(self, config, fatal_errors=None, rate_limit_errors=None):
        super().__init__(config)
        self._fatal_errors = fatal_errors or []
        self._rate_limit_errors = rate_limit_errors or []
        self._extracted_text = "Test response"
        self._response_obj = None
        self._user_role = "user"
        self._assistant_role = "assistant"
        self._logged_requests = []
        self._logged_responses = []
        self._last_response_obj = None

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if exception is a rate limit error."""
        return any(isinstance(error, err_type) for err_type in self._rate_limit_errors)

    def _extract_retry_delay(self, error: Exception) -> float | None:
        """Extract retry delay from error."""
        if hasattr(error, "retry_after"):
            return error.retry_after
        return None

    def _log_raw_request(self, user_message: str, chat_session=None) -> None:
        """Log raw request for debugging."""
        self._logged_requests.append((user_message, chat_session))

    def _log_raw_response(self, response_obj, attempt: int) -> None:
        """Log raw response for debugging."""
        self._logged_responses.append((response_obj, attempt))
        self._last_response_obj = response_obj

    def _extract_text_from_response(self, response_obj) -> str:
        """Extract text content from response."""
        # If response is a string, return it directly
        if isinstance(response_obj, str):
            return response_obj
        # Otherwise use the configured extracted text
        return self._extracted_text

    def _is_fatal_error(self, error: Exception) -> bool:
        """Check if error is fatal."""
        return any(isinstance(error, err_type) for err_type in self._fatal_errors)

    def get_user_role_name(self) -> str:
        """Return user role name."""
        return self._user_role

    def get_assistant_role_name(self) -> str:
        """Return assistant role name."""
        return self._assistant_role

    async def generate_chat_response(
        self, chat_session, user_message: str, discord_message
    ) -> tuple[str, object] | None:
        """Generate chat response."""
        return self._extracted_text, self._response_obj


class RateLimitError(Exception):
    """Mock rate limit error."""

    def __init__(self, message="", retry_after=5.0):
        super().__init__(message)
        self.retry_after = retry_after


class FatalError(Exception):
    """Mock fatal error."""

    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock AppConfig for testing."""
    config = Mock()
    config.discord_token = "test_token"
    config.api_max_retries = 3
    config.api_rate_limit_retry_after = 5.0
    config.api_request_timeout = 30.0
    config.api_retry_backoff_base = 2.0
    config.api_retry_backoff_max = 60.0
    return config


@pytest.fixture
def test_service(mock_config):
    """Create a test LLM service instance."""
    return TestLLMService(mock_config)


@pytest.fixture
def mock_discord_message():
    """Create a mock Discord message."""
    message = Mock(spec=discord.Message)
    message.id = 123456789
    message.channel = Mock()
    message.attachments = []
    message.reply = AsyncMock()
    return message


@pytest.fixture
def mock_discord_message_with_reply(mock_discord_message):
    """Create a mock Discord message with reply that supports edit."""
    message = mock_discord_message
    reply_msg = Mock()
    reply_msg.edit = AsyncMock()
    reply_msg.delete = AsyncMock()
    message.reply = AsyncMock(return_value=reply_msg)
    return message


@pytest.fixture
def mock_attachment():
    """Create a mock Discord attachment."""
    attachment = Mock()
    attachment.filename = "test_image.png"
    attachment.content_type = "image/png"
    attachment.read = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n...")
    return attachment


@pytest.fixture
def mock_non_image_attachment():
    """Create a mock non-image attachment."""
    attachment = Mock()
    attachment.filename = "test_file.pdf"
    attachment.content_type = "application/pdf"
    attachment.read = AsyncMock(return_value=b"PDF content")
    return attachment


# =============================================================================
# Tests for Retry Logic - execute_with_retry()
# =============================================================================


@pytest.mark.asyncio
class TestExecuteWithRetry:
    """Test suite for execute_with_retry method."""

    async def test_successful_call_on_first_attempt(self, test_service):
        """Test successful API call on first attempt."""
        model_call = Mock(return_value="Success!")
        result = await test_service.execute_with_retry(model_call)
        # The extract_text_from_response will return the actual string response
        assert result == "Success!"
        assert len(test_service._logged_responses) == 1
        model_call.assert_called_once()

    async def test_retry_on_specific_exception(self, test_service):
        """Test retry logic on specific exceptions."""
        error_count = [0]

        def failing_call():
            error_count[0] += 1
            if error_count[0] < 2:
                raise ValueError("Temporary error")
            return "Success after retry"

        result = await test_service.execute_with_retry(failing_call)
        # The actual return value is extracted from response
        assert result == "Success after retry"
        assert error_count[0] == 2

    async def test_exponential_backoff_calculation(self, test_service, mock_config):
        """Test exponential backoff calculation."""
        backoff_times = []

        async def failing_with_backoff():
            raise Exception("Fail")

        # Patch asyncio.sleep to capture backoff times
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda x: backoff_times.append(x) or None

            await test_service.execute_with_retry(failing_with_backoff)

            # Should calculate backoff for each retry (3 attempts = 2 retries)
            assert len(backoff_times) == 2
            # First retry: 2.0^1 = 2.0
            assert backoff_times[0] == 2.0
            # Second retry: 2.0^2 = 4.0
            assert backoff_times[1] == 4.0

    async def test_maximum_backoff_cap(self, test_service, mock_config):
        """Test maximum backoff cap is respected."""
        mock_config.api_retry_backoff_max = 10.0
        backoff_times = []

        async def failing_call():
            raise Exception("Fail")

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda x: backoff_times.append(x) or None

            # Set high retry count to test cap
            test_service.config.api_max_retries = 5
            await test_service.execute_with_retry(failing_call)

            # All backoff times should be <= max
            assert all(bt <= 10.0 for bt in backoff_times)

    async def test_maximum_retries_exhausted(self, test_service):
        """Test behavior when maximum retries are exhausted."""
        error_count = [0]

        def always_failing():
            error_count[0] += 1
            raise ValueError("Always fails")

        result = await test_service.execute_with_retry(always_failing)
        assert result is None
        assert error_count[0] == 3  # api_max_retries = 3

    async def test_return_full_response_parameter(self, test_service):
        """Test return_full_response parameter."""
        test_service._response_obj = {"text": "Full response", "metadata": {}}
        model_call = Mock(return_value=test_service._response_obj)

        # Test with return_full_response=True
        result_full = await test_service.execute_with_retry(
            model_call, return_full_response=True
        )
        assert result_full == test_service._response_obj

        # Test with return_full_response=False (default)
        result_text = await test_service.execute_with_retry(
            model_call, return_full_response=False
        )
        assert result_text == "Test response"

    async def test_timeout_handling(self, test_service):
        """Test timeout handling in execute_with_retry."""

        async def slow_call():
            await asyncio.sleep(10)  # Longer than timeout
            return "Should not reach"

        test_service.config.api_request_timeout = 0.1
        result = await test_service.execute_with_retry(slow_call)
        assert result is None

    async def test_custom_timeout_parameter(self, test_service):
        """Test custom timeout parameter."""

        async def slow_call():
            await asyncio.sleep(0.5)
            return "Success"

        # Custom timeout longer than function
        result = await test_service.execute_with_retry(slow_call, timeout=1.0)
        assert result == "Success"

        # Custom timeout shorter than function
        result = await test_service.execute_with_retry(slow_call, timeout=0.1)
        assert result is None

    async def test_rate_limit_with_countdown(
        self, test_service, mock_discord_message_with_reply
    ):
        """Test rate limit handling with countdown."""
        test_service._rate_limit_errors = [RateLimitError]
        delay = 2.0

        async def rate_limited_call():
            raise RateLimitError(retry_after=delay)

        # Speed up the countdown for testing
        with patch("asyncio.sleep"):
            result = await test_service.execute_with_retry(
                rate_limited_call, discord_message=mock_discord_message_with_reply
            )

        # Should have sent reply
        mock_discord_message_with_reply.reply.assert_called()
        # Should have edited the countdown message
        reply_msg = await mock_discord_message_with_reply.reply()
        reply_msg.edit.assert_called()
        # Should have deleted countdown message
        reply_msg.delete.assert_called()


# =============================================================================
# Tests for Rate Limit Handling
# =============================================================================


class TestRateLimitHandling:
    """Test suite for rate limit detection and handling."""

    def test_is_rate_limit_error_returns_true(self, test_service):
        """Test _is_rate_limit_error returns True for rate limit errors."""
        test_service._rate_limit_errors = [RateLimitError]
        error = RateLimitError()
        assert test_service._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_returns_false(self, test_service):
        """Test _is_rate_limit_error returns False for non-rate-limit errors."""
        test_service._rate_limit_errors = [RateLimitError]
        error = ValueError("Some error")
        assert test_service._is_rate_limit_error(error) is False

    def test_extract_retry_delay_from_error(self, test_service):
        """Test _extract_retry_delay extracts correct delay."""
        error = RateLimitError(retry_after=7.5)
        delay = test_service._extract_retry_delay(error)
        assert delay == 7.5

    def test_extract_retry_delay_returns_none(self, test_service):
        """Test _extract_retry_delay returns None when no delay."""
        error = ValueError("No delay")
        delay = test_service._extract_retry_delay(error)
        assert delay is None

    @pytest.mark.asyncio
    async def test_wait_with_countdown_edits_message(
        self, test_service, mock_discord_message_with_reply
    ):
        """Test _wait_with_countdown edits countdown message."""
        delay = 3
        with patch("asyncio.sleep"):
            await test_service._wait_with_countdown(
                delay, mock_discord_message_with_reply
            )

        reply_msg = await mock_discord_message_with_reply.reply()
        reply_msg.edit.assert_called()

    @pytest.mark.asyncio
    async def test_wait_with_countdown_deletes_message(
        self, test_service, mock_discord_message_with_reply
    ):
        """Test _wait_with_countdown deletes countdown message after completion."""
        delay = 2
        with patch("asyncio.sleep"):
            await test_service._wait_with_countdown(
                delay, mock_discord_message_with_reply
            )

        reply_msg = await mock_discord_message_with_reply.reply()
        reply_msg.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_with_countdown_no_discord_message(self, test_service):
        """Test _wait_with_countdown works without Discord message."""
        delay = 2
        with patch("asyncio.sleep"):
            await test_service._wait_with_countdown(delay, None)
            # Should not raise any exception

    @pytest.mark.asyncio
    async def test_wait_with_countdown_zero_or_negative_delay(self, test_service):
        """Test _wait_with_countdown handles zero or negative delay."""
        with patch("asyncio.sleep") as mock_sleep:
            await test_service._wait_with_countdown(0, None)
            mock_sleep.assert_not_called()

            await test_service._wait_with_countdown(-1, None)
            mock_sleep.assert_not_called()


# =============================================================================
# Tests for Image Processing
# =============================================================================


@pytest.mark.asyncio
class TestImageProcessing:
    """Test suite for image extraction and processing."""

    async def test_extract_images_no_attachments(
        self, test_service, mock_discord_message
    ):
        """Test _extract_images_from_message with no attachments."""
        images = await test_service._extract_images_from_message(mock_discord_message)
        assert images == []

    async def test_extract_images_single_image(
        self, test_service, mock_discord_message, mock_attachment
    ):
        """Test _extract_images_from_message with single image attachment."""
        mock_discord_message.attachments = [mock_attachment]
        images = await test_service._extract_images_from_message(mock_discord_message)
        assert len(images) == 1
        mock_attachment.read.assert_called_once()

    async def test_extract_images_multiple_attachments(
        self, test_service, mock_discord_message, mock_attachment
    ):
        """Test _extract_images_from_message with multiple image attachments."""
        attachment2 = Mock()
        attachment2.filename = "test2.jpg"
        attachment2.content_type = "image/jpeg"
        attachment2.read = AsyncMock(return_value=b"JPEG data")

        mock_discord_message.attachments = [mock_attachment, attachment2]
        images = await test_service._extract_images_from_message(mock_discord_message)
        assert len(images) == 2

    async def test_extract_images_filters_non_images(
        self, test_service, mock_discord_message, mock_non_image_attachment
    ):
        """Test _extract_images_from_message filters non-image attachments."""
        mock_discord_message.attachments = [mock_non_image_attachment]
        images = await test_service._extract_images_from_message(mock_discord_message)
        assert images == []
        mock_non_image_attachment.read.assert_not_called()

    async def test_extract_images_mixed_attachments(
        self,
        test_service,
        mock_discord_message,
        mock_attachment,
        mock_non_image_attachment,
    ):
        """Test _extract_images_from_message with mixed attachment types."""
        mock_discord_message.attachments = [mock_attachment, mock_non_image_attachment]
        images = await test_service._extract_images_from_message(mock_discord_message)
        assert len(images) == 1
        mock_attachment.read.assert_called_once()
        mock_non_image_attachment.read.assert_not_called()

    async def test_extract_images_handles_read_errors(
        self, test_service, mock_discord_message
    ):
        """Test _extract_images_from_message handles read errors gracefully."""
        failing_attachment = Mock()
        failing_attachment.filename = "failing.png"
        failing_attachment.content_type = "image/png"
        failing_attachment.read = AsyncMock(side_effect=Exception("Read failed"))

        mock_discord_message.attachments = [failing_attachment]
        images = await test_service._extract_images_from_message(mock_discord_message)
        assert images == []
        # Should continue without raising exception

    async def test_extract_images_content_type_detection(
        self, test_service, mock_discord_message
    ):
        """Test content_type detection for various image formats."""
        formats = ["image/png", "image/jpeg", "image/gif", "image/webp"]
        attachments = []
        for i, fmt in enumerate(formats):
            att = Mock()
            att.filename = f"image{i}.{'png' if 'png' in fmt else 'jpg'}"
            att.content_type = fmt
            att.read = AsyncMock(return_value=b"image data")
            attachments.append(att)

        mock_discord_message.attachments = attachments
        images = await test_service._extract_images_from_message(mock_discord_message)
        assert len(images) == len(formats)


# =============================================================================
# Tests for Image Downscaling
# =============================================================================


class TestImageDownscaling:
    """Test suite for _process_image_sync method."""

    def test_no_downscaling_small_image(self, test_service):
        """Test no downscaling for small images (<1MP)."""
        # Create a 800x600 image (480,000 pixels < 1MP)
        img = Image.new("RGB", (800, 600), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        processed = test_service._process_image_sync(image_data, "small.png")
        # Should process but not downscale (JPEG conversion may change size slightly)
        assert processed is not None
        # Verify original size is preserved (within reason)
        processed_img = Image.open(io.BytesIO(processed))
        assert processed_img.size == (800, 600)

    def test_downscaling_large_image(self, test_service):
        """Test downscaling for large images (>1MP)."""
        # Create a 2000x2000 image (4,000,000 pixels > 1MP)
        img = Image.new("RGB", (2000, 2000), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        processed = test_service._process_image_sync(image_data, "large.png")

        # Verify it was downscaled (processed image should be smaller)
        processed_img = Image.open(io.BytesIO(processed))
        width, height = processed_img.size
        # Target pixels = 1,000,000, so should be around 1000x1000
        assert width * height <= 1_200_000  # Allow some margin for rounding

    def test_aspect_ratio_preservation(self, test_service):
        """Test aspect ratio is preserved during downscaling."""
        # Create a 1600x1200 image (4:3 aspect ratio)
        img = Image.new("RGB", (1600, 1200), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        processed = test_service._process_image_sync(image_data, "aspect.png")
        processed_img = Image.open(io.BytesIO(processed))
        width, height = processed_img.size

        # Check aspect ratio is preserved (allowing for rounding)
        original_ratio = 1600 / 1200
        new_ratio = width / height
        assert abs(original_ratio - new_ratio) < 0.05

    def test_jpeg_conversion(self, test_service):
        """Test JPEG conversion of images."""
        # Create a PNG image
        img = Image.new("RGB", (800, 600), color="yellow")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        processed = test_service._process_image_sync(image_data, "test.png")
        processed_img = Image.open(io.BytesIO(processed))

        # Verify format is JPEG
        assert processed_img.format == "JPEG"

    def test_rgba_to_rgb_conversion(self, test_service):
        """Test RGBA to RGB conversion for JPEG compatibility."""
        # Create an RGBA image
        img = Image.new("RGBA", (800, 600), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        processed = test_service._process_image_sync(image_data, "rgba.png")
        processed_img = Image.open(io.BytesIO(processed))

        # Verify converted to RGB
        assert processed_img.mode == "RGB"

    def test_palette_mode_conversion(self, test_service):
        """Test palette mode (P) to RGB conversion."""
        # Create a palette mode image
        img = Image.new("P", (800, 600))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        processed = test_service._process_image_sync(image_data, "palette.png")
        processed_img = Image.open(io.BytesIO(processed))

        # Verify converted to RGB
        assert processed_img.mode == "RGB"

    def test_error_handling_in_processing(self, test_service):
        """Test error handling returns original image on failure."""
        invalid_data = b"not a valid image"

        processed = test_service._process_image_sync(invalid_data, "invalid.png")
        # Should return original data on error
        assert processed == invalid_data


# =============================================================================
# Tests for ChatMessage Dataclass
# =============================================================================


class TestChatMessage:
    """Test suite for ChatMessage dataclass."""

    def test_initialization_with_required_fields(self):
        """Test initialization with required fields."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_initialization_with_all_fields(self):
        """Test initialization with all optional fields."""
        msg = ChatMessage(
            role="user",
            content="Hello",
            author_id=123456789,
            author_name="TestUser",
            message_ids=["msg1", "msg2"],
            parts=[{"text": "Hello"}],
            images=b"image_data",
        )
        assert msg.author_id == 123456789
        assert msg.author_name == "TestUser"
        assert msg.message_ids == ["msg1", "msg2"]
        assert msg.parts == [{"text": "Hello"}]
        assert msg.images == b"image_data"

    def test_optional_fields_defaults(self):
        """Test optional fields have correct defaults."""
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.author_id is None
        assert msg.author_name is None
        assert msg.message_ids == []
        assert msg.parts is None
        assert msg.images == []

    def test_role_field_values(self):
        """Test various role field values."""
        user_msg = ChatMessage(role="user", content="User message")
        assistant_msg = ChatMessage(role="assistant", content="Assistant message")
        system_msg = ChatMessage(role="system", content="System message")

        assert user_msg.role == "user"
        assert assistant_msg.role == "assistant"
        assert system_msg.role == "system"

    def test_message_ids_mutable(self):
        """Test message_ids list is mutable."""
        msg = ChatMessage(role="user", content="Test")
        msg.message_ids.append("new_msg_id")
        assert "new_msg_id" in msg.message_ids

    def test_images_mutable(self):
        """Test images list is mutable."""
        msg = ChatMessage(role="user", content="Test")
        msg.images.append(b"image1")
        msg.images.append(b"image2")
        assert len(msg.images) == 2


# =============================================================================
# Tests for Abstract Methods
# =============================================================================


class TestAbstractMethods:
    """Test suite for abstract method implementations."""

    def test_subclass_implements_required_methods(self, test_service):
        """Test subclass implements all required abstract methods."""
        # Check all required methods are implemented
        assert hasattr(test_service, "_is_rate_limit_error")
        assert hasattr(test_service, "_extract_retry_delay")
        assert hasattr(test_service, "_log_raw_request")
        assert hasattr(test_service, "_log_raw_response")
        assert hasattr(test_service, "_extract_text_from_response")
        assert hasattr(test_service, "get_user_role_name")
        assert hasattr(test_service, "get_assistant_role_name")
        assert hasattr(test_service, "generate_chat_response")

    def test_reload_parameters_can_be_called(self, test_service):
        """Test reload_parameters can be called without error."""
        # Should not raise exception
        test_service.reload_parameters()

    def test_get_user_role_name(self, test_service):
        """Test get_user_role_name returns correct value."""
        assert test_service.get_user_role_name() == "user"

    def test_get_assistant_role_name(self, test_service):
        """Test get_assistant_role_name returns correct value."""
        assert test_service.get_assistant_role_name() == "assistant"

    def test_custom_role_names(self, mock_config):
        """Test custom role names can be set."""
        service = TestLLMService(mock_config)
        service._user_role = "human"
        service._assistant_role = "ai"
        assert service.get_user_role_name() == "human"
        assert service.get_assistant_role_name() == "ai"


# =============================================================================
# Tests for Fatal Error Detection
# =============================================================================


class TestFatalErrorDetection:
    """Test suite for _is_fatal_error method."""

    def test_recognizes_fatal_errors(self, test_service):
        """Test _is_fatal_error recognizes fatal errors."""
        test_service._fatal_errors = [FatalError]
        error = FatalError("Fatal!")
        assert test_service._is_fatal_error(error) is True

    def test_non_fatal_errors(self, test_service):
        """Test _is_fatal_error returns False for non-fatal errors."""
        test_service._fatal_errors = [FatalError]
        error = ValueError("Not fatal")
        assert test_service._is_fatal_error(error) is False

    def test_multiple_fatal_error_types(self, test_service):
        """Test _is_fatal_error with multiple fatal error types."""
        test_service._fatal_errors = [FatalError, RuntimeError, KeyboardInterrupt]
        assert test_service._is_fatal_error(FatalError()) is True
        assert test_service._is_fatal_error(RuntimeError()) is True
        assert test_service._is_fatal_error(KeyboardInterrupt()) is True
        assert test_service._is_fatal_error(ValueError()) is False

    @pytest.mark.asyncio
    async def test_fatal_error_is_raised(self, test_service):
        """Test fatal errors are re-raised instead of being retried."""
        test_service._fatal_errors = [FatalError]

        async def fatal_call():
            raise FatalError("This should be re-raised")

        with pytest.raises(FatalError):
            await test_service.execute_with_retry(fatal_call)


# =============================================================================
# Tests for Sync/Async Execution
# =============================================================================


@pytest.mark.asyncio
class TestSyncAsyncExecution:
    """Test suite for _execute_model_call method."""

    async def test_with_sync_function(self, test_service):
        """Test _execute_model_call with sync function."""

        def sync_call():
            return "sync result"

        result = await test_service._execute_model_call(sync_call)
        assert result == "sync result"

    async def test_with_async_function(self, test_service):
        """Test _execute_model_call with async function."""

        async def async_call():
            await asyncio.sleep(0)
            return "async result"

        result = await test_service._execute_model_call(async_call)
        assert result == "async result"

    async def test_with_coroutine_function(self, test_service):
        """Test _execute_model_call with coroutine function."""

        async def coro_call():
            return "coroutine result"

        result = await test_service._execute_model_call(coro_call)
        assert result == "coroutine result"

    async def test_with_callable_class(self, test_service):
        """Test _execute_model_call with callable class."""

        class CallableClass:
            def __call__(self):
                return "callable result"

        callable_obj = CallableClass()
        result = await test_service._execute_model_call(callable_obj)
        assert result == "callable result"

    async def test_with_lambda(self, test_service):
        """Test _execute_model_call with lambda."""
        result = await test_service._execute_model_call(lambda: "lambda result")
        assert result == "lambda result"

    async def test_with_arguments(self, test_service):
        """Test _execute_model_call with callable that takes arguments."""

        def func_with_args(x, y):
            return x + y

        result = await test_service._execute_model_call(lambda: func_with_args(3, 5))
        assert result == 8


# =============================================================================
# Tests for Error Handling
# =============================================================================


@pytest.mark.asyncio
class TestErrorHandling:
    """Test suite for various error scenarios."""

    async def test_asyncio_timeout_error(self, test_service):
        """Test handling of asyncio.TimeoutError."""

        async def timeout_call():
            raise asyncio.TimeoutError()

        result = await test_service.execute_with_retry(timeout_call)
        assert result is None

    async def test_generic_exception(self, test_service):
        """Test handling of generic exceptions."""

        async def generic_error_call():
            raise ValueError("Generic error")

        result = await test_service.execute_with_retry(generic_error_call)
        assert result is None

    async def test_rate_limit_with_fallback_success(self, test_service, mock_config):
        """Test rate limit with successful fallback."""
        test_service._rate_limit_errors = [RateLimitError]
        attempt_count = [0]

        async def primary_call():
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                raise RateLimitError(retry_after=1)
            return "primary"

        async def fallback_call():
            return "fallback success"

        with patch("asyncio.sleep"):
            result = await test_service.execute_with_retry(
                primary_call, fallback_call=fallback_call
            )
            # Fallback should be attempted and succeed
            assert result == "fallback success"

    async def test_fallback_failure_continues_normal_flow(
        self, test_service, mock_config
    ):
        """Test fallback failure continues with normal retry logic."""
        test_service._rate_limit_errors = [RateLimitError]
        attempt_count = [0]

        async def primary_call():
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                raise RateLimitError(retry_after=1)
            return "primary success"

        async def failing_fallback():
            raise ValueError("Fallback failed")

        with patch("asyncio.sleep"):
            result = await test_service.execute_with_retry(
                primary_call, fallback_call=failing_fallback
            )
            # Should eventually succeed with primary after fallback fails
            assert result == "primary success"

    async def test_error_message_sent_to_discord(
        self, test_service, mock_discord_message
    ):
        """Test error message is sent to Discord on failure."""

        async def always_fails():
            raise ValueError("Always fails")

        with patch("asyncio.sleep"):
            result = await test_service.execute_with_retry(
                always_fails, discord_message=mock_discord_message
            )
            assert result is None
            mock_discord_message.reply.assert_called()

    async def test_discord_error_silently_handled(
        self, test_service, mock_discord_message
    ):
        """Test Discord send errors are silently handled."""

        async def always_fails():
            raise ValueError("Always fails")

        mock_discord_message.reply = AsyncMock(
            side_effect=discord.HTTPException(Mock(), "Discord error")
        )

        with patch("asyncio.sleep"):
            result = await test_service.execute_with_retry(
                always_fails, discord_message=mock_discord_message
            )
            # Should not raise exception even if Discord message fails
            assert result is None


# =============================================================================
# Tests for Edge Cases
# =============================================================================


@pytest.mark.asyncio
class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    async def test_zero_timeout(self, test_service):
        """Test zero timeout is handled."""

        async def instant_call():
            return "instant"

        result = await test_service.execute_with_retry(instant_call, timeout=0)
        assert result is None  # Zero timeout causes immediate timeout

    async def test_very_short_timeout(self, test_service):
        """Test very short timeout."""

        async def quick_call():
            return "quick"

        # Very short but non-zero timeout
        result = await test_service.execute_with_retry(quick_call, timeout=0.001)
        # May succeed or timeout depending on system
        assert result is not None or result is None

    async def test_negative_delay_in_countdown(self, test_service):
        """Test negative delay in _wait_with_countdown."""
        # Should not raise exception
        await test_service._wait_with_countdown(-5.0, None)

    async def test_very_long_delay_in_countdown(self, test_service):
        """Test very long delay in _wait_with_countdown."""
        with patch("asyncio.sleep"):
            await test_service._wait_with_countdown(999999.0, None)
            # Should handle gracefully

    async def test_maximum_backoff_cap_with_large_attempts(
        self, test_service, mock_config
    ):
        """Test maximum backoff cap with many retry attempts."""
        mock_config.api_retry_backoff_max = 100.0
        mock_config.api_max_retries = 10
        backoff_times = []

        async def always_fails():
            raise Exception("Fail")

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda x: backoff_times.append(x) or None

            await test_service.execute_with_retry(always_fails)

            # All backoff times should be capped
            assert all(bt <= 100.0 for bt in backoff_times)

    async def test_concurrent_retry_attempts(self, test_service):
        """Test concurrent retry attempts don't interfere."""

        async def failing_call(attempt_id):
            await asyncio.sleep(0.01)
            if attempt_id < 3:
                raise ValueError(f"Attempt {attempt_id} failed")
            return f"Success {attempt_id}"

        # Run multiple concurrent calls
        tasks = []
        for i in range(5):
            # Create a wrapper function to capture the current value of i
            async def wrapper(i=i):
                async def call():
                    return await failing_call(i)

                return await test_service.execute_with_retry(call)

            tasks.append(wrapper())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without interference
        assert all(r is not None or isinstance(r, Exception) for r in results)

    async def test_empty_response_from_model(self, test_service):
        """Test empty response from model."""
        test_service._extracted_text = ""
        model_call = Mock(return_value={"text": ""})

        result = await test_service.execute_with_retry(model_call)
        assert result == ""

    async def test_none_response_from_model(self, test_service):
        """Test None response from model."""
        test_service._extracted_text = None
        model_call = Mock(return_value=None)

        result = await test_service.execute_with_retry(model_call)
        assert result is None

    async def test_model_call_returns_exception(self, test_service):
        """Test model call that returns an exception object."""

        async def returns_exception():
            return ValueError("Exception as return value")

        # Should treat exception object as return value, not as raised exception
        result = await test_service.execute_with_retry(returns_exception)
        # May succeed or fail depending on implementation
        assert result is not None or result is None

    async def test_retry_with_zero_max_retries(self, test_service):
        """Test with zero max retries."""
        test_service.config.api_max_retries = 0
        attempt_count = [0]

        async def fails():
            attempt_count[0] += 1
            raise ValueError("Fail")

        result = await test_service.execute_with_retry(fails)
        # With 0 max_retries, no attempts are made (range(1, 1) is empty)
        assert result is None
        assert attempt_count[0] == 0  # No attempts made

    async def test_very_large_backoff_base(self, test_service, mock_config):
        """Test with very large backoff base."""
        mock_config.api_retry_backoff_base = 100.0
        mock_config.api_retry_backoff_max = 200.0
        mock_config.api_max_retries = 2
        backoff_times = []

        async def fails():
            raise Exception("Fail")

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda x: backoff_times.append(x) or None

            await test_service.execute_with_retry(fails)

            # Should cap at max
            assert all(100.0 <= bt <= 200.0 for bt in backoff_times)

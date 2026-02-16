"""Tests for image_service.py module."""

import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from persbot.services.image_service import ImageGenerationError, ImageService


class TestImageGenerationError:
    """Tests for ImageGenerationError exception."""

    def test_is_exception(self):
        """ImageGenerationError is an Exception."""
        assert issubclass(ImageGenerationError, Exception)

    def test_can_be_raised_with_message(self):
        """ImageGenerationError can be raised with a message."""
        with pytest.raises(ImageGenerationError) as exc_info:
            raise ImageGenerationError("Test error message")
        assert "Test error message" in str(exc_info.value)


class TestImageServiceInit:
    """Tests for ImageService initialization."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        config = MagicMock()
        config.openrouter_api_key = "test-api-key"
        config.openrouter_image_model = "test-image-model"
        config.api_request_timeout = 30
        config.api_max_retries = 3
        config.api_retry_backoff_base = 1.0
        config.api_retry_backoff_max = 30.0
        config.api_rate_limit_retry_after = 5
        return config

    def test_init_creates_service(self, mock_config):
        """ImageService initializes with config."""
        service = ImageService(mock_config)
        assert service.config == mock_config
        assert service._client is None
        assert service._retry_handler is None

    def test_get_client_creates_openai_client(self, mock_config):
        """_get_client creates OpenAI client with correct config."""
        with patch("persbot.services.image_service.OpenAI") as mock_openai:
            service = ImageService(mock_config)
            client = service._get_client()

            mock_openai.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1",
                timeout=30,
            )

    def test_get_client_returns_same_instance(self, mock_config):
        """_get_client returns the same client instance on multiple calls."""
        with patch("persbot.services.image_service.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            service = ImageService(mock_config)
            client1 = service._get_client()
            client2 = service._get_client()

            assert client1 is client2
            mock_openai.assert_called_once()


class TestImageServiceRetryConfig:
    """Tests for ImageService retry configuration."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        config = MagicMock()
        config.openrouter_api_key = "test-api-key"
        config.openrouter_image_model = "test-image-model"
        config.api_request_timeout = 60
        config.api_max_retries = 3
        config.api_retry_backoff_base = 2.0
        config.api_retry_backoff_max = 60.0
        config.api_rate_limit_retry_after = 10
        return config

    def test_create_retry_config(self, mock_config):
        """_create_retry_config returns correct config."""
        service = ImageService(mock_config)
        retry_config = service._create_retry_config()

        assert retry_config.max_retries == 2
        assert retry_config.base_delay == 2.0
        assert retry_config.max_delay == 32.0
        assert retry_config.rate_limit_delay == 5
        assert retry_config.request_timeout == 60

    def test_create_retry_handler(self, mock_config):
        """_create_retry_handler returns OpenAIRetryHandler."""
        with patch("persbot.services.retry_handler.OpenAIRetryHandler") as mock_handler:
            service = ImageService(mock_config)
            handler = service._create_retry_handler()

            mock_handler.assert_called_once()


class TestImageServiceRateLimitError:
    """Tests for _is_rate_limit_error method."""

    @pytest.fixture
    def service(self):
        """Create ImageService instance."""
        config = MagicMock()
        config.openrouter_api_key = "test-key"
        config.api_request_timeout = 30
        return ImageService(config)

    def test_detects_rate_limit_in_message(self, service):
        """_is_rate_limit_error detects 'rate limit' in message."""
        error = Exception("Rate limit exceeded")
        assert service._is_rate_limit_error(error) is True

    def test_detects_429_in_message(self, service):
        """_is_rate_limit_error detects '429' in message."""
        error = Exception("Error 429: Too many requests")
        assert service._is_rate_limit_error(error) is True

    def test_detects_rate_limit_error_instance(self, service):
        """_is_rate_limit_error detects RateLimitError instance."""
        from openai import RateLimitError

        error = RateLimitError(
            message="Rate limit",
            response=MagicMock(status_code=429),
            body=None,
        )
        assert service._is_rate_limit_error(error) is True

    def test_returns_false_for_other_errors(self, service):
        """_is_rate_limit_error returns False for other errors."""
        error = Exception("Some other error")
        assert service._is_rate_limit_error(error) is False


class TestImageServiceLogging:
    """Tests for logging methods."""

    @pytest.fixture
    def service(self):
        """Create ImageService instance."""
        config = MagicMock()
        config.openrouter_api_key = "test-key"
        config.api_request_timeout = 30
        return ImageService(config)

    def test_log_raw_request_logs_when_debug_enabled(self, service):
        """_log_raw_request no longer logs (debug logging removed)."""
        with patch("persbot.services.image_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            service._log_raw_request("test prompt")
            # Debug logging has been removed, so debug should not be called
            mock_logger.debug.assert_not_called()

    def test_log_raw_request_skips_when_debug_disabled(self, service):
        """_log_raw_request skips logging when DEBUG level is disabled."""
        with patch("persbot.services.image_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = False
            service._log_raw_request("test prompt")
            mock_logger.debug.assert_not_called()

    def test_log_raw_response_logs_when_debug_enabled(self, service):
        """_log_raw_response no longer logs (debug logging removed)."""
        with patch("persbot.services.image_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            service._log_raw_response({"response": "data"}, attempt=1)
            # Debug logging has been removed, so debug should not be called
            mock_logger.debug.assert_not_called()

    def test_extract_text_from_response_returns_empty(self, service):
        """_extract_text_from_response returns empty string."""
        result = service._extract_text_from_response({"any": "response"})
        assert result == ""


class TestImageServiceRoleNames:
    """Tests for role name methods."""

    @pytest.fixture
    def service(self):
        """Create ImageService instance."""
        config = MagicMock()
        return ImageService(config)

    def test_get_user_role_name(self, service):
        """get_user_role_name returns 'user'."""
        assert service.get_user_role_name() == "user"

    def test_get_assistant_role_name(self, service):
        """get_assistant_role_name returns 'assistant'."""
        assert service.get_assistant_role_name() == "assistant"


class TestImageServiceGenerateImage:
    """Tests for generate_image method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        config = MagicMock()
        config.openrouter_api_key = "test-api-key"
        config.openrouter_image_model = "default-image-model"
        config.api_request_timeout = 30
        config.api_max_retries = 3
        config.api_retry_backoff_base = 1.0
        config.api_retry_backoff_max = 30.0
        config.api_rate_limit_retry_after = 5
        return config

    @pytest.fixture
    def service(self, mock_config):
        """Create ImageService with mocked dependencies."""
        service = ImageService(mock_config)
        service._retry_handler = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_when_cancel_event_set(self, service):
        """generate_image raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await service.generate_image("test prompt", cancel_event=cancel_event)

        assert "aborted by user" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_error_for_empty_prompt(self, service):
        """generate_image raises ImageGenerationError for empty prompt."""
        with pytest.raises(ImageGenerationError) as exc_info:
            await service.generate_image("")

        assert "cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_error_for_whitespace_prompt(self, service):
        """generate_image raises ImageGenerationError for whitespace-only prompt."""
        with pytest.raises(ImageGenerationError) as exc_info:
            await service.generate_image("   ")

        assert "cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_truncates_long_prompt(self, service, mock_config):
        """generate_image truncates prompts longer than 2000 characters."""
        long_prompt = "x" * 2500

        with patch.object(service, "execute_with_retry") as mock_retry:
            mock_retry.return_value = (b"fake image data", "png")

            await service.generate_image(long_prompt)

            # Check that the prompt was truncated
            call_args = mock_retry.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_uses_default_model_when_none_provided(self, service, mock_config):
        """generate_image uses config default model when model is None."""
        with patch.object(service, "execute_with_retry") as mock_retry:
            mock_retry.return_value = (b"image data", "png")

            await service.generate_image("test prompt", model=None)

            # Verify the model used was from config
            mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_custom_model_when_provided(self, service):
        """generate_image uses provided model instead of default."""
        with patch.object(service, "execute_with_retry") as mock_retry:
            mock_retry.return_value = (b"image data", "png")

            await service.generate_image("test prompt", model="custom-model")

            mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_aspect_ratio_to_api(self, service):
        """generate_image passes aspect_ratio to API call."""
        with patch.object(service, "execute_with_retry") as mock_retry:
            mock_retry.return_value = (b"image data", "png")

            await service.generate_image("test prompt", aspect_ratio="16:9")

            mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_image_input_for_image_to_image(self, service):
        """generate_image includes image_input in request for image-to-image."""
        with patch.object(service, "execute_with_retry") as mock_retry:
            mock_retry.return_value = (b"image data", "png")

            await service.generate_image(
                "test prompt",
                image_input="data:image/png;base64,abc123"
            )

            mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_result_is_none(self, service):
        """generate_image returns None when execute_with_retry returns None."""
        with patch.object(service, "execute_with_retry") as mock_retry:
            mock_retry.return_value = None

            result = await service.generate_image("test prompt")

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_image_bytes_and_format(self, service):
        """generate_image returns tuple of (bytes, format)."""
        with patch.object(service, "execute_with_retry") as mock_retry:
            mock_retry.return_value = (b"fake image bytes", "png")

            result = await service.generate_image("test prompt")

            assert result == (b"fake image bytes", "png")

    @pytest.mark.asyncio
    async def test_raises_image_generation_error_on_exception(self, service):
        """generate_image raises ImageGenerationError on API exception."""
        with patch.object(service, "execute_with_retry") as mock_retry:
            mock_retry.side_effect = Exception("API error")

            with pytest.raises(ImageGenerationError) as exc_info:
                await service.generate_image("test prompt")

            assert "Image generation failed" in str(exc_info.value)


class TestImageServiceProcessImageUrl:
    """Tests for _process_image_url method."""

    @pytest.fixture
    def service(self):
        """Create ImageService instance."""
        config = MagicMock()
        return ImageService(config)

    def test_processes_base64_data_url(self, service):
        """_process_image_url handles base64 data URLs."""
        image_data = base64.b64encode(b"fake png data").decode()
        data_url = f"data:image/png;base64,{image_data}"

        result = service._process_image_url(data_url, "hash123", 10)

        assert result == (b"fake png data", "png")

    def test_processes_jpeg_data_url(self, service):
        """_process_image_url handles JPEG data URLs."""
        image_data = base64.b64encode(b"fake jpeg data").decode()
        data_url = f"data:image/jpeg;base64,{image_data}"

        result = service._process_image_url(data_url, "hash123", 10)

        assert result == (b"fake jpeg data", "jpeg")

    def test_processes_webp_data_url(self, service):
        """_process_image_url handles WebP data URLs."""
        image_data = base64.b64encode(b"fake webp data").decode()
        data_url = f"data:image/webp;base64,{image_data}"

        result = service._process_image_url(data_url, "hash123", 10)

        assert result == (b"fake webp data", "webp")

    def test_returns_url_for_http_url(self, service):
        """_process_image_url returns URL tuple for HTTP URLs."""
        http_url = "https://example.com/image.png"

        result = service._process_image_url(http_url, "hash123", 10)

        assert result == (http_url.encode("utf-8"), "url")

    def test_raises_error_for_unknown_format(self, service):
        """_process_image_url raises error for unknown URL format."""
        with pytest.raises(ImageGenerationError) as exc_info:
            service._process_image_url("unknown://invalid", "hash123", 10)

        assert "Invalid image URL format" in str(exc_info.value)


class TestImageServiceDecodeBase64Image:
    """Tests for _decode_base64_image method."""

    @pytest.fixture
    def service(self):
        """Create ImageService instance."""
        config = MagicMock()
        return ImageService(config)

    def test_decodes_png_image(self, service):
        """_decode_base64_image decodes PNG data URL."""
        image_data = base64.b64encode(b"fake png data").decode()
        data_url = f"data:image/png;base64,{image_data}"

        result = service._decode_base64_image(data_url, "hash123", 10)

        assert result == (b"fake png data", "png")

    def test_decodes_jpeg_image(self, service):
        """_decode_base64_image decodes JPEG data URL."""
        image_data = base64.b64encode(b"fake jpeg data").decode()
        data_url = f"data:image/jpeg;base64,{image_data}"

        result = service._decode_base64_image(data_url, "hash123", 10)

        assert result == (b"fake jpeg data", "jpeg")

    def test_decodes_jpg_image(self, service):
        """_decode_base64_image handles image/jpg MIME type."""
        image_data = base64.b64encode(b"fake jpg data").decode()
        data_url = f"data:image/jpg;base64,{image_data}"

        result = service._decode_base64_image(data_url, "hash123", 10)

        assert result == (b"fake jpg data", "jpeg")

    def test_defaults_to_png_for_unknown_format(self, service):
        """_decode_base64_image defaults to PNG for unknown MIME type."""
        image_data = base64.b64encode(b"fake image data").decode()
        data_url = f"data:image/unknown;base64,{image_data}"

        result = service._decode_base64_image(data_url, "hash123", 10)

        assert result == (b"fake image data", "png")

    def test_raises_error_for_invalid_base64(self, service):
        """_decode_base64_image raises error for invalid base64."""
        data_url = "data:image/png;base64,invalid!@#$"

        with pytest.raises(ImageGenerationError) as exc_info:
            service._decode_base64_image(data_url, "hash123", 10)

        assert "Failed to decode" in str(exc_info.value)

    def test_raises_error_for_missing_comma(self, service):
        """_decode_base64_image raises error for malformed data URL."""
        data_url = "data:image/png;base64"  # Missing comma and data

        with pytest.raises(ImageGenerationError) as exc_info:
            service._decode_base64_image(data_url, "hash123", 10)

        assert "Failed to decode" in str(exc_info.value)


class TestImageServiceFetchImageFromUrl:
    """Tests for fetch_image_from_url method."""

    @pytest.fixture
    def service(self):
        """Create ImageService instance."""
        config = MagicMock()
        return ImageService(config)

    @pytest.mark.asyncio
    async def test_fetches_image_from_url(self, service):
        """fetch_image_from_url fetches image from HTTP URL."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"image bytes")

        # get() returns an async context manager
        mock_get_context = AsyncMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_context.__aexit__ = AsyncMock()

        mock_session_instance = AsyncMock()
        mock_session_instance.get = MagicMock(return_value=mock_get_context)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock()

        with patch("persbot.services.image_service.aiohttp.ClientSession") as mock_session:
            mock_session.return_value = mock_session_instance
            result = await service.fetch_image_from_url("https://example.com/image.png")

        assert result == b"image bytes"

    @pytest.mark.asyncio
    async def test_returns_none_for_non_200_status(self, service):
        """fetch_image_from_url returns None for non-200 status."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()

            mock_session.return_value = mock_session_instance

            with patch("persbot.services.image_service.aiohttp.ClientSession", mock_session):
                result = await service.fetch_image_from_url("https://example.com/notfound.png")

            assert result is None

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_when_cancel_event_set(self, service):
        """fetch_image_from_url raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service.fetch_image_from_url(
                "https://example.com/image.png",
                cancel_event=cancel_event
            )

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, service):
        """fetch_image_from_url returns None on exception."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.side_effect = Exception("Network error")

            with patch("persbot.services.image_service.aiohttp.ClientSession", mock_session):
                result = await service.fetch_image_from_url("https://example.com/image.png")

            assert result is None


class TestImageServiceGenerateImageWithFetch:
    """Tests for generate_image_with_fetch method."""

    @pytest.fixture
    def service(self):
        """Create ImageService with mocked methods."""
        config = MagicMock()
        config.openrouter_api_key = "test-key"
        config.openrouter_image_model = "test-model"
        config.api_request_timeout = 30
        service = ImageService(config)
        return service

    @pytest.mark.asyncio
    async def test_returns_none_when_generate_returns_none(self, service):
        """generate_image_with_fetch returns None when generate_image returns None."""
        with patch.object(service, "generate_image", return_value=None):
            result = await service.generate_image_with_fetch("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_bytes_for_base64_result(self, service):
        """generate_image_with_fetch returns bytes for base64 result."""
        with patch.object(service, "generate_image", return_value=(b"image bytes", "png")):
            result = await service.generate_image_with_fetch("test prompt")
            assert result == b"image bytes"

    @pytest.mark.asyncio
    async def test_fetches_url_result(self, service):
        """generate_image_with_fetch fetches URL result."""
        url = "https://example.com/image.png"

        with patch.object(service, "generate_image", return_value=(url.encode(), "url")):
            with patch.object(service, "fetch_image_from_url", return_value=b"fetched bytes"):
                result = await service.generate_image_with_fetch("test prompt")
                assert result == b"fetched bytes"


class TestImageServiceNotImplementedMethods:
    """Tests for NotImplementedError methods."""

    @pytest.fixture
    def service(self):
        """Create ImageService instance."""
        config = MagicMock()
        return ImageService(config)

    @pytest.mark.asyncio
    async def test_generate_chat_response_raises_not_implemented(self, service):
        """generate_chat_response raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            await service.generate_chat_response(None, "message", None)

        assert "does not support chat" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_tool_results_raises_not_implemented(self, service):
        """send_tool_results raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            await service.send_tool_results(None, [], None)

        assert "does not support tool" in str(exc_info.value)

    def test_get_tools_for_provider_returns_empty_list(self, service):
        """get_tools_for_provider returns empty list."""
        result = service.get_tools_for_provider([{"tool": "definition"}])
        assert result == []

    def test_extract_function_calls_returns_empty_list(self, service):
        """extract_function_calls returns empty list."""
        result = service.extract_function_calls({"response": "data"})
        assert result == []

    def test_format_function_results_returns_empty_list(self, service):
        """format_function_results returns empty list."""
        result = service.format_function_results([{"name": "tool", "result": "data"}])
        assert result == []


class TestImageServiceExecuteImageGeneration:
    """Tests for _execute_image_generation method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        config = MagicMock()
        config.openrouter_api_key = "test-api-key"
        config.openrouter_image_model = "test-model"
        config.api_request_timeout = 30
        return config

    @pytest.fixture
    def service(self, mock_config):
        """Create ImageService with mocked client."""
        service = ImageService(mock_config)
        return service

    def test_raises_error_for_no_choices(self, service):
        """_execute_image_generation raises error when no choices."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response

        service._client = mock_client

        with pytest.raises(ImageGenerationError) as exc_info:
            service._execute_image_generation(
                "test-model",
                "test prompt",
                {"aspect_ratio": "1:1"},
                "hash123",
                10
            )

        assert "No image generated" in str(exc_info.value)

    def test_raises_error_for_no_images_in_message(self, service):
        """_execute_image_generation raises error when no images in message."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        del mock_message.images  # No images attribute
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response

        service._client = mock_client

        with pytest.raises(ImageGenerationError) as exc_info:
            service._execute_image_generation(
                "test-model",
                "test prompt",
                {"aspect_ratio": "1:1"},
                "hash123",
                10
            )

        assert "No image generated" in str(exc_info.value)

    def test_raises_error_for_empty_images_list(self, service):
        """_execute_image_generation raises error when images list is empty."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.images = []
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response

        service._client = mock_client

        with pytest.raises(ImageGenerationError) as exc_info:
            service._execute_image_generation(
                "test-model",
                "test prompt",
                {"aspect_ratio": "1:1"},
                "hash123",
                10
            )

        assert "No image generated" in str(exc_info.value)

    def test_processes_dict_image_object(self, service):
        """_execute_image_generation processes dict image object."""
        image_data = base64.b64encode(b"test image").decode()
        data_url = f"data:image/png;base64,{image_data}"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.images = [{"image_url": {"url": data_url}}]
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response

        service._client = mock_client

        result = service._execute_image_generation(
            "test-model",
            "test prompt",
            {"aspect_ratio": "1:1"},
            "hash123",
            10
        )

        assert result == (b"test image", "png")

    def test_processes_object_image_object(self, service):
        """_execute_image_generation processes object with attributes."""
        image_data = base64.b64encode(b"test image").decode()
        data_url = f"data:image/png;base64,{image_data}"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_image_obj = MagicMock()
        mock_image_obj.image_url.url = data_url
        mock_message.images = [mock_image_obj]
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response

        service._client = mock_client

        result = service._execute_image_generation(
            "test-model",
            "test prompt",
            {"aspect_ratio": "1:1"},
            "hash123",
            10
        )

        assert result == (b"test image", "png")

    def test_calls_api_with_correct_parameters(self, service):
        """_execute_image_generation calls API with correct parameters."""
        image_data = base64.b64encode(b"test image").decode()
        data_url = f"data:image/png;base64,{image_data}"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.images = [{"image_url": {"url": data_url}}]
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response

        service._client = mock_client

        service._execute_image_generation(
            "test-model",
            "test prompt content",
            {"aspect_ratio": "16:9"},
            "hash123",
            18
        )

        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "test prompt content"}],
            modalities=["image"],
            extra_body={"image_config": {"aspect_ratio": "16:9"}}
        )

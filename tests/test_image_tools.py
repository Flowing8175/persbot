"""Tests for image_tools.py"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persbot.tools.api_tools.image_tools import generate_image, send_image, _get_image_service, _download_and_convert_image, _detect_mime_type
from persbot.tools.base import ToolResult


@pytest.mark.asyncio
async def test_generate_image_success():
    """Test successful image generation with ImageService."""
    # Mock ImageService
    mock_service = AsyncMock()
    mock_service.generate_image_with_fetch = AsyncMock(return_value=b"fake_image_bytes")

    # Mock config
    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    # Mock rate limiter
    mock_limiter = MagicMock()
    mock_rate_result = MagicMock()
    mock_rate_result.allowed = True
    mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
            with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                result = await generate_image("test prompt")

    # Verify success
    assert result.success is True
    assert result.data == "Image generated successfully"
    assert result.metadata.get("image_bytes") == b"fake_image_bytes"

    # Verify ImageService was called
    mock_service.generate_image_with_fetch.assert_called_once()


@pytest.mark.asyncio
async def test_generate_image_empty_prompt():
    """Test that empty prompt returns error."""
    result = await generate_image("")
    assert result.success is False
    assert "empty" in result.error.lower()


@pytest.mark.asyncio
async def test_generate_image_with_aspect_ratio():
    """Test image generation with custom aspect ratio."""
    mock_service = AsyncMock()
    mock_service.generate_image_with_fetch = AsyncMock(return_value=b"fake_image_bytes")

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    mock_limiter = MagicMock()
    mock_rate_result = MagicMock()
    mock_rate_result.allowed = True
    mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
            with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                result = await generate_image("test prompt", aspect_ratio="16:9")

    # Verify service was called with correct aspect ratio
    mock_service.generate_image_with_fetch.assert_called_once()
    call_kwargs = mock_service.generate_image_with_fetch.call_args[1]
    assert call_kwargs["aspect_ratio"] == "16:9"


@pytest.mark.asyncio
async def test_generate_image_with_model():
    """Test image generation with custom model."""
    mock_service = AsyncMock()
    mock_service.generate_image_with_fetch = AsyncMock(return_value=b"fake_image_bytes")

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "default-model"
    mock_config.api_request_timeout = 120.0

    mock_limiter = MagicMock()
    mock_rate_result = MagicMock()
    mock_rate_result.allowed = True
    mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
            with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                result = await generate_image("test prompt", model="custom-model")

    # Verify service was called with correct model
    mock_service.generate_image_with_fetch.assert_called_once()
    call_kwargs = mock_service.generate_image_with_fetch.call_args[1]
    assert call_kwargs["model"] == "custom-model"


@pytest.mark.asyncio
async def test_generate_image_rate_limited():
    """Test that rate limit returns error."""
    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    mock_limiter = MagicMock()
    mock_rate_result = MagicMock()
    mock_rate_result.allowed = False
    mock_rate_result.message = "Rate limited"
    mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

    mock_discord_context = MagicMock()
    mock_discord_context.author = MagicMock(id=12345)

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
            result = await generate_image("test prompt", discord_context=mock_discord_context)

    assert result.success is False
    assert "rate limited" in result.error.lower()


@pytest.mark.asyncio
async def test_generate_image_cancelled():
    """Test that cancellation returns appropriate error."""
    mock_service = AsyncMock()
    mock_service.generate_image_with_fetch = AsyncMock(
        side_effect=Exception("Image generation aborted by user")
    )

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    mock_limiter = MagicMock()
    mock_rate_result = MagicMock()
    mock_rate_result.allowed = True
    mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
            with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                result = await generate_image("test prompt")

    assert result.success is False
    assert "failed" in result.error.lower()


@pytest.mark.asyncio
async def test_generate_image_service_failure():
    """Test handling of ImageService failure."""
    from persbot.services.image_service import ImageGenerationError

    mock_service = AsyncMock()
    mock_service.generate_image_with_fetch = AsyncMock(
        side_effect=ImageGenerationError("API error")
    )

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    mock_limiter = MagicMock()
    mock_rate_result = MagicMock()
    mock_rate_result.allowed = True
    mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
            with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                result = await generate_image("test prompt")

    assert result.success is False
    assert "failed" in result.error.lower() or "api error" in result.error.lower()


# Tests for send_image function


@pytest.mark.asyncio
async def test_send_image_success():
    """Test successful image sending to Discord channel."""
    # Mock Discord message and channel
    mock_discord_context = MagicMock()
    mock_channel = AsyncMock()
    mock_discord_context.channel = mock_channel

    # Create a proper async context manager mock for the response
    class MockResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "image/png"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def read(self):
            return b"fake_image_bytes"

    # Create a proper async context manager mock for the session
    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return MockResponse()

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        result = await send_image(
            image_url="https://example.com/image.png",
            discord_context=mock_discord_context,
        )

    # Verify success
    assert result.success is True
    assert result.data == "Image sent successfully"

    # Verify channel.send was called
    mock_channel.send.assert_called_once()


@pytest.mark.asyncio
async def test_send_image_empty_url():
    """Test that empty URL returns error."""
    result = await send_image(image_url="", discord_context=MagicMock())
    assert result.success is False
    assert "empty" in result.error.lower()


@pytest.mark.asyncio
async def test_send_image_invalid_url_format():
    """Test that invalid URL format returns error."""
    mock_discord_context = MagicMock()
    mock_discord_context.channel = AsyncMock()

    result = await send_image(
        image_url="not-a-valid-url",
        discord_context=mock_discord_context,
    )
    assert result.success is False
    assert "invalid" in result.error.lower()


@pytest.mark.asyncio
async def test_send_image_no_discord_context():
    """Test that missing Discord context returns error."""
    result = await send_image(image_url="https://example.com/image.png", discord_context=None)
    assert result.success is False
    assert "no discord channel" in result.error.lower()


@pytest.mark.asyncio
async def test_send_image_fetch_failure():
    """Test handling of HTTP fetch failure."""
    mock_discord_context = MagicMock()
    mock_discord_context.channel = AsyncMock()

    # Create a proper async context manager mock for the response with 404 status
    class MockResponse:
        def __init__(self):
            self.status = 404
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return MockResponse()

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        result = await send_image(
            image_url="https://example.com/notfound.png",
            discord_context=mock_discord_context,
        )

    assert result.success is False
    assert "failed to fetch" in result.error.lower()


@pytest.mark.asyncio
async def test_send_image_non_image_content_type():
    """Test that non-image content type is rejected."""
    mock_discord_context = MagicMock()
    mock_discord_context.channel = AsyncMock()

    # Create a proper async context manager mock for the response with text content type
    class MockResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "text/html"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return MockResponse()

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        result = await send_image(
            image_url="https://example.com/page.html",
            discord_context=mock_discord_context,
        )

    assert result.success is False
    assert "does not point to an image" in result.error.lower()


# Tests for image_url parameter


@pytest.mark.asyncio
async def test_generate_image_with_url():
    """Test image generation with image_url parameter."""
    mock_service = AsyncMock()
    mock_service.generate_image_with_fetch = AsyncMock(return_value=b"fake_image_bytes")

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    mock_limiter = MagicMock()
    mock_rate_result = MagicMock()
    mock_rate_result.allowed = True
    mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

    # Mock image download
    async def mock_download(url):
        # Return a fake base64 data URL
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
            with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                with patch("persbot.tools.api_tools.image_tools._download_and_convert_image", side_effect=mock_download):
                    result = await generate_image("test prompt", image_url="https://example.com/image.png")

    # Verify success
    assert result.success is True
    assert result.data == "Image generated successfully"

    # Verify ImageService was called
    mock_service.generate_image_with_fetch.assert_called_once()
    # Verify image_input was passed (the downloaded image)
    call_kwargs = mock_service.generate_image_with_fetch.call_args[1]
    assert "image_input" in call_kwargs
    assert call_kwargs["image_input"].startswith("data:image/")


@pytest.mark.asyncio
async def test_generate_image_url_download_fails_uses_attachment():
    """Test that failed URL download falls back to attached image."""
    mock_service = AsyncMock()
    mock_service.generate_image_with_fetch = AsyncMock(return_value=b"fake_image_bytes")

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    mock_limiter = MagicMock()
    mock_rate_result = MagicMock()
    mock_rate_result.allowed = True
    mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

    # Mock Discord context with attachment
    mock_discord_context = MagicMock()
    mock_attachment = MagicMock()
    mock_attachment.content_type = "image/png"
    mock_attachment.filename = "test.png"
    mock_attachment.read = AsyncMock(return_value=b"fake_attachment_bytes")
    mock_discord_context.attachments = [mock_attachment]
    mock_discord_context.author = MagicMock(id=12345)

    # Mock failed URL download
    async def mock_download_fail(url):
        return None

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
            with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                with patch("persbot.tools.api_tools.image_tools._download_and_convert_image", side_effect=mock_download_fail):
                    with patch("persbot.tools.api_tools.image_tools.process_image_sync", return_value=b"processed_bytes"):
                        with patch("persbot.tools.api_tools.image_tools.get_mime_type", return_value="image/png"):
                            result = await generate_image(
                                "test prompt",
                                image_url="https://invalid.com/image.png",
                                discord_context=mock_discord_context
                            )

    # Verify success - should have used attachment
    assert result.success is True
    mock_service.generate_image_with_fetch.assert_called_once()


@pytest.mark.asyncio
async def test_download_and_convert_image_success():
    """Test successful image download and conversion."""
    # Create a proper async context manager mock for the response
    class MockResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "image/png"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def read(self):
            # Minimal PNG data (1x1 transparent pixel)
            return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

    class MockSession:
        def __init__(self):
            self._response = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            self._response = MockResponse()
            return self._response

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        with patch("persbot.tools.api_tools.image_tools.process_image_sync", return_value=b"\x89PNG\r\n\x1a\n"):
            result = await _download_and_convert_image("https://example.com/image.png")

    # Verify result is a data URL
    assert result is not None
    assert result.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_download_and_convert_image_timeout():
    """Test timeout handling in image download."""
    import asyncio

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            raise asyncio.TimeoutError()

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        result = await _download_and_convert_image("https://example.com/image.png")

    # Should return None on timeout
    assert result is None


@pytest.mark.asyncio
async def test_download_and_convert_image_404():
    """Test 404 error handling in image download."""
    class MockResponse:
        def __init__(self):
            self.status = 404
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return MockResponse()

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        result = await _download_and_convert_image("https://example.com/notfound.png")

    # Should return None on 404
    assert result is None


@pytest.mark.asyncio
async def test_download_and_convert_image_non_image_content():
    """Test non-image content type rejection."""
    class MockResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "text/html"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return MockResponse()

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        result = await _download_and_convert_image("https://example.com/page.html")

    # Should return None for non-image content
    assert result is None


def test_detect_mime_type_from_content_type():
    """Test MIME type detection from Content-Type header."""
    assert _detect_mime_type("https://example.com/image.png", "image/png", b"") == "image/png"
    assert _detect_mime_type("https://example.com/image.jpg", "image/jpeg", b"") == "image/jpeg"
    assert _detect_mime_type("https://example.com/image.webp", "image/webp", b"") == "image/webp"


def test_detect_mime_type_from_url():
    """Test MIME type detection from URL extension."""
    assert _detect_mime_type("https://example.com/image.png", "", b"") == "image/png"
    assert _detect_mime_type("https://example.com/image.jpg", "", b"") == "image/jpeg"
    assert _detect_mime_type("https://example.com/image.jpeg", "", b"") == "image/jpeg"
    assert _detect_mime_type("https://example.com/image.webp", "", b"") == "image/webp"
    assert _detect_mime_type("https://example.com/image.gif", "", b"") == "image/gif"


def test_detect_mime_type_from_url_with_query_params():
    """Test MIME type detection from URL with query parameters."""
    assert _detect_mime_type("https://example.com/image.png?123", "", b"") == "image/png"
    assert _detect_mime_type("https://example.com/image.jpg?v=1", "", b"") == "image/jpeg"


def test_detect_mime_type_fallback():
    """Test MIME type fallback to png."""
    # Unknown URL, no content type
    result = _detect_mime_type("https://example.com/unknown", "", b"")
    # Should fallback to "image/png" when imghdr fails on empty bytes
    assert result == "image/png"

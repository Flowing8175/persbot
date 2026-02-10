"""Tests for image_tools.py"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persbot.tools.api_tools.image_tools import generate_image, send_image, _get_image_service
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

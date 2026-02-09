"""Tests for image_tools.py"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persbot.tools.api_tools.image_tools import generate_image
from persbot.tools.base import ToolResult


@pytest.mark.asyncio
async def test_generate_image_success():
    """Test successful image generation with OpenRouter."""
    # Mock OpenAI client
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_message = AsyncMock()
    mock_image = AsyncMock()
    mock_image.image_url.url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    mock_message.images = [mock_image]
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Mock config
    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.OpenAI", return_value=mock_client):
            result = await generate_image("test prompt")

    # Verify success
    assert result.success is True
    assert result.data is not None
    assert isinstance(result.data, bytes)

    # Verify OpenRouter API called with correct parameters
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["modalities"] == ["image"]
    assert "anime style" in call_kwargs["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_generate_image_empty_prompt():
    """Test that empty prompt returns error."""
    result = await generate_image("")
    assert result.success is False
    assert "empty" in result.error.lower()


@pytest.mark.asyncio
async def test_generate_image_anime_prefix():
    """Test that anime style is added to prompts."""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_message = AsyncMock()
    mock_image = AsyncMock()
    mock_image.image_url.url = "data:image/png;base64,AAAA"
    mock_message.images = [mock_image]
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.OpenAI", return_value=mock_client):
            await generate_image("cat")

    # Verify prompt includes anime prefix
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]
    assert "anime style" in prompt.lower()
    assert "detailed artwork" in prompt.lower()


@pytest.mark.asyncio
async def test_generate_image_base64_decode():
    """Test that base64 data URL is correctly decoded."""
    # Valid 1x1 PNG in base64
    valid_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_message = AsyncMock()
    mock_image = AsyncMock()
    mock_image.image_url.url = f"data:image/png;base64,{valid_png_base64}"
    mock_message.images = [mock_image]
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.OpenAI", return_value=mock_client):
            result = await generate_image("test")

    # Verify bytes decoded correctly
    assert result.success is True
    assert result.data == base64.b64decode(valid_png_base64)


@pytest.mark.asyncio
async def test_generate_image_invalid_base64():
    """Test handling of invalid base64 in response."""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_message = AsyncMock()
    mock_image = AsyncMock()
    mock_image.image_url.url = "data:image/png;base64,invalid!!!"
    mock_message.images = [mock_image]
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.OpenAI", return_value=mock_client):
            result = await generate_image("test")

    # Should fail with decode error
    assert result.success is False
    assert "decode" in result.error.lower()


@pytest.mark.asyncio
async def test_generate_image_no_choices():
    """Test handling of empty choices in response."""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.choices = []
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_config = MagicMock()
    mock_config.openrouter_api_key = "test-key"
    mock_config.openrouter_image_model = "test-model"
    mock_config.api_request_timeout = 120.0

    with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
        with patch("persbot.tools.api_tools.image_tools.OpenAI", return_value=mock_client):
            result = await generate_image("test")

    assert result.success is False
    assert "no image generated" in result.error.lower()

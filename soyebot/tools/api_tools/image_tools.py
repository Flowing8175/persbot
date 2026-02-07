"""Image generation tools for SoyeBot AI."""

import hashlib
import logging
import time

import aiohttp
from openai import OpenAI
from openai import AuthenticationError, RateLimitError, APIStatusError

from soyebot.config import load_config
from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def generate_image(
    prompt: str,
    **kwargs,
) -> ToolResult:
    """Generate an image using Z.AI image generation API.

    Args:
        prompt: The text prompt for image generation.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        ToolResult with image bytes in data field on success,
        or error message on failure.
    """
    if not prompt or not prompt.strip():
        return ToolResult(success=False, error="Image prompt cannot be empty")

    # Calculate prompt hash for logging
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    prompt_length = len(prompt)

    try:
        # Record start time for performance logging
        start_time = time.time()

        # Load config to get API credentials
        config = load_config()

        # Initialize OpenAI client with Z.AI credentials
        client = OpenAI(
            api_key=config.zai_api_key,
            base_url=config.zai_base_url,
            timeout=config.api_request_timeout,
        )

        # Call OpenAI client to generate image
        api_response = client.images.generate(
            model="glm-image",
            prompt=prompt,
            size="1280x1280",
            quality="hd",
        )

        # Check if response data is empty
        if not api_response.data or len(api_response.data) == 0:
            logger.error(
                "Empty data array in image generation response (prompt_hash=%s, length=%d)",
                prompt_hash,
                prompt_length,
            )
            return ToolResult(
                success=False,
                error="No image generated",
            )

        # Check for content filter violations
        if hasattr(api_response, "content_filter") and api_response.content_filter:
            reasons = (
                ", ".join(api_response.content_filter)
                if api_response.content_filter
                else "unknown"
            )
            logger.warning(
                "Content filter violation (prompt_hash=%s, length=%d, reasons=%s)",
                prompt_hash,
                prompt_length,
                reasons,
            )
            return ToolResult(
                success=False,
                error=f"Content filter violation: {reasons}",
            )

        # Extract image URL from response
        image_url = api_response.data[0].url

        # Download image from returned URL using aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        logger.error(
                            "Failed to download image (prompt_hash=%s, length=%d, HTTP status=%d)",
                            prompt_hash,
                            prompt_length,
                            resp.status,
                        )
                        return ToolResult(
                            success=False,
                            error="Failed to download generated image",
                        )
                    image_bytes = await resp.read()
        except Exception as download_error:
            logger.error(
                "Download failed (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                download_error,
            )
            return ToolResult(
                success=False,
                error="Failed to download generated image",
            )

        # Calculate response time
        response_time = time.time() - start_time

        # Log successful generation
        logger.info(
            "Image generated successfully (prompt_hash=%s, length=%d, response_time=%.2fs)",
            prompt_hash,
            prompt_length,
            response_time,
        )

        return ToolResult(
            success=True,
            data=image_bytes,
        )

    except AuthenticationError as e:
        logger.error(
            "Authentication error (prompt_hash=%s, length=%d): %s",
            prompt_hash,
            prompt_length,
            e,
        )
        return ToolResult(
            success=False,
            error="API key invalid or missing",
        )

    except RateLimitError as e:
        logger.error(
            "Rate limit error (prompt_hash=%s, length=%d): %s",
            prompt_hash,
            prompt_length,
            e,
        )
        return ToolResult(
            success=False,
            error="Rate limited, please try again later",
        )

    except APIStatusError as e:
        if e.status_code == 401:
            logger.error(
                "Unauthorized error (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                e,
            )
            return ToolResult(
                success=False,
                error="API key invalid or missing",
            )
        elif e.status_code == 429:
            logger.error(
                "Rate limit error (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                e,
            )
            return ToolResult(
                success=False,
                error="Rate limited, please try again later",
            )
        elif e.status_code == 500:
            logger.error(
                "Server error (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                e,
            )
            return ToolResult(
                success=False,
                error="Image generation service unavailable",
            )
        else:
            logger.error(
                "API status error (prompt_hash=%s, length=%d, status=%d): %s",
                prompt_hash,
                prompt_length,
                e.status_code,
                e,
            )
            return ToolResult(
                success=False,
                error="Image generation failed",
            )

    except Exception as e:
        # Check if it's a rate limit error via error string (fallback pattern from zai_service.py)
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str:
            logger.error(
                "Rate limit detected via string match (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                e,
            )
            return ToolResult(
                success=False,
                error="Rate limited, please try again later",
            )

        # Generic exception handler
        logger.error(
            "Image generation failed (prompt_hash=%s, length=%d): %s",
            prompt_hash,
            prompt_length,
            e,
            exc_info=True,
        )
        return ToolResult(
            success=False,
            error="Image generation failed",
        )


def register_image_tools(registry):
    """Register image tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="generate_image",
            description="Generate an image using AI based on a text description. Use this when the user asks for an image, drawing, or visual content.",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description="The text description of image to generate",
                    required=True,
                ),
            ],
            handler=generate_image,
            rate_limit=30,
        )
    )

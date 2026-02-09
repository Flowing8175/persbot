"""Image generation tools for SoyeBot AI."""

import asyncio
import base64
import hashlib
import logging
import time

import aiohttp
from openai import APIStatusError, AuthenticationError, OpenAI, RateLimitError

from persbot.config import load_config
from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


async def generate_image(
    prompt: str,
    **kwargs,
) -> ToolResult:
    """Generate an image using OpenRouter image generation API.

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

    # Check prompt length before API call (OpenRouter has ~2000 char limit for image models)
    # Add anime prefix and validate total length
    enhanced_prompt = f"{prompt}, anime"
    total_prompt_length = len(enhanced_prompt)

    # OpenRouter character limit for image generation is approximately 2000 chars
    MAX_PROMPT_LENGTH = 2000

    if total_prompt_length > MAX_PROMPT_LENGTH:
        logger.warning(
            "Prompt too long (prompt_hash=%s, length=%d, max=%d): truncating",
            prompt_hash,
            prompt_length,
            MAX_PROMPT_LENGTH,
        )
        # Truncate to fit within limit, keep as much of user's original prompt as possible
        # Reserve room for ", anime" prefix
        available_chars = MAX_PROMPT_LENGTH - len(", anime")
        if available_chars > 0:
            # Truncate user's prompt to fit, leaving room for anime prefix
            truncated_prompt = prompt[:available_chars]
            enhanced_prompt = f"{truncated_prompt}, anime"
        else:
            # Even user's prompt is too long to add anime prefix, use truncated version
            enhanced_prompt = prompt[:MAX_PROMPT_LENGTH]
            logger.warning(
                "User prompt exceeds maximum, using truncated version (prompt_hash=%s)",
                prompt_hash,
            )

    try:
        # Record start time for performance logging
        start_time = time.time()

        # Load config to get API credentials
        config = load_config()

        # Initialize OpenAI client with OpenRouter credentials
        image_base_url = "https://openrouter.ai/api/v1"
        logger.info(
            "Initializing image generation with OpenRouter: %s (model=%s)",
            image_base_url,
            config.openrouter_image_model,
        )
        client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=image_base_url,
            timeout=config.api_request_timeout,
        )

        # Call OpenAI client to generate image via OpenRouter
        api_response = await asyncio.to_thread(
            client.chat.completions.create,
            model=config.openrouter_image_model,
            messages=[{"role": "user", "content": enhanced_prompt}],
            modalities=["image"],
            extra_body={"image_config": {"aspect_ratio": "1:1"}},
        )

        # Validate response structure
        if not api_response.choices or len(api_response.choices) == 0:
            logger.error(
                "No choices in image generation response (prompt_hash=%s, length=%d)",
                prompt_hash,
                prompt_length,
            )
            return ToolResult(
                success=False,
                error="No image generated",
            )

        message = api_response.choices[0].message
        if not hasattr(message, "images") or not message.images or len(message.images) == 0:
            logger.error(
                "No images in response message (prompt_hash=%s, length=%d)",
                prompt_hash,
                prompt_length,
            )
            return ToolResult(
                success=False,
                error="No image generated",
            )

        # Parse image URL: handle both base64 data URLs and HTTP/HTTPS URLs
        # Handle both object and dict response formats
        image_obj = message.images[0]
        if isinstance(image_obj, dict):
            image_data_url = image_obj["image_url"]["url"]
        else:
            image_data_url = image_obj.image_url.url

        # Check if URL is base64 data URL or HTTP/HTTPS URL
        if image_data_url.startswith("data:"):
            # Base64 data URL format: "data:image/png;base64,iVBORw0KG..."
            try:
                header, data = image_data_url.split(",", 1)
                image_bytes = base64.b64decode(data)
            except (ValueError, IndexError) as decode_error:
                logger.error(
                    "Failed to decode base64 image (prompt_hash=%s, length=%d): %s",
                    prompt_hash,
                    prompt_length,
                    decode_error,
                )
                return ToolResult(
                    success=False,
                    error="Failed to decode generated image",
                )
        elif image_data_url.startswith(("http://", "https://")):
            # HTTP/HTTPS URL - fetch the image bytes
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_data_url) as response:
                        if response.status != 200:
                            logger.error(
                                "Failed to fetch image from URL (prompt_hash=%s, length=%d, status=%d)",
                                prompt_hash,
                                prompt_length,
                                response.status,
                            )
                            return ToolResult(
                                success=False,
                                error=f"Failed to fetch generated image (HTTP {response.status})",
                            )
                        image_bytes = await response.read()
            except Exception as fetch_error:
                logger.error(
                    "Failed to fetch image from URL (prompt_hash=%s, length=%d): %s",
                    prompt_hash,
                    prompt_length,
                    fetch_error,
                )
                return ToolResult(
                    success=False,
                    error="Failed to fetch generated image",
                )
        else:
            logger.error(
                "Unknown image URL format (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                image_data_url[:50] if image_data_url else "empty",
            )
            return ToolResult(
                success=False,
                error="Invalid image URL format",
            )

        # Calculate response time
        response_time = time.time() - start_time

        # Log successful generation
        logger.info(
            "Image generated successfully via OpenRouter (prompt_hash=%s, length=%d, response_time=%.2fs)",
            prompt_hash,
            prompt_length,
            response_time,
        )

        # Return success message and store image in metadata for Discord sending
        # Don't return the binary data in the main result field to avoid LLM prompt length issues
        return ToolResult(
            success=True,
            data="Image generated successfully",
            metadata={"image_bytes": image_bytes},
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
        # Log raw API response for debugging rate limit issues
        logger.debug(
            "Raw API response for rate limit (prompt_hash=%s): %s",
            prompt_hash,
            str(e),
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
        # Check if it's a rate limit error via error string
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
            rate_limit=0,
        )
    )

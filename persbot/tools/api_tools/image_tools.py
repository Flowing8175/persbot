"""Image generation tools for SoyeBot AI."""

import asyncio
import base64
import hashlib
import io
import logging
import time
from typing import Optional

import aiohttp
import discord
from openai import APIStatusError, AuthenticationError, OpenAI, RateLimitError

from persbot.config import load_config
from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    model: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
    **kwargs,
) -> ToolResult:
    """Generate an image using OpenRouter image generation API.

    Args:
        prompt: The text prompt for image generation.
        aspect_ratio: Optional aspect ratio for the image. Common values:
            "1:1" (1024x1024), "16:9" (1344x768), "9:16" (768x1344),
            "4:3" (1184x864), "3:2" (1248x832), "2:3" (832x1248),
            "21:9" (1536x672). Defaults to "1:1".
        cancel_event: AsyncIO event to check for cancellation before API calls.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        ToolResult with image bytes in data field on success,
        or error message on failure.
    """
    if not prompt or not prompt.strip():
        return ToolResult(success=False, error="Image prompt cannot be empty")

    # Check for cancellation before starting
    if cancel_event and cancel_event.is_set():
        logger.info("Image generation aborted due to cancellation signal before API call")
        return ToolResult(success=False, error="Image generation aborted by user")

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

        # Use provided model or fall back to config default
        image_model = model or config.openrouter_image_model

        # Initialize OpenAI client with OpenRouter credentials
        image_base_url = "https://openrouter.ai/api/v1"
        logger.info(
            "Initializing image generation with OpenRouter: %s (model=%s)",
            image_base_url,
            image_model,
        )
        client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=image_base_url,
            timeout=config.api_request_timeout,
        )

        logger.info(
            "Generating image with aspect_ratio: %s, model: %s (prompt_hash=%s)",
            aspect_ratio,
            image_model,
            prompt_hash,
        )

        # Check for cancellation before API call
        if cancel_event and cancel_event.is_set():
            logger.info("Image generation aborted due to cancellation signal before API call")
            return ToolResult(success=False, error="Image generation aborted by user")

        # Build image config with aspect_ratio
        image_config = {"aspect_ratio": aspect_ratio}

        # Call OpenAI client to generate image via OpenRouter
        api_response = await asyncio.to_thread(
            client.chat.completions.create,
            model=image_model,
            messages=[{"role": "user", "content": enhanced_prompt}],
            modalities=["image"],
            extra_body={"image_config": image_config},
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
            # Check for cancellation before fetching image
            if cancel_event and cancel_event.is_set():
                logger.info("Image generation aborted due to cancellation signal before image fetch")
                return ToolResult(success=False, error="Image generation aborted by user")

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


async def send_image(
    image_url: str,
    discord_context: Optional[discord.Message] = None,
    **kwargs,
) -> ToolResult:
    """Send an image to the Discord channel by fetching from a URL.

    This tool is useful when you want to manually send an image to the user,
    for example if a previous image generation took too long and you want to
    retry or send a different image.

    Args:
        image_url: The URL of the image to fetch and send. Must be a valid HTTP/HTTPS URL.
        discord_context: Discord message context (automatically injected) to get the channel.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        ToolResult with success message or error details.
    """
    if not image_url or not image_url.strip():
        return ToolResult(success=False, error="Image URL cannot be empty")

    if not discord_context or not hasattr(discord_context, "channel"):
        return ToolResult(success=False, error="No Discord channel available")

    # Validate URL format
    if not image_url.startswith(("http://", "https://")):
        return ToolResult(success=False, error="Invalid image URL format. Must start with http:// or https://")

    try:
        # Fetch the image from the URL
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    logger.error(
                        "Failed to fetch image from URL (status=%d): %s",
                        response.status,
                        image_url[:100],
                    )
                    return ToolResult(
                        success=False,
                        error=f"Failed to fetch image (HTTP {response.status})",
                    )

                # Validate content type is an image
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    logger.warning(
                        "URL does not point to an image (content_type=%s): %s",
                        content_type,
                        image_url[:100],
                    )
                    return ToolResult(
                        success=False,
                        error=f"URL does not point to an image (content type: {content_type})",
                    )

                image_bytes = await response.read()

        # Send the image to the Discord channel
        channel = discord_context.channel
        img_file = discord.File(io.BytesIO(image_bytes), filename="image.png")

        await channel.send(file=img_file)

        logger.info("Successfully sent image from URL to channel %s", channel.id)

        return ToolResult(
            success=True,
            data="Image sent successfully",
        )

    except aiohttp.ClientError as e:
        logger.error("Failed to fetch image from URL: %s", e)
        return ToolResult(
            success=False,
            error="Failed to fetch image from URL",
        )
    except discord.Forbidden:
        logger.error("No permission to send image to channel %s", discord_context.channel.id)
        return ToolResult(
            success=False,
            error="No permission to send image to this channel",
        )
    except discord.HTTPException as e:
        logger.error("Discord API error sending image: %s", e)
        return ToolResult(
            success=False,
            error="Failed to send image via Discord API",
        )
    except Exception as e:
        logger.error("Unexpected error sending image: %s", e, exc_info=True)
        return ToolResult(
            success=False,
            error="Failed to send image",
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
                ToolParameter(
                    name="aspect_ratio",
                    type="string",
                    description='Aspect ratio for the image. Common values: "1:1" (1024x1024), "16:9" (1344x768), "9:16" (768x1344), "4:3" (1184x864), "3:2" (1248x832), "2:3" (832x1248), "21:9" (1536x672). Default is "1:1".',
                    required=False,
                ),
                ToolParameter(
                    name="model",
                    type="string",
                    description='Optional specific model to use for image generation (e.g., "sourceful/riverflow-v2-pro", "sourceful/riverflow-v2-fast", "black-forest-labs/flux.2-klein-4b"). If not provided, uses the channel default.',
                    required=False,
                ),
            ],
            handler=generate_image,
            rate_limit=0,
            timeout=120.0,
        )
    )

    registry.register(
        ToolDefinition(
            name="send_image",
            description="Send an image to the Discord channel by providing a URL. Use this to manually send an image if generation took too long, or to send an image from a specific URL.",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="image_url",
                    type="string",
                    description="The HTTP/HTTPS URL of the image to fetch and send to the Discord channel",
                    required=True,
                ),
            ],
            handler=send_image,
            rate_limit=0,
            timeout=60.0,
        )
    )

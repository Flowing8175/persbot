"""Image generation tools for SoyeBot AI."""

import asyncio
import base64
import io
import logging
from typing import Optional

import aiohttp
import discord

from persbot.config import load_config
from persbot.rate_limiter import get_image_rate_limiter
from persbot.services.image_service import ImageService, ImageGenerationError
from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult
from persbot.utils import get_mime_type, process_image_sync

logger = logging.getLogger(__name__)

# Global ImageService instance (lazy initialization)
_image_service: Optional[ImageService] = None


def _get_image_service() -> ImageService:
    """Get or create the global ImageService instance."""
    global _image_service
    if _image_service is None:
        config = load_config()
        _image_service = ImageService(config)
    return _image_service


async def _download_and_convert_image(url: str) -> Optional[str]:
    """Download image from URL and convert to base64 data URL.

    Args:
        url: Image URL (Discord CDN URL or external URL).

    Returns:
        Base64 data URL string on success, None on failure.
    """
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                if response.status != 200:
                    logger.error(
                        "Failed to download image from URL (status=%d): %s",
                        response.status,
                        url[:100],
                    )
                    return None

                # Validate content type is an image
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    logger.warning(
                        "URL does not point to an image (content_type=%s): %s",
                        content_type,
                        url[:100],
                    )
                    return None

                image_bytes = await response.read()

                # Detect MIME type from content type or URL extension
                mime_type = _detect_mime_type(url, content_type, image_bytes)

                # Downscale image to ~1MP to reduce API payload
                img_bytes_processed = await asyncio.to_thread(process_image_sync, image_bytes, url)

                b64_str = base64.b64encode(img_bytes_processed).decode("utf-8")
                return f"data:{mime_type};base64,{b64_str}"

    except asyncio.TimeoutError:
        logger.error("Timeout downloading image from URL: %s", url[:100])
        return None
    except aiohttp.ClientError as e:
        logger.error("Failed to download image from URL: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error downloading image: %s", e)
        return None


def _detect_mime_type(url: str, content_type: str, data: bytes) -> str:
    """Detect MIME type from URL extension, content type header, or content.

    Args:
        url: Image URL.
        content_type: Content-Type header from response.
        data: Image bytes.

    Returns:
        MIME type string (e.g., 'image/png').
    """
    # Priority 1: Use content type from response header
    if content_type and content_type.startswith("image/"):
        # Get the main MIME type without charset or other parameters
        return content_type.split(";")[0].strip()

    # Priority 2: Try from URL extension (handle query parameters)
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path.lower()  # Get path without query string
    if path.endswith((".png", ".png/")):
        return "image/png"
    elif path.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    elif path.endswith((".webp", ".webp/")):
        return "image/webp"
    elif path.endswith((".gif", ".gif/")):
        return "image/gif"

    # Priority 3: Fallback to detection from content using file signatures
    if len(data) >= 8:
        # Check for common image file signatures (magic bytes)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif data[:2] == b"\xff\xd8":
            return "image/jpeg"
        elif data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
            return "image/webp"
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"

    # Priority 4: Ultimate fallback
    return "image/png"


async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    model: Optional[str] = None,
    image_url: Optional[str] = None,
    discord_context: Optional[discord.Message] = None,
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
        image_url: Optional image URL (Discord CDN URL or external URL) to use
            as reference/input for image generation. The system will automatically
            download and convert to base64. Takes priority over attached image.
        discord_context: Discord message context (automatically injected) to extract
            attached images for use as input.
        cancel_event: AsyncIO event to check for cancellation before API calls.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        ToolResult with image bytes in data field on success,
        or error message on failure.
    """
    if not prompt or not prompt.strip():
        return ToolResult(success=False, error="Image prompt cannot be empty")

    # Check rate limits
    user_id = (
        discord_context.author.id
        if (discord_context and hasattr(discord_context, "author"))
        else "unknown"
    )
    rate_limiter = get_image_rate_limiter()
    rate_limit_result = await rate_limiter.check_rate_limit(user_id)

    if not rate_limit_result.allowed:
        logger.warning(
            "Image generation rate limited for user %s: %s",
            user_id,
            rate_limit_result.message,
        )
        return ToolResult(
            success=False,
            error=rate_limit_result.message,
        )

    # Check for cancellation before starting
    if cancel_event and cancel_event.is_set():
        logger.debug("Image generation aborted before API call")
        return ToolResult(success=False, error="Image generation aborted by user")

    # Process image reference with priority:
    # 1. Explicit image_url (URL to download and convert) - highest priority
    # 2. Attached image from Discord message - fallback
    image_input = None

    # Priority 1: URL to download
    if image_url:
        try:
            image_input = await _download_and_convert_image(image_url)
            if image_input:
                logger.info(
                    "Successfully downloaded and converted image from URL for image generation"
                )
        except Exception as e:
            logger.warning(
                "Failed to download/convert image from URL %s: %s",
                image_url[:100] if image_url else "unknown",
                e,
            )

    # Priority 2: Attached image (only if URL download failed or not provided)
    if not image_input and discord_context and discord_context.attachments:
        # Process first image attachment as input
        for attachment in discord_context.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    img_bytes = await attachment.read()
                    # Downscale image to ~1MP to reduce API payload
                    img_bytes_processed = await asyncio.to_thread(
                        process_image_sync, img_bytes, attachment.filename
                    )
                    b64_str = base64.b64encode(img_bytes_processed).decode("utf-8")
                    mime_type = get_mime_type(img_bytes_processed)
                    image_input = f"data:{mime_type};base64,{b64_str}"
                    logger.info("Using attached image as input for image generation")
                    break
                except Exception as e:
                    logger.warning(
                        "Failed to read attachment %s for image input: %s",
                        attachment.filename,
                        e,
                    )

    try:
        # Get ImageService and generate image
        image_service = _get_image_service()

        image_bytes = await image_service.generate_image_with_fetch(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            model=model,
            image_input=image_input,
            cancel_event=cancel_event,
        )

        if image_bytes is None:
            return ToolResult(
                success=False,
                error="Image generation failed",
            )

        # Return success message and store image in metadata for Discord sending
        # Don't return the binary data in the main result field to avoid LLM prompt length issues
        return ToolResult(
            success=True,
            data="Image generated successfully",
            metadata={"image_bytes": image_bytes},
        )

    except asyncio.CancelledError:
        logger.debug("Image generation cancelled")
        return ToolResult(success=False, error="Image generation aborted by user")
    except ImageGenerationError as e:
        logger.error("Image generation error: %s", e)
        return ToolResult(success=False, error=str(e))
    except Exception as e:
        logger.error(
            "Image generation failed: %s",
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
        return ToolResult(
            success=False, error="Invalid image URL format. Must start with http:// or https://"
        )

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


def register_image_tools(registry) -> None:
    """Register image tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="generate_image",
            description="Generate an image using AI based on a text description. Optionally use an image URL or an attached image as input/reference for image-to-image generation. Use this when the user asks for an image, drawing, or visual content.",
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
                ToolParameter(
                    name="image_url",
                    type="string",
                    description="Optional image URL (Discord CDN URL or external URL) to use as input/reference for image-to-image generation. The system will automatically download and convert to base64. Takes priority over attached image.",
                    required=False,
                ),
            ],
            handler=generate_image,
            timeout=300.0,
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
            timeout=60.0,
        )
    )

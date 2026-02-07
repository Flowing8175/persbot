"""Image generation tools for SoyeBot AI."""

import logging

import aiohttp
from openai import OpenAI

from soyebot.config import load_config
from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def generate_image(
    prompt: str,
    discord_user_id: int,
    **kwargs,
) -> ToolResult:
    """Generate an image using Z.AI image generation API.

    Args:
        prompt: The text prompt for image generation.
        discord_user_id: Discord user ID for user identification.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        ToolResult with image bytes in data field on success,
        or error message on failure.
    """
    if not prompt or not prompt.strip():
        return ToolResult(success=False, error="Image prompt cannot be empty")

    try:
        # Load config to get API credentials
        config = load_config()

        # Initialize OpenAI client with Z.AI credentials
        client = OpenAI(
            api_key=config.zai_api_key,
            base_url=config.zai_base_url,
            timeout=config.api_request_timeout,
        )

        # Call OpenAI client to generate image
        response = client.images.create(
            model="glm-image",
            prompt=prompt,
            size="1280x1280",
            quality="hd",
            user_id=str(discord_user_id),
        )

        # Extract image URL from response
        image_url = response.data[0].url

        # Download image from returned URL using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    return ToolResult(
                        success=False,
                        error=f"Failed to download image: HTTP {resp.status}",
                    )
                image_bytes = await resp.read()

        return ToolResult(
            success=True,
            data=image_bytes,
        )

    except Exception as e:
        logger.error("Error generating image: %s", e)
        return ToolResult(
            success=False,
            error=f"Image generation failed: {str(e)}",
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
                    description="The text description of the image to generate",
                    required=True,
                ),
            ],
            handler=generate_image,
            rate_limit=30,
        )
    )

"""Web content analysis tools for persona-based AI."""

import logging
import re
from typing import Any, Dict

import aiohttp
from bs4 import BeautifulSoup
from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def inspect_external_content(
    url: str,
    **kwargs,
) -> ToolResult:
    """Inspect and analyze external web content from a URL.

    This tool fetches and extracts text content from a webpage,
    including title and body text. For YouTube URLs, it attempts
    to extract video title and description.

    Args:
        url: The URL of the webpage to inspect.

    Returns:
        ToolResult with extracted content (title, description, body text).
    """
    if not url or not url.strip():
        return ToolResult(success=False, error="URL cannot be empty")

    # Check for YouTube URL
    youtube_pattern = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
    is_youtube = bool(re.match(youtube_pattern, url))

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status != 200:
                    return ToolResult(
                        success=False,
                        error=f"Failed to fetch URL: HTTP {response.status}",
                    )

                html = await response.text()

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "No title found"

        # Extract meta description
        description_tag = soup.find("meta", attrs={"name": "description"})
        description = description_tag.get("content", "") if description_tag else ""

        # For YouTube, try to get video description
        if is_youtube:
            youtube_title = title
            youtube_description = description

            # Try to extract additional YouTube metadata
            og_title = soup.find("meta", property="og:title")
            if og_title:
                youtube_title = og_title.get("content", youtube_title)

            og_description = soup.find("meta", property="og:description")
            if og_description:
                youtube_description = og_description.get("content", youtube_description)

            return ToolResult(
                success=True,
                data={
                    "url": url,
                    "type": "youtube",
                    "title": youtube_title,
                    "description": youtube_description[:1000]
                    if len(youtube_description) > 1000
                    else youtube_description,
                },
            )

        # For regular webpages, extract body text
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get body text
        body = soup.find("body")
        if body:
            body_text = body.get_text(separator=" ", strip=True)

            # Clean up whitespace
            body_text = re.sub(r"\s+", " ", body_text)

            # Limit to 1000 characters
            if len(body_text) > 1000:
                body_text = body_text[:1000] + "..."
        else:
            body_text = "No body content found"

        return ToolResult(
            success=True,
            data={
                "url": url,
                "type": "webpage",
                "title": title,
                "description": description[:500]
                if len(description) > 500
                else description,
                "body": body_text,
            },
        )

    except aiohttp.ClientError as e:
        logger.error("HTTP error fetching URL %s: %s", url, e)
        return ToolResult(success=False, error=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        logger.error(
            "Error inspecting external content from %s: %s", url, e, exc_info=True
        )
        return ToolResult(success=False, error=f"Failed to inspect content: {str(e)}")


def register_web_tools(registry):
    """Register web content tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="inspect_external_content",
            description=(
                "Inspect and extract content from a URL. Returns title, description, "
                "and body text (truncated to 1000 chars). "
                "Supports regular webpages and YouTube videos."
            ),
            category=ToolCategory.PERSONA_WEB,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The URL of the webpage or video to inspect",
                    required=True,
                ),
            ],
            handler=inspect_external_content,
            rate_limit=30,
        )
    )

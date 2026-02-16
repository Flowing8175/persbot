"""Web content inspection tools for SoyeBot AI.

This module provides external content analysis functionality for persona bots.
It allows inspecting and summarizing content from external URLs including web pages
and YouTube videos.
"""

import asyncio
import logging
import re
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# User agent to use for requests
USER_AGENT = "Mozilla/5.0 (compatible; SoyeBot/1.0; +https://github.com/persbot)"

# Maximum content length to return
MAX_CONTENT_LENGTH = 1000


def _parse_html_sync(html: str) -> dict:
    """Parse HTML content synchronously using BeautifulSoup.

    This CPU-intensive operation is run in a thread pool to avoid
    blocking the event loop.

    Args:
        html: The HTML content to parse.

    Returns:
        dict with parsed title, content, and meta_description.
    """
    # Parse HTML
    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = None
    if soup.title:
        title = soup.title.string.strip()

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Get main content
    # Try to find main content areas
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"class": re.compile(r"content|article|post", re.I)})
        or soup.body
    )

    if main_content:
        # Extract text paragraphs
        paragraphs = main_content.find_all("p")
        text_content = " ".join(
            [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        )
    else:
        text_content = soup.get_text(separator=" ", strip=True)

    # Clean up whitespace
    text_content = re.sub(r"\s+", " ", text_content).strip()

    # Truncate if too long
    if len(text_content) > MAX_CONTENT_LENGTH:
        text_content = text_content[:MAX_CONTENT_LENGTH] + "..."

    # Also extract meta description if available
    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = meta_tag.get("content")

    return {
        "title": title or "No title found",
        "content": text_content or "No content could be extracted",
        "meta_description": meta_description,
        "content_length": len(text_content),
    }


async def inspect_external_content(
    url: str,
    cancel_event: Optional[asyncio.Event] = None,
    **kwargs,
) -> ToolResult:
    """Inspect and analyze content from an external URL.

    This tool fetches the content from a web page, extracts the title and main text,
    and returns a summarized version suitable for AI processing.

    Args:
        url: The URL to inspect and analyze.
        cancel_event: AsyncIO event to check for cancellation before HTTP calls.

    Returns:
        ToolResult with the page's title, URL, and summarized content.
    """
    if not url or not url.strip():
        return ToolResult(success=False, error="URL cannot be empty")

    # Validate URL format
    if not url.startswith(("http://", "https://")):
        return ToolResult(success=False, error="URL must start with http:// or https://")

    # Check for cancellation before HTTP request
    if cancel_event and cancel_event.is_set():
        return ToolResult(success=False, error="Web content inspection aborted by user")

    try:
        # Check if this is a YouTube URL
        youtube_id = _extract_youtube_id(url)
        if youtube_id:
            return await _inspect_youtube_content(url, youtube_id, cancel_event)

        # Regular web page inspection
        return await _inspect_web_page(url)

    except Exception as e:
        logger.error("Error inspecting external content: %s", e, exc_info=True)
        return ToolResult(success=False, error=f"Failed to inspect URL: {str(e)}")


async def _inspect_web_page(url: str) -> ToolResult:
    """Inspect a regular web page and extract its content.

    Args:
        url: The URL of the web page to inspect.

    Returns:
        ToolResult with extracted page content.
    """
    headers = {
        "User-Agent": USER_AGENT,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status != 200:
                    return ToolResult(
                        success=False, error=f"HTTP {response.status}: Failed to fetch page content"
                    )

                html = await response.text()

        # Offload CPU-intensive HTML parsing to thread pool
        parsed_data = await asyncio.to_thread(_parse_html_sync, html)

        return ToolResult(
            success=True,
            data={
                "url": url,
                "title": parsed_data["title"],
                "content": parsed_data["content"],
                "meta_description": parsed_data["meta_description"],
                "content_length": parsed_data["content_length"],
                "type": "web_page",
            },
        )

    except asyncio.CancelledError:
        return ToolResult(success=False, error="Web content inspection aborted by user")
    except aiohttp.ClientError as e:
        logger.error("HTTP client error: %s", e)
        return ToolResult(success=False, error=f"Network error: {str(e)}")
    except Exception as e:
        logger.error("Error parsing web page: %s", e, exc_info=True)
        return ToolResult(success=False, error=f"Failed to parse page: {str(e)}")


async def _inspect_youtube_content(
    url: str,
    video_id: str,
    cancel_event: Optional[asyncio.Event] = None,
) -> ToolResult:
    """Inspect a YouTube video and extract its metadata.

    Args:
        url: The YouTube URL.
        video_id: The YouTube video ID.
        cancel_event: AsyncIO event to check for cancellation before HTTP calls.

    Returns:
        ToolResult with YouTube video information.
    """
    # Check for cancellation before HTTP request
    if cancel_event and cancel_event.is_set():
        return ToolResult(success=False, error="YouTube content inspection aborted by user")

    # Try to get video info via YouTube's oEmbed endpoint (no API key needed)
    oembed_url = (
        f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(oembed_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return ToolResult(
                        success=True,
                        data={
                            "url": url,
                            "title": data.get("title", "YouTube Video"),
                            "author": data.get("author_name", "Unknown"),
                            "description": f"Video by {data.get('author_name', 'Unknown')}",
                            "type": "youtube_video",
                            "video_id": video_id,
                        },
                    )

    except asyncio.CancelledError:
        return ToolResult(success=False, error="YouTube content inspection aborted by user")
    except Exception as e:
        pass  # Logging removed

    # Fallback: Return basic info
    return ToolResult(
        success=True,
        data={
            "url": url,
            "title": f"YouTube Video ({video_id})",
            "description": "Could not fetch video details. This appears to be a YouTube video.",
            "type": "youtube_video",
            "video_id": video_id,
        },
    )


def _extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from a URL.

    Args:
        url: The URL to parse.

    Returns:
        YouTube video ID if found, None otherwise.
    """
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",  # Direct video ID
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def register_web_tools(registry) -> None:
    """Register web content inspection tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="inspect_external_content",
            description="Inspect and analyze content from an external URL. Extracts the page title and main text content, summarizing for AI processing. Supports web pages and YouTube videos.",
            category=ToolCategory.PERSONA_WEB,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The URL to inspect and analyze (must start with http:// or https://).",
                    required=True,
                ),
            ],
            handler=inspect_external_content,
        )
    )

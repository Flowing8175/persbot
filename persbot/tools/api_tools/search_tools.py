"""Web search tools for SoyeBot AI."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ddgs import DDGS
from ddgs.exceptions import RatelimitException

from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


async def web_search(
    query: str,
    num_results: int = 5,
    search_api_key: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
    **kwargs,
) -> ToolResult:
    """Search the web for information.

    Args:
        query: The search query string.
        num_results: Number of results to return (default: 5, max: 10).
        search_api_key: Optional API key for search service (unused with DuckDuckGo).
        cancel_event: AsyncIO event to check for cancellation before API calls.

    Returns:
        ToolResult with search results.
    """
    if not query or not query.strip():
        return ToolResult(success=False, error="Search query cannot be empty")

    num_results = min(max(1, num_results), 10)  # Clamp between 1-10

    # Check for cancellation before API call
    if cancel_event and cancel_event.is_set():
        logger.info("Web search aborted due to cancellation signal before API call")
        return ToolResult(success=False, error="Web search aborted by user")

    # Use DuckDuckGo search via duckduckgo-search package
    try:
        # Run synchronous DDGS call in thread pool to avoid blocking
        results = await asyncio.to_thread(
            _perform_search,
            query,
            num_results,
        )

        if results:
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results[:num_results],
                    "count": len(results[:num_results]),
                },
            )
    except RatelimitException:
        return ToolResult(
            success=False,
            error="Rate limit exceeded. DuckDuckGo search is temporarily unavailable due to too many requests. Please try again in a few minutes.",
        )
    except Exception:
        pass

    # Fallback: Return a message about search limitations
    return ToolResult(
        success=False,
        error="Web search failed. The search service may be temporarily unavailable.",
    )


def _perform_search(query: str, num_results: int) -> List[Dict[str, str]]:
    """Perform synchronous DuckDuckGo search.

    Args:
        query: The search query string.
        num_results: Number of results to return.

    Returns:
        List of search result dictionaries.

    Raises:
        RatelimitException: When DuckDuckGo rate limits the request.
        Exception: For other search errors.
    """
    ddgs = DDGS()
    search_results = ddgs.text(
        query,
        max_results=num_results,
        region="ko-kr",
        safesearch="moderate",
        backend="auto",
    )

    # Transform DDGS results to match expected format
    results = []
    for result in search_results or []:
        results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", ""),
                "source": "DuckDuckGo",
            }
        )

    return results


def register_search_tools(registry):
    """Register search tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="web_search",
            description="Search the web for current information on any topic. Useful for finding facts, news, or answering questions about recent events.",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query to look up on the web",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    type="integer",
                    description="Number of search results to return (default: 5, max: 10)",
                    required=False,
                    default=5,
                ),
            ],
            handler=web_search,
        )
    )

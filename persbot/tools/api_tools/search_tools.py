"""Web search tools for SoyeBot AI."""

import asyncio
import logging
from typing import Optional

from persbot.services.search_service import SearchService, SearchError, SearchRateLimitError
from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Global SearchService instance (lazy initialization)
_search_service: Optional[SearchService] = None


def _get_search_service() -> SearchService:
    """Get or create the global SearchService instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


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

    try:
        # Get SearchService and perform search
        search_service = _get_search_service()

        results = await search_service.web_search(
            query=query,
            num_results=num_results,
            cancel_event=cancel_event,
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

        # No results found
        return ToolResult(
            success=False,
            error="No search results found",
        )

    except asyncio.CancelledError:
        logger.info("Web search cancelled by user")
        return ToolResult(success=False, error="Web search aborted by user")
    except SearchRateLimitError:
        return ToolResult(
            success=False,
            error="Rate limit exceeded. DuckDuckGo search is temporarily unavailable due to too many requests. Please try again in a few minutes.",
        )
    except SearchError as e:
        logger.error("Search error: %s", e)
        return ToolResult(
            success=False,
            error="Web search failed. The search service may be temporarily unavailable.",
        )
    except Exception as e:
        logger.error("Unexpected search error: %s", e, exc_info=True)
        return ToolResult(
            success=False,
            error="Web search failed",
        )


def register_search_tools(registry) -> None:
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

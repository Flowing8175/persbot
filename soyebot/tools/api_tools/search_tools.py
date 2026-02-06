"""Web search tools for SoyeBot AI."""

import logging
from typing import Any, Dict, List, Optional

import aiohttp
from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def web_search(
    query: str,
    num_results: int = 5,
    search_api_key: Optional[str] = None,
) -> ToolResult:
    """Search the web for information.

    Args:
        query: The search query string.
        num_results: Number of results to return (default: 5, max: 10).
        search_api_key: Optional API key for search service.

    Returns:
        ToolResult with search results.
    """
    if not query or not query.strip():
        return ToolResult(success=False, error="Search query cannot be empty")

    num_results = min(max(1, num_results), 10)  # Clamp between 1-10

    # Try DuckDuckGo instant answer API (no key required)
    try:
        async with aiohttp.ClientSession() as session:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
            }

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    results = []

                    # Extract abstract
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", query),
                            "url": data.get("AbstractURL", ""),
                            "snippet": data.get("Abstract", ""),
                            "source": "DuckDuckGo",
                        })

                    # Extract related topics
                    if data.get("RelatedTopics"):
                        for topic in data.get("RelatedTopics", [])[:num_results]:
                            if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                                results.append({
                                    "title": topic.get("Text", "").split(" - ")[0][:100],
                                    "url": topic.get("FirstURL", ""),
                                    "snippet": topic.get("Text", ""),
                                    "source": "DuckDuckGo",
                                })

                    if results:
                        return ToolResult(
                            success=True,
                            data={
                                "query": query,
                                "results": results[:num_results],
                                "count": len(results[:num_results]),
                            }
                        )
    except Exception as e:
        logger.debug("DuckDuckGo search failed: %s", e)

    # Fallback: Return a message about search limitations
    return ToolResult(
        success=False,
        error="Web search requires a search API key. Please configure SEARCH_API_KEY in your environment to enable web search functionality.",
    )


def register_search_tools(registry):
    """Register search tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(ToolDefinition(
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
        rate_limit=30,  # 30 seconds between searches
    ))

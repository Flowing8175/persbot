"""Episodic memory search tools for persona-based AI."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def search_episodic_memory(
    user_id: str,
    query: str,
    limit: int = 5,
    **kwargs,
) -> ToolResult:
    """Search episodic memory for past interactions, preferences, and facts.

    This tool searches through stored episodes to find relevant information about
    the user's past interactions, preferences, agreements, or facts.

    Args:
        user_id: Discord user ID to search memories for.
        query: Search query or keywords to find relevant episodes.
        limit: Maximum number of episodes to return (default: 5).

    Returns:
        ToolResult with list of relevant episodes in "date - content" format.
    """
    if not user_id:
        return ToolResult(success=False, error="user_id is required")

    if not query or not query.strip():
        return ToolResult(success=False, error="Query cannot be empty")

    limit = min(max(1, limit), 10)

    # Load mock memory data
    try:
        memory_file = Path("data/memory_vector_mock.json")
        if not memory_file.exists():
            # Return empty result if no memory file exists
            return ToolResult(
                success=True,
                data={
                    "user_id": user_id,
                    "query": query,
                    "episodes": [],
                    "message": "No episodic memory data found. This is a prototype - Vector DB integration pending.",
                },
            )

        with open(memory_file, "r", encoding="utf-8") as f:
            memory_data = json.load(f)

        # Simple keyword matching (prototype)
        query_lower = query.lower()
        query_keywords = set(query_lower.split())

        episodes = []
        for episode in memory_data.get("episodes", []):
            # Skip episodes for different users
            if episode.get("user_id") != user_id:
                continue

            # Keyword matching
            content = episode.get("content", "").lower()
            episode_keywords = set(content.split())

            # Calculate relevance score
            intersection = query_keywords.intersection(episode_keywords)
            score = len(intersection) / max(len(query_keywords), 1)

            if score > 0:
                episodes.append(
                    {
                        "score": score,
                        "date": episode.get("date", "Unknown"),
                        "content": episode.get("content", ""),
                        "type": episode.get("type", "general"),
                    }
                )

        # Sort by relevance score and limit results
        episodes.sort(key=lambda x: x["score"], reverse=True)
        episodes = episodes[:limit]

        # Format as "date - content"
        formatted_episodes = [f"{ep['date']} - {ep['content']}" for ep in episodes]

        return ToolResult(
            success=True,
            data={
                "user_id": user_id,
                "query": query,
                "episodes": formatted_episodes,
                "count": len(formatted_episodes),
            },
        )

    except json.JSONDecodeError as e:
        logger.error("Failed to parse memory data: %s", e)
        return ToolResult(success=False, error=f"Failed to parse memory data: {str(e)}")
    except Exception as e:
        logger.error("Error searching episodic memory: %s", e, exc_info=True)
        return ToolResult(
            success=False, error=f"Failed to search episodic memory: {str(e)}"
        )


def register_memory_tools(registry):
    """Register memory tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="search_episodic_memory",
            description=(
                "Search through the persona's episodic memory to find past interactions, "
                "preferences, agreements, or specific facts about the user. "
                "Returns episodes in 'date - content' format for context."
            ),
            category=ToolCategory.PERSONA_MEMORY,
            parameters=[
                ToolParameter(
                    name="user_id",
                    type="string",
                    description="Discord user ID to search memories for",
                    required=True,
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query or keywords to find relevant episodes",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of episodes to return (default: 5, max: 10)",
                    required=False,
                    default=5,
                ),
            ],
            handler=search_episodic_memory,
            rate_limit=10,
        )
    )

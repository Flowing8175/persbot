"""Episodic memory tools for SoyeBot AI.

This module provides RAG-based memory search functionality for persona bots.
It allows searching through past conversations, preferences, and specific events.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiofiles

from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Default path for memory vector data (can be overridden via config)
DEFAULT_MEMORY_PATH = "data/memory_vector_mock.json"


async def search_episodic_memory(
    user_id: str,
    query: str = "",
    limit: int = 5,
    global_search: bool = False,
    **kwargs,
) -> ToolResult:
    """Search through episodic memory for relevant past conversations and events.

    This tool searches through stored memories to find specific promises,
    preferences, facts, and past interactions with the user.

    Args:
        user_id: The Discord user ID to search memories for.
        query: The search query to find relevant memories (optional if global_search=True).
        limit: Maximum number of memory results to return (default: 5, max: 50 for global search).
        global_search: If True, list all memories for the user without query filtering.

    Returns:
        ToolResult with relevant episodic memories formatted as "Date - Content".
    """
    # Global search doesn't require a query
    if not global_search and (not query or not query.strip()):
        return ToolResult(success=False, error="Search query cannot be empty (use global_search=True to list all memories)")

    # Clamp limit: 1-10 for normal search, 1-50 for global search
    max_limit = 50 if global_search else 10
    limit = min(max(1, limit), max_limit)

    try:
        # Try to load from memory vector file
        memories = await _load_memory_vector()

        if not memories:
            # Return a helpful message if no memories are stored yet
            return ToolResult(
                success=True,
                data={
                    "user_id": user_id,
                    "query": query if not global_search else "*",
                    "global_search": global_search,
                    "memories": [],
                    "message": "No episodic memories found. The persona is still building their memory bank.",
                },
            )

        # Global search: list all memories for the user
        if global_search:
            relevant_memories = _get_all_memories_for_user(memories, user_id, limit)
        else:
            # Simple keyword-based matching (prototype - will be replaced with vector search)
            relevant_memories = _filter_memories_by_keywords(memories, user_id, query, limit)

        if not relevant_memories:
            message = (
                "No memories found for this user."
                if global_search
                else f"No specific memories found for '{query}'. The persona will remember this interaction for future conversations."
            )
            return ToolResult(
                success=True,
                data={
                    "user_id": user_id,
                    "query": query if not global_search else "*",
                    "global_search": global_search,
                    "memories": [],
                    "message": message,
                },
            )

        # Format memories as "Date - Content" strings
        formatted_memories = []
        for memory in relevant_memories:
            date_str = memory.get("date", datetime.now().strftime("%Y-%m-%d"))
            content = memory.get("content", memory.get("text", ""))
            formatted_memories.append(f"{date_str} - {content}")

        return ToolResult(
            success=True,
            data={
                "user_id": user_id,
                "query": query if not global_search else "*",
                "global_search": global_search,
                "count": len(formatted_memories),
                "memories": formatted_memories,
            },
        )

    except Exception as e:
        logger.error("Error searching episodic memory: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def save_episodic_memory(
    user_id: str,
    content: str,
    memory_type: str = "preference",
    tags: Optional[List[str]] = None,
    **kwargs,
) -> ToolResult:
    """Save a new episodic memory entry.

    This tool stores new memories including preferences, facts, promises,
    and important interactions for future retrieval.

    Args:
        user_id: The Discord user ID to save memory for.
        content: The memory content to store.
        memory_type: Type of memory (preference, fact, promise, interest).
        tags: Optional list of tags for categorization.

    Returns:
        ToolResult with confirmation of saved memory.
    """
    if not content or not content.strip():
        return ToolResult(success=False, error="Memory content cannot be empty")

    if not user_id:
        return ToolResult(success=False, error="User ID cannot be empty")

    try:
        # Load existing memories
        memories = await _load_memory_vector()

        # Create new memory entry
        new_memory = {
            "user_id": user_id,
            "content": content,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "type": memory_type.lower(),
            "tags": tags or [],
        }

        # Add to memories list
        memories.append(new_memory)

        # Save back to file
        await _save_memory_vector(memories)

        return ToolResult(
            success=True,
            data={
                "user_id": user_id,
                "content": content,
                "type": memory_type,
                "tags": tags or [],
                "date": new_memory["date"],
                "message": "Memory saved successfully",
            },
        )

    except Exception as e:
        logger.error("Error saving episodic memory: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def remove_episodic_memory(
    user_id: str,
    content: Optional[str] = None,
    memory_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs,
) -> ToolResult:
    """Remove episodic memory entries matching the given criteria.

    This tool removes memories from storage based on user_id and optional filters.
    If only user_id is provided, all memories for that user are removed.
    Additional filters (content, memory_type, tags) can be used for selective removal.

    Args:
        user_id: The Discord user ID to remove memories for.
        content: Optional content string to match for selective removal.
        memory_type: Optional memory type to filter by (preference, fact, promise, interest).
        tags: Optional list of tags to filter by.

    Returns:
        ToolResult with confirmation of removed memories.
    """
    if not user_id:
        return ToolResult(success=False, error="User ID cannot be empty")

    try:
        # Load existing memories
        memories = await _load_memory_vector()

        if not memories:
            return ToolResult(
                success=True,
                data={
                    "user_id": user_id,
                    "removed_count": 0,
                    "message": "No memories found to remove",
                },
            )

        # Filter memories to keep (those that DON'T match removal criteria)
        memories_to_keep = []
        removed_count = 0

        for memory in memories:
            # Skip if memory belongs to a different user
            memory_user_id = memory.get("user_id", "global")
            if memory_user_id != user_id and memory_user_id != "global":
                memories_to_keep.append(memory)
                continue

            # If user_id matches global, apply all filters
            # If user_id matches specific user, apply all filters
            should_remove = True

            # If content filter is provided, check for match
            if content is not None and content.strip():
                memory_content = memory.get("content", "")
                if content not in memory_content:
                    should_remove = False

            # If memory_type filter is provided, check for match
            if memory_type is not None and memory_type.strip():
                memory_type_value = memory.get("type", "")
                if memory_type.lower() != memory_type_value.lower():
                    should_remove = False

            # If tags filter is provided, check for at least one matching tag
            if tags is not None and tags:
                memory_tags = memory.get("tags", [])
                has_matching_tag = any(
                    tag.lower() in [t.lower() for t in memory_tags] for tag in tags
                )
                if not has_matching_tag:
                    should_remove = False

            if should_remove:
                removed_count += 1
            else:
                memories_to_keep.append(memory)

        # Save updated memories
        await _save_memory_vector(memories_to_keep)

        return ToolResult(
            success=True,
            data={
                "user_id": user_id,
                "removed_count": removed_count,
                "message": f"Removed {removed_count} memory entries",
            },
        )

    except Exception as e:
        logger.error("Error removing episodic memory: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def _load_memory_vector() -> List[Dict[str, Any]]:
    """Load memory vectors from the local JSON file.

    Returns:
        List of memory entries with user_id, content, timestamp, and metadata.
    """
    memory_path = os.environ.get("MEMORY_VECTOR_PATH", DEFAULT_MEMORY_PATH)

    if not os.path.exists(memory_path):
        # Create a sample memory file for demonstration
        await _create_sample_memory_file(memory_path)

    try:
        async with aiofiles.open(memory_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
            return data.get("memories", [])
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in memory file, returning empty list")
        return []
    except Exception as e:
        logger.error("Error loading memory file: %s", e)
        return []


async def _create_sample_memory_file(path: str) -> None:
    """Create a sample memory file for demonstration purposes.

    Args:
        path: Path where to create the sample memory file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sample_data = {
        "memories": [
            {
                "user_id": "global",
                "content": "User prefers short, concise responses rather than long explanations.",
                "date": "2024-01-15",
                "type": "preference",
                "tags": ["communication", "style"],
            },
            {
                "user_id": "global",
                "content": "User mentioned they enjoy late-night gaming sessions.",
                "date": "2024-01-20",
                "type": "fact",
                "tags": ["gaming", "hobby"],
            },
            {
                "user_id": "global",
                "content": "User asked about AI personality preferences - likes playful but helpful tone.",
                "date": "2024-02-01",
                "type": "preference",
                "tags": ["personality", "tone"],
            },
            {
                "user_id": "global",
                "content": "User expressed interest in learning about machine learning concepts.",
                "date": "2024-02-10",
                "type": "interest",
                "tags": ["ml", "learning"],
            },
            {
                "user_id": "global",
                "content": "User mentioned they work in software development.",
                "date": "2024-02-15",
                "type": "fact",
                "tags": ["work", "profession"],
            },
        ]
    }

    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(sample_data, ensure_ascii=False, indent=2))



async def _save_memory_vector(memories: List[Dict[str, Any]]) -> None:
    """Save memory vectors to the local JSON file.

    Args:
        memories: List of memory entries to save.
    """
    memory_path = os.environ.get("MEMORY_VECTOR_PATH", DEFAULT_MEMORY_PATH)

    try:
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)

        data = {"memories": memories}

        async with aiofiles.open(memory_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error("Error saving memory file: %s", e)
        raise


def _get_all_memories_for_user(
    memories: List[Dict[str, Any]],
    user_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Get all memories for a specific user (global search).

    Args:
        memories: List of all memory entries.
        user_id: User ID to filter by (memories with user_id or 'global').
        limit: Maximum results to return.

    Returns:
        List of memories belonging to the user or global, sorted by date (newest first).
    """
    user_memories = []
    for memory in memories:
        memory_user_id = memory.get("user_id", "global")
        if memory_user_id == user_id or memory_user_id == "global":
            user_memories.append(memory)

    # Sort by date descending (newest first)
    user_memories.sort(key=lambda m: m.get("date", ""), reverse=True)
    return user_memories[:limit]


def _filter_memories_by_keywords(
    memories: List[Dict[str, Any]],
    user_id: str,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Filter memories based on keyword matching (simple prototype implementation).

    Args:
        memories: List of all memory entries.
        user_id: User ID to filter by (use 'global' for universal memories).
        query: Search query keywords.
        limit: Maximum results to return.

    Returns:
        Filtered and sorted list of relevant memories.
    """
    query_lower = query.lower()
    keywords = query_lower.split()

    scored_memories = []

    for memory in memories:
        # Check if memory belongs to the user or is global
        memory_user_id = memory.get("user_id", "global")
        if memory_user_id != user_id and memory_user_id != "global":
            continue

        content = memory.get("content", "").lower()
        tags = [tag.lower() for tag in memory.get("tags", [])]
        memory_type = memory.get("type", "").lower()

        # Calculate relevance score
        score = 0

        # Exact content match
        if query_lower in content:
            score += 10

        # Keyword matches in content
        for keyword in keywords:
            if keyword in content:
                score += 2
            if keyword in tags:
                score += 3
            if keyword in memory_type:
                score += 1

        if score > 0:
            scored_memories.append((score, memory))

    # Sort by score descending and return top results
    scored_memories.sort(key=lambda x: x[0], reverse=True)
    return [memory for _, memory in scored_memories[:limit]]


def register_memory_tools(registry) -> None:
    """Register episodic memory tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="search_episodic_memory",
            description="Search through persona's episodic memory to find relevant past conversations, promises, preferences, and facts. Returns specific memories with dates. Use global_search=True to list all memories.",
            category=ToolCategory.PERSONA_MEMORY,
            parameters=[
                ToolParameter(
                    name="user_id",
                    type="string",
                    description="The Discord user ID to search memories for.",
                    required=True,
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query to find relevant memories (keywords, topics, or questions). Optional if global_search=True.",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of memory results to return (default: 5, max: 10 for normal search, 50 for global search).",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="global_search",
                    type="boolean",
                    description="If True, list all memories for the user without query filtering. Use to browse all stored memories.",
                    required=False,
                    default=False,
                ),
            ],
            handler=search_episodic_memory,
        )
    )

    registry.register(
        ToolDefinition(
            name="save_episodic_memory",
            description="Save a new episodic memory entry including preferences, facts, promises, and important interactions for future retrieval.",
            category=ToolCategory.PERSONA_MEMORY,
            parameters=[
                ToolParameter(
                    name="user_id",
                    type="string",
                    description="The Discord user ID to save memory for.",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="The memory content to store.",
                    required=True,
                ),
                ToolParameter(
                    name="memory_type",
                    type="string",
                    description="Type of memory (preference, fact, promise, interest).",
                    required=False,
                    default="preference",
                    enum=["preference", "fact", "promise", "interest"],
                ),
                ToolParameter(
                    name="tags",
                    type="array",
                    description="Optional list of tags for categorization.",
                    required=False,
                    items_type="string",
                ),
            ],
            handler=save_episodic_memory,
        )
    )

    registry.register(
        ToolDefinition(
            name="remove_episodic_memory",
            description="Remove episodic memory entries matching the given criteria. Can remove all memories for a user or filter by content, type, or tags.",
            category=ToolCategory.PERSONA_MEMORY,
            parameters=[
                ToolParameter(
                    name="user_id",
                    type="string",
                    description="The Discord user ID to remove memories for.",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Optional content string to match for selective removal. If provided, only memories containing this content will be removed.",
                    required=False,
                ),
                ToolParameter(
                    name="memory_type",
                    type="string",
                    description="Optional memory type to filter by (preference, fact, promise, interest).",
                    required=False,
                    enum=["preference", "fact", "promise", "interest"],
                ),
                ToolParameter(
                    name="tags",
                    type="array",
                    description="Optional list of tags to filter by. Memories matching any of these tags will be removed.",
                    required=False,
                    items_type="string",
                ),
            ],
            handler=remove_episodic_memory,
        )
    )

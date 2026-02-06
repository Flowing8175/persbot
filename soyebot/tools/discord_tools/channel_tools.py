"""Discord channel read-only tools."""

import logging
from typing import Any, Dict, List, Optional

import discord

from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def get_channel_info(
    channel_id: int,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get information about a Discord channel.

    Args:
        channel_id: The ID of the channel to get info for.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with channel information.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        channel = discord_context.guild.get_channel(channel_id)
        if not channel:
            return ToolResult(success=False, error=f"Channel {channel_id} not found")

        info = {
            "id": str(channel.id),
            "name": channel.name,
            "type": str(channel.type),
            "category": channel.category.name if channel.category else None,
            "position": channel.position,
            "nsfw": getattr(channel, "nsfw", False),
            "topic": getattr(channel, "topic", None),
            "slowmode_delay": getattr(channel, "slowmode_delay", 0),
        }

        return ToolResult(success=True, data=info)

    except Exception as e:
        logger.error("Error getting channel info: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def get_channel_history(
    channel_id: int,
    limit: int = 10,
    before_message_id: Optional[int] = None,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get recent messages from a Discord channel.

    Args:
        channel_id: The ID of the channel to get history from.
        limit: Maximum number of messages to retrieve (default: 10).
        before_message_id: Optional message ID to get history before.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with channel history.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        channel = discord_context.guild.get_channel(channel_id)
        if not channel:
            return ToolResult(success=False, error=f"Channel {channel_id} not found")

        if not hasattr(channel, "history"):
            return ToolResult(success=False, error="Channel does not support message history")

        # Build history kwargs
        kwargs = {"limit": min(limit, 100)}  # Cap at 100 for safety
        if before_message_id:
            kwargs["before"] = discord.Object(id=before_message_id)

        messages = []
        async for msg in channel.history(**kwargs):
            messages.append({
                "id": str(msg.id),
                "author": str(msg.author),
                "author_id": str(msg.author.id),
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
                "has_attachments": len(msg.attachments) > 0,
            })

        return ToolResult(success=True, data={"messages": messages, "count": len(messages)})

    except Exception as e:
        logger.error("Error getting channel history: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def get_message(
    message_id: int,
    channel_id: int,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get a specific Discord message.

    Args:
        message_id: The ID of the message to retrieve.
        channel_id: The ID of the channel the message is in.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with message information.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        channel = discord_context.guild.get_channel(channel_id)
        if not channel:
            return ToolResult(success=False, error=f"Channel {channel_id} not found")

        if not hasattr(channel, "fetch_message"):
            return ToolResult(success=False, error="Channel does not support fetching messages")

        msg = await channel.fetch_message(message_id)

        message_data = {
            "id": str(msg.id),
            "channel_id": str(msg.channel.id),
            "author": str(msg.author),
            "author_id": str(msg.author.id),
            "content": msg.content,
            "timestamp": msg.created_at.isoformat(),
            "edited_timestamp": msg.edited_at.isoformat() if msg.edited_at else None,
            "has_attachments": len(msg.attachments) > 0,
            "attachments": [
                {
                    "filename": a.filename,
                    "url": a.url,
                    "content_type": a.content_type,
                }
                for a in msg.attachments
            ] if msg.attachments else [],
            "reference": {
                "message_id": str(msg.reference.message_id),
                "channel_id": str(msg.reference.channel_id),
            } if msg.reference else None,
        }

        return ToolResult(success=True, data=message_data)

    except discord.NotFound:
        return ToolResult(success=False, error=f"Message {message_id} not found")
    except Exception as e:
        logger.error("Error getting message: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def list_channels(
    guild_id: int,
    channel_type: Optional[str] = None,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """List all channels in a Discord guild.

    Args:
        guild_id: The ID of the guild to list channels for.
        channel_type: Optional filter by channel type (text, voice, category).
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with list of channels.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        # Use the guild from context if it matches, otherwise try to get from client
        guild = discord_context.guild if discord_context.guild.id == guild_id else None
        if not guild and discord_context.bot:
            guild = discord_context.bot.get_guild(guild_id)

        if not guild:
            return ToolResult(success=False, error=f"Guild {guild_id} not found")

        channels = []
        for channel in guild.channels:
            # Filter by type if specified
            if channel_type:
                ch_type = str(channel.type).lower()
                if channel_type.lower() not in ch_type:
                    continue

            channel_info = {
                "id": str(channel.id),
                "name": channel.name,
                "type": str(channel.type),
                "category": channel.category.name if channel.category else None,
                "position": channel.position,
            }
            channels.append(channel_info)

        # Sort by position
        channels.sort(key=lambda x: x.get("position", 0))

        return ToolResult(success=True, data={"channels": channels, "count": len(channels)})

    except Exception as e:
        logger.error("Error listing channels: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


def register_channel_tools(registry):
    """Register all channel tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(ToolDefinition(
        name="get_channel_info",
        description="Get detailed information about a specific Discord channel including its name, type, category, and settings.",
        category=ToolCategory.DISCORD_CHANNEL,
        parameters=[
            ToolParameter(
                name="channel_id",
                type="integer",
                description="The ID of the channel to get information for",
                required=True,
            ),
        ],
        handler=get_channel_info,
        requires_permission="read_messages",
    ))

    registry.register(ToolDefinition(
        name="get_channel_history",
        description="Get recent messages from a Discord channel's history. Useful for reviewing recent conversations.",
        category=ToolCategory.DISCORD_CHANNEL,
        parameters=[
            ToolParameter(
                name="channel_id",
                type="integer",
                description="The ID of the channel to get history from",
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of messages to retrieve (default: 10, max: 100)",
                required=False,
                default=10,
            ),
            ToolParameter(
                name="before_message_id",
                type="integer",
                description="Optional message ID to get history before (for pagination)",
                required=False,
            ),
        ],
        handler=get_channel_history,
        requires_permission="read_message_history",
    ))

    registry.register(ToolDefinition(
        name="get_message",
        description="Get a specific Discord message by its ID. Useful for referencing or replying to a specific message.",
        category=ToolCategory.DISCORD_CHANNEL,
        parameters=[
            ToolParameter(
                name="message_id",
                type="integer",
                description="The ID of the message to retrieve",
                required=True,
            ),
            ToolParameter(
                name="channel_id",
                type="integer",
                description="The ID of the channel the message is in",
                required=True,
            ),
        ],
        handler=get_message,
        requires_permission="read_message_history",
    ))

    registry.register(ToolDefinition(
        name="list_channels",
        description="List all channels in a Discord guild. Can filter by channel type.",
        category=ToolCategory.DISCORD_CHANNEL,
        parameters=[
            ToolParameter(
                name="guild_id",
                type="integer",
                description="The ID of the guild to list channels for",
                required=True,
            ),
            ToolParameter(
                name="channel_type",
                type="string",
                description="Optional filter by channel type (e.g., 'text', 'voice', 'category')",
                required=False,
                enum=["text", "voice", "category", "news", "store"],
            ),
        ],
        handler=list_channels,
        requires_permission="read_messages",
    ))

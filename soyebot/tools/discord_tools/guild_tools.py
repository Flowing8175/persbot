"""Discord guild read-only tools."""

import logging
from typing import Any, Dict, List, Optional

import discord

from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def get_guild_info(
    guild_id: int,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get information about a Discord guild.

    Args:
        guild_id: The ID of the guild to get info for.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with guild information.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        guild = discord_context.guild if discord_context.guild.id == guild_id else None
        if not guild and discord_context.bot:
            guild = discord_context.bot.get_guild(guild_id)

        if not guild:
            return ToolResult(success=False, error=f"Guild {guild_id} not found")

        owner = guild.owner if guild.owner else None
        info = {
            "id": str(guild.id),
            "name": guild.name,
            "description": guild.description,
            "owner_id": str(guild.owner_id),
            "owner_name": str(owner) if owner else None,
            "region": str(guild.region) if hasattr(guild, "region") else None,
            "verification_level": str(guild.verification_level),
            "default_message_notifications": str(guild.default_notifications),
            "explicit_content_filter": str(guild.explicit_content_filter),
            "mfa_level": str(guild.mfa_level),
            "created_at": guild.created_at.isoformat(),
            "member_count": getattr(guild, "member_count", None),
            "role_count": len(guild.roles),
            "channel_count": len(guild.channels),
            "emoji_count": len(guild.emojis),
            "premium_tier": str(guild.premium_tier),
            "premium_subscription_count": guild.premium_subscription_count,
            "banner_url": guild.banner.url if guild.banner else None,
            "icon_url": guild.icon.url if guild.icon else None,
            "vanity_url_code": guild.vanity_url_code,
        }

        return ToolResult(success=True, data=info)

    except Exception as e:
        logger.error("Error getting guild info: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def get_guild_roles(
    guild_id: int,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get all roles in a Discord guild.

    Args:
        guild_id: The ID of the guild to get roles for.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with list of guild roles.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        guild = discord_context.guild if discord_context.guild.id == guild_id else None
        if not guild and discord_context.bot:
            guild = discord_context.bot.get_guild(guild_id)

        if not guild:
            return ToolResult(success=False, error=f"Guild {guild_id} not found")

        roles = []
        for role in guild.roles:
            role_info = {
                "id": str(role.id),
                "name": role.name,
                "color": str(role.color),
                "position": role.position,
                "permissions": str(role.permissions.value),
                "managed": role.managed,
                "mentionable": role.mentionable,
                "hoist": role.hoist,
                "is_default": role.is_default(),
                "tags": {
                    "bot_id": str(role.tags.bot_id) if role.tags and role.tags.bot_id else None,
                    "integration_id": str(role.tags.integration_id) if role.tags and role.tags.integration_id else None,
                    "premium_subscriber": role.tags.is_premium_subscriber() if role.tags else False,
                } if role.tags else None,
            }
            roles.append(role_info)

        # Sort by position (highest first)
        roles.sort(key=lambda x: -x.get("position", 0))

        return ToolResult(success=True, data={"roles": roles, "count": len(roles)})

    except Exception as e:
        logger.error("Error getting guild roles: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def get_guild_emojis(
    guild_id: int,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get custom emojis in a Discord guild.

    Args:
        guild_id: The ID of the guild to get emojis for.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with list of guild emojis.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        guild = discord_context.guild if discord_context.guild.id == guild_id else None
        if not guild and discord_context.bot:
            guild = discord_context.bot.get_guild(guild_id)

        if not guild:
            return ToolResult(success=False, error=f"Guild {guild_id} not found")

        emojis = []
        for emoji in guild.emojis:
            emoji_info = {
                "id": str(emoji.id),
                "name": emoji.name,
                "animated": emoji.animated,
                "available": emoji.available,
                "managed": emoji.managed,
                "require_colons": emoji.require_colons,
                "roles": [str(r) for r in emoji.roles] if emoji.roles else [],
                "url": str(emoji.url),
            }
            emojis.append(emoji_info)

        return ToolResult(success=True, data={"emojis": emojis, "count": len(emojis)})

    except Exception as e:
        logger.error("Error getting guild emojis: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


def register_guild_tools(registry):
    """Register all guild tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(ToolDefinition(
        name="get_guild_info",
        description="Get detailed information about a Discord guild including name, owner, member count, verification level, and features.",
        category=ToolCategory.DISCORD_GUILD,
        parameters=[
            ToolParameter(
                name="guild_id",
                type="integer",
                description="The ID of the guild to get information for",
                required=True,
            ),
        ],
        handler=get_guild_info,
        requires_permission="read_messages",
    ))

    registry.register(ToolDefinition(
        name="get_guild_roles",
        description="Get all roles in a Discord guild with their details including permissions, position, color, and special properties.",
        category=ToolCategory.DISCORD_GUILD,
        parameters=[
            ToolParameter(
                name="guild_id",
                type="integer",
                description="The ID of the guild to get roles for",
                required=True,
            ),
        ],
        handler=get_guild_roles,
        requires_permission="read_messages",
    ))

    registry.register(ToolDefinition(
        name="get_guild_emojis",
        description="Get all custom emojis in a Discord guild with their details including name, animation status, and URL.",
        category=ToolCategory.DISCORD_GUILD,
        parameters=[
            ToolParameter(
                name="guild_id",
                type="integer",
                description="The ID of the guild to get emojis for",
                required=True,
            ),
        ],
        handler=get_guild_emojis,
        requires_permission="read_messages",
    ))

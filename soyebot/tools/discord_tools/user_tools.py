"""Discord user/member read-only tools."""

import logging
from typing import Any, Dict, List, Optional

import discord

from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def get_user_info(
    user_id: int,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get information about a Discord user.

    Args:
        user_id: The ID of the user to get info for.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with user information.
    """
    if not discord_context or not discord_context.bot:
        return ToolResult(success=False, error="Discord context not available")

    try:
        user = await discord_context.bot.fetch_user(user_id)

        info = {
            "id": str(user.id),
            "username": user.name,
            "display_name": user.display_name,
            "discriminator": user.discriminator,
            "bot": user.bot,
            "created_at": user.created_at.isoformat(),
            "avatar_url": user.avatar.url if user.avatar else None,
            "global_name": user.global_name,
        }

        return ToolResult(success=True, data=info)

    except discord.NotFound:
        return ToolResult(success=False, error=f"User {user_id} not found")
    except Exception as e:
        logger.error("Error getting user info: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def get_member_info(
    user_id: int,
    guild_id: int,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get information about a Discord guild member.

    Args:
        user_id: The ID of the user to get member info for.
        guild_id: The ID of the guild.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with member information.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        guild = discord_context.guild if discord_context.guild.id == guild_id else None
        if not guild and discord_context.bot:
            guild = discord_context.bot.get_guild(guild_id)

        if not guild:
            return ToolResult(success=False, error=f"Guild {guild_id} not found")

        member = guild.get_member(user_id)
        if not member:
            # Try fetching if not in cache
            try:
                member = await guild.fetch_member(user_id)
            except discord.NotFound:
                return ToolResult(success=False, error=f"Member {user_id} not found in guild {guild_id}")

        info = {
            "id": str(member.id),
            "username": member.name,
            "display_name": member.display_name,
            "bot": member.bot,
            "joined_at": member.joined_at.isoformat() if member.joined_at else None,
            "premium_since": member.premium_since.isoformat() if member.premium_since else None,
            "pending": member.pending,
            "avatar_url": member.avatar.url if member.avatar else None,
            "guild_avatar_url": member.guild_avatar.url if member.guild_avatar else None,
            "is_owner": guild.owner_id == member.id,
        }

        return ToolResult(success=True, data=info)

    except Exception as e:
        logger.error("Error getting member info: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def get_member_roles(
    user_id: int,
    guild_id: int,
    discord_context: Optional[discord.Message] = None,
) -> ToolResult:
    """Get roles for a guild member.

    Args:
        user_id: The ID of the user to get roles for.
        guild_id: The ID of the guild.
        discord_context: Discord message context for accessing the client.

    Returns:
        ToolResult with member roles.
    """
    if not discord_context or not discord_context.guild:
        return ToolResult(success=False, error="Discord context not available or not in a guild")

    try:
        guild = discord_context.guild if discord_context.guild.id == guild_id else None
        if not guild and discord_context.bot:
            guild = discord_context.bot.get_guild(guild_id)

        if not guild:
            return ToolResult(success=False, error=f"Guild {guild_id} not found")

        member = guild.get_member(user_id)
        if not member:
            try:
                member = await guild.fetch_member(user_id)
            except discord.NotFound:
                return ToolResult(success=False, error=f"Member {user_id} not found in guild {guild_id}")

        roles = []
        for role in member.roles:
            # Skip @everyone role for cleaner output
            if role.is_default():
                continue

            role_info = {
                "id": str(role.id),
                "name": role.name,
                "color": str(role.color),
                "position": role.position,
                "permissions": str(role.permissions.value),
                "managed": role.managed,
                "mentionable": role.mentionable,
                "hoist": role.hoist,
            }
            roles.append(role_info)

        # Sort by position (highest first)
        roles.sort(key=lambda x: -x.get("position", 0))

        return ToolResult(success=True, data={"roles": roles, "count": len(roles)})

    except Exception as e:
        logger.error("Error getting member roles: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


def register_user_tools(registry):
    """Register all user tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(ToolDefinition(
        name="get_user_info",
        description="Get detailed information about a Discord user including username, display name, bot status, and account creation date.",
        category=ToolCategory.DISCORD_USER,
        parameters=[
            ToolParameter(
                name="user_id",
                type="integer",
                description="The ID of the user to get information for",
                required=True,
            ),
        ],
        handler=get_user_info,
    ))

    registry.register(ToolDefinition(
        name="get_member_info",
        description="Get detailed information about a Discord guild member including join date, premium status, and ownership status.",
        category=ToolCategory.DISCORD_USER,
        parameters=[
            ToolParameter(
                name="user_id",
                type="integer",
                description="The ID of the user to get member info for",
                required=True,
            ),
            ToolParameter(
                name="guild_id",
                type="integer",
                description="The ID of the guild the user is in",
                required=True,
            ),
        ],
        handler=get_member_info,
        requires_permission="read_messages",
    ))

    registry.register(ToolDefinition(
        name="get_member_roles",
        description="Get all roles assigned to a guild member, including role details like position, color, and permissions.",
        category=ToolCategory.DISCORD_USER,
        parameters=[
            ToolParameter(
                name="user_id",
                type="integer",
                description="The ID of the user to get roles for",
                required=True,
            ),
            ToolParameter(
                name="guild_id",
                type="integer",
                description="The ID of the guild the user is in",
                required=True,
            ),
        ],
        handler=get_member_roles,
        requires_permission="read_messages",
    ))

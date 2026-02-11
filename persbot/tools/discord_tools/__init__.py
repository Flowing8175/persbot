"""Discord read-only tools for SoyeBot AI.

This module provides tools for reading Discord information without
making any modifications (read-only operations).
"""

from persbot.tools.discord_tools.channel_tools import register_channel_tools
from persbot.tools.discord_tools.guild_tools import register_guild_tools
from persbot.tools.discord_tools.user_tools import register_user_tools

__all__ = [
    "register_channel_tools",
    "register_user_tools",
    "register_guild_tools",
]


def register_all_discord_tools(registry) -> None:
    """Register all Discord tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    register_channel_tools(registry)
    register_user_tools(registry)
    register_guild_tools(registry)

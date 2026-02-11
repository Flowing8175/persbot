"""External API tools for SoyeBot AI.

This module provides tools for accessing external APIs like search, weather, etc.
"""

from persbot.tools.api_tools.image_tools import register_image_tools
from persbot.tools.api_tools.search_tools import register_search_tools
from persbot.tools.api_tools.time_tools import register_time_tools
from persbot.tools.api_tools.weather_tools import register_weather_tools

__all__ = [
    "register_search_tools",
    "register_weather_tools",
    "register_time_tools",
    "register_image_tools",
]


def register_all_api_tools(registry) -> None:
    """Register all external API tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    register_search_tools(registry)
    register_weather_tools(registry)
    register_time_tools(registry)
    register_image_tools(registry)

"""Persona tools for SoyeBot AI.

This module provides tools for high-immersion persona bot functionality including:
- Episodic memory search (RAG)
- Situational snapshot generation
- Virtual routine status checking
- External content inspection
"""

from soyebot.tools.persona_tools.media_tools import register_media_tools
from soyebot.tools.persona_tools.memory_tools import register_memory_tools
from soyebot.tools.persona_tools.routine_tools import register_routine_tools
from soyebot.tools.persona_tools.web_tools import register_web_tools

__all__ = [
    "register_memory_tools",
    "register_media_tools",
    "register_routine_tools",
    "register_web_tools",
]


def register_all_persona_tools(registry):
    """Register all persona tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    register_memory_tools(registry)
    register_media_tools(registry)
    register_routine_tools(registry)
    register_web_tools(registry)

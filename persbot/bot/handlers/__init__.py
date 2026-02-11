"""Command handlers for SoyeBot.

This module contains handlers for Discord bot commands, separating
command logic from the cog classes.
"""

from .base_handler import BaseHandler
from .persona_handler import PersonaCommandHandler
from .model_handler import ModelCommandHandler

__all__ = [
    "BaseHandler",
    "PersonaCommandHandler",
    "ModelCommandHandler",
]

"""Prompts and persona configuration for SoyeBot."""

import logging
from pathlib import Path

from persbot.constants import (
    META_PROMPT,
    QUESTION_GENERATION_PROMPT,
    SUMMARY_SYSTEM_INSTRUCTION,
)

logger = logging.getLogger(__name__)

# Re-export prompts from constants for backward compatibility
__all__ = [
    "BOT_PERSONA_PROMPT",
    "SUMMARY_SYSTEM_INSTRUCTION",
    "META_PROMPT",
    "QUESTION_GENERATION_PROMPT",
    "load_persona",
]


# --- 페르소나 및 프롬프트 ---
def load_persona() -> str:
    try:
        path = Path("persbot/assets/persona.md")
        if not path.exists():
            # Try from root if running from elsewhere or inside container structure variants
            path = Path("assets/persona.md")

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return "System prompt could not be loaded."
    except Exception:
        logger.exception("Failed to load persona.md")
        return "System prompt error."


BOT_PERSONA_PROMPT = load_persona()

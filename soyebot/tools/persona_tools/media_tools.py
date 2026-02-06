"""Media generation tools for persona-based AI."""

import datetime as dt
import logging
from typing import Any, Dict, Optional

from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def generate_situational_snapshot(
    time_str: Optional[str] = None,
    location: Optional[str] = None,
    mood: Optional[str] = None,
    attire: Optional[str] = None,
    activity: Optional[str] = None,
    **kwargs,
) -> ToolResult:
    """Generate a detailed image generation prompt for the persona's current situation.

    This tool creates a structured prompt describing the persona's current
    circumstances (time, location, mood, attire) which can be used by
    the LLM to generate contextual responses or descriptions.

    Args:
        time_str: Time of day (e.g., "Morning", "Afternoon", "Night").
                  If not provided, inferred from current time.
        location: Current location (e.g., "Room", "Office", "Cafe").
        mood: Current mood state (e.g., "Happy", "Tired", "Excited").
        attire: Current clothing or outfit description.
        activity: Current activity being performed.

    Returns:
        ToolResult with a detailed image generation prompt string.
    """
    try:
        # Infer time if not provided
        if not time_str:
            hour = dt.datetime.now().hour
            if 5 <= hour < 12:
                time_str = "Morning"
            elif 12 <= hour < 18:
                time_str = "Afternoon"
            elif 18 <= hour < 22:
                time_str = "Evening"
            else:
                time_str = "Night"
            inferred_time = True
        else:
            inferred_time = False

        # Build prompt components
        prompt_parts = ["Image Generation Prompt:"]
        prompt_parts.append(f"[Time: {time_str}")
        prompt_parts.append(f"Location: {location or 'Indoors'}")

        if mood:
            prompt_parts.append(f"Mood: {mood}")

        if attire:
            prompt_parts.append(f"Attire: {attire}")

        if activity:
            prompt_parts.append(f"Activity: {activity}")

        # Add contextual details based on time
        if inferred_time:
            hour = dt.datetime.now().hour
            if 6 <= hour < 18:
                prompt_parts.append("Lighting: Natural sunlight")
            else:
                prompt_parts.append("Lighting: Warm indoor lighting")

        # Add atmosphere details based on mood
        if mood:
            mood_lower = mood.lower()
            if any(word in mood_lower for word in ["happy", "excited", "joyful"]):
                prompt_parts.append("Atmosphere: Vibrant, energetic")
            elif any(word in mood_lower for word in ["tired", "sleepy", "exhausted"]):
                prompt_parts.append("Atmosphere: Calm, peaceful")
            elif any(word in mood_lower for word in ["sad", "depressed", "lonely"]):
                prompt_parts.append("Atmosphere: Somber, contemplative")
            elif any(word in mood_lower for word in ["focused", "working", "studying"]):
                prompt_parts.append("Atmosphere: Concentrated, productive")
            else:
                prompt_parts.append("Atmosphere: Balanced, natural")

        # Close prompt
        prompt_parts.append("Style: Detailed, high quality]")

        full_prompt = ", ".join(prompt_parts)

        return ToolResult(
            success=True,
            data={
                "prompt": full_prompt,
                "time": time_str,
                "location": location or "Indoors",
                "mood": mood or "Neutral",
                "attire": attire or "Casual",
                "activity": activity or "Idle",
                "inferred_time": inferred_time,
            },
        )

    except Exception as e:
        logger.error("Error generating situational snapshot: %s", e, exc_info=True)
        return ToolResult(
            success=False, error=f"Failed to generate situational snapshot: {str(e)}"
        )


def register_media_tools(registry):
    """Register media tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="generate_situational_snapshot",
            description=(
                "Generate a detailed image generation prompt describing the persona's "
                "current situation (time, location, mood, attire, activity). "
                "Useful for contextualizing responses or creating visual descriptions."
            ),
            category=ToolCategory.PERSONA_MEDIA,
            parameters=[
                ToolParameter(
                    name="time_str",
                    type="string",
                    description="Time of day (Morning, Afternoon, Evening, Night). Inferred if not provided.",
                    required=False,
                ),
                ToolParameter(
                    name="location",
                    type="string",
                    description="Current location (e.g., Room, Office, Cafe). Default: Indoors.",
                    required=False,
                ),
                ToolParameter(
                    name="mood",
                    type="string",
                    description="Current mood state (e.g., Happy, Tired, Focused).",
                    required=False,
                ),
                ToolParameter(
                    name="attire",
                    type="string",
                    description="Current clothing or outfit description.",
                    required=False,
                ),
                ToolParameter(
                    name="activity",
                    type="string",
                    description="Current activity being performed (e.g., Reading, Working, Relaxing).",
                    required=False,
                ),
            ],
            handler=generate_situational_snapshot,
            rate_limit=5,
        )
    )

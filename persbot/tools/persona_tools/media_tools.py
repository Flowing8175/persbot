"""Media generation tools for SoyeBot AI.

This module provides situational snapshot generation functionality for persona bots.
It generates detailed image generation prompts based on the persona's current situation.
"""

import logging
from datetime import datetime
from typing import Optional

from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


async def generate_situational_snapshot(
    time_of_day: Optional[str] = None,
    location: Optional[str] = None,
    mood: Optional[str] = None,
    activity: Optional[str] = None,
    **kwargs,
) -> ToolResult:
    """Generate a detailed situational snapshot prompt for image generation.

    This tool creates a comprehensive image generation prompt based on the persona's
    current situation, including time, location, mood, and activity. The generated
    prompt can be used with image generation models to visualize the persona's state.

    Args:
        time_of_day: Current time of day (morning, afternoon, evening, night).
        location: Current location (room, desk, cafe, outdoors, etc.).
        mood: Current emotional state (happy, tired, focused, relaxed, etc.).
        activity: Current activity (working, reading, gaming, resting, etc.).

    Returns:
        ToolResult with a detailed image generation prompt.
    """
    try:
        # Infer current time if not provided
        if not time_of_day:
            hour = datetime.now().hour
            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

        # Default values for missing parameters
        location = location or "cozy room"
        mood = mood or "calm"
        activity = activity or "relaxing"

        # Build detailed prompt components
        time_descriptions = {
            "morning": "soft morning light streaming through window, fresh atmosphere",
            "afternoon": "bright afternoon sunlight, energetic atmosphere",
            "evening": "warm golden hour light, peaceful evening ambiance",
            "night": "dim ambient lighting, cozy nighttime atmosphere",
        }

        mood_styles = {
            "happy": "bright and cheerful expression, vibrant colors",
            "tired": "slightly droopy eyes, soft and muted colors",
            "focused": "intense gaze, organized and clean environment",
            "relaxed": "comfortable posture, warm and inviting colors",
            "sad": "melancholic expression, cooler and muted tones",
            "excited": "energetic pose, dynamic and bright composition",
            "calm": "serene expression, balanced and harmonious colors",
        }

        activity_details = {
            "working": "sitting at desk with computer, focused on work",
            "reading": "holding a book or e-reader, immersed in reading",
            "gaming": "at gaming setup, wearing headphones, engaged in game",
            "resting": "lying down or reclining, taking a break",
            "relaxing": "comfortable seating position, enjoying leisure time",
            "eating": "at table with food and drink, having a meal",
            "studying": "with books and notes, deep in concentration",
        }

        # Get descriptions or use generic fallbacks
        time_desc = time_descriptions.get(time_of_day.lower(), f"{time_of_day} time setting")
        mood_style = mood_styles.get(mood.lower(), f"{mood} emotional state")
        activity_detail = activity_details.get(activity.lower(), f"engaged in {activity}")

        # Construct the full prompt
        prompt_parts = [
            f"Image Generation Prompt for Persona Snapshot:",
            f"Time: {time_of_day.capitalize()} - {time_desc}",
            f"Location: {location} - detailed environment with appropriate furniture and decor",
            f"Mood: {mood} - {mood_style}",
            f"Activity: {activity} - {activity_detail}",
            "",
            "Visual Style: Anime/illustration style, high quality, detailed character design,",
            "consistent proportions, expressive eyes, appropriate clothing for the setting,",
            "proper lighting matching the time of day, atmospheric depth and background details.",
        ]

        full_prompt = "\n".join(prompt_parts)

        # Also return structured data for programmatic use
        structured_data = {
            "time_of_day": time_of_day,
            "location": location,
            "mood": mood,
            "activity": activity,
            "image_prompt": full_prompt,
            "timestamp": datetime.now().isoformat(),
        }

        return ToolResult(
            success=True,
            data=structured_data,
            metadata={"prompt_type": "image_generation", "format": "structured"},
        )

    except Exception as e:
        logger.error("Error generating situational snapshot: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def describe_scene_atmosphere(
    mood: str,
    setting: str,
    **kwargs,
) -> ToolResult:
    """Generate a detailed description of scene atmosphere for roleplay contexts.

    Args:
        mood: The emotional atmosphere (cozy, tense, romantic, mysterious, etc.).
        setting: The physical setting (bedroom, cafe, park, office, etc.).

    Returns:
        ToolResult with atmospheric description.
    """
    atmosphere_prompts = {
        "cozy": "warm soft lighting, comfortable furniture, plush textures, warm color palette, sense of safety and comfort",
        "tense": "dramatic lighting, sharp shadows, constrained space, cool color tones, sense of anticipation",
        "romantic": "soft warm lighting, intimate space, gentle colors, romantic atmosphere, emotional connection",
        "mysterious": "dim lighting, shadows, fog or mist, unusual elements, sense of intrigue and wonder",
        "peaceful": "natural light, open space, calming colors, sense of tranquility and balance",
        "energetic": "bright vivid lighting, dynamic composition, bold colors, sense of movement and excitement",
    }

    atmosphere = atmosphere_prompts.get(
        mood.lower(), f"{mood} atmosphere with appropriate lighting and mood elements"
    )

    description = (
        f"Scene Atmosphere Description:\nSetting: {setting}\nMood: {mood}\nAtmosphere: {atmosphere}"
    )

    return ToolResult(
        success=True,
        data={
            "mood": mood,
            "setting": setting,
            "atmosphere_description": description,
        },
    )


def register_media_tools(registry) -> None:
    """Register media generation tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="generate_situational_snapshot",
            description="Generate a detailed image generation prompt based on the persona's current situation (time, location, mood, activity). Returns a structured prompt for visualizing the persona's state.",
            category=ToolCategory.PERSONA_MEDIA,
            parameters=[
                ToolParameter(
                    name="time_of_day",
                    type="string",
                    description="Current time of day (morning, afternoon, evening, night). If not provided, inferred from actual time.",
                    required=False,
                    enum=["morning", "afternoon", "evening", "night"],
                ),
                ToolParameter(
                    name="location",
                    type="string",
                    description="Current location (e.g., room, desk, cafe, outdoors). Default: cozy room.",
                    required=False,
                ),
                ToolParameter(
                    name="mood",
                    type="string",
                    description="Current emotional state (e.g., happy, tired, focused, relaxed). Default: calm.",
                    required=False,
                ),
                ToolParameter(
                    name="activity",
                    type="string",
                    description="Current activity (e.g., working, reading, gaming, resting). Default: relaxing.",
                    required=False,
                ),
            ],
            handler=generate_situational_snapshot,
        )
    )

    registry.register(
        ToolDefinition(
            name="describe_scene_atmosphere",
            description="Generate a detailed description of scene atmosphere for roleplay contexts. Useful for setting the mood and visual tone.",
            category=ToolCategory.PERSONA_MEDIA,
            parameters=[
                ToolParameter(
                    name="mood",
                    type="string",
                    description="The emotional atmosphere (cozy, tense, romantic, mysterious, peaceful, energetic).",
                    required=True,
                    enum=["cozy", "tense", "romantic", "mysterious", "peaceful", "energetic"],
                ),
                ToolParameter(
                    name="setting",
                    type="string",
                    description="The physical setting (bedroom, cafe, park, office, etc.).",
                    required=True,
                ),
            ],
            handler=describe_scene_atmosphere,
        )
    )

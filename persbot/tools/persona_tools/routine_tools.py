"""Routine checking tools for SoyeBot AI.

This module provides virtual daily routine status checking for persona bots.
It determines the persona's availability and context based on time-based schedules.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Default path for persona schedule data
DEFAULT_SCHEDULE_PATH = "data/persona_schedule.json"


async def check_virtual_routine_status(
    **kwargs,
) -> ToolResult:
    """Check the persona's current virtual routine status based on time.

    This tool checks the persona's current activity and availability based on
    a predefined schedule. Returns status information including whether the
    persona is available, busy, or sleeping, along with appropriate response
    style recommendations.

    Returns:
        ToolResult with current routine status and context.
    """
    try:
        # Load the persona schedule
        schedule = await _load_persona_schedule()

        # Get current time
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        current_time = current_hour * 100 + current_minute  # Convert to HHMM format

        # Find the active schedule entry
        active_entry = None
        for entry in schedule.get("schedule", []):
            start_time = _time_to_minutes(entry.get("start", "0000"))
            end_time = _time_to_minutes(entry.get("end", "2359"))

            # Handle overnight schedules (e.g., 23:00 to 07:00)
            if start_time > end_time:
                # Schedule spans midnight
                if current_time >= start_time or current_time <= end_time:
                    active_entry = entry
                    break
            else:
                # Normal schedule within same day
                if start_time <= current_time <= end_time:
                    active_entry = entry
                    break

        # If no active entry found, use default
        if not active_entry:
            active_entry = {
                "status": "Available",
                "activity": "free time",
                "response_style": "Normal",
                "context": "The persona is currently available for conversation.",
            }

        # Get current day of week for additional context
        day_of_week = now.strftime("%A")
        is_weekend = day_of_week in ["Saturday", "Sunday"]

        # Build the response
        result = {
            "status": active_entry.get("status", "Available"),
            "activity": active_entry.get("activity", "unknown"),
            "response_style": active_entry.get("response_style", "Normal"),
            "context": active_entry.get("context", ""),
            "current_time": now.strftime("%H:%M"),
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "timestamp": now.isoformat(),
        }

        # Add special response recommendations
        if result["status"] == "Sleeping":
            result["response_recommendation"] = {
                "style": "Groggy/Delayed",
                "tone": "Sleepy, potentially confused, may take time to respond",
                "suggested_prefix": "(sleepily) ",
            }
        elif result["status"] == "Busy":
            result["response_recommendation"] = {
                "style": "Brief/Focused",
                "tone": "Concise, may seem distracted",
                "suggested_prefix": "(while busy) ",
            }
        elif result["status"] == "Available":
            result["response_recommendation"] = {
                "style": "Full/Engaged",
                "tone": "Attentive, responsive, conversational",
                "suggested_prefix": "",
            }

        return ToolResult(success=True, data=result)

    except Exception as e:
        logger.error("Error checking routine status: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def get_routine_schedule(
    day: Optional[str] = None,
    **kwargs,
) -> ToolResult:
    """Get the persona's full routine schedule or specific day's schedule.

    Args:
        day: Optional specific day of week (Monday, Tuesday, etc.) to filter by.

    Returns:
        ToolResult with the persona's schedule information.
    """
    try:
        schedule = await _load_persona_schedule()

        if day:
            # Filter by specific day
            day_schedule = [
                entry
                for entry in schedule.get("schedule", [])
                if day.lower() in [d.lower() for d in entry.get("days", ["Everyday"])]
            ]
            return ToolResult(
                success=True,
                data={
                    "day": day,
                    "schedule": day_schedule,
                },
            )
        else:
            # Return full schedule
            return ToolResult(
                success=True,
                data={
                    "persona_name": schedule.get("persona_name", "Unknown"),
                    "schedule": schedule.get("schedule", []),
                    "notes": schedule.get("notes", ""),
                },
            )

    except Exception as e:
        logger.error("Error getting routine schedule: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def _load_persona_schedule() -> Dict[str, Any]:
    """Load persona schedule from the local JSON file.

    Returns:
        Dictionary with persona schedule information.
    """
    schedule_path = os.environ.get("PERSONA_SCHEDULE_PATH", DEFAULT_SCHEDULE_PATH)

    if not os.path.exists(schedule_path):
        # Create a sample schedule file for demonstration
        await _create_sample_schedule_file(schedule_path)

    try:
        with open(schedule_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in schedule file, returning default schedule")
        return _get_default_schedule()
    except Exception as e:
        logger.error("Error loading schedule file: %s", e)
        return _get_default_schedule()


async def _create_sample_schedule_file(path: str) -> None:
    """Create a sample persona schedule file for demonstration purposes.

    Args:
        path: Path where to create the sample schedule file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sample_schedule = _get_default_schedule()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_schedule, f, ensure_ascii=False, indent=2)

    logger.info("Created sample schedule file at %s", path)


def _get_default_schedule() -> Dict[str, Any]:
    """Get the default persona schedule.

    Returns:
        Default schedule dictionary.
    """
    return {
        "persona_name": "Soye",
        "notes": "Default daily schedule for the persona",
        "schedule": [
            {
                "start": "0700",
                "end": "0900",
                "status": "Available",
                "activity": "morning routine",
                "response_style": "Normal",
                "days": ["Everyday"],
                "context": "The persona is waking up and preparing for the day.",
            },
            {
                "start": "0900",
                "end": "1200",
                "status": "Busy",
                "activity": "work/study",
                "response_style": "Brief",
                "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "context": "The persona is focused on work or study.",
            },
            {
                "start": "1200",
                "end": "1300",
                "status": "Available",
                "activity": "lunch break",
                "response_style": "Relaxed",
                "days": ["Everyday"],
                "context": "The persona is taking a lunch break.",
            },
            {
                "start": "1300",
                "end": "1800",
                "status": "Busy",
                "activity": "work/study",
                "response_style": "Brief",
                "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "context": "The persona is focused on work or study.",
            },
            {
                "start": "1800",
                "end": "2300",
                "status": "Available",
                "activity": "free time",
                "response_style": "Normal",
                "days": ["Everyday"],
                "context": "The persona is relaxing and available for conversation.",
            },
            {
                "start": "2300",
                "end": "0700",
                "status": "Sleeping",
                "activity": "sleeping",
                "response_style": "Groggy/Delayed",
                "days": ["Everyday"],
                "context": "The persona is sleeping. Responses will be delayed.",
            },
        ],
    }


def _time_to_minutes(time_str: str) -> int:
    """Convert time string (HHMM or HH:MM) to minutes since midnight.

    Args:
        time_str: Time string in format HHMM or HH:MM.

    Returns:
        Minutes since midnight.
    """
    time_str = time_str.replace(":", "")
    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    return hours * 60 + minutes


def register_routine_tools(registry) -> None:
    """Register routine checking tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="check_virtual_routine_status",
            description="Check the persona's current virtual routine status based on time of day. Returns availability (Available/Busy/Sleeping), current activity, and recommended response style.",
            category=ToolCategory.PERSONA_ROUTINE,
            parameters=[],
            handler=check_virtual_routine_status,
        )
    )

    registry.register(
        ToolDefinition(
            name="get_routine_schedule",
            description="Get the persona's full daily routine schedule or filter by specific day. Useful for understanding the persona's typical daily patterns.",
            category=ToolCategory.PERSONA_ROUTINE,
            parameters=[
                ToolParameter(
                    name="day",
                    type="string",
                    description="Optional specific day of week (Monday, Tuesday, etc.) to filter by. If not provided, returns full schedule.",
                    required=False,
                    enum=[
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ],
                ),
            ],
            handler=get_routine_schedule,
        )
    )

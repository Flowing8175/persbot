"""Routine and schedule tools for persona-based AI."""

import json
import logging
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def check_virtual_routine_status(
    **kwargs,
) -> ToolResult:
    """Check the persona's current routine status and availability.

    This tool examines the persona's schedule to determine current state
    (Available, Busy, Sleeping) and provides contextual information for
    appropriate response style.

    Args:
        None - Uses current time and schedule from persona_schedule.json.

    Returns:
        ToolResult with status, context, and response_style information.
    """
    try:
        # Load persona schedule
        schedule_file = Path("data/persona_schedule.json")
        if not schedule_file.exists():
            # Return default status if no schedule file exists
            current_hour = dt.datetime.now().hour
            if 6 <= current_hour < 23:
                return ToolResult(
                    success=True,
                    data={
                        "status": "Available",
                        "response_style": "Normal",
                        "context": "Using default routine - Persona is available",
                        "current_time": dt.datetime.now().strftime("%H:%M"),
                        "message": "No persona schedule found. Using default behavior.",
                    },
                )
            else:
                return ToolResult(
                    success=True,
                    data={
                        "status": "Sleeping",
                        "response_style": "Groggy/Delayed",
                        "context": "Using default routine - Persona is sleeping",
                        "current_time": dt.datetime.now().strftime("%H:%M"),
                        "message": "No persona schedule found. Using default sleep schedule.",
                    },
                )

        with open(schedule_file, "r", encoding="utf-8") as f:
            schedule_data = json.load(f)

        # Get current time in minutes since midnight
        now = dt.datetime.now()
        current_minutes = now.hour * 60 + now.minute
        current_time_str = now.strftime("%H:%M")

        # Get sleep schedule
        sleep_start_str = schedule_data.get("sleep_start", "23:00")
        sleep_end_str = schedule_data.get("sleep_end", "07:00")

        # Parse sleep times
        sleep_start_hour, sleep_start_min = map(int, sleep_start_str.split(":"))
        sleep_end_hour, sleep_end_min = map(int, sleep_end_str.split(":"))

        sleep_start = sleep_start_hour * 60 + sleep_start_min
        sleep_end = sleep_end_hour * 60 + sleep_end_min

        # Check if sleeping
        is_sleeping = False
        if sleep_start < sleep_end:
            # Sleep within same day (e.g., 23:00 - 07:00 crosses midnight)
            # Actually this is the same as below
            if sleep_start <= current_minutes < sleep_end:
                is_sleeping = False  # Sleep is 23:00 to 07:00 next day
            elif current_minutes >= sleep_start or current_minutes < sleep_end:
                is_sleeping = True
        else:
            # Sleep crosses midnight
            if current_minutes >= sleep_start or current_minutes < sleep_end:
                is_sleeping = True
            else:
                is_sleeping = False

        # Check busy periods
        busy_periods = schedule_data.get("busy_periods", [])
        is_busy = False
        busy_context = None

        for period in busy_periods:
            start_str = period.get("start")
            end_str = period.get("end")
            context = period.get("context", "Busy")

            if not start_str or not end_str:
                continue

            start_hour, start_min = map(int, start_str.split(":"))
            end_hour, end_min = map(int, end_str.split(":"))

            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min

            # Check if current time falls within busy period
            in_period = False
            if start_minutes < end_minutes:
                if start_minutes <= current_minutes < end_minutes:
                    in_period = True
            else:
                # Period crosses midnight
                if current_minutes >= start_minutes or current_minutes < end_minutes:
                    in_period = True

            if in_period:
                is_busy = True
                busy_context = context
                break

        # Determine status and response style
        if is_sleeping:
            status = "Sleeping"
            response_style = "Groggy/Delayed"
            context_msg = "Persona is currently sleeping and may respond slowly or with simple responses."
        elif is_busy:
            status = "Busy"
            response_style = "Brief/Focused"
            context_msg = (
                f"Persona is currently busy: {busy_context}. Responses may be shorter."
            )
        else:
            status = "Available"
            response_style = "Normal"
            context_msg = "Persona is available and ready to engage in conversation."

        return ToolResult(
            success=True,
            data={
                "status": status,
                "response_style": response_style,
                "context": context_msg,
                "current_time": current_time_str,
                "sleep_schedule": f"{sleep_start_str} - {sleep_end_str}",
                "busy_periods": busy_periods,
                "is_sleeping": is_sleeping,
                "is_busy": is_busy,
            },
        )

    except json.JSONDecodeError as e:
        logger.error("Failed to parse schedule data: %s", e)
        return ToolResult(
            success=False, error=f"Failed to parse schedule data: {str(e)}"
        )
    except Exception as e:
        logger.error("Error checking routine status: %s", e, exc_info=True)
        return ToolResult(
            success=False, error=f"Failed to check routine status: {str(e)}"
        )


def register_routine_tools(registry):
    """Register routine tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="check_virtual_routine_status",
            description=(
                "Check the persona's current routine status (Available, Busy, Sleeping). "
                "Returns appropriate response style (e.g., Groggy for sleep, Brief for busy). "
                "Use this to adjust response timing and tone based on persona's schedule."
            ),
            category=ToolCategory.PERSONA_ROUTINE,
            parameters=[],
            handler=check_virtual_routine_status,
            rate_limit=60,
        )
    )

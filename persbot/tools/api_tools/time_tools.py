"""Time and timezone tools for SoyeBot AI."""

import logging
from datetime import datetime, timezone
from typing import Optional

from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Common timezone mappings
TIMEZONE_MAPPINGS = {
    "seoul": "Asia/Seoul",
    "tokyo": "Asia/Tokyo",
    "new york": "America/New_York",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "moscow": "Europe/Moscow",
    "dubai": "Asia/Dubai",
    "mumbai": "Asia/Kolkata",
    "sydney": "Australia/Sydney",
    "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "denver": "America/Denver",
    "utc": "UTC",
    "gmt": "GMT",
}


async def get_time(
    timezone_str: Optional[str] = None,
    **kwargs,  # Accept extra kwargs for compatibility with executor
) -> ToolResult:
    """Get the current time in a specific timezone.

    Args:
        timezone_str: The timezone name (e.g., 'Asia/Seoul', 'America/New_York').
                     Can also be a city name like 'Seoul', 'Tokyo', 'New York'.

    Returns:
        ToolResult with time information.
    """
    try:
        # Normalize timezone input
        tz_str = timezone_str.lower().strip() if timezone_str else "utc"

        # Check for city name mappings
        if tz_str in TIMEZONE_MAPPINGS:
            tz_str = TIMEZONE_MAPPINGS[tz_str]
        elif "/" not in tz_str.upper():
            # Try to find a matching timezone
            import pytz

            for tz_name in pytz.all_timezones:
                if tz_str in tz_name.lower():
                    tz_str = tz_name
                    break

        # Import pytz for timezone handling
        import pytz

        try:
            tz = pytz.timezone(tz_str)
        except pytz.UnknownTimeZoneError:
            # Fallback to common mappings or UTC
            if tz_str in TIMEZONE_MAPPINGS.values():
                tz = pytz.timezone(tz_str)
            else:
                tz = pytz.UTC
                tz_str = "UTC"

        now = datetime.now(tz)

        time_info = {
            "timezone": str(tz),
            "datetime": now.isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "utc_offset": now.strftime("%z"),
            "unix_timestamp": int(now.timestamp()),
        }

        return ToolResult(success=True, data=time_info)

    except ImportError:
        # pytz not available, use basic timezone support
        return await get_time_basic(timezone_str)
    except Exception as e:
        logger.error("Error getting time: %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


async def get_time_basic(timezone_str: Optional[str] = None, **kwargs) -> ToolResult:
    """Get current time using basic timezone support (fallback)."""
    try:
        tz_str = timezone_str.lower().strip() if timezone_str else "utc"

        # Basic UTC offset mapping (limited)
        utc_offsets = {
            "seoul": 9 * 3600,
            "tokyo": 9 * 3600,
            "new york": -5 * 3600,
            "london": 0,
            "paris": 1 * 3600,
            "berlin": 1 * 3600,
            "moscow": 3 * 3600,
            "dubai": 4 * 3600,
            "mumbai": 5.5 * 3600,
            "sydney": 10 * 3600,
            "los angeles": -8 * 3600,
            "chicago": -6 * 3600,
            "denver": -7 * 3600,
        }

        offset = utc_offsets.get(tz_str, 0)

        if offset == 0 and tz_str != "utc" and tz_str != "gmt":
            # Unknown timezone, use UTC
            tz_str = "UTC"

        now = datetime.now(timezone(offset))

        time_info = {
            "timezone": tz_str.upper(),
            "datetime": now.isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "utc_offset": f"{int(offset / 3600):+03d}:00",
            "unix_timestamp": int(now.timestamp()),
        }

        return ToolResult(success=True, data=time_info)

    except Exception as e:
        logger.error("Error getting time (basic): %s", e, exc_info=True)
        return ToolResult(success=False, error=str(e))


def register_time_tools(registry) -> None:
    """Register time tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(
        ToolDefinition(
            name="get_time",
            description="Get the current time and date for any timezone or city worldwide. Returns the local time, date, day of week, and timezone information.",
            category=ToolCategory.API_TIME,
            parameters=[
                ToolParameter(
                    name="timezone",
                    type="string",
                    description="The timezone name (e.g., 'Asia/Seoul', 'America/New_York') or city name (e.g., 'Seoul', 'Tokyo', 'London'). Defaults to UTC.",
                    required=False,
                ),
            ],
            handler=get_time,
        )
    )

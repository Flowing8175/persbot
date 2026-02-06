"""Weather information tools for SoyeBot AI."""

import logging
from typing import Any, Dict, List, Optional

import aiohttp
from soyebot.tools.base import ToolDefinition, ToolParameter, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


async def get_weather(
    location: str,
    units: str = "metric",
    weather_api_key: Optional[str] = None,
    **kwargs,
) -> ToolResult:
    """Get current weather information for a location.

    Args:
        location: The location name (e.g., "Seoul", "New York", "Tokyo").
        units: Unit system - "metric" (Celsius) or "imperial" (Fahrenheit).
        weather_api_key: Optional API key for weather service.

    Returns:
        ToolResult with weather information.
    """
    if not location or not location.strip():
        return ToolResult(success=False, error="Location cannot be empty")

    units = units.lower()
    if units not in ("metric", "imperial"):
        units = "metric"

    # OpenWeatherMap API (requires API key)
    if weather_api_key:
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    "q": location,
                    "appid": weather_api_key,
                    "units": units,
                }

                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        weather_info = {
                            "location": data.get("name", location),
                            "country": data.get("sys", {}).get("country", ""),
                            "temperature": data.get("main", {}).get("temp"),
                            "feels_like": data.get("main", {}).get("feels_like"),
                            "humidity": data.get("main", {}).get("humidity"),
                            "pressure": data.get("main", {}).get("pressure"),
                            "wind_speed": data.get("wind", {}).get("speed"),
                            "wind_direction": data.get("wind", {}).get("deg"),
                            "visibility": data.get("visibility"),
                            "clouds": data.get("clouds", {}).get("all"),
                            "description": data.get("weather", [{}])[0].get("description", ""),
                            "icon": data.get("weather", [{}])[0].get("icon", ""),
                            "units": units,
                        }

                        return ToolResult(success=True, data=weather_info)
                    elif resp.status == 401:
                        return ToolResult(success=False, error="Invalid weather API key")
                    elif resp.status == 404:
                        return ToolResult(success=False, error=f"Location '{location}' not found")
                    else:
                        return ToolResult(success=False, error=f"Weather API error: {resp.status}")

        except aiohttp.ClientError as e:
            logger.error("Weather API request failed: %s", e)
            return ToolResult(success=False, error="Failed to fetch weather data")
        except Exception as e:
            logger.error("Weather lookup error: %s", e, exc_info=True)
            return ToolResult(success=False, error=str(e))

    # Fallback: Use wttr.in (no API key required, but limited)
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://wttr.in/{location}?format=j1"
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    current = data.get("current_condition", [{}])[0]

                    # Convert units if needed
                    temp = current.get("temp_C", "")
                    if units == "imperial" and temp:
                        try:
                            temp_f = float(temp) * 9 / 5 + 32
                            temp = f"{temp_f:.1f}"
                        except ValueError:
                            pass

                    weather_info = {
                        "location": location,
                        "temperature": temp,
                        "feels_like": current.get("FeelsLikeC", ""),
                        "humidity": current.get("humidity", ""),
                        "description": current.get("weatherDesc", [{}])[0].get("value", ""),
                        "wind_speed": current.get("windspeedKmph", ""),
                        "wind_direction": current.get("winddir16Point", ""),
                        "visibility": current.get("visibility", ""),
                        "uv_index": current.get("uvIndex", ""),
                        "units": "metric" if units == "metric" else "imperial",
                        "source": "wttr.in",
                    }

                    return ToolResult(success=True, data=weather_info)
    except Exception as e:
        logger.debug("wttr.in weather lookup failed: %s", e)

    return ToolResult(
        success=False,
        error="Weather information requires a weather API key. Please configure WEATHER_API_KEY in your environment to enable weather functionality.",
    )


def register_weather_tools(registry):
    """Register weather tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools with.
    """
    registry.register(ToolDefinition(
        name="get_weather",
        description="Get current weather information for any location worldwide. Returns temperature, humidity, wind conditions, and weather description.",
        category=ToolCategory.API_WEATHER,
        parameters=[
            ToolParameter(
                name="location",
                type="string",
                description="The location name (e.g., 'Seoul', 'New York', 'Tokyo', 'London')",
                required=True,
            ),
            ToolParameter(
                name="units",
                type="string",
                description="Unit system - 'metric' for Celsius or 'imperial' for Fahrenheit",
                required=False,
                default="metric",
                enum=["metric", "imperial"],
            ),
        ],
        handler=get_weather,
        rate_limit=60,  # 60 seconds between weather lookups
    ))

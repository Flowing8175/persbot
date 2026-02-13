"""Feature tests for weather tools module.

Tests focus on behavior using mocking:
- get_weather: get weather information
- register_weather_tools: register tools with registry
"""

import asyncio
import sys
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest


# Mock external dependencies before any imports
_mock_ddgs = MagicMock()
_mock_ddgs.DDGS = MagicMock
_mock_ddgs.exceptions = MagicMock()
_mock_ddgs.exceptions.RatelimitException = Exception
_mock_ddgs.exceptions.DDGSException = Exception
sys.modules['ddgs'] = _mock_ddgs
sys.modules['ddgs.exceptions'] = _mock_ddgs.exceptions

_mock_bs4 = MagicMock()
sys.modules['bs4'] = _mock_bs4


class TestGetWeather:
    """Tests for get_weather function."""

    @pytest.fixture
    def mock_aiohttp(self):
        """Mock aiohttp module."""
        mock_aiohttp = MagicMock()
        mock_session = AsyncMock()

        # Create async context manager for ClientSession
        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp.ClientSession.return_value = session_cm

        sys.modules['aiohttp'] = mock_aiohttp
        yield mock_aiohttp
        if 'aiohttp' in sys.modules:
            del sys.modules['aiohttp']

    @pytest.mark.asyncio
    async def test_get_weather_exists(self):
        """get_weather function exists."""
        from persbot.tools.api_tools.weather_tools import get_weather
        assert get_weather is not None

    @pytest.mark.asyncio
    async def test_get_weather_returns_tool_result(self):
        """get_weather returns ToolResult."""
        from persbot.tools.api_tools.weather_tools import get_weather
        from persbot.tools.base import ToolResult

        result = await get_weather(location="Seoul")
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_get_weather_empty_location_returns_error(self):
        """get_weather returns error for empty location."""
        from persbot.tools.api_tools.weather_tools import get_weather

        result = await get_weather(location="")
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_weather_whitespace_location_returns_error(self):
        """get_weather returns error for whitespace location."""
        from persbot.tools.api_tools.weather_tools import get_weather

        result = await get_weather(location="   ")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_weather_respects_cancel_event(self):
        """get_weather respects cancel_event."""
        from persbot.tools.api_tools.weather_tools import get_weather

        cancel_event = asyncio.Event()
        cancel_event.set()

        result = await get_weather(location="Seoul", cancel_event=cancel_event)
        assert result.success is False
        assert "aborted" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_weather_validates_units(self):
        """get_weather validates units parameter."""
        from persbot.tools.api_tools.weather_tools import get_weather

        # Invalid units should be normalized to metric
        # Without API key, it will fail but shouldn't crash
        result = await get_weather(location="Seoul", units="invalid")
        # Should not raise an exception
        assert hasattr(result, 'success')


class TestRegisterWeatherTools:
    """Tests for register_weather_tools function."""

    def test_register_weather_tools_exists(self):
        """register_weather_tools function exists."""
        from persbot.tools.api_tools.weather_tools import register_weather_tools
        assert register_weather_tools is not None

    def test_registers_get_weather_tool(self):
        """register_weather_tools registers get_weather tool."""
        from persbot.tools.api_tools.weather_tools import register_weather_tools

        mock_registry = MagicMock()
        register_weather_tools(mock_registry)

        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        tool_def = call_args[0][0]
        assert tool_def.name == "get_weather"

    def test_tool_has_required_parameters(self):
        """get_weather tool has required parameters."""
        from persbot.tools.api_tools.weather_tools import register_weather_tools

        mock_registry = MagicMock()
        register_weather_tools(mock_registry)

        call_args = mock_registry.register.call_args
        tool_def = call_args[0][0]

        param_names = [p.name for p in tool_def.parameters]
        assert "location" in param_names
        assert "units" in param_names

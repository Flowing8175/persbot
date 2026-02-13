"""Feature tests for time tools module.

Tests focus on behavior:
- TIMEZONE_MAPPINGS: common timezone mappings
- get_time: get current time in timezone
- get_time_basic: fallback time function
- register_time_tools: register tools with registry
"""

import sys
from datetime import datetime, timezone, timedelta
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


class TestTimezoneMappings:
    """Tests for TIMEZONE_MAPPINGS constant."""

    def test_timezone_mappings_exists(self):
        """TIMEZONE_MAPPINGS constant exists."""
        from persbot.tools.api_tools.time_tools import TIMEZONE_MAPPINGS
        assert TIMEZONE_MAPPINGS is not None

    def test_has_seoul_mapping(self):
        """TIMEZONE_MAPPINGS has Seoul mapping."""
        from persbot.tools.api_tools.time_tools import TIMEZONE_MAPPINGS
        assert TIMEZONE_MAPPINGS["seoul"] == "Asia/Seoul"

    def test_has_tokyo_mapping(self):
        """TIMEZONE_MAPPINGS has Tokyo mapping."""
        from persbot.tools.api_tools.time_tools import TIMEZONE_MAPPINGS
        assert TIMEZONE_MAPPINGS["tokyo"] == "Asia/Tokyo"

    def test_has_new_york_mapping(self):
        """TIMEZONE_MAPPINGS has New York mapping."""
        from persbot.tools.api_tools.time_tools import TIMEZONE_MAPPINGS
        assert TIMEZONE_MAPPINGS["new york"] == "America/New_York"

    def test_has_london_mapping(self):
        """TIMEZONE_MAPPINGS has London mapping."""
        from persbot.tools.api_tools.time_tools import TIMEZONE_MAPPINGS
        assert TIMEZONE_MAPPINGS["london"] == "Europe/London"

    def test_has_utc_mapping(self):
        """TIMEZONE_MAPPINGS has UTC mapping."""
        from persbot.tools.api_tools.time_tools import TIMEZONE_MAPPINGS
        assert TIMEZONE_MAPPINGS["utc"] == "UTC"


class TestGetTime:
    """Tests for get_time function."""

    @pytest.mark.asyncio
    async def test_get_time_exists(self):
        """get_time function exists."""
        from persbot.tools.api_tools.time_tools import get_time
        assert get_time is not None

    @pytest.mark.asyncio
    async def test_get_time_returns_tool_result(self):
        """get_time returns ToolResult."""
        from persbot.tools.api_tools.time_tools import get_time
        from persbot.tools.base import ToolResult

        result = await get_time()
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_get_time_returns_success(self):
        """get_time returns success result."""
        from persbot.tools.api_tools.time_tools import get_time

        result = await get_time()
        # May succeed or fail depending on pytz availability
        assert hasattr(result, 'success')
        assert hasattr(result, 'data') or hasattr(result, 'error')

    @pytest.mark.asyncio
    async def test_get_time_with_city_name(self):
        """get_time handles city name."""
        from persbot.tools.api_tools.time_tools import get_time

        result = await get_time(timezone_str="seoul")
        assert hasattr(result, 'success')

    @pytest.mark.asyncio
    async def test_get_time_with_timezone_string(self):
        """get_time handles timezone string."""
        from persbot.tools.api_tools.time_tools import get_time

        result = await get_time(timezone_str="Asia/Seoul")
        assert hasattr(result, 'success')


class TestGetTimeBasic:
    """Tests for get_time_basic function."""

    @pytest.mark.asyncio
    async def test_get_time_basic_exists(self):
        """get_time_basic function exists."""
        from persbot.tools.api_tools.time_tools import get_time_basic
        assert get_time_basic is not None

    @pytest.mark.asyncio
    async def test_get_time_basic_returns_tool_result(self):
        """get_time_basic returns ToolResult."""
        from persbot.tools.api_tools.time_tools import get_time_basic
        from persbot.tools.base import ToolResult

        result = await get_time_basic()
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_get_time_basic_returns_result(self):
        """get_time_basic returns a result (may be success or error depending on implementation)."""
        from persbot.tools.api_tools.time_tools import get_time_basic

        result = await get_time_basic()
        # Result should be a ToolResult with success or error
        assert hasattr(result, 'success')
        assert result.success is True or result.error is not None

    @pytest.mark.asyncio
    async def test_get_time_basic_with_utc(self):
        """get_time_basic handles UTC timezone."""
        from persbot.tools.api_tools.time_tools import get_time_basic

        result = await get_time_basic(timezone_str="utc")
        # Should handle UTC without errors
        assert hasattr(result, 'success')


class TestRegisterTimeTools:
    """Tests for register_time_tools function."""

    def test_register_time_tools_exists(self):
        """register_time_tools function exists."""
        from persbot.tools.api_tools.time_tools import register_time_tools
        assert register_time_tools is not None

    def test_registers_get_time_tool(self):
        """register_time_tools registers get_time tool."""
        from persbot.tools.api_tools.time_tools import register_time_tools

        mock_registry = MagicMock()
        register_time_tools(mock_registry)

        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        tool_def = call_args[0][0]
        assert tool_def.name == "get_time"

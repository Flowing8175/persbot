"""Feature tests for routine tools module.

Tests focus on behavior using mocking:
- check_virtual_routine_status: check persona's current routine status
- register_routine_tools: register tools with registry
"""

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


class TestCheckVirtualRoutineStatus:
    """Tests for check_virtual_routine_status function."""

    @pytest.mark.asyncio
    async def test_check_virtual_routine_status_exists(self):
        """check_virtual_routine_status function exists."""
        from persbot.tools.persona_tools.routine_tools import check_virtual_routine_status
        assert check_virtual_routine_status is not None

    @pytest.mark.asyncio
    async def test_returns_status_with_default_schedule(self):
        """check_virtual_routine_status returns status with default schedule."""
        from persbot.tools.persona_tools.routine_tools import check_virtual_routine_status

        with patch('persbot.tools.persona_tools.routine_tools._load_persona_schedule', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {"schedule": []}

            result = await check_virtual_routine_status()

            assert result.success is True
            assert "status" in result.data
            assert "activity" in result.data

    @pytest.mark.asyncio
    async def test_includes_current_time(self):
        """check_virtual_routine_status includes current time."""
        from persbot.tools.persona_tools.routine_tools import check_virtual_routine_status

        with patch('persbot.tools.persona_tools.routine_tools._load_persona_schedule', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {"schedule": []}

            result = await check_virtual_routine_status()

            assert result.success is True
            assert "current_time" in result.data

    @pytest.mark.asyncio
    async def test_includes_day_of_week(self):
        """check_virtual_routine_status includes day of week."""
        from persbot.tools.persona_tools.routine_tools import check_virtual_routine_status

        with patch('persbot.tools.persona_tools.routine_tools._load_persona_schedule', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {"schedule": []}

            result = await check_virtual_routine_status()

            assert result.success is True
            assert "day_of_week" in result.data

    @pytest.mark.asyncio
    async def test_includes_weekend_flag(self):
        """check_virtual_routine_status includes weekend flag."""
        from persbot.tools.persona_tools.routine_tools import check_virtual_routine_status

        with patch('persbot.tools.persona_tools.routine_tools._load_persona_schedule', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {"schedule": []}

            result = await check_virtual_routine_status()

            assert result.success is True
            assert "is_weekend" in result.data
            assert isinstance(result.data["is_weekend"], bool)

    @pytest.mark.asyncio
    async def test_matches_schedule_entry(self):
        """check_virtual_routine_status matches schedule entry."""
        from persbot.tools.persona_tools.routine_tools import check_virtual_routine_status

        schedule = {
            "schedule": [
                {
                    "start": "0000",
                    "end": "2359",
                    "status": "Available",
                    "activity": "free time",
                    "response_style": "Normal",
                    "context": "Always available."
                }
            ]
        }

        with patch('persbot.tools.persona_tools.routine_tools._load_persona_schedule', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = schedule

            result = await check_virtual_routine_status()

            assert result.success is True
            assert result.data["status"] == "Available"

    @pytest.mark.asyncio
    async def test_includes_response_recommendation_for_sleeping(self):
        """check_virtual_routine_status includes response recommendation for sleeping."""
        from persbot.tools.persona_tools.routine_tools import check_virtual_routine_status

        schedule = {
            "schedule": [
                {
                    "start": "0000",
                    "end": "2359",
                    "status": "Sleeping",
                    "activity": "sleeping",
                    "response_style": "Groggy"
                }
            ]
        }

        with patch('persbot.tools.persona_tools.routine_tools._load_persona_schedule', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = schedule

            result = await check_virtual_routine_status()

            assert result.success is True
            assert "response_recommendation" in result.data


class TestRegisterRoutineTools:
    """Tests for register_routine_tools function."""

    def test_register_routine_tools_exists(self):
        """register_routine_tools function exists."""
        from persbot.tools.persona_tools.routine_tools import register_routine_tools
        assert register_routine_tools is not None

    def test_registers_tools(self):
        """register_routine_tools registers tools."""
        from persbot.tools.persona_tools.routine_tools import register_routine_tools

        mock_registry = MagicMock()
        register_routine_tools(mock_registry)

        # Should register at least 1 tool
        assert mock_registry.register.call_count >= 1

    def test_registers_check_virtual_routine_status(self):
        """register_routine_tools registers check_virtual_routine_status."""
        from persbot.tools.persona_tools.routine_tools import register_routine_tools

        mock_registry = MagicMock()
        register_routine_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[0]
        tool_def = call_args[0][0]
        assert tool_def.name == "check_virtual_routine_status"

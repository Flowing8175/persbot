"""Feature tests for Discord user tools module.

Tests focus on behavior using mocking:
- get_user_info: get user information
- get_member_info: get member information
- get_member_roles: get member roles
- register_user_tools: register tools with registry
"""

import sys
from unittest.mock import Mock, MagicMock, AsyncMock

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


class TestGetUserInfo:
    """Tests for get_user_info function."""

    @pytest.mark.asyncio
    async def test_get_user_info_exists(self):
        """get_user_info function exists."""
        from persbot.tools.discord_tools.user_tools import get_user_info
        assert get_user_info is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_user_info returns error without Discord context."""
        from persbot.tools.discord_tools.user_tools import get_user_info

        result = await get_user_info(user_id=123, discord_context=None)
        assert result.success is False
        assert "context" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_error_without_user_id(self):
        """get_user_info returns error when no user_id and no context."""
        from persbot.tools.discord_tools.user_tools import get_user_info

        result = await get_user_info(discord_context=None)
        assert result.success is False


class TestGetMemberInfo:
    """Tests for get_member_info function."""

    @pytest.mark.asyncio
    async def test_get_member_info_exists(self):
        """get_member_info function exists."""
        from persbot.tools.discord_tools.user_tools import get_member_info
        assert get_member_info is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_member_info returns error without Discord context."""
        from persbot.tools.discord_tools.user_tools import get_member_info

        result = await get_member_info(user_id=123, discord_context=None)
        assert result.success is False


class TestGetMemberRoles:
    """Tests for get_member_roles function."""

    @pytest.mark.asyncio
    async def test_get_member_roles_exists(self):
        """get_member_roles function exists."""
        from persbot.tools.discord_tools.user_tools import get_member_roles
        assert get_member_roles is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_member_roles returns error without Discord context."""
        from persbot.tools.discord_tools.user_tools import get_member_roles

        result = await get_member_roles(user_id=123, discord_context=None)
        assert result.success is False


class TestRegisterUserTools:
    """Tests for register_user_tools function."""

    def test_register_user_tools_exists(self):
        """register_user_tools function exists."""
        from persbot.tools.discord_tools.user_tools import register_user_tools
        assert register_user_tools is not None

    def test_registers_tools(self):
        """register_user_tools registers tools."""
        from persbot.tools.discord_tools.user_tools import register_user_tools

        mock_registry = MagicMock()
        register_user_tools(mock_registry)

        # Should register 3 tools
        assert mock_registry.register.call_count == 3

    def test_registers_get_user_info(self):
        """register_user_tools registers get_user_info."""
        from persbot.tools.discord_tools.user_tools import register_user_tools

        mock_registry = MagicMock()
        register_user_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[0]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_user_info"

    def test_registers_get_member_info(self):
        """register_user_tools registers get_member_info."""
        from persbot.tools.discord_tools.user_tools import register_user_tools

        mock_registry = MagicMock()
        register_user_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[1]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_member_info"

    def test_registers_get_member_roles(self):
        """register_user_tools registers get_member_roles."""
        from persbot.tools.discord_tools.user_tools import register_user_tools

        mock_registry = MagicMock()
        register_user_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[2]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_member_roles"

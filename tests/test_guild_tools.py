"""Feature tests for Discord guild tools module.

Tests focus on behavior using mocking:
- get_guild_info: get guild information
- get_guild_roles: get guild roles
- get_guild_emojis: get guild emojis
- register_guild_tools: register tools with registry
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


class TestGetGuildInfo:
    """Tests for get_guild_info function."""

    @pytest.mark.asyncio
    async def test_get_guild_info_exists(self):
        """get_guild_info function exists."""
        from persbot.tools.discord_tools.guild_tools import get_guild_info
        assert get_guild_info is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_guild_info returns error without Discord context."""
        from persbot.tools.discord_tools.guild_tools import get_guild_info

        result = await get_guild_info(guild_id=123, discord_context=None)
        assert result.success is False
        assert "context" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_error_without_guild(self):
        """get_guild_info returns error when not in guild."""
        from persbot.tools.discord_tools.guild_tools import get_guild_info

        mock_context = MagicMock()
        mock_context.guild = None

        result = await get_guild_info(guild_id=123, discord_context=mock_context)
        assert result.success is False


class TestGetGuildRoles:
    """Tests for get_guild_roles function."""

    @pytest.mark.asyncio
    async def test_get_guild_roles_exists(self):
        """get_guild_roles function exists."""
        from persbot.tools.discord_tools.guild_tools import get_guild_roles
        assert get_guild_roles is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_guild_roles returns error without Discord context."""
        from persbot.tools.discord_tools.guild_tools import get_guild_roles

        result = await get_guild_roles(discord_context=None)
        assert result.success is False


class TestGetGuildEmojis:
    """Tests for get_guild_emojis function."""

    @pytest.mark.asyncio
    async def test_get_guild_emojis_exists(self):
        """get_guild_emojis function exists."""
        from persbot.tools.discord_tools.guild_tools import get_guild_emojis
        assert get_guild_emojis is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_guild_emojis returns error without Discord context."""
        from persbot.tools.discord_tools.guild_tools import get_guild_emojis

        result = await get_guild_emojis(discord_context=None)
        assert result.success is False


class TestRegisterGuildTools:
    """Tests for register_guild_tools function."""

    def test_register_guild_tools_exists(self):
        """register_guild_tools function exists."""
        from persbot.tools.discord_tools.guild_tools import register_guild_tools
        assert register_guild_tools is not None

    def test_registers_tools(self):
        """register_guild_tools registers tools."""
        from persbot.tools.discord_tools.guild_tools import register_guild_tools

        mock_registry = MagicMock()
        register_guild_tools(mock_registry)

        # Should register 3 tools
        assert mock_registry.register.call_count == 3

    def test_registers_get_guild_info(self):
        """register_guild_tools registers get_guild_info."""
        from persbot.tools.discord_tools.guild_tools import register_guild_tools

        mock_registry = MagicMock()
        register_guild_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[0]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_guild_info"

    def test_registers_get_guild_roles(self):
        """register_guild_tools registers get_guild_roles."""
        from persbot.tools.discord_tools.guild_tools import register_guild_tools

        mock_registry = MagicMock()
        register_guild_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[1]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_guild_roles"

    def test_registers_get_guild_emojis(self):
        """register_guild_tools registers get_guild_emojis."""
        from persbot.tools.discord_tools.guild_tools import register_guild_tools

        mock_registry = MagicMock()
        register_guild_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[2]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_guild_emojis"

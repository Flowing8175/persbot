"""Feature tests for Discord channel tools module.

Tests focus on behavior using mocking:
- get_channel_info: get channel information
- get_channel_history: get channel history
- get_message: get specific message
- list_channels: list channels in guild
- register_channel_tools: register tools with registry
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


class TestGetChannelInfo:
    """Tests for get_channel_info function."""

    @pytest.mark.asyncio
    async def test_get_channel_info_exists(self):
        """get_channel_info function exists."""
        from persbot.tools.discord_tools.channel_tools import get_channel_info
        assert get_channel_info is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_channel_info returns error without Discord context."""
        from persbot.tools.discord_tools.channel_tools import get_channel_info

        result = await get_channel_info(channel_id=123, discord_context=None)
        assert result.success is False
        assert "context" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_error_without_guild(self):
        """get_channel_info returns error when not in guild."""
        from persbot.tools.discord_tools.channel_tools import get_channel_info

        mock_context = MagicMock()
        mock_context.guild = None

        result = await get_channel_info(channel_id=123, discord_context=mock_context)
        assert result.success is False


class TestGetChannelHistory:
    """Tests for get_channel_history function."""

    @pytest.mark.asyncio
    async def test_get_channel_history_exists(self):
        """get_channel_history function exists."""
        from persbot.tools.discord_tools.channel_tools import get_channel_history
        assert get_channel_history is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_channel_history returns error without Discord context."""
        from persbot.tools.discord_tools.channel_tools import get_channel_history

        result = await get_channel_history(channel_id=123, discord_context=None)
        assert result.success is False


class TestGetMessage:
    """Tests for get_message function."""

    @pytest.mark.asyncio
    async def test_get_message_exists(self):
        """get_message function exists."""
        from persbot.tools.discord_tools.channel_tools import get_message
        assert get_message is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """get_message returns error without Discord context."""
        from persbot.tools.discord_tools.channel_tools import get_message

        result = await get_message(message_id=123, discord_context=None)
        assert result.success is False


class TestListChannels:
    """Tests for list_channels function."""

    @pytest.mark.asyncio
    async def test_list_channels_exists(self):
        """list_channels function exists."""
        from persbot.tools.discord_tools.channel_tools import list_channels
        assert list_channels is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_context(self):
        """list_channels returns error without Discord context."""
        from persbot.tools.discord_tools.channel_tools import list_channels

        result = await list_channels(discord_context=None)
        assert result.success is False


class TestRegisterChannelTools:
    """Tests for register_channel_tools function."""

    def test_register_channel_tools_exists(self):
        """register_channel_tools function exists."""
        from persbot.tools.discord_tools.channel_tools import register_channel_tools
        assert register_channel_tools is not None

    def test_registers_tools(self):
        """register_channel_tools registers tools."""
        from persbot.tools.discord_tools.channel_tools import register_channel_tools

        mock_registry = MagicMock()
        register_channel_tools(mock_registry)

        # Should register 4 tools
        assert mock_registry.register.call_count == 4

    def test_registers_get_channel_info(self):
        """register_channel_tools registers get_channel_info."""
        from persbot.tools.discord_tools.channel_tools import register_channel_tools

        mock_registry = MagicMock()
        register_channel_tools(mock_registry)

        # Check first registration
        call_args = mock_registry.register.call_args_list[0]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_channel_info"

    def test_registers_get_channel_history(self):
        """register_channel_tools registers get_channel_history."""
        from persbot.tools.discord_tools.channel_tools import register_channel_tools

        mock_registry = MagicMock()
        register_channel_tools(mock_registry)

        # Check second registration
        call_args = mock_registry.register.call_args_list[1]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_channel_history"

    def test_registers_get_message(self):
        """register_channel_tools registers get_message."""
        from persbot.tools.discord_tools.channel_tools import register_channel_tools

        mock_registry = MagicMock()
        register_channel_tools(mock_registry)

        # Check third registration
        call_args = mock_registry.register.call_args_list[2]
        tool_def = call_args[0][0]
        assert tool_def.name == "get_message"

    def test_registers_list_channels(self):
        """register_channel_tools registers list_channels."""
        from persbot.tools.discord_tools.channel_tools import register_channel_tools

        mock_registry = MagicMock()
        register_channel_tools(mock_registry)

        # Check fourth registration
        call_args = mock_registry.register.call_args_list[3]
        tool_def = call_args[0][0]
        assert tool_def.name == "list_channels"

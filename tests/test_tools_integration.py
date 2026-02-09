"""Integration tests for tool system end-to-end workflow."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from soyebot.tools import ToolManager
from soyebot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult
from soyebot.tools.executor import ToolExecutor
from soyebot.tools.registry import ToolRegistry


class TestToolManager:
    """Tests for ToolManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock()
        config.enable_tools = True
        config.enable_discord_tools = True
        config.enable_api_tools = True
        config.tool_rate_limit = 0  # No rate limiting in tests
        config.tool_timeout = 10.0
        config.weather_api_key = "test_weather_key"
        config.search_api_key = "test_search_key"
        config.no_check_permission = True
        return config

    @pytest.fixture
    def tool_manager(self, mock_config):
        """Create a ToolManager instance."""
        return ToolManager(mock_config)

    def test_initialization(self, tool_manager):
        """Test ToolManager initialization."""
        assert tool_manager.registry is not None
        assert tool_manager.executor is not None
        assert tool_manager.is_enabled()

    def test_initialization_tools_disabled(self):
        """Test ToolManager with tools disabled."""
        config = Mock()
        config.enable_tools = False
        config.enable_discord_tools = True
        config.enable_api_tools = True
        config.tool_rate_limit = 0
        config.tool_timeout = 10.0

        manager = ToolManager(config)
        assert not manager.is_enabled()
        # Registry should still be created but empty
        assert manager.registry is not None

    def test_get_enabled_tools(self, tool_manager):
        """Test getting all enabled tools."""
        tools = tool_manager.get_enabled_tools()
        assert isinstance(tools, dict)
        # Should have tools from registration
        assert len(tools) > 0

    def test_get_tools_by_category(self, tool_manager):
        """Test getting tools by category."""
        # Test API search category
        search_tools = tool_manager.get_tools_by_category(ToolCategory.API_SEARCH)
        assert isinstance(search_tools, list)
        # Should have web_search at minimum
        assert any(t.name == "web_search" for t in search_tools)

        # Test API weather category
        weather_tools = tool_manager.get_tools_by_category(ToolCategory.API_WEATHER)
        assert isinstance(weather_tools, list)
        assert any(t.name == "get_weather" for t in weather_tools)

        # Test API time category
        time_tools = tool_manager.get_tools_by_category(ToolCategory.API_TIME)
        assert isinstance(time_tools, list)
        assert any(t.name == "get_time" for t in time_tools)

    def test_get_discord_tools_by_category(self, tool_manager):
        """Test getting Discord tools by category."""
        # Test channel tools
        channel_tools = tool_manager.get_tools_by_category(ToolCategory.DISCORD_CHANNEL)
        assert isinstance(channel_tools, list)
        expected_tools = ["get_channel_info", "get_channel_history", "get_message", "list_channels"]
        for expected in expected_tools:
            assert any(t.name == expected for t in channel_tools), f"Missing {expected}"

        # Test user tools
        user_tools = tool_manager.get_tools_by_category(ToolCategory.DISCORD_USER)
        assert isinstance(user_tools, list)
        expected_tools = ["get_user_info", "get_member_info", "get_member_roles"]
        for expected in expected_tools:
            assert any(t.name == expected for t in user_tools)

        # Test guild tools
        guild_tools = tool_manager.get_tools_by_category(ToolCategory.DISCORD_GUILD)
        assert isinstance(guild_tools, list)
        expected_tools = ["get_guild_info", "get_guild_roles", "get_guild_emojis"]
        for expected in expected_tools:
            assert any(t.name == expected for t in guild_tools)

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, tool_manager):
        """Test successful tool execution."""
        # Execute get_time which doesn't require external dependencies
        # Note: parameter name is 'timezone_str' not 'timezone'
        result = await tool_manager.execute_tool("get_time", {"timezone_str": "UTC"})

        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, tool_manager):
        """Test executing non-existent tool."""
        result = await tool_manager.execute_tool("nonexistent_tool", {})

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self, tool_manager):
        """Test executing multiple tools in parallel."""
        tool_calls = [
            {"name": "get_time", "parameters": {"timezone_str": "UTC"}},
            {"name": "get_time", "parameters": {"timezone_str": "Asia/Seoul"}},
            {"name": "get_time", "parameters": {"timezone_str": "America/New_York"}},
        ]

        results = await tool_manager.execute_tools(tool_calls, None)

        assert len(results) == 3
        assert all("name" in r for r in results)

    def test_set_tool_enabled(self, tool_manager):
        """Test enabling/disabling a tool."""
        # Disable get_time
        result = tool_manager.set_tool_enabled("get_time", False)
        assert result is True

        # Check that it's disabled
        tools = tool_manager.get_enabled_tools()
        assert "get_time" not in tools

        # Re-enable
        result = tool_manager.set_tool_enabled("get_time", True)
        assert result is True

        # Check that it's enabled
        tools = tool_manager.get_enabled_tools()
        assert "get_time" in tools

    def test_get_metrics(self, tool_manager):
        """Test getting execution metrics."""
        metrics = tool_manager.get_metrics()
        assert isinstance(metrics, dict)

    def test_clear_rate_limits(self, tool_manager):
        """Test clearing rate limits."""
        # Should not raise any errors
        tool_manager.clear_rate_limits()
        tool_manager.clear_rate_limits(user_id=12345)

    @pytest.mark.asyncio
    async def test_time_tool_execution_flow(self, tool_manager):
        """Test complete execution flow for time tool."""
        # Define tool call
        tool_calls = [
            {
                "name": "get_time",
                "parameters": {"timezone_str": "UTC"},
            }
        ]

        # Execute tools
        results = await tool_manager.execute_tools(tool_calls, None)

        assert len(results) == 1
        assert results[0]["name"] == "get_time"
        assert results[0]["result"] is not None

    @pytest.mark.asyncio
    async def test_multiple_timezones_execution(self, tool_manager):
        """Test executing time tool for multiple timezones."""
        timezones = ["UTC", "Asia/Seoul", "America/New_York", "Europe/London"]

        tool_calls = [{"name": "get_time", "parameters": {"timezone_str": tz}} for tz in timezones]

        results = await tool_manager.execute_tools(tool_calls, None)

        assert len(results) == len(timezones)
        for i, result in enumerate(results):
            assert result["name"] == "get_time"
            assert result["result"] is not None
            assert "timezone" in result["result"]


class TestToolRegistration:
    """Tests for tool registration workflow."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ToolRegistry()

    def test_register_custom_tool(self, registry):
        """Test registering a custom tool."""

        async def custom_handler(param1: str) -> str:
            return f"Custom: {param1}"

        tool = ToolDefinition(
            name="custom_tool",
            description="A custom tool",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="First parameter",
                    required=True,
                ),
            ],
            handler=custom_handler,
        )

        registry.register(tool)
        assert registry.has_tool("custom_tool")

    @pytest.mark.asyncio
    async def test_execute_custom_tool(self, registry):
        """Test executing a custom tool through executor."""

        async def custom_handler(param1: str) -> str:
            return f"Custom: {param1}"

        tool = ToolDefinition(
            name="custom_tool",
            description="A custom tool",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="First parameter",
                    required=True,
                ),
            ],
            handler=custom_handler,
        )

        registry.register(tool)

        config = Mock()
        config.no_check_permission = True
        config.tool_rate_limit = 0
        config.tool_timeout = 10.0

        executor = ToolExecutor(config, registry)
        result = await executor.execute_tool("custom_tool", {"param1": "test"}, None)

        assert result.success is True
        assert result.data == "Custom: test"


class TestToolErrorHandling:
    """Tests for error handling in tool execution."""

    @pytest.fixture
    def error_tool_registry(self):
        """Create a registry with tools that can fail."""
        registry = ToolRegistry()

        # Failing tool
        async def failing_handler():
            raise ValueError("Intentional failure")

        registry.register(
            ToolDefinition(
                name="failing_tool",
                description="Always fails",
                category=ToolCategory.API_SEARCH,
                parameters=[],
                handler=failing_handler,
            )
        )

        # Timeout tool
        async def timeout_handler():
            import asyncio

            await asyncio.sleep(15)

        registry.register(
            ToolDefinition(
                name="timeout_tool",
                description="Times out",
                category=ToolCategory.API_SEARCH,
                parameters=[],
                handler=timeout_handler,
            )
        )

        return registry

    @pytest.mark.asyncio
    async def test_failing_tool_error(self, error_tool_registry):
        """Test that failing tools return proper error results."""
        config = Mock()
        config.no_check_permission = True
        config.tool_rate_limit = 0
        config.tool_timeout = 10.0

        executor = ToolExecutor(config, error_tool_registry)
        result = await executor.execute_tool("failing_tool", {}, None)

        assert result.success is False
        assert "Intentional failure" in result.error

    @pytest.mark.asyncio
    async def test_timeout_tool_error(self, error_tool_registry):
        """Test that timeout tools return proper error results."""
        config = Mock()
        config.no_check_permission = True
        config.tool_rate_limit = 0
        config.tool_timeout = 1.0  # 1 second timeout

        executor = ToolExecutor(config, error_tool_registry)
        result = await executor.execute_tool("timeout_tool", {}, None)

        assert result.success is False
        # Check for timeout-related error message (be flexible with formatting)
        error_lower = result.error.lower()
        assert (
            "timeout" in error_lower
            or "timed out" in error_lower
            or "tool execution" in error_lower
        )


class TestToolParameters:
    """Tests for tool parameter handling."""

    @pytest.mark.asyncio
    async def test_tool_with_optional_parameters(self):
        """Test tool execution with optional parameters."""

        async def optional_handler(required: str, optional: int = 10):
            return f"{required}, {optional}"

        tool = ToolDefinition(
            name="optional_test",
            description="Test optional parameters",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="required",
                    type="string",
                    description="Required parameter",
                    required=True,
                ),
                ToolParameter(
                    name="optional",
                    type="integer",
                    description="Optional parameter",
                    required=False,
                    default=10,
                ),
            ],
            handler=optional_handler,
        )

        # Execute with only required parameter
        result = await tool.execute(required="test")
        assert result.success is True
        assert result.data == "test, 10"

        # Execute with both parameters
        result = await tool.execute(required="test", optional=20)
        assert result.success is True
        assert result.data == "test, 20"

    @pytest.mark.asyncio
    async def test_tool_with_enum_parameters(self):
        """Test tool execution with enum parameters."""

        async def enum_handler(choice: str):
            return f"You chose: {choice}"

        tool = ToolDefinition(
            name="enum_test",
            description="Test enum parameters",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="choice",
                    type="string",
                    description="A choice",
                    required=True,
                    enum=["option1", "option2", "option3"],
                ),
            ],
            handler=enum_handler,
        )

        # Execute with valid enum value
        result = await tool.execute(choice="option1")
        assert result.success is True
        assert result.data == "You chose: option1"


class TestToolResults:
    """Tests for ToolResult handling."""

    def test_result_metadata(self):
        """Test ToolResult metadata handling."""
        result = ToolResult(
            success=True,
            data="test data",
            metadata={
                "execution_time": 0.5,
                "cache_hit": True,
                "api_calls": 1,
            },
        )

        assert result.metadata["execution_time"] == 0.5
        assert result.metadata["cache_hit"] is True
        assert result.metadata["api_calls"] == 1

    @pytest.mark.asyncio
    async def test_handler_returning_tool_result(self):
        """Test handler that returns ToolResult directly."""

        async def result_handler(message: str):
            return ToolResult(
                success=True,
                data=f"Processed: {message}",
                metadata={"length": len(message)},
            )

        tool = ToolDefinition(
            name="result_tool",
            description="Returns ToolResult",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Message to process",
                    required=True,
                ),
            ],
            handler=result_handler,
        )

        result = await tool.execute(message="test")
        assert result.success is True
        assert result.data == "Processed: test"
        assert result.metadata["length"] == 4

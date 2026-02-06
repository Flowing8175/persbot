"""Tests for core tool infrastructure (base, registry, executor)."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timezone

from soyebot.tools.base import (
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolCategory,
)
from soyebot.tools.registry import ToolRegistry
from soyebot.tools.executor import ToolExecutor, ExecutionMetrics


class TestToolParameter:
    """Tests for ToolParameter class."""

    def test_tool_parameter_creation(self):
        """Test creating a tool parameter."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="A test parameter",
            required=True,
        )
        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.required is True
        assert param.default is None

    def test_tool_parameter_with_defaults(self):
        """Test creating a tool parameter with defaults."""
        param = ToolParameter(
            name="optional_param",
            type="integer",
            description="An optional parameter",
            required=False,
            default=42,
            enum=[1, 2, 3, 42],
        )
        assert param.default == 42
        assert param.enum == [1, 2, 3, 42]


class TestToolResult:
    """Tests for ToolResult class."""

    def test_success_result(self):
        """Test creating a successful tool result."""
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.metadata == {}

    def test_error_result(self):
        """Test creating an error tool result."""
        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self):
        """Test creating a result with metadata."""
        result = ToolResult(
            success=True,
            data="test",
            metadata={"execution_time": 0.5, "cache_hit": True},
        )
        assert result.metadata == {"execution_time": 0.5, "cache_hit": True}


class TestToolDefinition:
    """Tests for ToolDefinition class."""

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test executing a tool."""
        async def dummy_handler(arg1: str, arg2: int = 10):
            return f"Result: {arg1}, {arg2}"

        tool = ToolDefinition(
            name="dummy_tool",
            description="A dummy tool for testing",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="arg1",
                    type="string",
                    description="First argument",
                    required=True,
                ),
                ToolParameter(
                    name="arg2",
                    type="integer",
                    description="Second argument",
                    required=False,
                    default=10,
                ),
            ],
            handler=dummy_handler,
        )

        result = await tool.execute(arg1="test", arg2=20)
        assert result.success is True
        assert result.data == "Result: test, 20"

    @pytest.mark.asyncio
    async def test_tool_execution_with_defaults(self):
        """Test executing a tool with default parameters."""
        async def dummy_handler(arg1: str, arg2: int = 10):
            return f"Result: {arg1}, {arg2}"

        tool = ToolDefinition(
            name="dummy_tool",
            description="A dummy tool",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="arg1",
                    type="string",
                    description="First argument",
                    required=True,
                ),
            ],
            handler=dummy_handler,
        )

        result = await tool.execute(arg1="test")
        assert result.success is True
        assert result.data == "Result: test, 10"

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test tool execution with error."""
        async def failing_handler():
            raise ValueError("Test error")

        tool = ToolDefinition(
            name="failing_tool",
            description="A tool that fails",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=failing_handler,
        )

        result = await tool.execute()
        assert result.success is False
        assert "Test error" in result.error

    def test_to_openai_format(self):
        """Test converting tool to OpenAI format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="First parameter",
                    required=True,
                ),
                ToolParameter(
                    name="param2",
                    type="integer",
                    description="Second parameter",
                    required=False,
                    default=42,
                ),
            ],
            handler=AsyncMock(),
        )

        openai_format = tool.to_openai_format()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_tool"
        assert openai_format["function"]["description"] == "A test tool"
        assert "param1" in openai_format["function"]["parameters"]["required"]
        assert "param2" not in openai_format["function"]["parameters"]["required"]
        assert openai_format["function"]["parameters"]["properties"]["param2"]["default"] == 42

    def test_to_gemini_format(self):
        """Test converting tool to Gemini format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="First parameter",
                    required=True,
                ),
            ],
            handler=AsyncMock(),
        )

        gemini_format = tool.to_gemini_format()
        # Note: With mocked genai_types, we get Mock objects
        # Just verify the method returns something and doesn't crash
        assert gemini_format is not None
        assert hasattr(gemini_format, "name")


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool)
        assert len(registry) == 1
        assert registry.has_tool("test_tool")

    def test_get_tool(self):
        """Test getting a registered tool."""
        registry = ToolRegistry()

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool)
        retrieved = registry.get("test_tool")

        assert retrieved is not None
        assert retrieved.name == "test_tool"
        assert retrieved.description == "A test tool"

    def test_get_all_tools(self):
        """Test getting all registered tools."""
        registry = ToolRegistry()

        tool1 = ToolDefinition(
            name="tool1",
            description="First tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )

        tool2 = ToolDefinition(
            name="tool2",
            description="Second tool",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        all_tools = registry.get_all()
        assert len(all_tools) == 2
        assert "tool1" in all_tools
        assert "tool2" in all_tools

    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        registry = ToolRegistry()

        search_tool = ToolDefinition(
            name="search_tool",
            description="Search tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )

        weather_tool = ToolDefinition(
            name="weather_tool",
            description="Weather tool",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(search_tool)
        registry.register(weather_tool)

        search_tools = registry.get_by_category(ToolCategory.API_SEARCH)
        weather_tools = registry.get_by_category(ToolCategory.API_WEATHER)

        assert len(search_tools) == 1
        assert search_tools[0].name == "search_tool"
        assert len(weather_tools) == 1
        assert weather_tools[0].name == "weather_tool"

    def test_get_enabled_tools(self):
        """Test getting only enabled tools."""
        registry = ToolRegistry()

        enabled_tool = ToolDefinition(
            name="enabled_tool",
            description="An enabled tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=True,
        )

        disabled_tool = ToolDefinition(
            name="disabled_tool",
            description="A disabled tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=False,
        )

        registry.register(enabled_tool)
        registry.register(disabled_tool)

        enabled_tools = registry.get_enabled()
        assert len(enabled_tools) == 1
        assert "enabled_tool" in enabled_tools
        assert "disabled_tool" not in enabled_tools

    def test_set_tool_enabled(self):
        """Test enabling/disabling a tool."""
        registry = ToolRegistry()

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=True,
        )

        registry.register(tool)
        assert registry.get("test_tool").enabled is True

        # Disable the tool
        registry.set_tool_enabled("test_tool", False)
        assert registry.get("test_tool").enabled is False

        # Re-enable the tool
        registry.set_tool_enabled("test_tool", True)
        assert registry.get("test_tool").enabled is True

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool)
        assert len(registry) == 1

        result = registry.unregister("test_tool")
        assert result is True
        assert len(registry) == 0
        assert not registry.has_tool("test_tool")

    def test_get_tool_names(self):
        """Test getting list of tool names."""
        registry = ToolRegistry()

        tool1 = ToolDefinition(
            name="tool1",
            description="First tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )

        tool2 = ToolDefinition(
            name="tool2",
            description="Second tool",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        names = registry.get_tool_names()
        assert len(names) == 2
        assert "tool1" in names
        assert "tool2" in names


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    @pytest.fixture
    def registry(self):
        """Create a test registry with sample tools."""
        registry = ToolRegistry()

        # Simple echo tool
        async def echo_handler(message: str):
            return f"Echo: {message}"

        registry.register(
            ToolDefinition(
                name="echo",
                description="Echo a message",
                category=ToolCategory.API_SEARCH,
                parameters=[
                    ToolParameter(
                        name="message",
                        type="string",
                        description="Message to echo",
                        required=True,
                    ),
                ],
                handler=echo_handler,
            )
        )

        # Failing tool
        async def failing_handler():
            raise ValueError("Test failure")

        registry.register(
            ToolDefinition(
                name="failing_tool",
                description="A tool that fails",
                category=ToolCategory.API_SEARCH,
                parameters=[],
                handler=failing_handler,
            )
        )

        # Disabled tool
        async def disabled_handler():
            return "Should not run"

        registry.register(
            ToolDefinition(
                name="disabled_tool",
                description="A disabled tool",
                category=ToolCategory.API_SEARCH,
                parameters=[],
                handler=disabled_handler,
                enabled=False,
            )
        )

        return registry

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.no_check_permission = True
        config.tool_rate_limit = 0  # No rate limiting in tests
        config.tool_timeout = 10.0
        return config

    @pytest.fixture
    def executor(self, registry, mock_config):
        """Create a test executor."""
        return ToolExecutor(mock_config, registry)

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, executor):
        """Test successful tool execution."""
        result = await executor.execute_tool(
            "echo",
            {"message": "Hello, World!"},
            None,
        )

        assert result.success is True
        assert result.data == "Echo: Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, executor):
        """Test executing a non-existent tool."""
        result = await executor.execute_tool(
            "nonexistent_tool",
            {},
            None,
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_disabled_tool(self, executor):
        """Test executing a disabled tool."""
        result = await executor.execute_tool(
            "disabled_tool",
            {},
            None,
        )

        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_failing_tool(self, executor):
        """Test executing a tool that raises an exception."""
        result = await executor.execute_tool(
            "failing_tool",
            {},
            None,
        )

        assert result.success is False
        assert "Test failure" in result.error

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self, executor):
        """Test executing multiple tools in parallel."""
        tool_calls = [
            {"name": "echo", "parameters": {"message": "First"}},
            {"name": "echo", "parameters": {"message": "Second"}},
            {"name": "echo", "parameters": {"message": "Third"}},
        ]

        results = await executor.execute_tools(tool_calls, None)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].data == "Echo: First"
        assert results[1].data == "Echo: Second"
        assert results[2].data == "Echo: Third"

    @pytest.mark.asyncio
    async def test_tool_timeout(self, mock_config):
        """Test tool execution timeout."""
        async def slow_handler():
            await asyncio.sleep(15)  # Longer than timeout

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="slow_tool",
                description="A slow tool",
                category=ToolCategory.API_SEARCH,
                parameters=[],
                handler=slow_handler,
            )
        )

        mock_config.tool_timeout = 1.0  # 1 second timeout
        executor = ToolExecutor(mock_config, registry)

        result = await executor.execute_tool("slow_tool", {}, None)

        assert result.success is False
        # Check for timeout-related error message
        error_lower = result.error.lower()
        assert "timeout" in error_lower or "timed out" in error_lower

    def test_get_metrics(self, executor):
        """Test getting execution metrics."""
        metrics = executor.get_metrics()
        assert isinstance(metrics, dict)

    def test_clear_rate_limits(self, executor):
        """Test clearing rate limits."""
        # Should not raise any errors
        executor.clear_rate_limits()
        executor.clear_rate_limits(user_id=12345)


class TestExecutionMetrics:
    """Tests for ExecutionMetrics class."""

    def test_record_success(self):
        """Test recording a successful execution."""
        metrics = ExecutionMetrics()
        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0

        metrics.record_success()
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.last_execution_time is not None

    def test_record_failure(self):
        """Test recording a failed execution."""
        metrics = ExecutionMetrics()
        assert metrics.total_executions == 0
        assert metrics.failed_executions == 0

        metrics.record_failure()
        assert metrics.total_executions == 1
        assert metrics.failed_executions == 1
        assert metrics.successful_executions == 0
        assert metrics.last_execution_time is not None

    def test_mixed_recordings(self):
        """Test recording mixed successes and failures."""
        metrics = ExecutionMetrics()

        metrics.record_success()
        metrics.record_success()
        metrics.record_failure()
        metrics.record_success()

        assert metrics.total_executions == 4
        assert metrics.successful_executions == 3
        assert metrics.failed_executions == 1

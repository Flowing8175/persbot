"""Feature tests for tool executor.

Tests focus on behavior rather than implementation details:
- ToolExecutor: tool execution with permission checks and timeouts
- ExecutionMetrics: execution metrics tracking
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persbot.config import AppConfig
from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult
from persbot.tools.executor import ExecutionMetrics, ToolExecutor
from persbot.tools.registry import ToolRegistry


# Helper to create tools for testing
def make_tool(
    name: str,
    category: ToolCategory = ToolCategory.API_SEARCH,
    enabled: bool = True,
    requires_permission: str | None = None,
    timeout: float | None = None,
):
    """Create a tool definition for testing."""

    async def handler(**kwargs):
        return ToolResult(success=True, data=f"{name}_result")

    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        category=category,
        parameters=[],
        handler=handler,
        enabled=enabled,
        requires_permission=requires_permission,
        timeout=timeout,
    )


def make_failing_tool(name: str, error_message: str):
    """Create a tool that fails."""

    async def handler(**kwargs):
        raise RuntimeError(error_message)

    return ToolDefinition(
        name=name,
        description=f"Failing tool {name}",
        category=ToolCategory.API_SEARCH,
        parameters=[],
        handler=handler,
    )


def make_slow_tool(name: str, delay: float):
    """Create a tool that takes time to execute."""

    async def handler(**kwargs):
        await asyncio.sleep(delay)
        return ToolResult(success=True, data=f"{name}_result")

    return ToolDefinition(
        name=name,
        description=f"Slow tool {name}",
        category=ToolCategory.API_SEARCH,
        parameters=[],
        handler=handler,
    )


def make_config(**overrides):
    """Create a config for testing."""
    defaults = {
        "discord_token": "test_token",
        "tool_timeout": 5.0,
        "no_check_permission": False,
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def make_discord_message(guild=None, is_member=True, has_permission=True):
    """Create a mock Discord message.

    Note: When has_permission=True and guild is not None, tests should use
    @patch to make isinstance check pass for discord.Member.
    """
    message = MagicMock()
    message.guild = guild

    if is_member and guild:
        author = MagicMock()
        # Set up permissions object with the permission attribute
        perms_mock = MagicMock()
        perms_mock.read_messages = has_permission
        author.guild_permissions = perms_mock
        message.author = author
    else:
        message.author = MagicMock()

    return message


# ==============================================================================
# ExecutionMetrics Feature Tests
# ==============================================================================

class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_creates_with_defaults(self):
        """ExecutionMetrics creates with zero counts."""
        metrics = ExecutionMetrics()
        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0
        assert metrics.last_execution_time is None

    def test_record_success_increments_counters(self):
        """record_success increments relevant counters."""
        metrics = ExecutionMetrics()
        metrics.record_success()

        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.last_execution_time is not None

    def test_record_failure_increments_counters(self):
        """record_failure increments relevant counters."""
        metrics = ExecutionMetrics()
        metrics.record_failure()

        assert metrics.total_executions == 1
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 1
        assert metrics.last_execution_time is not None

    def test_multiple_executions_accumulate(self):
        """Metrics accumulate over multiple executions."""
        metrics = ExecutionMetrics()
        metrics.record_success()
        metrics.record_success()
        metrics.record_failure()
        metrics.record_success()

        assert metrics.total_executions == 4
        assert metrics.successful_executions == 3
        assert metrics.failed_executions == 1


# ==============================================================================
# ToolExecutor Creation Feature Tests
# ==============================================================================

class TestToolExecutorCreation:
    """Tests for ToolExecutor instantiation."""

    def test_creates_with_config_and_registry(self):
        """ToolExecutor creates with config and registry."""
        config = make_config()
        registry = ToolRegistry()

        executor = ToolExecutor(config, registry)

        assert executor.config is config
        assert executor.registry is registry

    def test_creates_with_empty_metrics(self):
        """ToolExecutor starts with empty metrics."""
        config = make_config()
        registry = ToolRegistry()

        executor = ToolExecutor(config, registry)

        assert executor.get_metrics() == {}


# ==============================================================================
# ToolExecutor Execute Tool Feature Tests
# ==============================================================================

class TestToolExecutorExecuteTool:
    """Tests for ToolExecutor.execute_tool behavior."""

    @pytest.mark.asyncio
    async def test_execute_tool_returns_result(self):
        """execute_tool returns the tool's result."""
        config = make_config()
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        result = await executor.execute_tool("search", {})

        assert result.success is True
        assert result.data == "search_result"

    @pytest.mark.asyncio
    async def test_execute_tool_passes_parameters(self):
        """execute_tool passes parameters to the handler."""
        config = make_config()
        registry = ToolRegistry()

        received = {}

        async def handler(**kwargs):
            received.update(kwargs)
            return ToolResult(success=True)

        tool = ToolDefinition(
            name="search",
            description="Search",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        await executor.execute_tool("search", {"query": "test", "count": 5})

        assert received["query"] == "test"
        assert received["count"] == 5

    @pytest.mark.asyncio
    async def test_execute_tool_returns_error_for_unknown_tool(self):
        """execute_tool returns error for unknown tool."""
        config = make_config()
        registry = ToolRegistry()
        executor = ToolExecutor(config, registry)

        result = await executor.execute_tool("unknown", {})

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_returns_error_for_disabled_tool(self):
        """execute_tool returns error for disabled tool."""
        config = make_config()
        registry = ToolRegistry()
        tool = make_tool("search", enabled=False)
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        result = await executor.execute_tool("search", {})

        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_returns_error_on_handler_exception(self):
        """execute_tool returns error when handler raises exception."""
        config = make_config()
        registry = ToolRegistry()
        tool = make_failing_tool("failing", "Handler failed")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        result = await executor.execute_tool("failing", {})

        assert result.success is False
        assert "Handler failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_records_success_metrics(self):
        """execute_tool records metrics for successful execution."""
        config = make_config()
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        await executor.execute_tool("search", {})

        metrics = executor.get_metrics("search")
        assert metrics["search"].total_executions == 1
        assert metrics["search"].successful_executions == 1
        assert metrics["search"].failed_executions == 0

    @pytest.mark.asyncio
    async def test_execute_tool_records_failure_metrics(self):
        """execute_tool records metrics for failed execution."""
        config = make_config()
        registry = ToolRegistry()
        tool = make_failing_tool("failing", "Error")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        await executor.execute_tool("failing", {})

        metrics = executor.get_metrics("failing")
        assert metrics["failing"].total_executions == 1
        assert metrics["failing"].failed_executions == 1


# ==============================================================================
# ToolExecutor Permission Feature Tests
# ==============================================================================

class TestToolExecutorPermissions:
    """Tests for ToolExecutor permission checking."""

    @pytest.mark.asyncio
    async def test_allows_tool_without_permission_requirement(self):
        """execute_tool allows tools without permission requirement."""
        config = make_config(no_check_permission=False)
        registry = ToolRegistry()
        tool = make_tool("search", requires_permission=None)
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        message = make_discord_message(guild=MagicMock())
        result = await executor.execute_tool("search", {}, discord_context=message)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_allows_when_bypass_enabled(self):
        """execute_tool allows when no_check_permission is True."""
        config = make_config(no_check_permission=True)
        registry = ToolRegistry()
        tool = make_tool("search", requires_permission="read_messages")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        message = make_discord_message(guild=MagicMock(), has_permission=False)
        result = await executor.execute_tool("search", {}, discord_context=message)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_allows_in_dm_context(self):
        """execute_tool allows tools in DM context."""
        config = make_config(no_check_permission=False)
        registry = ToolRegistry()
        tool = make_tool("search", requires_permission="read_messages")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        # DM context has guild=None
        message = make_discord_message(guild=None)
        result = await executor.execute_tool("search", {}, discord_context=message)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_denies_when_missing_permission(self):
        """execute_tool denies when user lacks permission."""
        config = make_config(no_check_permission=False)
        registry = ToolRegistry()
        tool = make_tool("search", requires_permission="read_messages")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        message = make_discord_message(guild=MagicMock(), has_permission=False)
        result = await executor.execute_tool("search", {}, discord_context=message)

        assert result.success is False
        assert "permission" in result.error.lower()

    @pytest.mark.asyncio
    async def test_allows_when_has_permission(self):
        """execute_tool allows when user has permission."""
        config = make_config(no_check_permission=False)
        registry = ToolRegistry()
        tool = make_tool("search", requires_permission="read_messages")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        message = make_discord_message(guild=MagicMock(), has_permission=True)

        # Patch isinstance to treat our mock as discord.Member
        with patch("persbot.tools.executor.isinstance") as mock_isinstance:
            # Make isinstance return True for discord.Member check, False otherwise
            def isinstance_side_effect(obj, cls):
                if hasattr(cls, "__name__") and cls.__name__ == "Member":
                    return True
                return isinstance(obj, cls)
            mock_isinstance.side_effect = isinstance_side_effect

            result = await executor.execute_tool("search", {}, discord_context=message)

        assert result.success is True


# ==============================================================================
# ToolExecutor Timeout Feature Tests
# ==============================================================================

class TestToolExecutorTimeout:
    """Tests for ToolExecutor timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        """execute_tool returns error on timeout."""
        config = make_config(tool_timeout=0.1)
        registry = ToolRegistry()
        tool = make_slow_tool("slow", delay=1.0)  # Takes 1 second
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        result = await executor.execute_tool("slow", {})

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tool_specific_timeout_overrides_config(self):
        """Tool-specific timeout overrides config timeout."""
        config = make_config(tool_timeout=5.0)
        registry = ToolRegistry()
        tool = make_slow_tool("slow", delay=0.3)
        tool.timeout = 0.1  # Tool-specific timeout
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        result = await executor.execute_tool("slow", {})

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """execute_tool completes when within timeout."""
        config = make_config(tool_timeout=5.0)
        registry = ToolRegistry()
        tool = make_slow_tool("slow", delay=0.1)
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        result = await executor.execute_tool("slow", {})

        assert result.success is True

    @pytest.mark.asyncio
    async def test_timeout_records_failure_metrics(self):
        """Timeout records failure in metrics."""
        config = make_config(tool_timeout=0.1)
        registry = ToolRegistry()
        tool = make_slow_tool("slow", delay=1.0)
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        await executor.execute_tool("slow", {})

        metrics = executor.get_metrics("slow")
        assert metrics["slow"].failed_executions == 1


# ==============================================================================
# ToolExecutor Cancellation Feature Tests
# ==============================================================================

class TestToolExecutorCancellation:
    """Tests for ToolExecutor cancellation handling."""

    @pytest.mark.asyncio
    async def test_cancel_event_set_before_execution(self):
        """execute_tool returns error when cancel_event is set."""
        config = make_config()
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        cancel_event = asyncio.Event()
        cancel_event.set()

        result = await executor.execute_tool("search", {}, cancel_event=cancel_event)

        assert result.success is False
        assert "aborted" in result.error.lower()

    @pytest.mark.asyncio
    async def test_cancel_event_not_set_proceeds(self):
        """execute_tool proceeds when cancel_event is not set."""
        config = make_config()
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        cancel_event = asyncio.Event()

        result = await executor.execute_tool("search", {}, cancel_event=cancel_event)

        assert result.success is True


# ==============================================================================
# ToolExecutor Parallel Execution Feature Tests
# ==============================================================================

class TestToolExecutorExecuteTools:
    """Tests for ToolExecutor.execute_tools parallel execution."""

    @pytest.mark.asyncio
    async def test_executes_multiple_tools(self):
        """execute_tools executes multiple tools in parallel."""
        config = make_config()
        registry = ToolRegistry()
        registry.register(make_tool("search"))
        registry.register(make_tool("weather"))
        executor = ToolExecutor(config, registry)

        results = await executor.execute_tools([
            {"name": "search", "parameters": {"q": "test"}},
            {"name": "weather", "parameters": {"city": "NYC"}},
        ])

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_returns_results_in_order(self):
        """execute_tools returns results in the same order as calls."""
        config = make_config()
        registry = ToolRegistry()
        registry.register(make_tool("tool1"))
        registry.register(make_tool("tool2"))
        registry.register(make_tool("tool3"))
        executor = ToolExecutor(config, registry)

        results = await executor.execute_tools([
            {"name": "tool1", "parameters": {}},
            {"name": "tool2", "parameters": {}},
            {"name": "tool3", "parameters": {}},
        ])

        assert results[0].data == "tool1_result"
        assert results[1].data == "tool2_result"
        assert results[2].data == "tool3_result"

    @pytest.mark.asyncio
    async def test_handles_mixed_success_and_failure(self):
        """execute_tools handles mixed success and failure."""
        config = make_config()
        registry = ToolRegistry()
        registry.register(make_tool("good"))
        registry.register(make_failing_tool("bad", "Failed"))
        registry.register(make_tool("another_good"))
        executor = ToolExecutor(config, registry)

        results = await executor.execute_tools([
            {"name": "good", "parameters": {}},
            {"name": "bad", "parameters": {}},
            {"name": "another_good", "parameters": {}},
        ])

        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    @pytest.mark.asyncio
    async def test_empty_tool_calls_returns_empty_list(self):
        """execute_tools returns empty list for no calls."""
        config = make_config()
        registry = ToolRegistry()
        executor = ToolExecutor(config, registry)

        results = await executor.execute_tools([])

        assert results == []

    @pytest.mark.asyncio
    async def test_passes_discord_context_to_all_tools(self):
        """execute_tools passes discord_context to all tool calls."""
        config = make_config()
        registry = ToolRegistry()

        # Use a tool that requires permission
        registry.register(make_tool("tool1", requires_permission="read_messages"))
        executor = ToolExecutor(config, registry)

        # Valid context with permission
        message = make_discord_message(guild=MagicMock(), has_permission=True)

        # Patch isinstance to treat our mock as discord.Member
        with patch("persbot.tools.executor.isinstance") as mock_isinstance:
            def isinstance_side_effect(obj, cls):
                if hasattr(cls, "__name__") and cls.__name__ == "Member":
                    return True
                return isinstance(obj, cls)
            mock_isinstance.side_effect = isinstance_side_effect

            results = await executor.execute_tools(
                [{"name": "tool1", "parameters": {}}],
                discord_context=message,
            )

        assert results[0].success is True


# ==============================================================================
# ToolExecutor Get Metrics Feature Tests
# ==============================================================================

class TestToolExecutorGetMetrics:
    """Tests for ToolExecutor.get_metrics behavior."""

    @pytest.mark.asyncio
    async def test_get_metrics_returns_all_metrics(self):
        """get_metrics returns metrics for all tools."""
        config = make_config()
        registry = ToolRegistry()
        registry.register(make_tool("tool1"))
        registry.register(make_tool("tool2"))
        executor = ToolExecutor(config, registry)

        await executor.execute_tool("tool1", {})
        await executor.execute_tool("tool2", {})

        all_metrics = executor.get_metrics()

        assert "tool1" in all_metrics
        assert "tool2" in all_metrics

    @pytest.mark.asyncio
    async def test_get_metrics_for_specific_tool(self):
        """get_metrics returns metrics for a specific tool."""
        config = make_config()
        registry = ToolRegistry()
        registry.register(make_tool("search"))
        executor = ToolExecutor(config, registry)

        await executor.execute_tool("search", {})

        metrics = executor.get_metrics("search")

        assert "search" in metrics
        assert metrics["search"].total_executions == 1

    @pytest.mark.asyncio
    async def test_get_metrics_returns_empty_for_unknown_tool(self):
        """get_metrics returns empty metrics for unknown tool."""
        config = make_config()
        registry = ToolRegistry()
        executor = ToolExecutor(config, registry)

        metrics = executor.get_metrics("unknown")

        assert "unknown" in metrics
        assert metrics["unknown"].total_executions == 0

    @pytest.mark.asyncio
    async def test_metrics_accumulate_across_executions(self):
        """Metrics accumulate across multiple executions."""
        config = make_config()
        registry = ToolRegistry()
        registry.register(make_tool("search"))
        executor = ToolExecutor(config, registry)

        await executor.execute_tool("search", {})
        await executor.execute_tool("search", {})
        await executor.execute_tool("search", {})

        metrics = executor.get_metrics("search")

        assert metrics["search"].total_executions == 3
        assert metrics["search"].successful_executions == 3


# ==============================================================================
# ToolExecutor Edge Cases Feature Tests
# ==============================================================================

class TestToolExecutorEdgeCases:
    """Tests for edge cases in ToolExecutor."""

    @pytest.mark.asyncio
    async def test_missing_parameters_defaults_to_empty(self):
        """execute_tool handles missing parameters gracefully."""
        config = make_config()
        registry = ToolRegistry()

        async def handler(**kwargs):
            return ToolResult(success=True, data=kwargs)

        tool = ToolDefinition(
            name="flexible",
            description="Flexible tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        registry.register(tool)
        executor = ToolExecutor(config, registry)

        # Call without parameters dict
        result = await executor.execute_tool("flexible", {})

        assert result.success is True

    @pytest.mark.asyncio
    async def test_concurrent_executions_are_safe(self):
        """ToolExecutor handles concurrent executions safely."""
        config = make_config(tool_timeout=10.0)
        registry = ToolRegistry()
        registry.register(make_tool("search"))
        executor = ToolExecutor(config, registry)

        async def execute():
            return await executor.execute_tool("search", {})

        results = await asyncio.gather(*[execute() for _ in range(10)])

        assert all(r.success for r in results)

        metrics = executor.get_metrics("search")
        assert metrics["search"].total_executions == 10

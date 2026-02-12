"""Tests for tools/executor.py module.

This module provides comprehensive test coverage for:
- ExecutionMetrics dataclass
- ToolExecutor class
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

import discord

from persbot.tools.executor import (
    ToolExecutor,
    ExecutionMetrics,
)
from persbot.tools.base import ToolDefinition, ToolCategory, ToolParameter


# =============================================================================
# ExecutionMetrics Dataclass Tests
# =============================================================================


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_init_default_values(self):
        """Test ExecutionMetrics initialization with defaults."""
        metrics = ExecutionMetrics()

        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0
        assert metrics.last_execution_time is None

    def test_record_success_increments_counters(self):
        """Test record_success increments counters."""
        metrics = ExecutionMetrics()

        metrics.record_success()

        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.last_execution_time is not None

    def test_record_failure_increments_counters(self):
        """Test record_failure increments counters."""
        metrics = ExecutionMetrics()

        metrics.record_failure()

        assert metrics.total_executions == 1
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 1
        assert metrics.last_execution_time is not None

    def test_multiple_record_operations(self):
        """Test multiple record operations."""
        metrics = ExecutionMetrics()

        metrics.record_success()
        metrics.record_success()
        metrics.record_failure()

        assert metrics.total_executions == 3
        assert metrics.successful_executions == 2
        assert metrics.failed_executions == 1


# =============================================================================
# ToolExecutor Class Tests
# =============================================================================


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        from types import SimpleNamespace

        return SimpleNamespace(
            no_check_permission=True,
            tool_timeout=10.0,
        )

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool definition."""
        return ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.API_SEARCH,
            enabled=True,
            handler=AsyncMock(return_value=Mock(success=True, output="Result")),
        )

    @pytest.fixture
    def mock_registry(self, mock_tool):
        """Create a mock ToolRegistry."""
        registry = Mock()
        registry.get = Mock(return_value=mock_tool)
        return registry

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_config, mock_registry):
        """Test successful tool execution."""
        executor = ToolExecutor(mock_config(), mock_registry)
        tool = mock_registry.get()

        result = await executor.execute_tool(
            tool_name="test_tool",
            parameters={"query": "test"},
        )

        assert result.success is True
        assert result.output == "Result"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mock_config, mock_registry):
        """Test executing tool that is not registered."""
        executor = ToolExecutor(mock_config(), mock_registry)
        mock_registry.get = Mock(return_value=None)

        result = await executor.execute_tool(
            tool_name="nonexistent_tool",
            parameters={},
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_disabled(self, mock_config, mock_registry):
        """Test executing disabled tool."""
        executor = ToolExecutor(mock_config(), mock_registry)
        tool = mock_registry.get()
        tool.enabled = False

        result = await executor.execute_tool(
            tool_name="disabled_tool",
            parameters={},
        )

        assert result.success is False
        assert "disabled" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, mock_config, mock_registry):
        """Test tool execution timeout."""
        executor = ToolExecutor(mock_config(), mock_registry)
        tool = mock_registry.get()
        tool.handler = AsyncMock()

        # Make asyncio.wait_for timeout
        with patch("asyncio.wait_for") as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()

            result = await executor.execute_tool(
                tool_name="test_tool",
                parameters={},
            )

        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_cancellation_before_execution(self, mock_config, mock_registry):
        """Test cancellation before tool execution."""
        executor = ToolExecutor(mock_config(), mock_registry)
        cancel_event = asyncio.Event()
        cancel_event.set()

        result = await executor.execute_tool(
            tool_name="test_tool",
            parameters={},
            cancel_event=cancel_event,
        )

        assert result.success is False
        assert "aborted" in result.error

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self, mock_config, mock_registry):
        """Test parallel execution of multiple tools."""
        executor = ToolExecutor(mock_config(), mock_registry)

        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            enabled=True,
            handler=AsyncMock(return_value=Mock(success=True, output="Result1")),
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.API_SEARCH,
            enabled=True,
            handler=AsyncMock(return_value=Mock(success=True, output="Result2")),
        )

        mock_registry.get = Mock(side_effect=lambda n: [tool1, tool2][["tool1", "tool2"].index(n)])

        results = await executor.execute_tools(
            tool_calls=[
                {"name": "tool1", "parameters": {}},
                {"name": "tool2", "parameters": {}},
            ],
        )

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is True
        assert results[0].output == "Result1"
        assert results[1].output == "Result2"

    @pytest.mark.asyncio
    async def test_get_metrics_single_tool(self, mock_config, mock_registry):
        """Test getting metrics for a single tool."""
        executor = ToolExecutor(mock_config(), mock_registry)
        tool = mock_registry.get()

        # Execute tool once
        await executor.execute_tool(tool_name="test_tool", parameters={})

        metrics = executor.get_metrics("test_tool")

        assert "test_tool" in metrics
        assert metrics["test_tool"].total_executions == 1

    @pytest.mark.asyncio
    async def test_get_metrics_all_tools(self, mock_config, mock_registry):
        """Test getting metrics for all tools."""
        executor = ToolExecutor(mock_config(), mock_registry)

        await executor.execute_tool(tool_name="tool1", parameters={})
        await executor.execute_tool(tool_name="tool2", parameters={})

        metrics = executor.get_metrics()

        assert "tool1" in metrics
        assert "tool2" in metrics

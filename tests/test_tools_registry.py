"""Tests for tools/registry.py module.

This module provides comprehensive test coverage for:
- ToolRegistry class with all methods
- Tool registration and retrieval
- Category filtering
- Enabled/disabled state management
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from persbot.tools.base import (
    ToolDefinition,
    ToolCategory,
    ToolParameter,
    ToolResult,
)
from persbot.tools.registry import ToolRegistry


# =============================================================================
# ToolRegistry Class Tests
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool definition."""
        return ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                )
            ],
            handler=AsyncMock(return_value=ToolResult(success=True, data="result")),
        )

    @pytest.fixture
    def registry(self):
        """Create a ToolRegistry instance."""
        return ToolRegistry()

    def test_init_empty_registry(self, registry):
        """Test registry initialization."""
        assert registry._tools == {}
        # _by_category is initialized with all categories as empty lists
        assert ToolCategory.API_SEARCH in registry._by_category
        assert ToolCategory.API_WEATHER in registry._by_category

    def test_register_tool(self, registry, sample_tool):
        """Test registering a tool."""
        registry.register(sample_tool)

        assert "test_tool" in registry._tools
        assert registry._tools["test_tool"] == sample_tool
        assert ToolCategory.API_SEARCH in registry._by_category
        assert "test_tool" in registry._by_category[ToolCategory.API_SEARCH]

    def test_register_duplicate_tool(self, registry, sample_tool):
        """Test registering duplicate tool silently skips."""
        registry.register(sample_tool)
        # Should not raise, just skip
        registry.register(sample_tool)

        # Tool should only be registered once
        assert registry._by_category[ToolCategory.API_SEARCH].count("test_tool") == 1

    def test_register_multiple_tools(self, registry):
        """Test registering multiple tools."""
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        assert "tool1" in registry._tools
        assert "tool2" in registry._tools
        assert ToolCategory.API_SEARCH in registry._by_category
        assert ToolCategory.API_WEATHER in registry._by_category

    def test_get_existing_tool(self, registry, sample_tool):
        """Test getting an existing tool."""
        registry.register(sample_tool)

        result = registry.get("test_tool")

        assert result == sample_tool

    def test_get_nonexistent_tool(self, registry):
        """Test getting a non-existent tool returns None."""
        result = registry.get("nonexistent")

        assert result is None

    def test_has_tool_true(self, registry, sample_tool):
        """Test has_tool returns True for existing tool."""
        registry.register(sample_tool)

        assert registry.has_tool("test_tool") is True

    def test_has_tool_false(self, registry):
        """Test has_tool returns False for non-existent tool."""
        assert registry.has_tool("nonexistent") is False

    def test_get_all_empty(self, registry):
        """Test getting all tools from empty registry."""
        result = registry.get_all()

        assert result == {}

    def test_get_all_multiple_tools(self, registry):
        """Test getting all tools."""
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=True,
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
            enabled=False,
        )

        registry.register(tool1)
        registry.register(tool2)

        result = registry.get_all()

        assert len(result) == 2
        assert "tool1" in result
        assert "tool2" in result

    def test_get_by_category(self, registry):
        """Test getting tools by category."""
        tool1 = ToolDefinition(
            name="search_tool",
            description="Search",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )
        tool2 = ToolDefinition(
            name="weather_tool",
            description="Weather",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        result = registry.get_by_category(ToolCategory.API_SEARCH)

        assert len(result) == 1
        assert result[0].name == "search_tool"

    def test_get_enabled_only(self, registry):
        """Test getting only enabled tools."""
        tool1 = ToolDefinition(
            name="enabled_tool",
            description="Enabled",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=True,
        )
        tool2 = ToolDefinition(
            name="disabled_tool",
            description="Disabled",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
            enabled=False,
        )

        registry.register(tool1)
        registry.register(tool2)

        result = registry.get_enabled()

        assert len(result) == 1
        assert "enabled_tool" in result
        assert "disabled_tool" not in result

    def test_get_enabled_by_category(self, registry):
        """Test getting enabled tools by category."""
        tool1 = ToolDefinition(
            name="enabled_search",
            description="Enabled Search",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=True,
        )
        tool2 = ToolDefinition(
            name="disabled_search",
            description="Disabled Search",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=False,
        )

        registry.register(tool1)
        registry.register(tool2)

        result = registry.get_enabled_by_category(ToolCategory.API_SEARCH)

        assert len(result) == 1
        assert result[0].name == "enabled_search"

    def test_get_tool_names(self, registry):
        """Test getting all tool names."""
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        result = registry.get_tool_names()

        assert len(result) == 2
        assert "tool1" in result
        assert "tool2" in result

    def test_unregister_tool(self, registry, sample_tool):
        """Test unregistering a tool."""
        registry.register(sample_tool)

        result = registry.unregister("test_tool")

        assert result is True
        assert "test_tool" not in registry._tools
        assert registry.get("test_tool") is None

    def test_unregister_nonexistent_tool(self, registry):
        """Test unregistering non-existent tool returns False."""
        result = registry.unregister("nonexistent")

        assert result is False

    def test_set_tool_enabled_enable(self, registry):
        """Test enabling a disabled tool."""
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=False,
        )
        registry.register(tool)

        result = registry.set_tool_enabled("test_tool", True)

        assert result is True
        assert registry.get("test_tool").enabled is True

    def test_set_tool_enabled_disable(self, registry):
        """Test disabling an enabled tool."""
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
            enabled=True,
        )
        registry.register(tool)

        result = registry.set_tool_enabled("test_tool", False)

        assert result is True
        assert registry.get("test_tool").enabled is False

    def test_set_tool_enabled_nonexistent(self, registry):
        """Test setting enabled on non-existent tool returns False."""
        result = registry.set_tool_enabled("nonexistent", True)

        assert result is False

    def test_len_operator(self, registry):
        """Test __len__ returns correct count."""
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=AsyncMock(),
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.API_WEATHER,
            parameters=[],
            handler=AsyncMock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 2

    def test_contains_operator_true(self, registry, sample_tool):
        """Test __contains__ returns True for existing tool."""
        registry.register(sample_tool)

        assert "test_tool" in registry

    def test_contains_operator_false(self, registry):
        """Test __contains__ returns False for non-existent tool."""
        assert "nonexistent" not in registry

    def test_tool_definition_to_openai_format(self):
        """Test converting tool definition to OpenAI format."""
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                )
            ],
            handler=AsyncMock(),
        )

        result = tool.to_openai_format()

        assert result["type"] == "function"
        assert "function" in result
        assert result["function"]["name"] == "test_tool"
        assert "parameters" in result["function"]

    def test_tool_definition_to_gemini_format(self):
        """Test converting tool definition to Gemini format."""
        from persbot.tools.base import HAS_GENAI
        if not HAS_GENAI:
            pytest.skip("google.genai not installed")

        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                )
            ],
            handler=AsyncMock(),
        )

        result = tool.to_gemini_format()

        # Just verify the method can be called and returns object with name attribute
        assert hasattr(result, "name")

    def test_tool_definition_execute_success(self):
        """Test ToolDefinition execute with success."""
        async def mock_handler(**kwargs):
            return ToolResult(success=True, data="result")

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=mock_handler,
        )

        result = asyncio.run(tool.execute())

        assert result.success is True
        assert result.data == "result"

    def test_tool_definition_execute_with_error(self):
        """Test ToolDefinition execute with error."""
        async def mock_handler(**kwargs):
            raise ValueError("Test error")

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=mock_handler,
        )

        result = asyncio.run(tool.execute())

        assert result.success is False
        assert "Test error" in result.error

    def test_tool_definition_execute_with_cancel_event(self):
        """Test ToolDefinition execute respects cancel_event."""
        async def mock_handler(**kwargs):
            return ToolResult(success=True, data="result")

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=mock_handler,
        )

        cancel_event = asyncio.Event()
        cancel_event.set()

        result = asyncio.run(tool.execute(cancel_event=cancel_event))

        assert result.success is False
        assert "aborted" in result.error

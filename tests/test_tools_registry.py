"""Tests for tools/registry.py module.

This module provides comprehensive test coverage for:
- ToolRegistry class
"""

import pytest
from unittest.mock import Mock

from persbot.tools.registry import ToolRegistry
from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter


# =============================================================================
# ToolRegistry Class Tests
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_init_empty_registry(self):
        """Test ToolRegistry initialization."""
        registry = ToolRegistry()

        assert len(registry.get_all()) == 0
        assert len(registry.get_by_category(ToolCategory.API_SEARCH)) == 0

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool description",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool)

        # Verify tool was registered
        retrieved = registry.get("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"
        assert "test_tool" in registry.get_tool_names()

    def test_register_duplicate_tool(self):
        """Test that registering duplicate tool doesn't add it twice."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool description",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool)
        registry.register(tool)  # Register again

        # Should only have one entry
        assert registry.get_all()["test_tool"] is tool
        assert len(registry.get_all()) == 1

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool description",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool)

        # Unregister
        result = registry.unregister("test_tool")

        assert result is True
        assert "test_tool" not in registry.get_tool_names()
        assert "test_tool" not in registry.get_all()

    def test_unregister_nonexistent_tool(self):
        """Test unregistering non-existent tool returns False."""
        registry = ToolRegistry()

        result = registry.unregister("nonexistent_tool")

        assert result is False

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool description",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool)

        retrieved = registry.get("test_tool")

        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_get_tool_not_found(self):
        """Test getting non-existent tool returns None."""
        registry = ToolRegistry()

        retrieved = registry.get("nonexistent_tool")

        assert retrieved is None

    def test_get_all(self):
        """Test getting all registered tools."""
        registry = ToolRegistry()
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        all_tools = registry.get_all()

        assert len(all_tools) == 2
        assert "tool1" in all_tools
        assert "tool2" in all_tools

    def test_get_by_category(self):
        """Test getting tools by category."""
        registry = ToolRegistry()
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.DISCORD,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        api_tools = registry.get_by_category(ToolCategory.API_SEARCH)

        assert len(api_tools) == 1
        assert api_tools[0].name == "tool1"
        assert tool2 not in api_tools

    def test_get_enabled(self):
        """Test getting only enabled tools."""
        registry = ToolRegistry()
        tool_enabled = ToolDefinition(
            name="enabled_tool",
            description="Enabled tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )
        tool_disabled = ToolDefinition(
            name="disabled_tool",
            description="Disabled tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=False,
            handler=Mock(),
        )

        registry.register(tool_enabled)
        registry.register(tool_disabled)

        enabled_tools = registry.get_enabled()

        assert len(enabled_tools) == 1
        assert enabled_tools[0].name == "enabled_tool"
        assert tool_disabled.name not in [t.name for t in enabled_tools.values()]

    def test_get_enabled_by_category(self):
        """Test getting enabled tools by category."""
        registry = ToolRegistry()
        tool1 = ToolDefinition(
            name="enabled_tool1",
            description="Enabled tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )
        tool2 = ToolDefinition(
            name="disabled_tool1",
            description="Disabled tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=False,
            handler=Mock(),
        )
        tool3 = ToolDefinition(
            name="enabled_tool2",
            description="Enabled tool 2",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        enabled_api_tools = registry.get_enabled_by_category(ToolCategory.API_SEARCH)

        assert len(enabled_api_tools) == 2
        assert all(t.enabled for t in enabled_api_tools)

    def test_set_tool_enabled(self):
        """Test setting tool enabled status."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool)

        # Disable the tool
        result = registry.set_tool_enabled("test_tool", False)

        assert result is True
        assert registry.get("test_tool").enabled is False

        # Re-enable the tool
        result = registry.set_tool_enabled("test_tool", True)

        assert result is True
        assert registry.get("test_tool").enabled is True

    def test_set_tool_enabled_nonexistent(self):
        """Test setting enabled status on non-existent tool returns False."""
        registry = ToolRegistry()

        result = registry.set_tool_enabled("nonexistent_tool", True)

        assert result is False

    def test_get_tool_names(self):
        """Test getting list of tool names."""
        registry = ToolRegistry()
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        tool_names = registry.get_tool_names()

        assert len(tool_names) == 2
        assert "tool1" in tool_names
        assert "tool2" in tool_names

    def test_has_tool(self):
        """Test checking if tool is registered."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool)

        assert registry.has_tool("test_tool") is True
        assert registry.has_tool("nonexistent_tool") is False

    def test_len_operator(self):
        """Test len() operator."""
        registry = ToolRegistry()
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 2

    def test_contains_operator(self):
        """Test 'in' operator."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            enabled=True,
            handler=Mock(),
        )

        registry.register(tool)

        assert "test_tool" in registry
        assert "nonexistent_tool" not in registry

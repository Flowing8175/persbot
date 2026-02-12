"""Feature tests for tool registry.

Tests focus on behavior rather than implementation details:
- ToolRegistry: tool registration and management
"""

import pytest

from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult
from persbot.tools.registry import ToolRegistry


# Helper to create tools for testing
async def dummy_handler(**kwargs):
    return ToolResult(success=True)


def make_tool(name: str, category: ToolCategory = ToolCategory.API_SEARCH, enabled: bool = True):
    """Create a simple tool for testing."""
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        category=category,
        parameters=[],
        handler=dummy_handler,
        enabled=enabled,
    )


# ==============================================================================
# ToolRegistry Creation Feature Tests
# ==============================================================================

class TestToolRegistryCreation:
    """Tests for ToolRegistry instantiation."""

    def test_creates_empty_registry(self):
        """ToolRegistry starts empty."""
        registry = ToolRegistry()
        assert len(registry) == 0

    def test_creates_with_category_index(self):
        """ToolRegistry creates indices for all categories."""
        registry = ToolRegistry()

        # Internal check: all categories should have empty lists
        for category in ToolCategory:
            assert registry.get_by_category(category) == []


# ==============================================================================
# ToolRegistry Register Feature Tests
# ==============================================================================

class TestToolRegistryRegister:
    """Tests for ToolRegistry.register behavior."""

    def test_register_adds_tool(self):
        """register adds a tool to the registry."""
        registry = ToolRegistry()
        tool = make_tool("search")

        registry.register(tool)

        assert registry.get("search") is tool
        assert len(registry) == 1

    def test_register_skips_duplicate(self, caplog):
        """register skips tools with duplicate names."""
        registry = ToolRegistry()
        tool1 = make_tool("search")
        tool2 = make_tool("search")

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 1
        assert registry.get("search") is tool1
        assert "already registered" in caplog.text.lower()

    def test_register_adds_to_category_index(self):
        """register adds tool to category index."""
        registry = ToolRegistry()
        tool = make_tool("search", ToolCategory.API_SEARCH)

        registry.register(tool)

        tools_in_category = registry.get_by_category(ToolCategory.API_SEARCH)
        assert tool in tools_in_category

    def test_register_multiple_tools_in_same_category(self):
        """register handles multiple tools in same category."""
        registry = ToolRegistry()
        tool1 = make_tool("search", ToolCategory.API_SEARCH)
        tool2 = make_tool("weather", ToolCategory.API_SEARCH)

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.get_by_category(ToolCategory.API_SEARCH)
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_register_tools_in_different_categories(self):
        """register handles tools in different categories."""
        registry = ToolRegistry()
        tool1 = make_tool("search", ToolCategory.API_SEARCH)
        tool2 = make_tool("user_info", ToolCategory.DISCORD_USER)

        registry.register(tool1)
        registry.register(tool2)

        search_tools = registry.get_by_category(ToolCategory.API_SEARCH)
        user_tools = registry.get_by_category(ToolCategory.DISCORD_USER)

        assert len(search_tools) == 1
        assert len(user_tools) == 1


# ==============================================================================
# ToolRegistry Unregister Feature Tests
# ==============================================================================

class TestToolRegistryUnregister:
    """Tests for ToolRegistry.unregister behavior."""

    def test_unregister_removes_tool(self):
        """unregister removes a tool from the registry."""
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)

        result = registry.unregister("search")

        assert result is True
        assert registry.get("search") is None
        assert len(registry) == 0

    def test_unregister_returns_false_for_unknown_tool(self):
        """unregister returns False for unknown tools."""
        registry = ToolRegistry()

        result = registry.unregister("unknown")

        assert result is False

    def test_unregister_removes_from_category_index(self):
        """unregister removes tool from category index."""
        registry = ToolRegistry()
        tool = make_tool("search", ToolCategory.API_SEARCH)
        registry.register(tool)

        registry.unregister("search")

        tools = registry.get_by_category(ToolCategory.API_SEARCH)
        assert tool not in tools

    def test_unregister_preserves_other_tools(self):
        """unregister preserves other tools in registry."""
        registry = ToolRegistry()
        tool1 = make_tool("search")
        tool2 = make_tool("weather")
        registry.register(tool1)
        registry.register(tool2)

        registry.unregister("search")

        assert registry.get("search") is None
        assert registry.get("weather") is tool2


# ==============================================================================
# ToolRegistry Get Feature Tests
# ==============================================================================

class TestToolRegistryGet:
    """Tests for ToolRegistry.get behavior."""

    def test_get_returns_tool_by_name(self):
        """get returns the tool with the given name."""
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)

        result = registry.get("search")

        assert result is tool

    def test_get_returns_none_for_unknown_tool(self):
        """get returns None for unknown tools."""
        registry = ToolRegistry()

        result = registry.get("unknown")

        assert result is None

    def test_get_all_returns_copy(self):
        """get_all returns a copy of the tools dictionary."""
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)

        all_tools = registry.get_all()

        assert all_tools["search"] is tool
        # Verify it's a copy
        all_tools["new_tool"] = make_tool("new")
        assert "new_tool" not in registry.get_all()


# ==============================================================================
# ToolRegistry Get By Category Feature Tests
# ==============================================================================

class TestToolRegistryGetByCategory:
    """Tests for ToolRegistry.get_by_category behavior."""

    def test_returns_tools_in_category(self):
        """get_by_category returns tools in the specified category."""
        registry = ToolRegistry()
        tool1 = make_tool("search", ToolCategory.API_SEARCH)
        tool2 = make_tool("weather", ToolCategory.API_WEATHER)
        tool3 = make_tool("time", ToolCategory.API_TIME)
        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        result = registry.get_by_category(ToolCategory.API_SEARCH)

        assert len(result) == 1
        assert tool1 in result

    def test_returns_empty_list_for_empty_category(self):
        """get_by_category returns empty list for category with no tools."""
        registry = ToolRegistry()
        tool = make_tool("search", ToolCategory.API_SEARCH)
        registry.register(tool)

        result = registry.get_by_category(ToolCategory.DISCORD_GUILD)

        assert result == []

    def test_returns_all_tools_in_category(self):
        """get_by_category returns all tools in the category."""
        registry = ToolRegistry()
        tool1 = make_tool("search1", ToolCategory.API_SEARCH)
        tool2 = make_tool("search2", ToolCategory.API_SEARCH)
        tool3 = make_tool("search3", ToolCategory.API_SEARCH)
        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        result = registry.get_by_category(ToolCategory.API_SEARCH)

        assert len(result) == 3


# ==============================================================================
# ToolRegistry Get Enabled Feature Tests
# ==============================================================================

class TestToolRegistryGetEnabled:
    """Tests for ToolRegistry.get_enabled behavior."""

    def test_returns_only_enabled_tools(self):
        """get_enabled returns only enabled tools."""
        registry = ToolRegistry()
        tool1 = make_tool("enabled_tool", enabled=True)
        tool2 = make_tool("disabled_tool", enabled=False)
        registry.register(tool1)
        registry.register(tool2)

        enabled = registry.get_enabled()

        assert "enabled_tool" in enabled
        assert "disabled_tool" not in enabled
        assert len(enabled) == 1

    def test_returns_all_when_all_enabled(self):
        """get_enabled returns all tools when all are enabled."""
        registry = ToolRegistry()
        tool1 = make_tool("tool1", enabled=True)
        tool2 = make_tool("tool2", enabled=True)
        registry.register(tool1)
        registry.register(tool2)

        enabled = registry.get_enabled()

        assert len(enabled) == 2

    def test_returns_empty_when_all_disabled(self):
        """get_enabled returns empty dict when all tools disabled."""
        registry = ToolRegistry()
        tool1 = make_tool("tool1", enabled=False)
        tool2 = make_tool("tool2", enabled=False)
        registry.register(tool1)
        registry.register(tool2)

        enabled = registry.get_enabled()

        assert enabled == {}

    def test_get_enabled_by_category_filters_correctly(self):
        """get_enabled_by_category returns only enabled tools in category."""
        registry = ToolRegistry()
        tool1 = make_tool("enabled", ToolCategory.API_SEARCH, enabled=True)
        tool2 = make_tool("disabled", ToolCategory.API_SEARCH, enabled=False)
        registry.register(tool1)
        registry.register(tool2)

        result = registry.get_enabled_by_category(ToolCategory.API_SEARCH)

        assert len(result) == 1
        assert tool1 in result
        assert tool2 not in result


# ==============================================================================
# ToolRegistry Set Enabled Feature Tests
# ==============================================================================

class TestToolRegistrySetEnabled:
    """Tests for ToolRegistry.set_tool_enabled behavior."""

    def test_enables_tool(self):
        """set_tool_enabled enables a tool."""
        registry = ToolRegistry()
        tool = make_tool("search", enabled=False)
        registry.register(tool)

        result = registry.set_tool_enabled("search", True)

        assert result is True
        assert tool.enabled is True

    def test_disables_tool(self):
        """set_tool_enabled disables a tool."""
        registry = ToolRegistry()
        tool = make_tool("search", enabled=True)
        registry.register(tool)

        result = registry.set_tool_enabled("search", False)

        assert result is True
        assert tool.enabled is False

    def test_returns_false_for_unknown_tool(self):
        """set_tool_enabled returns False for unknown tools."""
        registry = ToolRegistry()

        result = registry.set_tool_enabled("unknown", True)

        assert result is False


# ==============================================================================
# ToolRegistry Contains Feature Tests
# ==============================================================================

class TestToolRegistryContains:
    """Tests for ToolRegistry.has_tool and __contains__ behavior."""

    def test_has_tool_returns_true_for_registered(self):
        """has_tool returns True for registered tools."""
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)

        assert registry.has_tool("search") is True

    def test_has_tool_returns_false_for_unknown(self):
        """has_tool returns False for unknown tools."""
        registry = ToolRegistry()

        assert registry.has_tool("unknown") is False

    def test_contains_operator_works(self):
        """__contains__ allows 'in' operator."""
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)

        assert "search" in registry
        assert "unknown" not in registry


# ==============================================================================
# ToolRegistry Length Feature Tests
# ==============================================================================

class TestToolRegistryLength:
    """Tests for ToolRegistry.__len__ behavior."""

    def test_len_returns_zero_for_empty_registry(self):
        """__len__ returns 0 for empty registry."""
        registry = ToolRegistry()
        assert len(registry) == 0

    def test_len_returns_count_of_registered_tools(self):
        """__len__ returns count of registered tools."""
        registry = ToolRegistry()
        registry.register(make_tool("tool1"))
        registry.register(make_tool("tool2"))
        registry.register(make_tool("tool3"))

        assert len(registry) == 3

    def test_len_decreases_after_unregister(self):
        """__len__ decreases after unregistering a tool."""
        registry = ToolRegistry()
        registry.register(make_tool("tool1"))
        registry.register(make_tool("tool2"))

        registry.unregister("tool1")

        assert len(registry) == 1


# ==============================================================================
# ToolRegistry Get Names Feature Tests
# ==============================================================================

class TestToolRegistryGetNames:
    """Tests for ToolRegistry.get_tool_names behavior."""

    def test_returns_list_of_tool_names(self):
        """get_tool_names returns list of registered tool names."""
        registry = ToolRegistry()
        registry.register(make_tool("search"))
        registry.register(make_tool("weather"))

        names = registry.get_tool_names()

        assert "search" in names
        assert "weather" in names
        assert len(names) == 2

    def test_returns_empty_list_for_empty_registry(self):
        """get_tool_names returns empty list for empty registry."""
        registry = ToolRegistry()

        names = registry.get_tool_names()

        assert names == []


# ==============================================================================
# ToolRegistry Edge Cases Feature Tests
# ==============================================================================

class TestToolRegistryEdgeCases:
    """Tests for edge cases in ToolRegistry."""

    def test_register_unregister_same_tool_multiple_times(self):
        """register and unregister handle repeated operations."""
        registry = ToolRegistry()
        tool = make_tool("search")

        registry.register(tool)
        registry.unregister("search")
        registry.register(tool)

        assert registry.get("search") is tool

    def test_unregister_already_unregistered_tool(self):
        """unregister returns False for already unregistered tool."""
        registry = ToolRegistry()
        tool = make_tool("search")
        registry.register(tool)

        result1 = registry.unregister("search")
        result2 = registry.unregister("search")

        assert result1 is True
        assert result2 is False

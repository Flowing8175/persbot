"""Tool registry for managing available tools."""

import logging
from typing import Dict, List, Optional

from persbot.tools.base import ToolCategory, ToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tool definitions and lookups."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._by_category: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool in the registry.

        Args:
            tool: The tool definition to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            logger.warning("Tool %s already registered, skipping", tool.name)
            return

        self._tools[tool.name] = tool
        self._by_category[tool.category].append(tool.name)
        logger.debug("Registered tool: %s (category: %s)", tool.name, tool.category.value)

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister.

        Returns:
            True if the tool was unregistered, False if it wasn't found.
        """
        if tool_name not in self._tools:
            return False

        tool = self._tools[tool_name]
        self._by_category[tool.category].remove(tool_name)
        del self._tools[tool_name]
        logger.debug("Unregistered tool: %s", tool_name)
        return True

    def get(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name.

        Args:
            tool_name: Name of the tool to retrieve.

        Returns:
            The tool definition if found, None otherwise.
        """
        return self._tools.get(tool_name)

    def get_all(self) -> Dict[str, ToolDefinition]:
        """Get all registered tools.

        Returns:
            Dictionary mapping tool names to their definitions.
        """
        return self._tools.copy()

    def get_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all tools in a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of tool definitions in the category.
        """
        tool_names = self._by_category.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_enabled(self) -> Dict[str, ToolDefinition]:
        """Get all enabled tools.

        Returns:
            Dictionary mapping tool names to their enabled definitions.
        """
        return {name: tool for name, tool in self._tools.items() if tool.enabled}

    def get_enabled_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all enabled tools in a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of enabled tool definitions in the category.
        """
        return [tool for tool in self.get_by_category(category) if tool.enabled]

    def set_tool_enabled(self, tool_name: str, enabled: bool) -> bool:
        """Enable or disable a tool.

        Args:
            tool_name: Name of the tool.
            enabled: Whether to enable or disable the tool.

        Returns:
            True if the tool was found and updated, False otherwise.
        """
        tool = self._tools.get(tool_name)
        if tool:
            tool.enabled = enabled
            logger.debug("Tool %s %s", tool_name, "enabled" if enabled else "disabled")
            return True
        return False

    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is registered, False otherwise.
        """
        return tool_name in self._tools

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered using 'in' operator."""
        return tool_name in self._tools

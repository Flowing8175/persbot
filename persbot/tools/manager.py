"""Tool manager for integrating tools with the chat system."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import discord

from persbot.config import AppConfig
from persbot.tools.api_tools import register_all_api_tools
from persbot.tools.base import ToolCategory, ToolDefinition
from persbot.tools.discord_tools import register_all_discord_tools
from persbot.tools.executor import ToolExecutor
from persbot.tools.persona_tools import register_all_persona_tools
from persbot.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolManager:
    """Manages tool registration, execution, and provider integration."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(config, self.registry)

        # Register tools based on configuration
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all enabled tools."""
        if not getattr(self.config, "enable_tools", True):
            logger.info("Tools are disabled in configuration")
            return

        # Register Discord tools
        if getattr(self.config, "enable_discord_tools", True):
            register_all_discord_tools(self.registry)
            logger.info("Registered Discord read-only tools")

        # Register API tools
        if getattr(self.config, "enable_api_tools", True):
            # Register with API keys if available
            register_all_api_tools(self.registry)
            logger.info("Registered external API tools")

        # Register Persona tools (for Zeta.ai-style high-immersion persona bot)
        if getattr(self.config, "enable_persona_tools", True):
            register_all_persona_tools(self.registry)
            logger.info("Registered persona immersion tools")

        logger.info("Total tools registered: %d", len(self.registry))

    def get_enabled_tools(self) -> Dict[str, ToolDefinition]:
        """Get all enabled tools.

        Returns:
            Dictionary of enabled tool definitions.
        """
        return self.registry.get_enabled()

    def get_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all enabled tools in a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of enabled tool definitions in the category.
        """
        return self.registry.get_enabled_by_category(category)

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        discord_context: Optional[discord.Message] = None,
    ):
        """Execute a single tool.

        Args:
            tool_name: Name of the tool to execute.
            parameters: Parameters to pass to the tool.
            discord_context: Discord message context for permission checks.

        Returns:
            ToolResult containing the execution result.
        """
        # Inject discord_context and config into parameters for tool handlers
        if "discord_context" not in parameters and discord_context:
            parameters["discord_context"] = discord_context

        # Inject API keys for API tools - only if tool category requires them
        tool = self.registry.get(tool_name)
        if tool and tool.category == ToolCategory.API_SEARCH:
            if "search_api_key" not in parameters:
                parameters["search_api_key"] = getattr(self.config, "search_api_key", None)
        if tool and tool.category == ToolCategory.API_WEATHER:
            if "weather_api_key" not in parameters:
                parameters["weather_api_key"] = getattr(self.config, "weather_api_key", None)

        return await self.executor.execute_tool(tool_name, parameters, discord_context)

    async def execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        discord_context: Optional[discord.Message] = None,
    ) -> List[Any]:
        """Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of tool call dictionaries with 'name' and 'parameters'.
            discord_context: Discord message context for permission checks.

        Returns:
            List of tool execution results formatted for the AI provider.
        """
        # Execute all tools in parallel using asyncio.gather
        tasks = [self._execute_and_format_tool(call, discord_context) for call in tool_calls]

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def _execute_and_format_tool(
        self,
        call: Dict[str, Any],
        discord_context: Optional[discord.Message],
    ) -> Dict[str, Any]:
        """Execute a single tool and format its result.

        Args:
            call: Tool call dictionary with 'name' and 'parameters'.
            discord_context: Discord message context for permission checks.

        Returns:
            Formatted tool result dictionary.
        """
        tool_name = call.get("name")
        if not tool_name:
            raise ValueError("Tool call missing 'name' field")

        parameters = call.get("parameters", {})

        # Add call ID if present (for OpenAI/Z.AI format)
        call_id = call.get("id")

        # Execute the tool
        result = await self.execute_tool(tool_name, parameters, discord_context)

        # Format result for provider
        if call_id:
            # OpenAI/Z.AI format with ID
            formatted = {
                "id": call_id,
                "name": tool_name,
                "result": result.data if result.success else None,
                "error": result.error if not result.success else None,
            }
        else:
            # Gemini format
            formatted = {
                "name": tool_name,
                "result": result.data if result.success else None,
                "error": result.error if not result.success else None,
            }

        # Preserve image data if present in metadata
        if result.success and result.metadata.get("image_bytes"):
            formatted["image_bytes"] = result.metadata["image_bytes"]

        return formatted

    def set_tool_enabled(self, tool_name: str, enabled: bool) -> bool:
        """Enable or disable a specific tool.

        Args:
            tool_name: Name of the tool.
            enabled: Whether to enable or disable.

        Returns:
            True if successful, False otherwise.
        """
        return self.registry.set_tool_enabled(tool_name, enabled)

    def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution metrics for tools.

        Args:
            tool_name: Optional specific tool name.

        Returns:
            Dictionary with tool metrics.
        """
        return self.executor.get_metrics(tool_name)

    def is_enabled(self) -> bool:
        """Check if tools are enabled in configuration.

        Returns:
            True if tools are enabled.
        """
        return getattr(self.config, "enable_tools", True)

    def clear_rate_limits(self, user_id: Optional[int] = None) -> None:
        """Clear rate limit records.

        Args:
            user_id: Optional specific user ID.
        """
        self.executor.clear_rate_limits(user_id)

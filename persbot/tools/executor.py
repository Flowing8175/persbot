"""Tool executor for executing AI tool calls."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import discord

from persbot.config import AppConfig
from persbot.tools.base import ToolDefinition, ToolResult
from persbot.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Metrics tracking for tool executions."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    last_execution_time: Optional[float] = None

    def record_success(self) -> None:
        """Record a successful execution."""
        self.total_executions += 1
        self.successful_executions += 1
        self.last_execution_time = time.time()

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.total_executions += 1
        self.failed_executions += 1
        self.last_execution_time = time.time()


class ToolExecutor:
    """Executes tools with permission checks and error handling."""

    def __init__(self, config: AppConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry
        self._metrics: Dict[str, ExecutionMetrics] = {}

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        discord_context: Optional[discord.Message] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ToolResult:
        """Execute a tool with permission checks and rate limiting.

        Args:
            tool_name: Name of the tool to execute.
            parameters: Parameters to pass to the tool.
            discord_context: Discord message context for permission checks.
            cancel_event: AsyncIO event to check for cancellation before execution.

        Returns:
            ToolResult containing the execution result.
        """
        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
            )

        if not tool.enabled:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' is disabled",
            )

        # Check permissions
        if discord_context:
            perm_check = self._check_permissions(tool, discord_context)
            if not perm_check:
                return ToolResult(
                    success=False,
                    error=f"Missing required permissions for tool '{tool_name}'",
                )

        # Inject user_id from Discord context if tool requires it but it's not provided
        if discord_context:
            # Check if tool has user_id parameter and it's not already provided
            has_user_id_param = any(p.name == "user_id" for p in tool.parameters)
            if has_user_id_param and "user_id" not in parameters:
                parameters["user_id"] = str(discord_context.author.id)

        # Execute the tool
        return await self._execute_with_timeout(tool, parameters, cancel_event)

    def _check_permissions(self, tool: ToolDefinition, message: discord.Message) -> bool:
        """Check if the user has required permissions for the tool.

        Args:
            tool: The tool to check permissions for.
            message: Discord message to check permissions against.

        Returns:
            True if user has required permissions, False otherwise.
        """
        if not tool.requires_permission:
            return True

        # Bypass permission check if NO_CHECK_PERMISSION is set
        if self.config.no_check_permission:
            return True

        # For DM contexts, we need to be more permissive
        if not message.guild:
            # In DMs, channel permissions don't apply
            # We should be more restrictive or permissive based on the tool
            # For read-only tools, we can allow them in DMs
            return True

        if isinstance(message.author, discord.Member):
            # Check if user has the required permission
            perm_name = tool.requires_permission
            if hasattr(message.author.guild_permissions, perm_name):
                return bool(getattr(message.author.guild_permissions, perm_name))
            else:
                logger.warning(
                    "Tool '%s' requires invalid permission '%s'. "
                    "Check tool configuration for valid permission names.",
                    tool.name,
                    perm_name,
                )

        return False

    async def _execute_with_timeout(
        self,
        tool: ToolDefinition,
        parameters: Dict[str, Any],
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ToolResult:
        """Execute a tool with timeout and cancellation support.

        Args:
            tool: The tool to execute.
            parameters: Parameters to pass to the tool.
            cancel_event: AsyncIO event to check for cancellation.

        Returns:
            ToolResult containing the execution result.
        """
        timeout = (
            tool.timeout if tool.timeout is not None else getattr(self.config, "tool_timeout", 10.0)
        )

        try:
            # Check for cancellation before executing tool
            if cancel_event and cancel_event.is_set():
                return ToolResult(
                    success=False,
                    error=f"Tool '{tool.name}' execution aborted by user",
                )

            result = await asyncio.wait_for(
                tool.execute(**parameters, cancel_event=cancel_event),
                timeout=timeout,
            )

            # Track metrics
            if tool.name not in self._metrics:
                self._metrics[tool.name] = ExecutionMetrics()

            if result.success:
                self._metrics[tool.name].record_success()
            else:
                self._metrics[tool.name].record_failure()
                logger.error("Tool '%s' execution failed: %s", tool.name, result.error)

            return result

        except asyncio.TimeoutError:
            if tool.name not in self._metrics:
                self._metrics[tool.name] = ExecutionMetrics()
            self._metrics[tool.name].record_failure()

            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {timeout} seconds",
            )
        except Exception as e:
            if tool.name not in self._metrics:
                self._metrics[tool.name] = ExecutionMetrics()
            self._metrics[tool.name].record_failure()

            logger.error("Error executing tool %s: %s", tool.name, e, exc_info=True)
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
            )

    async def execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        discord_context: Optional[discord.Message] = None,
    ) -> List[ToolResult]:
        """Execute multiple tools in parallel.

        Args:
            tool_calls: List of tool call dictionaries with 'name' and 'parameters'.
            discord_context: Discord message context for permission checks.

        Returns:
            List of ToolResults in the same order as tool_calls.
        """
        tasks = [
            self.execute_tool(
                call["name"],
                call.get("parameters", {}),
                discord_context,
            )
            for call in tool_calls
        ]

        return await asyncio.gather(*tasks)

    def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, ExecutionMetrics]:
        """Get execution metrics for tools.

        Args:
            tool_name: Optional specific tool name to get metrics for.
                     If None, returns metrics for all tools.

        Returns:
            Dictionary mapping tool names to their metrics.
        """
        if tool_name:
            return {tool_name: self._metrics.get(tool_name, ExecutionMetrics())}
        return self._metrics.copy()

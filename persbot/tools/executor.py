"""Tool executor for executing AI tool calls."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import discord

from persbot.config import AppConfig
from persbot.tools.base import ToolCategory, ToolDefinition, ToolResult
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
    """Executes tools with permission checks, rate limiting, and error handling."""

    def __init__(self, config: AppConfig, registry: ToolRegistry):
        self.config = config
        self.registry = registry
        self._rate_limits: Dict[str, float] = {}  # user_id -> last_execution_time
        self._metrics: Dict[str, ExecutionMetrics] = {}

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        discord_context: Optional[discord.Message] = None,
    ) -> ToolResult:
        """Execute a tool with permission checks and rate limiting.

        Args:
            tool_name: Name of the tool to execute.
            parameters: Parameters to pass to the tool.
            discord_context: Discord message context for permission checks.

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

        # Check rate limits
        if discord_context:
            rate_limit_check = self._check_rate_limit(tool, discord_context.author.id)
            if not rate_limit_check:
                return ToolResult(
                    success=False,
                    error=f"Rate limit exceeded for tool '{tool_name}'",
                )

        # Execute the tool
        return await self._execute_with_timeout(tool, parameters)

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
                return getattr(message.author.guild_permissions, perm_name)

        return False

    def _check_rate_limit(self, tool: ToolDefinition, user_id: int) -> bool:
        """Check if the user is rate limited for this tool.

        Args:
            tool: The tool to check rate limits for.
            user_id: ID of the user attempting to use the tool.

        Returns:
            True if user can use the tool, False if rate limited.
        """
        # Use the tool's rate limit if set, otherwise use default
        default_limit = getattr(self.config, "tool_rate_limit", 60)
        limit = tool.rate_limit if tool.rate_limit is not None else default_limit

        if limit <= 0:
            return True  # No rate limiting

        key = f"{user_id}:{tool.name}"
        last_execution = self._rate_limits.get(key, 0)
        current_time = time.time()

        if current_time - last_execution < limit:
            logger.debug(
                "User %s rate limited for tool %s (last: %s, now: %s, limit: %s)",
                user_id,
                tool.name,
                last_execution,
                current_time,
                limit,
            )
            return False

        self._rate_limits[key] = current_time
        return True

    async def _execute_with_timeout(
        self,
        tool: ToolDefinition,
        parameters: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool with timeout.

        Args:
            tool: The tool to execute.
            parameters: Parameters to pass to the tool.

        Returns:
            ToolResult containing the execution result.
        """
        timeout = getattr(self.config, "tool_timeout", 10.0)

        try:
            result = await asyncio.wait_for(
                tool.execute(**parameters),
                timeout=timeout,
            )

            # Track metrics
            if tool.name not in self._metrics:
                self._metrics[tool.name] = ExecutionMetrics()

            if result.success:
                self._metrics[tool.name].record_success()
                logger.debug("Tool %s executed successfully", tool.name)
            else:
                self._metrics[tool.name].record_failure()
                logger.warning("Tool %s execution failed: %s", tool.name, result.error)

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

    def clear_rate_limits(self, user_id: Optional[int] = None) -> None:
        """Clear rate limit records.

        Args:
            user_id: Optional specific user ID to clear limits for.
                    If None, clears all rate limits.
        """
        if user_id:
            keys_to_remove = [k for k in self._rate_limits if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self._rate_limits[key]
        else:
            self._rate_limits.clear()

    def cleanup_old_rate_limits(self, max_age_seconds: int = 3600) -> None:
        """Clean up old rate limit entries to prevent memory leaks.

        Args:
            max_age_seconds: Maximum age of rate limit entries in seconds (default: 1 hour).
        """
        current_time = time.time()
        keys_to_remove = [
            key
            for key, last_exec in self._rate_limits.items()
            if current_time - last_exec > max_age_seconds
        ]
        for key in keys_to_remove:
            del self._rate_limits[key]

        if keys_to_remove:
            logger.debug("Cleaned up %d old rate limit entries", len(keys_to_remove))

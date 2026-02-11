"""Base tool adapter with common functionality for all providers."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from persbot.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class BaseToolAdapter(ABC):
    """Abstract base class for tool format adapters with common functionality."""

    @staticmethod
    @abstractmethod
    def convert_tools(tools: List[ToolDefinition]) -> Any:
        """Convert tool definitions to provider-specific format.

        Args:
            tools: List of ToolDefinition objects.

        Returns:
            Provider-specific tool format.
        """
        pass

    @staticmethod
    @abstractmethod
    def extract_function_calls(response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from provider response.

        Args:
            response: Provider response object.

        Returns:
            List of function call dictionaries with 'name' and 'parameters'.
        """
        pass

    @staticmethod
    @abstractmethod
    def format_results(results: List[Dict[str, Any]]) -> Any:
        """Format function results for sending back to provider.

        Args:
            results: List of dicts with 'name', 'result', and optionally 'error'.

        Returns:
            Provider-specific formatted results.
        """
        pass

    @staticmethod
    def _extract_openai_style_function_calls(response: Any) -> List[Dict[str, Any]]:
        """
        Extract function calls from OpenAI-style responses (used by OpenAI and Z.AI).

        This method handles the common pattern used by OpenAI-compatible APIs
        where tool calls are in response.choices[x].message.tool_calls.
        """
        function_calls = []

        try:
            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    message = getattr(choice, "message", None)
                    if message and hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            fc_data = {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "parameters": {},
                            }

                            # Parse arguments
                            try:
                                args = json.loads(tool_call.function.arguments)
                                fc_data["parameters"] = args
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Failed to parse tool arguments: %s",
                                    tool_call.function.arguments,
                                )

                            function_calls.append(fc_data)

        except Exception as e:
            logger.error(
                "Error extracting function calls from OpenAI-style response: %s",
                e,
                exc_info=True,
            )

        return function_calls

    @staticmethod
    def _create_openai_style_tool_messages(
        results: List[Dict[str, Any]], handle_binary_data: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create tool messages in OpenAI format (used by OpenAI and Z.AI).

        Args:
            results: List of dicts with 'id', 'name', 'result', and optionally 'error'.
            handle_binary_data: If True, handle binary data specially.

        Returns:
            List of message dictionaries in OpenAI format.
        """
        messages = []

        for result_item in results:
            call_id = result_item.get("id")
            tool_name = result_item.get("name")
            result_data = result_item.get("result")
            error = result_item.get("error")

            if error:
                content = f"Error: {error}"
            elif handle_binary_data and isinstance(result_data, bytes):
                content = f"Binary data ({len(result_data)} bytes)"
            elif result_data is not None:
                content = str(result_data)
                # Truncate very long strings to avoid prompt length limits
                MAX_CONTENT_LENGTH = 10000
                if len(content) > MAX_CONTENT_LENGTH:
                    content = content[:MAX_CONTENT_LENGTH] + "... [truncated]"
            else:
                content = ""

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": content,
                }
            )

        return messages

    @staticmethod
    def _format_result_as_string(result: Any) -> str:
        """Convert result data to string representation."""
        if isinstance(result, dict):
            return str(result)
        elif isinstance(result, (list, tuple)):
            return str(result)
        else:
            return str(result) if result is not None else ""


class OpenAIStyleAdapter(BaseToolAdapter):
    """
    Base adapter for OpenAI-style APIs (OpenAI, Z.AI, etc.).

    This class implements the common pattern used by OpenAI-compatible APIs
    for function calling and tool messages.
    """

    @staticmethod
    def convert_tools(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tool definitions to OpenAI format."""
        if not tools:
            return []
        return [tool.to_openai_format() for tool in tools if tool.enabled]

    @staticmethod
    def extract_function_calls(response: Any) -> List[Dict[str, Any]]:
        """Extract function calls using OpenAI-style pattern."""
        return BaseToolAdapter._extract_openai_style_function_calls(response)

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format results as OpenAI-style tool messages."""
        return BaseToolAdapter._create_openai_style_tool_messages(results)

    @staticmethod
    def format_function_result(tool_name: str, result: Any, call_id: str) -> Dict[str, Any]:
        """Format a single function result for OpenAI-style APIs."""
        result_str = BaseToolAdapter._format_result_as_string(result)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result_str,
        }

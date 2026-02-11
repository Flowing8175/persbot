"""OpenAI tool format adapter.

This adapter converts between the bot's tool format and OpenAI's
function calling format (also used by Z.AI and other OpenAI-compatible APIs).
"""

import json
import logging
from typing import Any, Dict, List

from persbot.providers.adapters.base_adapter import BaseToolAdapter
from persbot.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class OpenAIToolAdapter(BaseToolAdapter):
    """Adapter for OpenAI-style function calling format.

    This format is used by OpenAI, Z.AI, and other OpenAI-compatible APIs.
    """

    @staticmethod
    def convert_tools(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tool definitions to OpenAI format.

        Args:
            tools: List of tool definitions to convert.

        Returns:
            List of tool dictionaries in OpenAI format.
        """
        converted = []

        for tool in tools:
            try:
                tool_dict = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters or {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
                converted.append(tool_dict)
            except Exception as e:
                logger.warning(f"Failed to convert tool {tool.name}: {e}")

        return converted

    @staticmethod
    def extract_function_calls(
        response: Any
    ) -> List[Dict[str, Any]]:
        """Extract function calls from an OpenAI-style response.

        Args:
            response: The OpenAI response object.

        Returns:
            List of function call dictionaries.
        """
        calls = []

        try:
            if hasattr(response, "choices"):
                for choice in response.choices:
                    if hasattr(choice, "message") and hasattr(choice.message, "tool_calls"):
                        for tool_call in choice.message.tool_calls or []:
                            call_dict = {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "parameters": json.loads(tool_call.function.arguments),
                            }
                            calls.append(call_dict)
        except Exception as e:
            logger.error(f"Error extracting function calls: {e}")

        return calls

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format function results for OpenAI.

        Args:
            results: List of tool execution results with 'id', 'name', 'result'.

        Returns:
            List of tool result messages in OpenAI format.
        """
        messages = []

        for result_item in results:
            call_id = result_item.get("id")
            tool_name = result_item.get("name")
            result_data = result_item.get("result")
            error = result_item.get("error")

            if error:
                content = f"Error: {error}"
            elif isinstance(result_data, dict):
                content = json.dumps(result_data, ensure_ascii=False)
            elif isinstance(result_data, bytes):
                # Handle binary data (Z.AI specific)
                import base64
                content = base64.b64encode(result_data).decode("utf-8")
            else:
                content = str(result_data) if result_data is not None else ""

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": content,
            })

        return messages

"""OpenAI tool format adapter."""

import logging
from typing import Any, Dict, List, Optional

from soyebot.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class OpenAIToolAdapter:
    """Adapter for converting tool definitions to/from OpenAI format."""

    @staticmethod
    def convert_tools(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert a list of tool definitions to OpenAI format.

        Args:
            tools: List of ToolDefinition objects.

        Returns:
            List of tool dictionaries in OpenAI function calling format.
        """
        if not tools:
            return []

        return [tool.to_openai_format() for tool in tools if tool.enabled]

    @staticmethod
    def extract_function_calls(response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from an OpenAI response.

        Args:
            response: OpenAI response object (from chat.completions.create).

        Returns:
            List of function call dictionaries with 'name' and 'parameters'.
        """
        function_calls = []

        try:
            if hasattr(response, 'choices') and response.choices:
                for choice in response.choices:
                    message = getattr(choice, 'message', None)
                    if message and hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            fc_data = {
                                'id': tool_call.id,
                                'name': tool_call.function.name,
                                'parameters': {},
                            }

                            # Parse arguments
                            import json
                            try:
                                args = json.loads(tool_call.function.arguments)
                                fc_data['parameters'] = args
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse tool arguments: %s", tool_call.function.arguments)

                            function_calls.append(fc_data)

        except Exception as e:
            logger.error("Error extracting function calls from OpenAI response: %s", e, exc_info=True)

        return function_calls

    @staticmethod
    def format_function_result(tool_name: str, result: Any, call_id: str) -> Dict[str, Any]:
        """Format a function result for sending back to OpenAI.

        Args:
            tool_name: Name of the tool that was executed.
            result: Result data from the tool execution.
            call_id: ID of the original tool call.

        Returns:
            Dictionary in OpenAI tool message format.
        """
        # Convert result to string representation
        if isinstance(result, dict):
            result_str = str(result)
        elif isinstance(result, (list, tuple)):
            result_str = str(result)
        else:
            result_str = str(result) if result is not None else ""

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result_str,
        }

    @staticmethod
    def create_tool_messages(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a list of tool messages from tool execution results.

        Args:
            results: List of dicts with 'id', 'name', 'result', and optionally 'error'.

        Returns:
            List of message dictionaries in OpenAI format.
        """
        messages = []

        for result_item in results:
            call_id = result_item.get('id')
            tool_name = result_item.get('name')
            result_data = result_item.get('result')
            error = result_item.get('error')

            if error:
                content = f"Error: {error}"
            else:
                content = str(result_data) if result_data is not None else ""

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": content,
            })

        return messages

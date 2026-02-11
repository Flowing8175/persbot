"""OpenAI tool format adapter."""

import logging
from typing import Any, Dict, List

from persbot.tools.adapters.base_adapter import OpenAIStyleAdapter

logger = logging.getLogger(__name__)


class OpenAIToolAdapter(OpenAIStyleAdapter):
    """Adapter for converting tool definitions to/from OpenAI format.

    Inherits common functionality from OpenAIStyleAdapter.
    """

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a list of tool messages from tool execution results.

        Args:
            results: List of dicts with 'id', 'name', 'result', and optionally 'error'.

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
            else:
                content = str(result_data) if result_data is not None else ""

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": content,
                }
            )

        return messages

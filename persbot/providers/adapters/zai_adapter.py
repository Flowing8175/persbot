"""Z.AI tool format adapter.

This adapter converts between the bot's tool format and Z.AI's
function calling format (OpenAI-compatible with binary data support).
"""

import logging
from typing import Any, Dict, List

from persbot.providers.adapters.openai_adapter import OpenAIToolAdapter

logger = logging.getLogger(__name__)


class ZAIToolAdapter(OpenAIToolAdapter):
    """Adapter for Z.AI (GLM) function calling format.

    Z.AI uses an OpenAI-compatible API with additional support for
    binary data in tool results (useful for image generation).
    """

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format function results for Z.AI with binary data support.

        Args:
            results: List of tool execution results.

        Returns:
            List of tool result messages in Z.AI format.
        """
        messages = []

        for result_item in results:
            call_id = result_item.get("id")
            tool_name = result_item.get("name")
            result_data = result_item.get("result")
            error = result_item.get("error")

            if error:
                content = f"Error: {error}"
            elif isinstance(result_data, bytes):
                # Z.AI supports binary data directly
                content = result_data
            elif isinstance(result_data, dict):
                import json

                content = json.dumps(result_data, ensure_ascii=False)
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

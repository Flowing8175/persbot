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

        Delegates to base class method with binary handling disabled.

        Args:
            results: List of dicts with 'id', 'name', 'result', and optionally 'error'.

        Returns:
            List of message dictionaries in OpenAI format.
        """
        return OpenAIStyleAdapter._create_openai_style_tool_messages(
            results, handle_binary_data=False
        )

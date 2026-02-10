"""Z.AI tool format adapter."""

import logging
from typing import Any, Dict, List

from persbot.tools.adapters.base_adapter import OpenAIStyleAdapter
from persbot.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class ZAIToolAdapter(OpenAIStyleAdapter):
    """Adapter for converting tool definitions to/from Z.AI (GLM) format.

    Z.AI uses an OpenAI-compatible API with function calling support.
    Inherits common functionality from OpenAIStyleAdapter and adds
    Z.AI-specific handling like binary data support.
    """

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a list of tool messages from tool execution results.

        Args:
            results: List of dicts with 'id', 'name', 'result', and optionally 'error'.

        Returns:
            List of message dictionaries in Z.AI format.
        """
        # Use the base class method with binary data handling enabled
        return OpenAIStyleAdapter._create_openai_style_tool_messages(
            results, handle_binary_data=True
        )

    # create_tool_messages is an alias for format_results for backward compatibility
    create_tool_messages = format_results

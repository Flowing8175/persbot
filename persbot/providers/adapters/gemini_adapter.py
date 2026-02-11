"""Gemini tool format adapter.

This adapter converts between the bot's tool format and Google Gemini's
function calling format.
"""

import json
import logging
from typing import Any, Dict, List

from google.genai import types as genai_types

from persbot.providers.adapters.base_adapter import BaseToolAdapter
from persbot.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class GeminiToolAdapter(BaseToolAdapter):
    """Adapter for Gemini function calling format.

    Gemini uses a declarative function calling format with JSON schemas.
    """

    @staticmethod
    def convert_tools(tools: List[ToolDefinition]) -> List[genai_types.Tool]:
        """Convert tool definitions to Gemini format.

        Args:
            tools: List of tool definitions to convert.

        Returns:
            List of genai_types.Tool objects.
        """
        converted = []
        for tool in tools:
            try:
                function_decl = GeminiToolAdapter._create_function_declaration(tool)
                converted.append(genai_types.Tool(function_declarations=function_decl))
            except Exception as e:
                logger.warning(f"Failed to convert tool {tool.name}: {e}")

        return converted

    @staticmethod
    def _create_function_declaration(
        tool: ToolDefinition
    ) -> genai_types.FunctionDeclaration:
        """Create a FunctionDeclaration from a ToolDefinition.

        Args:
            tool: The tool definition.

        Returns:
            A FunctionDeclaration object.
        """
        # Use the tool's built-in conversion method to get proper FunctionDeclaration
        return tool.to_gemini_format()

    @staticmethod
    def extract_function_calls(
        response: Any
    ) -> List[Dict[str, Any]]:
        """Extract function calls from a Gemini response.

        Args:
            response: The Gemini response object.

        Returns:
            List of function call dictionaries.
        """
        calls = []

        try:
            if hasattr(response, "candidates"):
                for candidate in response.candidates:
                    if hasattr(candidate, "content"):
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                call_dict = {
                                    "name": fc.name,
                                    "parameters": dict(fc.args) if hasattr(fc, "args") else {},
                                }
                                calls.append(call_dict)
        except Exception as e:
            logger.error(f"Error extracting function calls: {e}")

        return calls

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> List[genai_types.Part]:
        """Format function results for Gemini.

        Args:
            results: List of tool execution results.

        Returns:
            List of genai_types.Part objects with function responses.
        """
        parts = []

        for result_item in results:
            tool_name = result_item.get("name")
            result_data = result_item.get("result")
            error = result_item.get("error")

            if error:
                response_content = f"Error: {error}"
            else:
                # Handle different result types
                if isinstance(result_data, dict):
                    response_content = json.dumps(result_data, ensure_ascii=False)
                elif result_data is not None:
                    response_content = str(result_data)
                else:
                    response_content = ""

            parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=tool_name,
                        response={"result": response_content}
                    )
                )
            )

        return parts

"""Gemini tool format adapter."""

import logging
from typing import Any, Dict, List

from google.genai import types as genai_types

from persbot.tools.adapters.base_adapter import BaseToolAdapter
from persbot.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class GeminiToolAdapter(BaseToolAdapter):
    """Adapter for converting tool definitions to Gemini format."""

    @staticmethod
    def convert_tools(tools: List[ToolDefinition]) -> List[genai_types.Tool]:
        """Convert a list of tool definitions to Gemini format.

        Args:
            tools: List of ToolDefinition objects.

        Returns:
            List of genai_types.Tool objects.
        """
        if not tools:
            return []

        # Group by category for better organization
        function_declarations = [tool.to_gemini_format() for tool in tools if tool.enabled]

        if not function_declarations:
            return []

        return [genai_types.Tool(function_declarations=function_declarations)]

    @staticmethod
    def extract_function_calls(response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from a Gemini response.

        Args:
            response: Gemini response object.

        Returns:
            List of function call dictionaries with 'name' and 'parameters'.
        """
        function_calls = []

        try:
            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            # Check for function_call attribute
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                call_data = {
                                    "name": fc.name,
                                    "parameters": dict(fc.args) if hasattr(fc, "args") else {},
                                }
                                function_calls.append(call_data)
                            # Also check for functionResponses (results from previous calls)
                            elif hasattr(part, "function_response") and part.function_response:
                                # This is a response, not a call - skip
                                pass

        except Exception as e:
            logger.error(
                "Error extracting function calls from Gemini response: %s", e, exc_info=True
            )

        return function_calls

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> List[genai_types.Part]:
        """Create a list of function response parts from tool execution results.

        Args:
            results: List of dicts with 'name', 'result', and optionally 'error'.

        Returns:
            List of genai_types.Part objects.
        """
        parts = []

        for result_item in results:
            tool_name = result_item.get("name")
            result_data = result_item.get("result")
            error = result_item.get("error")

            if error:
                response_data = {"error": error}
            else:
                response_data = {"result": str(result_data) if result_data is not None else ""}

            parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=tool_name, response=response_data
                    )
                )
            )

        return parts

    @staticmethod
    def format_function_result(tool_name: str, result: Any) -> genai_types.Part:
        """Format a function result for sending back to Gemini.

        Args:
            tool_name: Name of the tool that was executed.
            result: Result data from the tool execution.

        Returns:
            genai_types.Part containing the function response.
        """
        result_str = BaseToolAdapter._format_result_as_string(result)
        return genai_types.Part(
            function_response=genai_types.FunctionResponse(
                name=tool_name, response={"result": result_str}
            )
        )

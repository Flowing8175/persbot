"""Provider adapters for tool format conversion.

This module contains adapters that convert between the bot's internal
tool format and the format required by specific LLM providers.
"""

from .base_adapter import (
    BaseToolAdapter,
    FunctionCall,
    FunctionResult,
    ToolAdapterRegistry,
    get_tool_adapter,
)

__all__ = [
    "BaseToolAdapter",
    "FunctionCall",
    "FunctionResult",
    "ToolAdapterRegistry",
    "get_tool_adapter",
]

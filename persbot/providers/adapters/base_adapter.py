"""Base adapter for tool conversion.

This module provides the base interface and common logic for converting
between the bot's tool format and provider-specific formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from persbot.tools.base import ToolDefinition


@dataclass
class FunctionCall:
    """Represents a function call from an LLM."""

    name: str
    parameters: Dict[str, Any]
    id: Optional[str] = None  # For OpenAI/Z.AI format


@dataclass
class FunctionResult:
    """Represents the result of a function execution."""

    name: str
    result: Any
    error: Optional[str] = None
    id: Optional[str] = None  # For OpenAI/Z.AI format


class BaseToolAdapter(ABC):
    """Base class for tool adapters.

    Tool adapters convert between the bot's internal tool format
    and the format required by specific LLM providers.
    """

    @staticmethod
    @abstractmethod
    def convert_tools(tools: List[ToolDefinition]) -> Any:
        """Convert tool definitions to provider format.

        Args:
            tools: List of tool definitions to convert.

        Returns:
            Provider-specific tool format.
        """
        pass

    @staticmethod
    @abstractmethod
    def extract_function_calls(response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from a provider response.

        Args:
            response: The provider's response object.

        Returns:
            List of function call dictionaries with 'name' and 'parameters'.
        """
        pass

    @staticmethod
    @abstractmethod
    def format_results(results: List[Dict[str, Any]]) -> Any:
        """Format function results for the provider.

        Args:
            results: List of result dicts with 'name', 'result', and optionally 'error'.

        Returns:
            Provider-specific formatted results.
        """
        pass


class ToolAdapterRegistry:
    """Registry for tool adapters by provider type."""

    _adapters: Dict[str, BaseToolAdapter] = {}

    @classmethod
    def register(cls, provider: str, adapter: BaseToolAdapter) -> None:
        """Register an adapter for a provider.

        Args:
            provider: The provider name ('gemini', 'openai', 'zai').
            adapter: The adapter instance.
        """
        cls._adapters[provider.lower()] = adapter

    @classmethod
    def get(cls, provider: str) -> Optional[BaseToolAdapter]:
        """Get the adapter for a provider.

        Args:
            provider: The provider name.

        Returns:
            The adapter, or None if not found.
        """
        return cls._adapters.get(provider.lower())

    @classmethod
    def get_or_create(cls, provider: str) -> BaseToolAdapter:
        """Get existing adapter or create default.

        Args:
            provider: The provider name.

        Returns:
            An adapter instance.

        Raises:
            ValueError: If the provider is unknown.
        """
        adapter = cls.get(provider)
        if adapter:
            return adapter

        # Create default adapter based on provider
        if provider.lower() == "gemini":
            from persbot.providers.adapters.gemini_adapter import (
                GeminiToolAdapter,
            )

            adapter = GeminiToolAdapter()
        elif provider.lower() == "openai":
            from persbot.providers.adapters.openai_adapter import (
                OpenAIToolAdapter,
            )

            adapter = OpenAIToolAdapter()
        elif provider.lower() == "zai":
            from persbot.providers.adapters.zai_adapter import ZAIToolAdapter

            adapter = ZAIToolAdapter()
        else:
            raise ValueError(f"Unknown provider: {provider}")

        cls.register(provider, adapter)
        return adapter


def get_tool_adapter(provider: str) -> BaseToolAdapter:
    """Convenience function to get a tool adapter.

    Args:
        provider: The provider name.

    Returns:
        The adapter for the provider.
    """
    return ToolAdapterRegistry.get_or_create(provider)

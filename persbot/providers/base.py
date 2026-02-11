"""Base provider interface for LLM providers.

This module defines the abstract interface that all LLM providers
must implement, ensuring consistent behavior across different backends.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import discord

from persbot.config import AppConfig
from persbot.services.base import BaseLLMService
from persbot.services.prompt_service import PromptService


class BaseLLMProvider(ABC):
    """Abstract base class for LLM provider implementations.

    All LLM providers (Gemini, OpenAI, Z.AI, etc.) must inherit from
    this class and implement its methods. This ensures consistent
    behavior and makes it easier to add new providers.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        assistant_model_name: str,
        summary_model_name: Optional[str] = None,
        prompt_service: PromptService,
    ):
        """Initialize the provider.

        Args:
            config: Application configuration.
            assistant_model_name: Default model for chat.
            summary_model_name: Default model for summarization.
            prompt_service: Service for managing prompts.
        """
        self.config = config
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

    @abstractmethod
    def create_assistant_model(
        self, system_instruction: str, use_cache: bool = True
    ) -> Any:
        """Create a model for chat interactions.

        Args:
            system_instruction: The system prompt/instruction.
            use_cache: Whether to use context caching (if supported).

        Returns:
            A model instance that can be used for chat.
        """
        pass

    @abstractmethod
    def create_summary_model(self, system_instruction: str) -> Any:
        """Create a model for text summarization.

        Args:
            system_instruction: The system prompt for summarization.

        Returns:
            A model instance for summarization.
        """
        pass

    @abstractmethod
    async def generate_chat_response(
        self,
        chat_session: Any,
        user_message: str,
        discord_message: Union[discord.Message, List[discord.Message]],
        model_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        cancel_event: Optional[Callable] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Generate a chat response.

        Args:
            chat_session: The chat session to use.
            user_message: The user's message.
            discord_message: The Discord message(s) for context.
            model_name: Optional specific model to use.
            tools: Optional list of tools for function calling.
            cancel_event: Optional callable to check for cancellation.

        Returns:
            Tuple of (response_text, response_obj) or None if failed.
        """
        pass

    @abstractmethod
    async def send_tool_results(
        self,
        chat_session: Any,
        tool_rounds: List[Tuple[Any, List[Dict[str, Any]]]],
        tools: Optional[List[Any]] = None,
        discord_message: Optional[discord.Message] = None,
        cancel_event: Optional[Callable] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Send tool results back to the model and get continuation.

        Args:
            chat_session: The chat session.
            tool_rounds: List of (response_obj, tool_results) tuples.
            tools: Optional tool definitions.
            discord_message: Optional Discord message for errors.
            cancel_event: Optional cancellation check.

        Returns:
            Tuple of (response_text, response_obj) or None.
        """
        pass

    @abstractmethod
    def get_user_role_name(self) -> str:
        """Get the role name for user messages.

        Returns:
            The role name (e.g., 'user', 'human').
        """
        pass

    @abstractmethod
    def get_assistant_role_name(self) -> str:
        """Get the role name for assistant messages.

        Returns:
            The role name (e.g., 'assistant', 'model').
        """
        pass

    @abstractmethod
    def get_tools_for_provider(self, tools: List[Any]) -> Any:
        """Convert tool definitions to provider-specific format.

        Args:
            tools: List of tool definitions.

        Returns:
            Provider-specific tool format.
        """
        pass

    @abstractmethod
    def extract_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from a response.

        Args:
            response: The provider's response object.

        Returns:
            List of function call dictionaries.
        """
        pass

    @abstractmethod
    def format_function_results(self, results: List[Dict[str, Any]]) -> Any:
        """Format function results for the provider.

        Args:
            results: List of tool execution results.

        Returns:
            Provider-specific formatted results.
        """
        pass

    # Optional methods with default implementations

    def reload_parameters(self) -> None:
        """Reload model parameters from config.

        Default implementation does nothing. Override if the provider
        needs to reload parameters dynamically.
        """
        pass

    async def summarize_text(self, text: str) -> Optional[str]:
        """Summarize the given text.

        Default implementation uses the summary model. Override if
        the provider has specialized summarization capabilities.

        Args:
            text: The text to summarize.

        Returns:
            The summarized text, or None if failed.
        """
        if not text.strip():
            return "요약할 메시지가 없습니다."

        summary_model = self.create_summary_model(
            self.prompt_service.get_summary_prompt()
        )
        # This is a simplified version - providers may override
        return await self._generate_summary(summary_model, text)

    async def _generate_summary(
        self, summary_model: Any, text: str
    ) -> Optional[str]:
        """Generate summary using the summary model.

        Args:
            summary_model: The summary model instance.
            text: The text to summarize.

        Returns:
            The summarized text, or None if failed.
        """
        # Default implementation - providers should override
        raise NotImplementedError(
            "Provider must implement _generate_summary or summarize_text"
        )

    # Stream support (optional)

    async def generate_chat_response_stream(
        self,
        chat_session: Any,
        user_message: str,
        discord_message: Union[discord.Message, List[discord.Message]],
        model_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        cancel_event: Optional[Callable] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response.

        Args:
            chat_session: The chat session.
            user_message: The user's message.
            discord_message: The Discord message(s).
            model_name: Optional specific model.
            tools: Optional list of tools.
            cancel_event: Optional cancellation check.

        Yields:
            Response text chunks as they are generated.

        Note:
            Default implementation falls back to non-streaming.
            Providers should override for true streaming support.
        """
        # Default: fall back to non-streaming
        result = await self.generate_chat_response(
            chat_session,
            user_message,
            discord_message,
            model_name,
            tools,
            cancel_event,
        )
        if result:
            yield result[0]

    @property
    def assistant_model_name(self) -> str:
        """Get the default assistant model name."""
        return self._assistant_model_name

    @property
    def summary_model_name(self) -> str:
        """Get the default summary model name."""
        return self._summary_model_name


class ProviderCapabilities:
    """Describes the capabilities of a provider."""

    def __init__(
        self,
        supports_streaming: bool = False,
        supports_function_calling: bool = False,
        supports_vision: bool = False,
        supports_context_cache: bool = False,
        supports_thinking: bool = False,
        max_tokens: Optional[int] = None,
        max_image_count: int = 0,
    ):
        """Initialize provider capabilities.

        Args:
            supports_streaming: Whether provider supports streaming responses.
            supports_function_calling: Whether provider supports function calling.
            supports_vision: Whether provider supports vision/image inputs.
            supports_context_cache: Whether provider supports context caching.
            supports_thinking: Whether provider supports thinking mode.
            max_tokens: Maximum tokens for context.
            max_image_count: Maximum images per request.
        """
        self.supports_streaming = supports_streaming
        self.supports_function_calling = supports_function_calling
        self.supports_vision = supports_vision
        self.supports_context_cache = supports_context_cache
        self.supports_thinking = supports_thinking
        self.max_tokens = max_tokens
        self.max_image_count = max_image_count


# Provider capability descriptions
class ProviderCaps:
    """Predefined capability sets for common providers."""

    GEMINI = ProviderCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_context_cache=True,
        supports_thinking=True,
        max_tokens=1048576,
        max_image_count=16,
    )

    OPENAI = ProviderCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_context_cache=False,
        supports_thinking=False,
        max_tokens=128000,
        max_image_count=10,
    )

    ZAI = ProviderCapabilities(
        supports_streaming=False,
        supports_function_calling=True,
        supports_vision=True,
        supports_context_cache=False,
        supports_thinking=False,
        max_tokens=128000,
        max_image_count=1,
    )

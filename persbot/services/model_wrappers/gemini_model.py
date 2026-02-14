"""Gemini cached model wrapper for managing model instances."""

import logging
from typing import Any, AsyncIterator, List, Optional, Union

import google.genai as genai
from google.genai import types as genai_types

from persbot.services.session_wrappers.gemini_session import GeminiChatSession

logger = logging.getLogger(__name__)


class GeminiCachedModel:
    """
    Lightweight wrapper that mimics the GenerativeModel interface.

    This wrapper provides a consistent interface for model interactions
    while supporting cached content and configuration management.
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        config: genai_types.GenerateContentConfig,
    ):
        """
        Initialize the cached model wrapper.

        Args:
            client: The Gemini client instance.
            model_name: The name of the model to use.
            config: The generation content configuration.
        """
        self._client = client
        self._model_name = model_name
        self._config = config

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    def generate_content(
        self, contents: Union[str, List[Any]], tools: Optional[List[Any]] = None
    ) -> Any:
        """
        Generate content with optional tools override.

        Args:
            contents: The content to generate.
            tools: Optional override for tools configuration.
                   Note: Cannot override tools when using cached_content.

        Returns:
            The API response.
        """
        if tools is not None:
            config = self._build_config_with_tools(tools)
        else:
            config = self._config

        return self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )

    async def generate_content_stream(
        self, contents: Union[str, List[Any]], tools: Optional[List[Any]] = None
    ) -> AsyncIterator[Any]:
        """
        Generate content with async streaming.

        Args:
            contents: The content to generate.
            tools: Optional override for tools configuration.
                   Note: Cannot override tools when using cached_content.

        Yields:
            Response chunks as they arrive from the API.
        """
        if tools is not None:
            config = self._build_config_with_tools(tools)
        else:
            config = self._config

        async for chunk in self._client.aio.models.generate_content_stream(
            model=self._model_name,
            contents=contents,
            config=config,
        ):
            yield chunk

    def _build_config_with_tools(self, tools: List[Any]) -> genai_types.GenerateContentConfig:
        """Build config with tools override, handling cache correctly."""
        # Check if we're using cached_content - if so, cannot override tools
        has_cached_content = getattr(self._config, "cached_content", None) is not None

        if has_cached_content:
            # When using cached_content, tools are already baked in
            # Rebuild config without tools to avoid API error
            config_kwargs = {
                "temperature": getattr(self._config, "temperature", 1.0),
                "top_p": getattr(self._config, "top_p", 1.0),
                "cached_content": self._config.cached_content,
            }
            # Add thinking config if present
            if hasattr(self._config, "thinking_config") and self._config.thinking_config:
                config_kwargs["thinking_config"] = self._config.thinking_config

            logger.warning(
                "Ignoring tools override when using cached_content. Tools are already in the cache."
            )
        else:
            # Not using cache, can override tools normally
            config_kwargs = {
                "temperature": getattr(self._config, "temperature", 1.0),
                "top_p": getattr(self._config, "top_p", 1.0),
                "system_instruction": getattr(self._config, "system_instruction", None),
                "tools": tools,
            }
            # Add thinking config if present
            if hasattr(self._config, "thinking_config") and self._config.thinking_config:
                config_kwargs["thinking_config"] = self._config.thinking_config

        return genai_types.GenerateContentConfig(**config_kwargs)

    def start_chat(self, system_instruction: str) -> GeminiChatSession:
        """
        Start a new chat session.

        Args:
            system_instruction: The system instruction for the chat.

        Returns:
            A new GeminiChatSession instance.
        """
        return GeminiChatSession(system_instruction, self)

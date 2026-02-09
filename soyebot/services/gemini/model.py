"""Model and text extraction utilities for Gemini service."""

import logging
from typing import Any, Optional, Tuple, Union

import google.genai as genai
from google.genai import types as genai_types

from .constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_P

logger = logging.getLogger(__name__)


def extract_clean_text(response_obj: Any) -> str:
    """Extract text content from Gemini response, filtering out thoughts."""
    try:
        text_parts = []
        if hasattr(response_obj, "candidates") and response_obj.candidates:
            for candidate in response_obj.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        # Skip parts that are marked as thoughts
                        if getattr(part, "thought", False):
                            continue

                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)

        if text_parts:
            return " ".join(text_parts).strip()

        return ""

    except Exception as e:
        logger.error(f"Failed to extract text from response: {e}", exc_info=True)
        return ""


class _CachedModel:
    """Lightweight wrapper that mimics the old GenerativeModel interface."""

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        config: genai_types.GenerateContentConfig,
    ):
        self._client = client
        self._model_name = model_name
        self._config = config

    def generate_content(self, contents: Union[str, list], tools: Optional[list] = None):
        """Generate content with optional tools override.

        Args:
            contents: The content to generate.
            tools: Optional override for tools configuration.
                   Note: Cannot override tools when using cached_content.

        Returns:
            The API response.
        """
        if tools is not None:
            # Check if we're using cached_content - if so, cannot override tools
            has_cached_content = getattr(self._config, "cached_content", None) is not None

            if has_cached_content:
                # When using cached_content, tools are already baked in
                # Rebuild config without tools to avoid API error
                config_kwargs = {
                    "temperature": getattr(self._config, "temperature", DEFAULT_TEMPERATURE),
                    "top_p": getattr(self._config, "top_p", DEFAULT_TOP_P),
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
                    "temperature": getattr(self._config, "temperature", DEFAULT_TEMPERATURE),
                    "top_p": getattr(self._config, "top_p", DEFAULT_TOP_P),
                    "system_instruction": getattr(self._config, "system_instruction", None),
                    "tools": tools,
                }
                # Add thinking config if present
                if hasattr(self._config, "thinking_config") and self._config.thinking_config:
                    config_kwargs["thinking_config"] = self._config.thinking_config

            config = genai_types.GenerateContentConfig(**config_kwargs)
        else:
            config = self._config

        return self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )

    def start_chat(self, system_instruction: str):
        from .session import _ChatSession

        # No underlying chat needed
        return _ChatSession(system_instruction, self)

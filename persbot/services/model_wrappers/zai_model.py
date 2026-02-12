"""Z.AI model wrapper for managing model instances."""

import logging
from typing import Any, Optional

from openai import OpenAI

from persbot.services.session_wrappers.zai_session import ZAIChatSession

logger = logging.getLogger(__name__)


class ZAIChatModel:
    """
    Wrapper for Z.AI chat model.

    Provides methods to create chat sessions with history management.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int = 50,
        text_extractor: Optional[Any] = None,
    ):
        """
        Initialize the Z.AI model wrapper.

        Args:
            client: The OpenAI-compatible client instance (configured for Z.AI).
            model_name: The name of the model to use.
            system_instruction: The system instruction for the chat.
            temperature: Temperature for response generation.
            top_p: Top-p for response generation.
            max_messages: Maximum number of messages to keep in history.
            text_extractor: Optional callback to extract text from responses.
        """
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._top_p = top_p
        self._max_messages = max_messages
        self._text_extractor = text_extractor

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    def start_chat(self, system_instruction: Optional[str] = None) -> ZAIChatSession:
        """Start a new chat session.

        Args:
            system_instruction: Optional override for system instruction.

        Returns:
            A new ZAIChatSession instance.
        """
        return ZAIChatSession(
            client=self._client,
            model_name=self._model_name,
            system_instruction=system_instruction or self._system_instruction,
            temperature=self._temperature,
            top_p=self._top_p,
            max_messages=self._max_messages,
            text_extractor=self._text_extractor,
        )

"""OpenAI model wrapper for managing model instances."""

import logging
from typing import Any, List, Optional

from openai import OpenAI

from persbot.services.session_wrappers.openai_session import (
    BaseOpenAISession,
    ChatCompletionSession,
    ResponseSession,
)

logger = logging.getLogger(__name__)


class OpenAIChatCompletionModel:
    """
    Wrapper for OpenAI chat completion model.

    Provides methods to create different types of sessions
    (chat completion for standard use, responses for low latency).
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int = 50,
        service_tier: Optional[str] = None,
    ):
        """
        Initialize the OpenAI model wrapper.

        Args:
            client: The OpenAI client instance.
            model_name: The name of the model to use.
            system_instruction: The system instruction for the chat.
            temperature: Temperature for response generation.
            top_p: Top-p for response generation.
            max_messages: Maximum number of messages to keep in history.
            service_tier: Optional service tier for specialized behavior.
        """
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._top_p = top_p
        self._max_messages = max_messages
        self._service_tier = service_tier

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    def create_chat_completion_session(self) -> ChatCompletionSession:
        """Create a standard chat completion session with history."""
        return ChatCompletionSession(
            client=self._client,
            model_name=self._model_name,
            system_instruction=self._system_instruction,
            temperature=self._temperature,
            top_p=self._top_p,
            max_messages=self._max_messages,
            service_tier=self._service_tier,
        )

    def create_response_session(self) -> ResponseSession:
        """Create a low-latency responses session (no history)."""
        return ResponseSession(
            client=self._client,
            model_name=self._model_name,
            system_instruction=self._system_instruction,
            temperature=self._temperature,
            top_p=self._top_p,
            max_messages=0,  # Responses API doesn't maintain history
            service_tier=self._service_tier,
        )

    def start_chat(self, use_responses_api: bool = False) -> BaseOpenAISession:
        """
        Start a new chat session.

        Args:
            use_responses_api: If True, use the low-latency responses API.
                              If False, use the standard chat completion API.

        Returns:
            A new session instance.
        """
        if use_responses_api:
            return self.create_response_session()
        return self.create_chat_completion_session()

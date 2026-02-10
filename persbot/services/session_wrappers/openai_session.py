"""OpenAI chat session wrapper for managing chat with history tracking."""

import asyncio
import base64
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Union

import discord
from openai import OpenAI

from persbot.services.base import ChatMessage
from persbot.utils import get_mime_type

logger = logging.getLogger(__name__)


def encode_image_to_url(img_bytes: bytes) -> Dict[str, str]:
    """Convert image bytes to OpenAI image_url format."""
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    mime_type = get_mime_type(img_bytes)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64_str}"},
    }


@dataclass
class OpenAIMessage:
    """A single message in OpenAI format."""

    role: str
    content: Optional[str]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Any]] = None


class BaseOpenAISession:
    """Base class for OpenAI chat sessions."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int,
        service_tier: Optional[str] = None,
    ):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._top_p = top_p
        self._max_messages = max_messages
        self._service_tier = service_tier
        self._history: Deque[ChatMessage] = deque(maxlen=max_messages)

    @property
    def history(self) -> List[ChatMessage]:
        """Get list of chat messages in history."""
        return list(self._history)

    @history.setter
    def history(self, new_history: List[ChatMessage]]) -> None:
        """Replace the history with a new list."""
        self._history.clear()
        self._history.extend(new_history)

    def _append_history(
        self,
        role: str,
        content: str,
        author_id: Optional[int] = None,
        author_name: Optional[str] = None,
        message_ids: List[str] = None,
    ) -> None:
        """Append a message to history if content is not empty."""
        if not content:
            return
        self._history.append(
            ChatMessage(
                role=role,
                content=content,
                author_id=author_id,
                author_name=author_name,
                message_ids=message_ids or [],
            )
        )

    def _format_messages(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
    ) -> List[Dict[str, Any]]:
        """Format messages for the OpenAI API call."""
        messages = []

        # Add system instruction first if available
        if self._system_instruction:
            messages.append({
                "role": "system",
                "content": self._system_instruction,
            })

        # Add history (excluding system messages if we just added it)
        for msg in self._history:
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.author_name and msg.role == "user":
                # Add author name to user messages for context
                msg_dict["name"] = msg.author_name
            messages.append(msg_dict)

        # Add current user message
        user_content = [{"type": "text", "text": user_message}]
        if images:
            for img_data in images:
                user_content.append(encode_image_to_url(img_data))

        user_msg_dict = {"role": "user", "content": user_content}
        if author_name:
            user_msg_dict["name"] = author_name

        messages.append(user_msg_dict)

        return messages

    async def send_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
        tools: Optional[List[Any]] = None,
    ) -> Tuple[ChatMessage, ChatMessage, Any]:
        """Send a message to the model and get the response."""
        messages = self._format_messages(
            user_message, author_id, author_name, message_ids, images
        )

        # Make API call
        response = await self._make_api_call(messages, tools)

        # Extract response content
        assistant_message = self._extract_response_content(response)

        # Create ChatMessage objects
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            author_id=author_id,
            author_name=author_name,
            message_ids=message_ids or [],
            images=images or [],
        )

        model_msg = ChatMessage(
            role="assistant",
            content=assistant_message,
            author_id=None,
        )

        return user_msg, model_msg, response

    async def _make_api_call(self, messages: List[Dict[str, Any]], tools: Optional[List[Any]]):
        """Make the actual API call - to be implemented by subclasses."""
        raise NotImplementedError

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from response - to be implemented by subclasses."""
        raise NotImplementedError


class ChatCompletionSession(BaseOpenAISession):
    """Session for standard chat completion API with history management."""

    async def _make_api_call(self, messages: List[Dict[str, Any]], tools: Optional[List[Any]]):
        """Make chat completion API call."""
        kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

        if tools:
            kwargs["tools"] = tools

        return await self._client.chat.completions.create(**kwargs)

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from chat completion response."""
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content or ""
        return ""


class ResponseSession(BaseOpenAISession):
    """Session for responses API (lower latency, no history)."""

    async def _make_api_call(self, messages: List[Dict[str, Any]], tools: Optional[List[Any]]):
        """Make responses API call (low latency)."""
        messages_for_response = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # Responses API doesn't support system messages
            messages_for_response.append(msg)

        kwargs = {
            "model": self._model_name,
            "messages": messages_for_response,
        }

        return await self._client.responses.create(**kwargs)

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from responses API response."""
        return response.output_text or ""

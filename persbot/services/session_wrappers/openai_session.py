"""OpenAI chat session wrapper for managing chat with history tracking."""

import base64
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterator, List, Optional, Tuple, Callable

from openai import OpenAI

if TYPE_CHECKING:
    from openai import Stream
    from openai.types.chat import ChatCompletionChunk

from persbot.services.base import ChatMessage
from persbot.utils import get_mime_type

logger = logging.getLogger(__name__)


# --- Stream adapter for Responses API -> Chat Completions format ---
@dataclass
class FakeDelta:
    """Mimics Chat Completions delta object."""
    content: Optional[str] = None


@dataclass
class FakeChoice:
    """Mimics Chat Completions choice object."""
    delta: FakeDelta = field(default_factory=FakeDelta)


@dataclass
class FakeChunk:
    """Mimics Chat Completions chunk object for Responses API stream."""
    choices: List[FakeChoice] = field(default_factory=list)


class ResponsesAPIStreamAdapter:
    """Adapts Responses API stream to Chat Completions API format.

    The Responses API stream yields different event types:
    - response.created, response.output_item.added, etc.
    - response.output_text.delta contains the actual text content

    This adapter extracts text deltas and wraps them in FakeChunk objects
    that have the same .choices[0].delta.content structure.
    """

    def __init__(self, raw_stream: Iterator[Any]):
        self._raw_stream = raw_stream
        self._buffer: List[FakeChunk] = []

    def __iter__(self) -> Iterator[FakeChunk]:
        for event in self._raw_stream:
            # Responses API uses event types
            if hasattr(event, 'type'):
                event_type = event.type

                # Extract text from output_text.delta events
                if event_type == 'response.output_text.delta':
                    if hasattr(event, 'delta') and event.delta:
                        chunk = FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=event.delta))])
                        yield chunk

                # Also check for content in the event data
                elif hasattr(event, 'data'):
                    data = event.data
                    if isinstance(data, dict):
                        # Some events have delta in data
                        delta_text = data.get('delta', {}).get('text', '')
                        if delta_text:
                            chunk = FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=delta_text))])
                            yield chunk
            else:
                # Fallback: try to extract content directly for unknown formats
                # This handles cases where the SDK might normalize the response
                if hasattr(event, 'choices') and event.choices:
                    yield event
                elif hasattr(event, 'content'):
                    text = event.content if isinstance(event.content, str) else str(event.content)
                    if text:
                        chunk = FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=text))])
                        yield chunk

    def close(self):
        """Close the underlying stream if it has a close method."""
        if hasattr(self._raw_stream, 'close'):
            self._raw_stream.close()


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

    # Models that don't support top_p parameter
    _MODELS_WITHOUT_TOP_P = frozenset(["gpt-5", "o1", "o3", "o4"])

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int,
        service_tier: Optional[str] = None,
        text_extractor: Optional[Callable[[Any], str]] = None,
    ):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._top_p = top_p
        self._max_messages = max_messages
        self._service_tier = service_tier
        self._text_extractor = text_extractor or (lambda x: "")
        self._history: Deque[ChatMessage] = deque(maxlen=max_messages)

    def _supports_top_p(self) -> bool:
        """Check if the current model supports top_p parameter."""
        model_lower = self._model_name.lower()
        return not any(model_lower.startswith(prefix) for prefix in self._MODELS_WITHOUT_TOP_P)

    @property
    def history(self) -> List[ChatMessage]:
        """Get list of chat messages in history."""
        return list(self._history)

    @history.setter
    def history(self, new_history: List[ChatMessage]) -> None:
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

    def _create_user_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
    ) -> ChatMessage:
        """Create a ChatMessage for user input."""
        return ChatMessage(
            role="user",
            content=user_message,
            author_id=author_id,
            author_name=author_name,
            message_ids=message_ids or [],
            images=images or [],
        )


class ChatCompletionSession(BaseOpenAISession):
    """Chat Completion API-backed chat session for fine-tuned models."""

    def send_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
        tools: Optional[Any] = None,
    ) -> Tuple[ChatMessage, ChatMessage, Any]:
        """Send a message and get response."""
        user_msg = self._create_user_message(
            user_message, author_id, author_name, message_ids, images
        )

        # Build messages list
        messages = self._build_system_message()
        messages.extend(self._convert_history_to_api_format())
        messages.append(self._build_user_content(user_message, images))

        api_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "service_tier": self._service_tier,
        }
        if self._supports_top_p():
            api_kwargs["top_p"] = self._top_p

        if tools:
            api_kwargs["tools"] = tools

        response = self._client.chat.completions.create(**api_kwargs)

        message_content = self._extract_response_content(response)
        model_msg = ChatMessage(role="assistant", content=message_content)

        return user_msg, model_msg, response

    def send_message_stream(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
        tools: Optional[Any] = None,
    ) -> Tuple["Stream[ChatCompletionChunk]", ChatMessage]:
        """Send a message and get a streaming response.

        Returns a stream that yields ChatCompletionChunk objects.
        The caller is responsible for iterating and collecting the response.
        """
        user_msg = self._create_user_message(
            user_message, author_id, author_name, message_ids, images
        )

        # Build messages list
        messages = self._build_system_message()
        messages.extend(self._convert_history_to_api_format())
        messages.append(self._build_user_content(user_message, images))

        api_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "service_tier": self._service_tier,
            "stream": True,
        }
        if self._supports_top_p():
            api_kwargs["top_p"] = self._top_p

        if tools:
            api_kwargs["tools"] = tools

        return self._client.chat.completions.create(**api_kwargs), user_msg

    def _build_system_message(self) -> List[Dict[str, Any]]:
        """Build system message list."""
        if self._system_instruction:
            return [{"role": "system", "content": self._system_instruction}]
        return []

    def _convert_history_to_api_format(self) -> List[Dict[str, Any]]:
        """Convert chat history to OpenAI API format."""
        api_history = []
        for msg in self._history:
            if msg.images:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for img_bytes in msg.images:
                    content_blocks.append(encode_image_to_url(img_bytes))
                api_history.append({"role": msg.role, "content": content_blocks})
            else:
                api_history.append({"role": msg.role, "content": msg.content})
        return api_history

    def _build_user_content(
        self, user_message: str, images: Optional[List[bytes]]
    ) -> Dict[str, Any]:
        """Build user message content for API."""
        if images:
            content_list = []
            if user_message:
                content_list.append({"type": "text", "text": user_message})
            for img_bytes in images:
                content_list.append(encode_image_to_url(img_bytes))
            return {"role": "user", "content": content_list}
        return {"role": "user", "content": user_message}

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from response."""
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return self._text_extractor(response)

    def send_tool_results(
        self, tool_rounds: List[Tuple[Any, Any]], tools: Optional[Any] = None
    ) -> Tuple[ChatMessage, Any]:
        """Send tool execution results back to model and get continuation.

        Args:
            tool_rounds: List of (response_obj, tool_results) tuples from each round.
            tools: Tools for the next API call.

        Returns:
            Tuple of (model_msg, response_obj).
        """
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        # Build base messages from history
        messages = self._build_system_message()
        messages.extend(self._convert_history_to_api_format())

        # Remove last assistant entry (text-only from initial response)
        if messages and messages[-1].get("role") == "assistant":
            messages.pop()

        # Add each tool round: assistant response (with tool_calls) + tool results
        for resp_obj, results in tool_rounds:
            assistant_msg = resp_obj.choices[0].message
            tool_calls_data = []
            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    tool_calls_data.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": tool_calls_data,
                }
            )

            # Add tool result messages
            messages.extend(OpenAIToolAdapter.format_results(results))

        # Call API
        api_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "service_tier": self._service_tier,
        }
        if self._supports_top_p():
            api_kwargs["top_p"] = self._top_p
        if tools:
            api_kwargs["tools"] = tools

        response = self._client.chat.completions.create(**api_kwargs)

        message_content = self._extract_response_content(response)
        model_msg = ChatMessage(role="assistant", content=message_content)

        return model_msg, response


class ResponseSession(BaseOpenAISession):
    """Response API-backed chat session with a bounded context window."""

    def _build_input_payload(self) -> List[Dict[str, Any]]:
        """Build input payload for Responses API."""
        payload = []
        if self._system_instruction:
            payload.append(
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._system_instruction,
                        }
                    ],
                }
            )

        for entry in self._history:
            content_type = "output_text" if entry.role == "assistant" else "input_text"
            payload.append(
                {
                    "type": "message",
                    "role": entry.role,
                    "content": [
                        {
                            "type": content_type,
                            "text": entry.content,
                        }
                    ],
                }
            )
        return payload

    def send_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
        tools: Optional[Any] = None,
    ) -> Tuple[ChatMessage, ChatMessage, Any]:
        """Send a message and get response."""
        user_msg = self._create_user_message(
            user_message, author_id, author_name, message_ids, images
        )

        # Build current message content
        current_payload = self._build_input_payload()

        content_list = []
        if user_message:
            content_list.append({"type": "input_text", "text": user_message})

        if images:
            logger.warning(
                "Images provided to ResponseSession (OpenAI Responses API), but image support is not fully implemented for this endpoint. Ignoring images."
            )

        # Append user message to payload
        current_payload.append({"type": "message", "role": "user", "content": content_list})

        api_kwargs = {
            "model": self._model_name,
            "input": current_payload,
            "temperature": self._temperature,
            "service_tier": self._service_tier,
        }
        if self._supports_top_p():
            api_kwargs["top_p"] = self._top_p

        if tools:
            api_kwargs["tools"] = tools

        response = self._client.responses.create(**api_kwargs)

        message_content = self._text_extractor(response)
        model_msg = ChatMessage(role="assistant", content=message_content)

        return user_msg, model_msg, response

    def send_message_stream(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
        tools: Optional[Any] = None,
    ) -> Tuple[Any, ChatMessage]:
        """Send a message and get a streaming response.

        Note: The Responses API uses SSE streaming with different chunk format
        than Chat Completions API. Returns a stream and user message.
        """
        user_msg = self._create_user_message(
            user_message, author_id, author_name, message_ids, images
        )

        # Build current message content
        current_payload = self._build_input_payload()

        content_list = []
        if user_message:
            content_list.append({"type": "input_text", "text": user_message})

        if images:
            logger.warning(
                "Images provided to ResponseSession streaming, but image support is not fully implemented for this endpoint. Ignoring images."
            )

        # Append user message to payload
        current_payload.append({"type": "message", "role": "user", "content": content_list})

        api_kwargs = {
            "model": self._model_name,
            "input": current_payload,
            "temperature": self._temperature,
            "service_tier": self._service_tier,
            "stream": True,
        }
        if self._supports_top_p():
            api_kwargs["top_p"] = self._top_p

        if tools:
            api_kwargs["tools"] = tools

        raw_stream = self._client.responses.create(**api_kwargs)
        # Wrap the stream to produce Chat Completions-compatible chunks
        adapted_stream = ResponsesAPIStreamAdapter(raw_stream)
        return adapted_stream, user_msg

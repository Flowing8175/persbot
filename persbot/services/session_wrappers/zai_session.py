"""Z.AI chat session wrapper for managing chat with history tracking."""

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

from openai import OpenAI

from persbot.services.base import ChatMessage
from persbot.services.session_wrappers.openai_session import BaseOpenAISession
from persbot.utils import get_mime_type

logger = logging.getLogger(__name__)


def encode_image_to_zai_format(img_bytes: bytes) -> Dict[str, str]:
    """Convert image bytes to Z.AI image_url format."""
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    mime_type = get_mime_type(img_bytes)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64_str}"},
    }


class ZAIChatSession(BaseOpenAISession):
    """Z.AI chat session with history management.

    Z.AI uses an OpenAI-compatible API but with some differences:
    - No service_tier parameter
    - Supports both standard and coding plan APIs
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int,
        text_extractor: Optional[Callable[[Any], str]] = None,
    ):
        # Pass None for service_tier as Z.AI doesn't support it
        super().__init__(
            client=client,
            model_name=model_name,
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_messages=max_messages,
            service_tier=None,
            text_extractor=text_extractor,
        )

    def send_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
        tools: Optional[Any] = None,
    ) -> Tuple[ChatMessage, ChatMessage, Any]:
        """Send message to Z.AI API and get response."""
        user_msg = self._create_user_message(
            user_message, author_id, author_name, message_ids, images
        )

        # Build messages list
        messages = self._build_messages(user_message, images)

        # Call API
        api_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

        if tools:
            api_kwargs["tools"] = tools

        response = self._client.chat.completions.create(**api_kwargs)

        # Extract response content
        message_content = self._text_extractor(response)
        model_msg = ChatMessage(role="assistant", content=message_content)

        return user_msg, model_msg, response

    def _build_messages(
        self, user_message: str, images: Optional[List[bytes]]
    ) -> List[Dict[str, Any]]:
        """Build messages list for Z.AI API."""
        messages = []

        if self._system_instruction:
            messages.append({"role": "system", "content": self._system_instruction})

        for msg in self._history:
            if msg.images:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for img_bytes in msg.images:
                    content_blocks.append(encode_image_to_zai_format(img_bytes))
                messages.append({"role": msg.role, "content": content_blocks})
            else:
                messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        user_content = []
        if user_message:
            user_content.append({"type": "text", "text": user_message})

        if images:
            for img_bytes in images:
                user_content.append(encode_image_to_zai_format(img_bytes))

        if user_content:
            messages.append({"role": "user", "content": user_content})

        return messages

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
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        # Build messages
        messages = []
        if self._system_instruction:
            messages.append({"role": "system", "content": self._system_instruction})

        # Add history (will remove last assistant msg below)
        for msg in self._history:
            if msg.images:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for img_bytes in msg.images:
                    content_blocks.append(encode_image_to_zai_format(img_bytes))
                messages.append({"role": msg.role, "content": content_blocks})
            else:
                messages.append({"role": msg.role, "content": msg.content})

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
            messages.extend(ZAIToolAdapter.format_results(results))

        # Call API
        api_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }
        if tools:
            api_kwargs["tools"] = tools

        response = self._client.chat.completions.create(**api_kwargs)

        message_content = self._text_extractor(response)
        model_msg = ChatMessage(role="assistant", content=message_content)

        return model_msg, response

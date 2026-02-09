"""Chat session management for Gemini service."""

from collections import deque
from typing import Optional

import google.genai as genai
from google.genai import types as genai_types

from persbot.services.base import BaseLLMService, ChatMessage
from persbot.utils import get_mime_type

from .model import extract_clean_text


class _ChatSession:
    """A wrapper for a Gemini chat session to manage history with author tracking."""

    def __init__(self, system_instruction: str, factory: "_CachedModel"):
        self._system_instruction = system_instruction
        self._factory = factory
        # We will manage the history manually to include author_id
        # Use deque with maxlen for automatic memory management
        max_history = 50  # Default max history size
        self.history: deque[ChatMessage] = deque(maxlen=max_history)

    def _get_api_history(self) -> list[dict]:
        """Convert local history to API format."""
        api_history = []
        for msg in self.history:
            final_parts = []

            # Add existing text/content parts
            if msg.parts:
                for p in msg.parts:
                    if isinstance(p, dict) and "text" in p:
                        final_parts.append(p)
                    elif hasattr(p, "text") and p.text:  # It's a Part object
                        final_parts.append(p)

            # Reconstruct image parts from stored bytes
            if hasattr(msg, "images") and msg.images:
                for img_data in msg.images:
                    mime_type = get_mime_type(img_data)
                    final_parts.append(
                        genai_types.Part.from_bytes(data=img_data, mime_type=mime_type)
                    )

            api_history.append({"role": msg.role, "parts": final_parts})
        return api_history

    def send_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[list[str]] = None,
        images: list[bytes] = None,
        tools: Optional[list] = None,
    ):
        # 1. Build the full content list for this turn (History + Current Message)
        contents = self._get_api_history()

        current_parts = []
        if user_message:
            current_parts.append({"text": user_message})

        if images:
            for img_data in images:
                mime_type = get_mime_type(img_data)
                current_parts.append(
                    genai_types.Part.from_bytes(data=img_data, mime_type=mime_type)
                )

        contents.append({"role": "user", "parts": current_parts})

        # 2. Call generate_content directly (Stateless)
        # Pass tools to generate_content to enable function calling
        response = self._factory.generate_content(contents=contents, tools=tools)

        # 3. Create ChatMessage objects but do NOT append to self.history yet.
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            parts=[
                {"text": user_message}
            ],  # We store text part only in parts for compatibility/simplicity?
            # Or we should store the text part. Images are stored in 'images' field.
            images=images or [],
            author_id=author_id,
            author_name=author_name,
            message_ids=message_ids or [],
        )

        clean_content = extract_clean_text(response)
        model_msg = ChatMessage(
            role="model",
            content=clean_content,
            parts=[{"text": clean_content}],
            author_id=None,  # Bot messages have no author
        )

        # Return the new messages and the raw response
        return user_msg, model_msg, response

    def send_tool_results(self, tool_rounds, tools=None):
        """Send tool execution results back to model and get continuation.

        Args:
            tool_rounds: List of (response_obj, tool_results) tuples from each round.
            tools: Tools to pass for the next API call.

        Returns:
            Tuple of (model_msg, response_obj).
        """
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        # Build contents from history
        contents = self._get_api_history()

        # The last entry is the text-only model msg from initial response - remove it
        if contents and contents[-1]["role"] == "model":
            contents.pop()

        # Add each tool round: model response (with function_call) + function results
        for resp_obj, results in tool_rounds:
            # Add model's response with function_call parts
            model_content = resp_obj.candidates[0].content
            contents.append({"role": "model", "parts": list(model_content.parts)})

            # Add function response parts
            fn_parts = GeminiToolAdapter.create_function_response_parts(results)
            contents.append({"role": "user", "parts": fn_parts})

        # Call generate_content
        response = self._factory.generate_content(contents=contents, tools=tools)

        # Create model message
        clean_content = extract_clean_text(response)
        model_msg = ChatMessage(
            role="model",
            content=clean_content,
            parts=[{"text": clean_content}],
        )

        return model_msg, response

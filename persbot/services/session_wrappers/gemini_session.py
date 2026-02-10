"""Gemini chat session wrapper for managing history with author tracking."""

import logging
from collections import deque
from typing import Any, Dict, List, Optional

from google.genai import types as genai_types

from persbot.services.base import ChatMessage
from persbot.utils import get_mime_type

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
        logger.error("Failed to extract text from response: %s", e, exc_info=True)
        return ""


class GeminiChatSession:
    """A wrapper for a Gemini chat session to manage history with author tracking."""

    def __init__(self, system_instruction: str, factory: "_CachedModel", max_history: int = 50):
        """
        Initialize the chat session.

        Args:
            system_instruction: The system instruction for the chat.
            factory: The cached model factory for generating content.
            max_history: Maximum number of messages to keep in history.
        """
        self._system_instruction = system_instruction
        self._factory = factory
        self.history: deque[ChatMessage] = deque(maxlen=max_history)

    def _get_api_history(self) -> List[Dict[str, Any]]:
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
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
        tools: Optional[List[Any]] = None,
    ) -> tuple[ChatMessage, ChatMessage, Any]:
        """
        Send a message to the model and get the response.

        Args:
            user_message: The user's message content.
            author_id: The Discord user ID of the author.
            author_name: Optional Discord username of the author.
            message_ids: Optional list of Discord message IDs.
            images: Optional list of image bytes.
            tools: Optional list of tools for function calling.

        Returns:
            Tuple of (user_message, model_message, raw_response).
        """
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
        response = self._factory.generate_content(contents=contents, tools=tools)

        # 3. Create ChatMessage objects but do NOT append to self.history yet.
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            parts=[{"text": user_message}],
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

        return user_msg, model_msg, response

    def send_tool_results(
        self, tool_rounds: List[tuple[Any, List[Dict[str, Any]]]], tools: Optional[List[Any]] = None
    ) -> tuple[ChatMessage, Any]:
        """
        Send tool execution results back to model and get continuation.

        Args:
            tool_rounds: List of (response_obj, tool_results) tuples from each round.
            tools: Tools to pass for the next API call.

        Returns:
            Tuple of (model_message, response_obj).
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

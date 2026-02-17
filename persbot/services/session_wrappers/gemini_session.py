"""Gemini chat session wrapper for managing history with author tracking."""

import logging
from collections import deque
from typing import Any, AsyncIterator, Dict, List, Optional

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
        """Convert local history to API format.

        Returns parts as dicts (not Part objects) to ensure JSON serialization works.
        """
        import base64

        api_history = []
        for msg in self.history:
            final_parts = []

            # Add existing text/content parts as dicts
            if msg.parts:
                for p in msg.parts:
                    if isinstance(p, dict) and "text" in p:
                        final_parts.append(p)
                    elif hasattr(p, "text") and p.text:  # It's a Part object
                        # Convert Part to dict
                        final_parts.append({"text": p.text})

            # Reconstruct image parts from stored bytes as dicts
            if hasattr(msg, "images") and msg.images:
                for img_data in msg.images:
                    mime_type = get_mime_type(img_data)
                    # Use inline_data dict format instead of Part object
                    final_parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(img_data).decode("utf-8")
                        }
                    })

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

    async def send_message_stream(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        images: Optional[List[bytes]] = None,
        tools: Optional[List[Any]] = None,
    ) -> tuple[ChatMessage, AsyncIterator[Any]]:
        """
        Send a message and get an async streaming response.

        Args:
            user_message: The user's message content.
            author_id: The Discord user ID of the author.
            author_name: Optional Discord username of the author.
            message_ids: Optional list of Discord message IDs.
            images: Optional list of image bytes.
            tools: Optional list of tools for function calling.

        Returns:
            Tuple of (user_message, async_stream_iterator).
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

        # 2. Get async streaming iterator
        try:
            stream = self._factory.generate_content_stream(contents=contents, tools=tools)
        except Exception as e:
            logger.error("generate_content_stream failed: %s", e, exc_info=True)
            raise

        # 3. Create user ChatMessage (model message created after stream completes)
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            parts=[{"text": user_message}],
            images=images or [],
            author_id=author_id,
            author_name=author_name,
            message_ids=message_ids or [],
        )

        return user_msg, stream

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
        # Format: (resp_obj, results, function_calls) - function_calls may be provided for streaming
        for tool_round in tool_rounds:
            # Unpack with optional third element for function_calls
            if len(tool_round) == 3:
                resp_obj, results, function_calls = tool_round
            else:
                resp_obj, results = tool_round
                function_calls = None

            # Build model parts from either response object or function_calls
            model_parts = None
            if resp_obj is not None and resp_obj.candidates:
                model_content = resp_obj.candidates[0].content
                # Convert Part objects to dicts for JSON serialization
                model_parts = []
                for part in model_content.parts:
                    if hasattr(part, "text") and part.text:
                        model_parts.append({"text": part.text})
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        model_parts.append({
                            "function_call": {
                                "name": getattr(fc, "name", None),
                                "args": dict(getattr(fc, "args", {}) or {})
                            }
                        })
                    elif hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        model_parts.append({
                            "function_response": {
                                "name": getattr(fr, "name", None),
                                "response": dict(getattr(fr, "response", {}) or {})
                            }
                        })
            elif function_calls:
                # Streaming case: construct parts from extracted function_calls as dicts
                model_parts = []
                for fc in function_calls:
                    # Build function_call part as dict (JSON-serializable)
                    model_parts.append({
                        "function_call": {
                            "name": fc.get("name"),
                            "args": fc.get("parameters") or fc.get("args") or {}
                        }
                    })

            if model_parts is None:
                logger.error("send_tool_results: no response object and no function_calls provided")
                continue

            contents.append({"role": "model", "parts": model_parts})

            # Add function response parts as dicts (not Part objects)
            for result_item in results:
                tool_name = result_item.get("name")
                result_data = result_item.get("result")
                error = result_item.get("error")
                if error:
                    response_data = {"error": error}
                else:
                    response_data = {"result": str(result_data) if result_data is not None else ""}
                fn_part = {"function_response": {"name": tool_name, "response": response_data}}
                contents.append({"role": "user", "parts": [fn_part]})

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

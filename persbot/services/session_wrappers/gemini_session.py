"""Gemini chat session wrapper for managing history with author tracking."""

import base64
import logging
from collections import deque
from typing import Any, AsyncIterator, List, Optional

from google.genai import types as genai_types

from persbot.services.base import ChatMessage
from persbot.utils import get_mime_type

logger = logging.getLogger(__name__)


def _build_content(role: str, parts: List[genai_types.Part]) -> genai_types.Content:
    """Build a Content object with role and parts.

    Args:
        role: The role ('user' or 'model').
        parts: List of Part objects.

    Returns:
        A Content object.
    """
    return genai_types.Content(role=role, parts=parts)


def _build_text_part(text: str) -> genai_types.Part:
    """Build a Part object from text.

    Args:
        text: The text content.

    Returns:
        A Part object with text.
    """
    return genai_types.Part(text=text)


def _build_function_call_part(name: str, args: dict) -> genai_types.Part:
    """Build a Part object containing a function call.

    Args:
        name: The function name.
        args: The function arguments.

    Returns:
        A Part object with function_call.
    """
    return genai_types.Part.from_function_call(name=name, args=args)


def _build_function_response_part(name: str, response: dict) -> genai_types.Part:
    """Build a Part object containing a function response.

    Args:
        name: The function name.
        response: The response dictionary (may contain 'result' or 'error').

    Returns:
        A Part object with function_response.
    """
    return genai_types.Part.from_function_response(name=name, response=response)


def _build_inline_data_part(data: bytes, mime_type: str) -> genai_types.Part:
    """Build a Part object containing inline data (e.g., images).

    Args:
        data: The raw binary data.
        mime_type: The MIME type of the data.

    Returns:
        A Part object with inline_data.
    """
    return genai_types.Part(
        inline_data=genai_types.Blob(
            mime_type=mime_type,
            data=base64.b64encode(data).decode("utf-8")
        )
    )


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

    def _get_api_history(self) -> List[genai_types.Content]:
        """Convert local history to API format.

        Returns a list of Content objects for the Google GenAI SDK.
        """
        api_history = []
        for msg in self.history:
            final_parts: List[genai_types.Part] = []

            # Add existing text/content parts
            # Handle both ChatMessage (has parts) and SimpleMessage (has only content)
            if hasattr(msg, 'parts') and msg.parts:
                for p in msg.parts:
                    if isinstance(p, dict) and "text" in p:
                        final_parts.append(_build_text_part(p["text"]))
                    elif hasattr(p, "text") and p.text:
                        final_parts.append(_build_text_part(p.text))
            elif hasattr(msg, 'content') and msg.content:
                # Fallback for SimpleMessage or messages without parts
                final_parts.append(_build_text_part(msg.content))

            # Reconstruct image parts from stored bytes
            if hasattr(msg, "images") and msg.images:
                for img_data in msg.images:
                    mime_type = get_mime_type(img_data)
                    final_parts.append(_build_inline_data_part(img_data, mime_type))

            if final_parts:
                api_history.append(_build_content(msg.role, final_parts))
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
        contents: List[genai_types.Content] = self._get_api_history()

        current_parts: List[genai_types.Part] = []
        if user_message:
            current_parts.append(_build_text_part(user_message))

        if images:
            for img_data in images:
                mime_type = get_mime_type(img_data)
                current_parts.append(_build_inline_data_part(img_data, mime_type))

        contents.append(_build_content("user", current_parts))

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
        contents: List[genai_types.Content] = self._get_api_history()

        current_parts: List[genai_types.Part] = []
        if user_message:
            current_parts.append(_build_text_part(user_message))

        if images:
            for img_data in images:
                mime_type = get_mime_type(img_data)
                current_parts.append(_build_inline_data_part(img_data, mime_type))

        contents.append(_build_content("user", current_parts))

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
        self, tool_rounds: List[tuple], tools: Optional[List[Any]] = None
    ) -> tuple[ChatMessage, Any]:
        """
        Send tool execution results back to model and get continuation.

        Args:
            tool_rounds: List of (response_obj, tool_results) or (response_obj, tool_results, function_calls) tuples.
            tools: Tools to pass for the next API call.

        Returns:
            Tuple of (model_message, response_obj).
        """
        # Build contents from history
        contents: List[genai_types.Content] = self._get_api_history()

        # The last entry is the text-only model msg from initial response - remove it
        if contents and contents[-1].role == "model":
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

            # Build model content from either response object or function_calls
            # Per SDK docs, we should use response.candidates[0].content directly
            # rather than rebuilding parts, to avoid serialization issues
            if resp_obj is not None and resp_obj.candidates:
                model_content = resp_obj.candidates[0].content
                # Filter out thought parts if present (they cause validation errors)
                if any(getattr(p, "thought", False) for p in model_content.parts):
                    filtered_parts = [
                        p for p in model_content.parts
                        if not getattr(p, "thought", False)
                    ]
                    if filtered_parts:
                        contents.append(genai_types.Content(role="model", parts=filtered_parts))
                else:
                    # Use the original content directly - it's already properly typed
                    contents.append(model_content)
            elif function_calls:
                # Streaming case: convert function_calls to Part objects
                model_parts: List[genai_types.Part] = []
                for fc in function_calls:
                    model_parts.append(_build_function_call_part(
                        name=fc.get("name", ""),
                        args=fc.get("parameters") or fc.get("args") or {}
                    ))
                if model_parts:
                    contents.append(_build_content("model", model_parts))
                else:
                    logger.error("send_tool_results: no function_calls to build")
                    continue
            else:
                logger.error("send_tool_results: no response object and no function_calls provided")
                continue

            # Add function response parts using proper SDK types
            for result_item in results:
                tool_name = result_item.get("name")
                result_data = result_item.get("result")
                error = result_item.get("error")
                if error:
                    response_data = {"error": error}
                else:
                    response_data = {"result": str(result_data) if result_data is not None else ""}

                # Use proper Part.from_function_response with role='tool' as per SDK docs
                func_response_part = _build_function_response_part(tool_name, response_data)
                contents.append(_build_content("tool", [func_response_part]))

        # Call generate_content with properly typed contents
        response = self._factory.generate_content(contents=contents, tools=tools)

        # Create model message
        clean_content = extract_clean_text(response)
        model_msg = ChatMessage(
            role="model",
            content=clean_content,
            parts=[{"text": clean_content}],
        )

        return model_msg, response

"""Z.AI API service for SoyeBot.

Supports both Standard API and Coding Plan API endpoints:
- Standard API: https://api.z.ai/api/paas/v4/ (pay-as-you-go, token-based)
- Coding Plan API: https://api.z.ai/api/coding/paas/v4/ (subscription-based, prompt-based)

Enable Coding Plan API by setting ZAI_CODING_PLAN=true in environment.
"""

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import discord
from openai import OpenAI

if TYPE_CHECKING:
    pass

from persbot.config import AppConfig
from persbot.constants import LLMDefaults
from persbot.services.base import BaseLLMServiceCore, ChatMessage
from persbot.services.model_wrappers.zai_model import ZAIChatModel
from persbot.services.prompt_service import PromptService
from persbot.services.retry_handler import (
    ZAIRetryHandler,
)
from persbot.services.session_wrappers.zai_session import ZAIChatSession
from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

logger = logging.getLogger(__name__)

# Models that don't support vision - will auto-switch to vision model
NON_VISION_MODELS = {"glm-4.7", "glm-4-flash"}
VISION_MODEL = "glm-4.6v"


class ZAIService(BaseLLMServiceCore):
    """Z.AI Coding Plan API와의 모든 상호작용을 관리합니다."""

    def __init__(
        self,
        config: AppConfig,
        *,
        assistant_model_name: str,
        summary_model_name: Optional[str] = None,
        prompt_service: PromptService,
    ):
        super().__init__(config)
        # Initialize OpenAI client with Z.AI base URL and timeout
        self.client = OpenAI(
            api_key=config.zai_api_key,
            base_url=config.zai_base_url,
            timeout=config.zai_request_timeout,
        )
        self._assistant_cache: Dict[int, ZAIChatModel] = {}
        self._max_messages = 7
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

        # Preload default assistant model
        self.assistant_model = self._get_or_create_assistant(
            self._assistant_model_name,
            self.prompt_service.get_active_assistant_prompt(),
        )

    def _create_retry_handler(self) -> ZAIRetryHandler:
        """Create retry handler for Z.AI API."""
        config = self._create_retry_config_core()
        return ZAIRetryHandler(config)

    def _get_or_create_assistant(self, model_name: str, system_instruction: str) -> ZAIChatModel:
        """Get or create a chat model for the given model."""
        key = hash((model_name, system_instruction))
        if key not in self._assistant_cache:
            self._assistant_cache[key] = ZAIChatModel(
                client=self.client,
                model_name=model_name,
                system_instruction=system_instruction,
                temperature=getattr(self.config, "temperature", LLMDefaults.TEMPERATURE),
                top_p=getattr(self.config, "top_p", LLMDefaults.TOP_P),
                max_messages=self._max_messages,
                text_extractor=self._extract_text_from_response,
            )
        return self._assistant_cache[key]

    def create_assistant_model(
        self, system_instruction: str, use_cache: bool = True
    ) -> ZAIChatModel:
        """Create a chat model with the given system instruction."""
        return self._get_or_create_assistant(self._assistant_model_name, system_instruction)

    def reload_parameters(self) -> None:
        """Reload parameters by clearing assistant cache."""
        self._assistant_cache.clear()

    def get_user_role_name(self) -> str:
        """Return role name for user messages."""
        return "user"

    def get_assistant_role_name(self) -> str:
        """Return role name for assistant messages."""
        return "assistant"

    def _get_model_for_images(self, model_name: str, has_images: bool) -> str:
        """Get the appropriate model name based on whether images are present.

        GLM-4.7 doesn't support vision, so we auto-switch to glm-4.6v when images are detected.
        """
        if has_images and model_name in NON_VISION_MODELS:
            return VISION_MODEL
        return model_name

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if exception is a rate limit error."""
        error_str = str(error).lower()
        return "rate limit" in error_str or "429" in error_str

    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        """Log raw request for debugging."""
        pass

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw response for debugging."""
        pass

    def _extract_text_from_response(self, response_obj: Any) -> str:
        """Extract text content from response object."""
        try:
            choices = getattr(response_obj, "choices", []) or []
            for choice in choices:
                message = getattr(choice, "message", None)
                if message and hasattr(message, "content") and message.content:
                    # Handle both string content and list of content blocks
                    content = message.content
                    if isinstance(content, str):
                        return content.strip()
                    elif isinstance(content, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                text_parts.append(block["text"])
                            elif hasattr(block, "text"):
                                text_parts.append(str(block.text))
                        return " ".join(text_parts).strip()
        except Exception:
            logger.exception("Failed to extract text from Z.AI response")
        return ""

    async def summarize_text(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[str]:
        """Summarize text using Z.AI model.

        Args:
            text: Text to summarize.
            cancel_event: Optional event to check for abort signals.

        Returns:
            Summarized text or None if cancelled/failed.
        """
        if not text.strip():
            return "요약할 메시지가 없습니다."

        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("LLM API call aborted by user")

        prompt = f"Discord 대화 내용:\n{text}"
        return await self.execute_with_retry(
            lambda: self.client.chat.completions.create(
                model=self._summary_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompt_service.get_summary_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=getattr(self.config, "temperature", 1.0),
                top_p=getattr(self.config, "top_p", 1.0),
            ),
            error_prefix="요약",
            cancel_event=cancel_event,
        )

    async def generate_chat_response(
        self,
        chat_session: ZAIChatSession,
        user_message: str,
        discord_message: Union[discord.Message, List[discord.Message]],
        model_name: Optional[str] = None,
        tools: Optional[Any] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Generate chat response."""
        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("LLM API call aborted by user")

        self._log_raw_request(user_message, chat_session)

        if isinstance(discord_message, list):
            primary_msg = discord_message[0]
            message_ids = [str(m.id) for m in discord_message]
        else:
            primary_msg = discord_message
            message_ids = [str(discord_message.id)]

        author_id = primary_msg.author.id
        author_name = getattr(primary_msg.author, "name", str(author_id))

        # Extract images from messages first (before model selection)
        images = await self._extract_images_from_messages(discord_message)

        # Determine the actual model to use (switch to vision model if images present)
        actual_model = self._get_model_for_images(
            model_name
            if model_name
            else getattr(chat_session, "_model_name", self._assistant_model_name),
            len(images) > 0,
        )

        # Check for model switch
        current_model_name = getattr(chat_session, "_model_name", None)
        if actual_model and current_model_name != actual_model:
            chat_session._model_name = actual_model

        # Convert tools to Z.AI (OpenAI-compatible) format if provided
        converted_tools = ZAIToolAdapter.convert_tools(tools) if tools else None

        result = await self.execute_with_retry(
            lambda: chat_session.send_message(
                user_message,
                author_id,
                author_name=author_name,
                message_ids=message_ids,
                images=images,
                tools=converted_tools,
            ),
            "응답 생성",
            return_full_response=True,
            discord_message=primary_msg,
        )

        if result is None:
            return None

        user_msg, model_msg, response = result

        # Update history safely
        chat_session._history.append(user_msg)
        chat_session._history.append(model_msg)

        return model_msg.content, response

    async def generate_chat_response_stream(
        self,
        chat_session: ZAIChatSession,
        user_message: str,
        discord_message: Union[discord.Message, List[discord.Message]],
        model_name: Optional[str] = None,
        tools: Optional[Any] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response from Z.AI.

        Yields text chunks as they arrive from the API.
        Chunks are yielded after each line break for faster initial response.

        Args:
            chat_session: The Z.AI chat session.
            user_message: The user's message.
            discord_message: The Discord message(s) for context.
            model_name: Optional specific model to use.
            tools: Optional tools for function calling.
            cancel_event: Optional event to check for cancellation.

        Yields:
            Text chunks as they are generated.
        """
        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("LLM API call aborted by user")

        self._log_raw_request(user_message, chat_session)

        if isinstance(discord_message, list):
            primary_msg = discord_message[0]
            message_ids = [str(m.id) for m in discord_message]
        else:
            primary_msg = discord_message
            message_ids = [str(discord_message.id)]

        author_id = primary_msg.author.id
        author_name = getattr(primary_msg.author, "name", str(author_id))

        # Extract images from messages first (before model selection)
        images = await self._extract_images_from_messages(discord_message)

        # Determine the actual model to use (switch to vision model if images present)
        actual_model = self._get_model_for_images(
            model_name
            if model_name
            else getattr(chat_session, "_model_name", self._assistant_model_name),
            len(images) > 0,
        )

        # Check for model switch
        current_model_name = getattr(chat_session, "_model_name", None)
        if actual_model and current_model_name != actual_model:
            chat_session._model_name = actual_model

        # Convert tools to Z.AI format if provided
        converted_tools = ZAIToolAdapter.convert_tools(tools) if tools else None

        # Start streaming - handle both sync and async send_message_stream methods
        # Gemini sessions have async send_message_stream, ZAI sessions have sync
        is_async_stream = inspect.iscoroutinefunction(chat_session.send_message_stream)
        if is_async_stream:
            # Async method (e.g., GeminiChatSession) - returns (user_msg, stream)
            user_msg, stream = await chat_session.send_message_stream(
                user_message,
                author_id,
                author_name=author_name,
                message_ids=message_ids,
                images=images,
                tools=converted_tools,
            )
        else:
            # Sync method (e.g., ZAIChatSession) - returns (stream, user_msg)
            stream, user_msg = await asyncio.to_thread(
                chat_session.send_message_stream,
                user_message,
                author_id,
                author_name=author_name,
                message_ids=message_ids,
                images=images,
                tools=converted_tools,
            )

        # Buffer for streaming - yield on newline for natural line breaks
        buffer = ""
        full_content = ""

        try:
            if is_async_stream:
                # Async iterator (Gemini)
                async for chunk in stream:
                    # Check for cancellation
                    if cancel_event and cancel_event.is_set():
                        # Close the stream to stop server-side generation
                        if hasattr(stream, 'aclose'):
                            await stream.aclose()
                        elif hasattr(stream, 'close'):
                            stream.close()
                        raise asyncio.CancelledError("LLM streaming aborted by user")

                    # Extract text from Gemini chunk format
                    text = self._extract_text_from_stream_chunk(chunk)
                    if text:
                        buffer += text
                        full_content += text

                        # Yield when we see a line break
                        if "\n" in buffer:
                            lines = buffer.split("\n")
                            for line in lines[:-1]:
                                if line:
                                    yield line + "\n"
                            buffer = lines[-1]
            else:
                # Sync iterator (ZAI/OpenAI)
                for chunk in stream:
                    # Check for cancellation
                    if cancel_event and cancel_event.is_set():
                        # Close the stream to stop server-side generation
                        try:
                            stream.close()
                        except Exception:
                            pass
                        raise asyncio.CancelledError("LLM streaming aborted by user")

                    # Extract text delta from OpenAI chunk format
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            text = delta.content
                            buffer += text
                            full_content += text

                            # Yield when we see a line break
                            if "\n" in buffer:
                                lines = buffer.split("\n")
                                for line in lines[:-1]:
                                    if line:
                                        yield line + "\n"
                                buffer = lines[-1]

            # Yield any remaining content in buffer
            if buffer:
                yield buffer

        except asyncio.CancelledError:
            # Ensure stream is closed on cancellation to stop server-side generation
            try:
                if hasattr(stream, 'aclose'):
                    await stream.aclose()
                elif hasattr(stream, 'close'):
                    stream.close()
            except BaseException:
                pass
            raise
        except Exception:
            # Close stream on any exception to prevent resource leaks
            try:
                if hasattr(stream, 'aclose'):
                    await stream.aclose()
                elif hasattr(stream, 'close'):
                    stream.close()
            except BaseException:
                pass
            raise
        finally:
            # Always try to close stream as a safety net
            try:
                if hasattr(stream, 'aclose'):
                    await stream.aclose()
                elif hasattr(stream, 'close'):
                    stream.close()
            except BaseException:
                pass

        # Update history with the full conversation
        model_msg = ChatMessage(role="assistant", content=full_content)
        chat_session._history.append(user_msg)
        chat_session._history.append(model_msg)

    def _extract_text_from_stream_chunk(self, chunk: Any) -> str:
        """Extract text from a Gemini streaming chunk, filtering out thoughts.

        This method handles Gemini-style chunks when a Gemini session
        is accidentally routed to this service.
        """
        try:
            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            # Skip parts that are marked as thoughts
                            if getattr(part, "thought", False):
                                continue

                            if hasattr(part, "text") and part.text:
                                return part.text
        except Exception:
            pass
        return ""

    async def send_tool_results(
        self,
        chat_session: ZAIChatSession,
        tool_rounds: List[Tuple[Any, Any]],
        tools: Optional[Any] = None,
        discord_message: Optional[discord.Message] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Send tool results back to model and get continuation response.

        Args:
            chat_session: The Z.AI chat session.
            tool_rounds: List of (response_obj, tool_results) tuples.
            tools: Original tool definitions (will be converted to Z.AI format).
            discord_message: Discord message for error notifications.
            cancel_event: Optional event to check for abort signals.

        Returns:
            Tuple of (response_text, response_obj) or None.
        """
        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("LLM API call aborted by user")

        converted_tools = ZAIToolAdapter.convert_tools(tools) if tools else None

        result = await self.execute_with_retry(
            lambda: chat_session.send_tool_results(tool_rounds, tools=converted_tools),
            "tool 결과 전송",
            return_full_response=True,
            discord_message=discord_message,
        )

        if result is None:
            return None

        model_msg, response = result

        # Update the last model entry in history with the final response
        if chat_session._history and chat_session._history[-1].role == "assistant":
            chat_session._history[-1] = model_msg

        return model_msg.content, response

    # Tool support methods
    def get_tools_for_provider(self, tools: List[Any]) -> Any:
        """Convert tool definitions to Z.AI format.

        Args:
            tools: List of ToolDefinition objects to convert.

        Returns:
            List of tool dictionaries in Z.AI function calling format.
        """
        return ZAIToolAdapter.convert_tools(tools)

    def extract_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from Z.AI response.

        Args:
            response: Z.AI response object (from chat.completions.create).

        Returns:
            List of function call dictionaries with 'id', 'name', and 'parameters'.
        """
        return ZAIToolAdapter.extract_function_calls(response)

    def format_function_results(self, results: List[Dict[str, Any]]) -> Any:
        """Format function results for sending back to Z.AI.

        Args:
            results: List of dicts with 'id', 'name', 'result', and optionally 'error'.

        Returns:
            List of message dictionaries in Z.AI tool format.
        """
        return ZAIToolAdapter.format_results(results)

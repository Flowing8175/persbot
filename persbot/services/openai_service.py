"""OpenAI API service for SoyeBot."""

import asyncio
import inspect
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import discord
from openai import OpenAI, RateLimitError

if TYPE_CHECKING:
    pass

from persbot.config import AppConfig
from persbot.constants import LLMDefaults
from persbot.services.base import BaseLLMServiceCore, ChatMessage
from persbot.services.model_wrappers.openai_model import OpenAIChatCompletionModel
from persbot.services.prompt_service import PromptService
from persbot.services.retry_handler import (
    OpenAIRetryHandler,
)
from persbot.services.session_wrappers.openai_session import (
    ChatCompletionSession,
    ResponseSession,
)
from persbot.providers.adapters.openai_adapter import OpenAIToolAdapter

logger = logging.getLogger(__name__)


class OpenAIService(BaseLLMServiceCore):
    """OpenAI API와의 모든 상호작용을 관리합니다."""

    # Models that don't support top_p parameter
    _MODELS_WITHOUT_TOP_P = frozenset(["gpt-5", "o1", "o3", "o4"])

    @staticmethod
    def _supports_top_p(model_name: str) -> bool:
        """Check if a model supports the top_p parameter."""
        model_lower = model_name.lower()
        return not any(model_lower.startswith(prefix) for prefix in OpenAIService._MODELS_WITHOUT_TOP_P)

    def __init__(
        self,
        config: AppConfig,
        *,
        assistant_model_name: str,
        summary_model_name: Optional[str] = None,
        prompt_service: PromptService,
    ):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.openai_api_key,
            timeout=config.api_request_timeout,
        )
        # Use OrderedDict for LRU eviction with max size limit
        self._assistant_cache: OrderedDict[int, OpenAIChatCompletionModel] = OrderedDict()
        self._cache_max_size = 10  # Limit cache to 10 model instances
        self._max_messages = 7
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

        # Preload default response model
        self.assistant_model = self._get_or_create_assistant(
            self._assistant_model_name,
            self.prompt_service.get_active_assistant_prompt(),
        )
        logger.debug("OpenAI 모델 '%s' 준비 완료.", self._assistant_model_name)

    def _create_retry_handler(self) -> OpenAIRetryHandler:
        """Create retry handler for OpenAI API."""
        config = self._create_retry_config_core()
        return OpenAIRetryHandler(config)

    def _get_or_create_assistant(
        self, model_name: str, system_instruction: str
    ) -> OpenAIChatCompletionModel:
        """Get or create an assistant model wrapper with LRU eviction."""
        key = hash((model_name, system_instruction))

        # Check if already cached - move to end for LRU
        if key in self._assistant_cache:
            self._assistant_cache.move_to_end(key)
            return self._assistant_cache[key]

        # Evict oldest entry if at capacity
        if len(self._assistant_cache) >= self._cache_max_size:
            evicted_key, _ = self._assistant_cache.popitem(last=False)
            logger.debug("Evicted assistant cache entry %s (LRU)", evicted_key)

        # Select model wrapper based on configuration (Fine-tuned models use Chat Completions)
        use_finetuned_logic = (
            self.config.openai_finetuned_model
            and model_name == self.config.openai_finetuned_model
        )

        service_tier = "flex"
        if use_finetuned_logic:
            service_tier = "default"

        self._assistant_cache[key] = OpenAIChatCompletionModel(
            client=self.client,
            model_name=model_name,
            system_instruction=system_instruction,
            temperature=getattr(self.config, "temperature", LLMDefaults.TEMPERATURE),
            top_p=getattr(self.config, "top_p", LLMDefaults.TOP_P),
            max_messages=self._max_messages,
            service_tier=service_tier,
        )
        return self._assistant_cache[key]

    def create_assistant_model(self, system_instruction: str) -> OpenAIChatCompletionModel:
        """Create an assistant model with custom system instruction."""
        return self._get_or_create_assistant(self._assistant_model_name, system_instruction)

    def reload_parameters(self) -> None:
        """Reload parameters by clearing the assistant cache."""
        self._assistant_cache.clear()
        logger.debug("OpenAI assistant cache cleared to apply new parameters.")

    def get_user_role_name(self) -> str:
        """Return the role name for user messages."""
        return "user"

    def get_assistant_role_name(self) -> str:
        """Return the role name for assistant messages."""
        return "assistant"

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if exception is a rate limit error."""
        if isinstance(error, RateLimitError):
            return True
        error_str = str(error).lower()
        return "rate limit" in error_str or "429" in error_str

    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        """Log raw request details for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug("[RAW REQUEST] User message preview: %r", user_message[:200])
            if chat_session and hasattr(chat_session, "history"):
                history = chat_session.history
                formatted = []
                for msg in history[-5:]:
                    role = msg.role
                    content = str(msg.content)

                    # Clean up content display if it starts with "Name: "
                    author_label = str(msg.author_name or msg.author_id or "bot")
                    display_content = content
                    if msg.author_name and content.startswith(f"{msg.author_name}:"):
                        display_content = content[len(msg.author_name) + 1 :].strip()

                    truncated = display_content[:100].replace("\n", " ")
                    formatted.append(f"{role} (author:{author_label}) {truncated}")
                if formatted:
                    logger.debug("[RAW REQUEST] Recent history:\n%s", "\n".join(formatted))
        except Exception:
            logger.exception("[RAW REQUEST] Error logging raw request")

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw response details for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug("[RAW RESPONSE %s] %s", attempt, response_obj)
        except Exception:
            logger.exception("[RAW RESPONSE %s] Error logging raw response", attempt)

    def _extract_text_from_response(self, response_obj: Any) -> str:
        """Extract text content from OpenAI chat completion response."""
        try:
            choices = getattr(response_obj, "choices", []) or []
            for choice in choices:
                message = getattr(choice, "message", None)
                if message and getattr(message, "content", None):
                    return str(message.content).strip()
        except Exception:
            logger.exception("Failed to extract text from OpenAI response")

        return self._extract_text_from_response_output(response_obj)

    def _extract_text_from_response_output(self, response_obj: Any) -> str:
        """Extract text from Responses API output."""
        try:
            text_fragments = []
            seen_fragments = set()
            output_text = getattr(response_obj, "output_text", None)
            if output_text:
                if isinstance(output_text, str):
                    normalized = str(output_text).strip()
                    if normalized and normalized not in seen_fragments:
                        text_fragments.append(normalized)
                        seen_fragments.add(normalized)
                else:
                    try:
                        for part in output_text:
                            normalized = str(part).strip()
                            if normalized and normalized not in seen_fragments:
                                text_fragments.append(normalized)
                                seen_fragments.add(normalized)
                    except TypeError:
                        normalized = str(output_text).strip()
                        if normalized and normalized not in seen_fragments:
                            text_fragments.append(normalized)
                            seen_fragments.add(normalized)

            output_items = getattr(response_obj, "output", None) or []
            for item in output_items:
                content_list = getattr(item, "content", None) or []
                for content in content_list:
                    text_value = getattr(content, "text", None)
                    if text_value:
                        normalized = str(text_value).strip()
                        if normalized and normalized not in seen_fragments:
                            text_fragments.append(normalized)
                            seen_fragments.add(normalized)
            return "\n".join(text_fragments).strip()
        except Exception:
            logger.exception("Failed to extract text from response output")
        return ""

    async def summarize_text(
        self,
        text: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[str]:
        """Summarize a text using the summary model.

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
            logger.debug("Summary API call aborted")
            raise asyncio.CancelledError("LLM API call aborted by user")

        prompt = f"Discord 대화 내용:\n{text}"

        def _create_summary_request():
            api_kwargs = {
                "model": self._summary_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": self.prompt_service.get_summary_prompt(),
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": getattr(self.config, "temperature", 1.0),
                "service_tier": getattr(self.config, "service_tier", "flex"),
            }
            if self._supports_top_p(self._summary_model_name):
                api_kwargs["top_p"] = getattr(self.config, "top_p", 1.0)
            return self.client.chat.completions.create(**api_kwargs)

        return await self.execute_with_retry(
            _create_summary_request,
            "요약",
            extract_text=self._extract_text_from_response,
            cancel_event=cancel_event,
        )

    async def generate_chat_response(
        self,
        chat_session: Union[ChatCompletionSession, ResponseSession],
        user_message: str,
        discord_message: Union[discord.Message, List[discord.Message]],
        model_name: Optional[str] = None,
        tools: Optional[Any] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Generate a chat response."""
        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            logger.debug("API call aborted")
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

        # Check for model switch
        current_model_name = getattr(chat_session, "_model_name", None)
        if model_name and current_model_name != model_name:
            logger.debug("Model: %s → %s", current_model_name, model_name)
            chat_session._model_name = model_name

        # Extract images from message(s) - supports both single and list of messages
        images = await self._extract_images_from_messages(discord_message)

        # Convert tools to appropriate format based on session type
        if tools:
            # ResponseSession uses Responses API which expects flatter format
            if isinstance(chat_session, ResponseSession):
                converted_tools = OpenAIToolAdapter.convert_tools_for_responses_api(tools)
            else:
                # ChatCompletionSession uses Chat Completions API
                converted_tools = OpenAIToolAdapter.convert_tools(tools)
        else:
            converted_tools = None

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
        chat_session: ChatCompletionSession,
        user_message: str,
        discord_message: Union[discord.Message, List[discord.Message]],
        model_name: Optional[str] = None,
        tools: Optional[Any] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response.

        Yields text chunks as they arrive from the API.
        Chunks are yielded after each line break for faster initial response.

        Args:
            chat_session: The chat session.
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
            logger.debug("Streaming API call aborted")
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

        # Check for model switch
        current_model_name = getattr(chat_session, "_model_name", None)
        if model_name and current_model_name != model_name:
            logger.debug("Model: %s → %s", current_model_name, model_name)
            chat_session._model_name = model_name

        # Extract images from message(s) - supports both single and list of messages
        images = await self._extract_images_from_messages(discord_message)

        # Convert tools to appropriate format based on session type
        if tools:
            # ResponseSession uses Responses API which expects flatter format
            if isinstance(chat_session, ResponseSession):
                converted_tools = OpenAIToolAdapter.convert_tools_for_responses_api(tools)
            else:
                # ChatCompletionSession uses Chat Completions API
                converted_tools = OpenAIToolAdapter.convert_tools(tools)
        else:
            converted_tools = None

        # Start streaming - get stream object
        # Check if send_message_stream is async (e.g., from GeminiChatSession passed incorrectly)
        is_async_stream = inspect.iscoroutinefunction(chat_session.send_message_stream)

        if is_async_stream:
            # Async version - call directly with await
            # Note: async version returns (user_msg, stream) not (stream, user_msg)
            user_msg, stream = await chat_session.send_message_stream(
                user_message,
                author_id,
                author_name=author_name,
                message_ids=message_ids,
                images=images,
                tools=converted_tools,
            )
        else:
            # Sync version - run in thread to avoid blocking
            stream, user_msg = await asyncio.to_thread(
                chat_session.send_message_stream,
                user_message,
                author_id,
                author_name=author_name,
                message_ids=message_ids,
                images=images,
                tools=converted_tools,
            )

        # Buffer for streaming - yield chunks immediately for faster first response
        buffer = ""
        full_content = ""

        async def _iterate_stream():
            """Iterate the stream and yield chunks.

            Handles both sync streams (from OpenAI sessions) and async streams
            (from Gemini sessions that were incorrectly passed here).
            """
            if is_async_stream:
                # Direct async iteration for async streams (e.g., GeminiChatSession)
                try:
                    async for chunk in stream:
                        # Check for cancellation
                        if cancel_event and cancel_event.is_set():
                            logger.debug("Streaming aborted")
                            raise asyncio.CancelledError("LLM streaming aborted by user")
                        yield chunk
                except asyncio.CancelledError:
                    logger.debug("Streaming response cancelled")
                    raise
            else:
                # Sync stream - iterate in thread to avoid blocking
                queue: asyncio.Queue = asyncio.Queue()
                sentinel = object()  # Sentinel to signal end of stream
                # Thread-safe flag for cancellation (accessible from sync thread)
                cancel_flag = threading.Event()

                def _sync_iterate():
                    """Run in thread: iterate stream and put chunks in queue."""
                    try:
                        for chunk in stream:
                            # Check for cancellation from within the thread
                            # This ensures we stop reading from httpx immediately
                            if cancel_flag.is_set():
                                logger.debug("Sync stream iteration aborted in thread")
                                break
                            # Put chunk in queue for async consumption
                            # Use put_nowait since queue is unbounded (no maxsize)
                            queue.put_nowait(chunk)
                    except Exception as e:
                        logger.debug("Stream iteration error: %s", e)
                    finally:
                        # Signal end of stream
                        try:
                            queue.put_nowait(sentinel)
                        except Exception:
                            pass
                        stream.close()

                # Start sync iteration in thread
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(None, _sync_iterate)

                try:
                    while True:
                        # Check for cancellation
                        if cancel_event and cancel_event.is_set():
                            logger.debug("Streaming aborted")
                            cancel_flag.set()  # Signal thread to stop
                            stream.close()
                            raise asyncio.CancelledError("LLM streaming aborted by user")

                        # Get chunk from queue with timeout
                        try:
                            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            # No chunk yet, check cancellation and continue
                            continue

                        # Check for end of stream
                        if chunk is sentinel:
                            break

                        yield chunk
                finally:
                    # Ensure thread is done - set cancel flag first to accelerate cleanup
                    cancel_flag.set()
                    try:
                        await asyncio.wait_for(future, timeout=1.0)
                    except (asyncio.TimeoutError, Exception):
                        pass

        try:
            async for chunk in _iterate_stream():
                # Extract text delta from chunk
                # Handle both OpenAI Chat Completions format and potential foreign formats
                text = None
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and hasattr(delta, 'content') and delta.content:
                        text = delta.content
                elif hasattr(chunk, 'text'):
                    # Direct text format (some simplified streaming formats)
                    text = chunk.text
                elif isinstance(chunk, str):
                    # Plain string chunks
                    text = chunk

                if text:
                    buffer += text
                    full_content += text

                    # Yield immediately when we have content with line break,
                    # OR yield after accumulating reasonable content for faster first response
                    if "\n" in buffer:
                        lines = buffer.split("\n")
                        # Yield all complete lines
                        for line in lines[:-1]:
                            if line:  # Skip empty lines
                                yield line + "\n"
                        # Keep the last incomplete line in buffer
                        buffer = lines[-1]
                    elif len(buffer) > 50:
                        # Yield partial content for faster first response
                        yield buffer
                        buffer = ""

            # Yield any remaining content in buffer
            if buffer:
                yield buffer

        except asyncio.CancelledError:
            logger.debug("Streaming response cancelled")
            raise

        # Update history with the full conversation
        model_msg = ChatMessage(role="assistant", content=full_content)
        chat_session._history.append(user_msg)
        chat_session._history.append(model_msg)

    async def send_tool_results(
        self,
        chat_session: ChatCompletionSession,
        tool_rounds: List[Tuple[Any, Any]],
        tools: Optional[Any] = None,
        discord_message: Optional[discord.Message] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Send tool results back to model and get continuation response.

        Args:
            chat_session: The OpenAI chat completion session.
            tool_rounds: List of (response_obj, tool_results) tuples.
            tools: Original tool definitions (will be converted to OpenAI format).
            discord_message: Discord message for error notifications.
            cancel_event: Optional event to check for abort signals.

        Returns:
            Tuple of (response_text, response_obj) or None.
        """
        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            logger.debug("Tool results API call aborted")
            raise asyncio.CancelledError("LLM API call aborted by user")

        # Convert tools to appropriate format based on session type
        if tools:
            if isinstance(chat_session, ResponseSession):
                converted_tools = OpenAIToolAdapter.convert_tools_for_responses_api(tools)
            else:
                converted_tools = OpenAIToolAdapter.convert_tools(tools)
        else:
            converted_tools = None

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
        """Convert tool definitions to OpenAI format.

        Args:
            tools: List of ToolDefinition objects to convert.

        Returns:
            List of tool dictionaries in OpenAI function calling format.
        """
        return OpenAIToolAdapter.convert_tools(tools)

    def extract_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from OpenAI response.

        Args:
            response: OpenAI response object (from chat.completions.create).

        Returns:
            List of function call dictionaries with 'id', 'name', and 'parameters'.
        """
        return OpenAIToolAdapter.extract_function_calls(response)

    def format_function_results(self, results: List[Dict[str, Any]]) -> Any:
        """Format function results for sending back to OpenAI.

        Args:
            results: List of dicts with 'id', 'name', 'result', and optionally 'error'.

        Returns:
            List of message dictionaries in OpenAI tool format.
        """
        return OpenAIToolAdapter.format_results(results)

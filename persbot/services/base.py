"""Base LLM Service for SoyeBot."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import discord

from persbot.config import AppConfig
from persbot.services.retry_handler import (
    BackoffStrategy,
    RetryConfig,
    RetryHandler,
)
from persbot.utils import (
    process_image_sync,
)

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single message in the chat history."""

    role: str
    content: str
    author_id: Optional[int] = None
    author_name: Optional[str] = None
    message_ids: List[str] = field(default_factory=list)
    # For Gemini, content is stored in 'parts'
    parts: Optional[list[dict[str, str]]] = None
    # For storing image data (bytes)
    images: List[bytes] = field(default_factory=list)


class BaseLLMService(ABC):
    """Abstract base class for LLM services handling retries, logging, and common behavior."""

    def __init__(self, config: AppConfig, retry_handler: Optional[RetryHandler] = None):
        self.config = config
        self._retry_handler = retry_handler

    @abstractmethod
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the exception is a rate limit error."""
        pass

    def _extract_retry_delay(self, error: Exception) -> Optional[float]:
        """Extract retry delay from error, if available."""
        return None

    @abstractmethod
    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        """Log the raw request for debugging."""
        pass

    @abstractmethod
    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log the raw response for debugging."""
        pass

    @abstractmethod
    def _extract_text_from_response(self, response_obj: Any) -> str:
        """Extract the text content from the response object."""
        pass

    @abstractmethod
    def get_user_role_name(self) -> str:
        """Return the name for the 'user' role in the chat history."""
        pass

    @abstractmethod
    def get_assistant_role_name(self) -> str:
        """Return the name for the 'assistant' role in the chat history."""
        pass

    @abstractmethod
    async def generate_chat_response(
        self,
        chat_session: Any,
        user_message: str,
        discord_message: discord.Message | List[discord.Message],
    ) -> Optional[Tuple[str, Any]]:
        """
        Generate chat response.
        Should be implemented by subclasses if they support chat generation.
        """
        pass

    @abstractmethod
    def get_tools_for_provider(self, tools: List[Any]) -> Any:
        """
        Convert tool definitions to provider-specific format.

        Args:
            tools: List of tool definitions to convert.

        Returns:
            Provider-specific tool format.
        """
        pass

    @abstractmethod
    def extract_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract function calls from provider response.

        Args:
            response: Provider response object.

        Returns:
            List of function call dictionaries with 'name' and 'parameters'.
        """
        pass

    @abstractmethod
    def format_function_results(self, results: List[Dict[str, Any]]) -> Any:
        """
        Format function results for sending back to provider.

        Args:
            results: List of dicts with 'name', 'result', and optionally 'error'.

        Returns:
            Provider-specific formatted results.
        """
        pass

    def reload_parameters(self) -> None:
        """Reload service parameters (e.g. clear caches). To be overridden."""
        pass

    def _create_retry_config(self) -> RetryConfig:
        """Create retry configuration from AppConfig. Override for custom behavior."""
        return RetryConfig(
            max_retries=self.config.api_max_retries,
            base_delay=self.config.api_retry_backoff_base,
            max_delay=self.config.api_retry_backoff_max,
            rate_limit_delay=self.config.api_rate_limit_retry_after,
            request_timeout=self.config.api_request_timeout,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        )

    @abstractmethod
    def _create_retry_handler(self) -> RetryHandler:
        """Create a retry handler instance. Must be implemented by subclasses."""
        pass

    def get_retry_handler(self) -> RetryHandler:
        """Get the retry handler for this service, creating if needed."""
        if self._retry_handler is None:
            self._retry_handler = self._create_retry_handler()
        return self._retry_handler

    def _is_fatal_error(self, error: Exception) -> bool:
        """Check if the exception is a fatal error that requires immediate intervention."""
        return False

    async def _extract_images_from_message(self, message: discord.Message) -> List[bytes]:
        """Extract image bytes from message attachments, downscaling to ~1MP."""
        images = []
        if not message.attachments:
            return images

        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    image_data = await attachment.read()

                    # Run CPU-bound image processing in a separate thread
                    processed_data = await asyncio.to_thread(
                        process_image_sync, image_data, attachment.filename
                    )
                    images.append(processed_data)

                except Exception:
                    logger.exception("Failed to read/process attachment %s", attachment.filename)

        return images

    async def _execute_model_call(self, model_call: Callable[[], Any | Awaitable[Any]]) -> Any:
        """Execute a model call, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(model_call):
            return await model_call()
        return await asyncio.to_thread(model_call)

    async def _wait_with_countdown(
        self, delay: float, discord_message: Optional[discord.Message]
    ) -> None:
        """Wait for a specified delay with a countdown message in Discord."""
        if delay <= 0:
            return

        sent_message: Optional[discord.Message] = None

        remaining = int(delay)
        if discord_message:
            try:
                sent_message = await discord_message.reply(
                    f"⏳ 소예봇 뇌 과부하! {remaining}초만 기다려 주세요.",
                    mention_author=False,
                )
            except discord.HTTPException:
                logger.warning("Failed to send rate limit message.")

        while remaining > 0:
            if remaining % 10 == 0 or remaining <= 3:
                countdown_message = f"⏳ 소예봇 뇌 과부하! {remaining}초만 기다려 주세요."
                if sent_message:
                    try:
                        await sent_message.edit(content=countdown_message)
                    except discord.HTTPException:
                        pass  # Ignore edit errors
            await asyncio.sleep(1)
            remaining -= 1

        if sent_message:
            try:
                await sent_message.delete()
            except discord.HTTPException:
                pass

    async def execute_with_retry(
        self,
        model_call: Callable[[], Any | Awaitable[Any]],
        error_prefix: str = "요청",
        return_full_response: bool = False,
        discord_message: Optional[discord.Message] = None,
        timeout: Optional[float] = None,
        fallback_call: Optional[Callable[[], Any | Awaitable[Any]]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Any]:
        """
        Execute API call with retries, logging, countdown notifications, and fallback logic.

        Uses the RetryHandler for retry logic with exponential backoff.
        """
        retry_handler = self.get_retry_handler()
        return await retry_handler.execute_with_retry(
            api_call=model_call,
            error_prefix=error_prefix,
            return_full_response=return_full_response,
            discord_message=discord_message,
            cancel_event=cancel_event,
            timeout=timeout,
            fallback_call=fallback_call,
            log_response=self._log_raw_response,
            extract_text=None if return_full_response else self._extract_text_from_response,
        )


class BaseLLMServiceCore(BaseLLMService):
    """Core LLM service with shared retry logic and utilities.

    Provides common methods for retry configuration, image extraction,
    and request/response logging that are shared across all providers.
    """

    def _create_retry_config_core(self) -> Any:
        """Create retry configuration from AppConfig.

        This method provides a common implementation that all providers
        can use or override for provider-specific behavior.
        """
        from persbot.services.retry_handler import (
            BackoffStrategy,
            RetryConfig,
        )

        return RetryConfig(
            max_retries=self.config.api_max_retries,
            base_delay=self.config.api_retry_backoff_base,
            max_delay=self.config.api_retry_backoff_max,
            rate_limit_delay=self.config.api_rate_limit_retry_after,
            request_timeout=self.config.api_request_timeout,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        )

    def _create_retry_config(self) -> Any:
        """Create retry configuration from AppConfig.

        Override this method in subclasses for custom retry config.
        """
        return self._create_retry_config_core()

    def _log_raw_request_core(
        self, user_message: str, chat_session: Any = None, prefix: str = "[RAW API REQUEST]"
    ) -> None:
        """Log raw API request data being sent (debug level only).

        Args:
            user_message: The user message being sent.
            chat_session: Optional chat session with history.
            prefix: Log prefix for the provider.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            from persbot.constants import DisplayConfig


            if chat_session and hasattr(chat_session, "history"):
                history = chat_session.history
                formatted_history = []
                for msg in history[-DisplayConfig.HISTORY_DISPLAY_LIMIT :]:
                    role = msg.role
                    # Handle different content formats
                    content = str(getattr(msg, "content", ""))
                    if hasattr(msg, "parts"):
                        texts = [part.get("text", "") for part in msg.parts]
                        content = " ".join(texts)

                    # Clean up content display if it starts with "Name: "
                    author_label = str(msg.author_name or msg.author_id or "bot")
                    display_content = content
                    if msg.author_name and content.startswith(f"{msg.author_name}:"):
                        display_content = content[len(msg.author_name) + 1 :].strip()

                    formatted_history.append(f"{role} (author:{author_label}) {display_content}")
        except Exception as e:
            logger.error(f"{prefix} Error logging raw request: {e}", exc_info=True)

    def _log_raw_response_core(
        self, response_obj: Any, attempt: int, prefix: str = "[RAW API RESPONSE]"
    ) -> None:
        """Log raw API response data for debugging.

        Args:
            response_obj: The response object to log.
            attempt: The attempt number for logging.
            prefix: Log prefix for the provider.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            pass  # Response logging removed
        except Exception as e:
            logger.error(f"{prefix} {attempt} Error logging raw response: {e}", exc_info=True)

    async def _extract_images_from_messages(
        self, discord_message: Union[discord.Message, List[discord.Message]]
    ) -> List[bytes]:
        """Extract image bytes from message(s), downscaling to ~1MP.

        Args:
            discord_message: Single Discord message or list of messages.

        Returns:
            List of processed image bytes.
        """
        images = []
        if isinstance(discord_message, list):
            for msg in discord_message:
                imgs = await self._extract_images_from_message(msg)
                images.extend(imgs)
        else:
            images = await self._extract_images_from_message(discord_message)
        return images

"""Base LLM Service for SoyeBot."""

import asyncio
import io
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import discord
from PIL import Image

from persbot.config import AppConfig
from persbot.utils import ERROR_API_TIMEOUT, ERROR_RATE_LIMIT, GENERIC_ERROR_MESSAGE

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

    def __init__(self, config: AppConfig):
        self.config = config

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
        discord_message: Union[discord.Message, List[discord.Message]],
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

    def _is_fatal_error(self, error: Exception) -> bool:
        """Check if the exception is a fatal error that requires immediate intervention."""
        return False

    def _process_image_sync(self, image_data: bytes, filename: str) -> bytes:
        """Process image synchronously with Pillow (CPU-bound)."""
        target_pixels = 1_000_000  # 1 Megapixel
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                # Check current dimensions
                width, height = img.size
                pixels = width * height

                if pixels > target_pixels:
                    # Calculate scaling factor
                    ratio = (target_pixels / pixels) ** 0.5
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)

                    logger.info(
                        f"Downscaling image {filename} from {width}x{height} to {new_width}x{new_height}"
                    )

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert to JPEG (compatible and efficient)
                output_buffer = io.BytesIO()
                # Convert to RGB if needed (e.g. RGBA -> RGB for JPEG)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                img.save(output_buffer, format="JPEG", quality=85)
                return output_buffer.getvalue()
        except Exception as img_err:
            logger.error(f"Failed to process image {filename}: {img_err}")
            # Fallback to original if processing fails
            return image_data

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
                        self._process_image_sync, image_data, attachment.filename
                    )
                    images.append(processed_data)

                except Exception as e:
                    logger.error(f"Failed to read/process attachment {attachment.filename}: {e}")

        return images

    async def _execute_model_call(
        self, model_call: Callable[[], Union[Any, Awaitable[Any]]]
    ) -> Any:
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

        logger.info("⏳ 레이트 제한 감지. %s초 대기 중...", int(delay))
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
                logger.info(countdown_message)
            await asyncio.sleep(1)
            remaining -= 1

        if sent_message:
            try:
                await sent_message.delete()
            except discord.HTTPException:
                pass

    async def execute_with_retry(
        self,
        model_call: Callable[[], Union[Any, Awaitable[Any]]],
        error_prefix: str = "요청",
        return_full_response: bool = False,
        discord_message: Optional[discord.Message] = None,
        timeout: Optional[float] = None,
        fallback_call: Optional[Callable[[], Union[Any, Awaitable[Any]]]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Any]:
        """
        Execute API call with retries, logging, countdown notifications, and fallback logic.
        """
        last_error: Optional[Exception] = None
        current_timeout = timeout if timeout is not None else self.config.api_request_timeout

        for attempt in range(1, self.config.api_max_retries + 1):
            try:
                # Check for cancellation before attempting API call
                if cancel_event and cancel_event.is_set():
                    logger.info("API call aborted due to cancellation signal before attempt")
                    raise asyncio.CancelledError("LLM API call aborted by user")

                response = await asyncio.wait_for(
                    self._execute_model_call(model_call),
                    timeout=current_timeout,
                )

                self._log_raw_response(response, attempt)

                if return_full_response:
                    return response
                return self._extract_text_from_response(response)

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError()
                logger.warning(
                    "%s API 타임아웃 (%s/%s)",
                    self.__class__.__name__,
                    attempt,
                    self.config.api_max_retries,
                )

                # Check for fallback on timeout? Usually timeout is temp, but maybe.
                # For now just retry.
                if attempt < self.config.api_max_retries:
                    logger.info("API 타임아웃, 재시도 중...")
                    continue
                break

            except Exception as e:
                last_error = e
                logger.error(
                    "%s API 에러 (%s/%s): %s",
                    self.__class__.__name__,
                    attempt,
                    self.config.api_max_retries,
                    e,
                    exc_info=True,
                )

                if self._is_rate_limit_error(e):
                    # Check if we should fallback
                    if fallback_call:
                        logger.info("Rate limit hit. Attempting fallback model...")
                        try:
                            # Execute fallback once without retry loop for simplicity, or we could recurse
                            # Assuming fallback is "lighter" and less likely to fail or different quota
                            response = await asyncio.wait_for(
                                self._execute_model_call(fallback_call), timeout=current_timeout
                            )
                            self._log_raw_response(response, attempt)
                            if return_full_response:
                                return response
                            return self._extract_text_from_response(response)
                        except Exception as fb_e:
                            logger.error(f"Fallback model failed: {fb_e}")
                            # Continue to normal wait logic if fallback fails

                    delay = self._extract_retry_delay(e) or self.config.api_rate_limit_retry_after
                    await self._wait_with_countdown(delay, discord_message)
                    continue

                if self._is_fatal_error(e):
                    logger.warning(
                        "%s encountered a fatal error. Re-throwing to allow recovery.",
                        self.__class__.__name__,
                    )
                    raise e

                if attempt >= self.config.api_max_retries:
                    break

                # Exponential backoff
                backoff = min(
                    self.config.api_retry_backoff_base**attempt,
                    self.config.api_retry_backoff_max,
                )
                logger.info("에러 발생, %.1f초 후 재시도", backoff)
                await asyncio.sleep(backoff)

        msg_content = GENERIC_ERROR_MESSAGE
        if isinstance(last_error, asyncio.TimeoutError):
            logger.error("❌ 에러: API 요청 시간 초과")
            msg_content = ERROR_API_TIMEOUT
        else:
            logger.error(
                "❌ 에러: 최대 재시도 횟수(%s)를 초과했습니다. (%s)",
                self.config.api_max_retries,
                error_prefix,
            )

        if discord_message:
            try:
                await discord_message.reply(msg_content, mention_author=False)
            except discord.HTTPException:
                pass

        return None

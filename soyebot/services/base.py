"""Base LLM Service for SoyeBot."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Union

import discord

from soyebot.config import AppConfig
from soyebot.metrics import get_metrics
from soyebot.utils import GENERIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single message in the chat history."""
    role: str
    content: str
    author_id: Optional[int] = None
    message_id: Optional[str] = None
    # For Gemini, content is stored in 'parts'
    parts: Optional[list[dict[str, str]]] = None

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

    async def _execute_model_call(self, model_call: Callable[[], Union[Any, Awaitable[Any]]]) -> Any:
        """Execute a model call, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(model_call):
            return await model_call()
        return await asyncio.to_thread(model_call)

    async def _wait_with_countdown(self, delay: float, discord_message: Optional[discord.Message]) -> None:
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
                countdown_message = (
                    f"⏳ 소예봇 뇌 과부하! {remaining}초만 기다려 주세요."
                )
                if sent_message:
                    try:
                        await sent_message.edit(content=countdown_message)
                    except discord.HTTPException:
                         pass # Ignore edit errors
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
    ) -> Optional[Any]:
        """
        Execute the API call with retries, logging, and countdown notifications.
        """
        metrics = get_metrics()
        request_start = time.perf_counter()
        metrics.increment_counter('api_requests_total')

        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.api_max_retries + 1):
            try:
                try:
                    response = await asyncio.wait_for(
                        self._execute_model_call(model_call),
                        timeout=self.config.api_request_timeout,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # Reraise to let outer try/except handle retry logic
                    raise e
                
                self._log_raw_response(response, attempt)

                duration_ms = (time.perf_counter() - request_start) * 1000
                metrics.record_latency('llm_api', duration_ms)
                metrics.increment_counter('api_requests_success')

                if return_full_response:
                    return response
                return self._extract_text_from_response(response)

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError()
                logger.warning("%s API 타임아웃 (%s/%s)", self.__class__.__name__, attempt, self.config.api_max_retries)
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
                    delay = self._extract_retry_delay(e) or self.config.api_rate_limit_retry_after
                    await self._wait_with_countdown(delay, discord_message)
                    continue

                if attempt >= self.config.api_max_retries:
                    break

                # Exponential backoff
                backoff = min(
                    self.config.api_retry_backoff_base ** attempt,
                    self.config.api_retry_backoff_max,
                )
                logger.info("에러 발생, %.1f초 후 재시도", backoff)
                await asyncio.sleep(backoff)

        if isinstance(last_error, asyncio.TimeoutError):
             logger.error("❌ 에러: API 요청 시간 초과")
        else:
            logger.error(
                "❌ 에러: 최대 재시도 횟수(%s)를 초과했습니다. (%s)",
                self.config.api_max_retries,
                error_prefix,
            )

        metrics.increment_counter('api_requests_error')

        if discord_message:
            try:
                await discord_message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)
            except discord.HTTPException:
                pass

        return None

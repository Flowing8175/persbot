"""Retry handler for API calls with configurable backoff strategies."""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import discord

from persbot.exceptions import APIException, FatalError, RateLimitException
from persbot.utils import ERROR_API_TIMEOUT, ERROR_RATE_LIMIT, GENERIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff calculation strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 2
    base_delay: float = 2.0
    max_delay: float = 32.0
    rate_limit_delay: int = 5
    request_timeout: float = 120.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL


class RetryHandler(ABC):
    """
    Abstract base class for handling API retries with configurable strategies.

    Subclasses must implement:
    - _is_rate_limit_error()
    - _is_fatal_error()
    - _extract_retry_delay() (optional, has default)
    """

    def __init__(self, config: RetryConfig):
        self.config = config
        self._retry_count = 0

    @abstractmethod
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the exception is a rate limit error."""
        pass

    def _is_fatal_error(self, error: Exception) -> bool:
        """Check if the exception is a fatal error that should stop retrying."""
        return isinstance(error, FatalError)

    def _extract_retry_delay(self, error: Exception) -> Optional[float]:
        """Extract retry delay from error message if available."""
        error_str = str(error)
        # Try common patterns for retry delay
        match = re.search(r"Please retry in ([0-9.]+)s", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"seconds:\s*(\d+)", error_str)
        if match:
            return float(match.group(1))
        return None

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff delay based on strategy and attempt number."""
        if self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            return min(self.config.base_delay**attempt, self.config.max_delay)
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            return min(self.config.base_delay * attempt, self.config.max_delay)
        else:  # FIXED
            return self.config.base_delay

    async def _wait_with_countdown(
        self,
        delay: float,
        discord_message: Optional[discord.Message] = None,
        cancel_event: Optional[asyncio.Event] = None,
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
                        pass
                logger.info(countdown_message)

            # Check for cancellation before sleep
            if cancel_event and cancel_event.is_set():
                logger.info("Rate limit wait aborted due to cancellation signal")
                if sent_message:
                    try:
                        await sent_message.delete()
                    except discord.HTTPException:
                        pass
                raise asyncio.CancelledError("LLM API call aborted by user")

            await asyncio.sleep(1)
            remaining -= 1

        if sent_message:
            try:
                await sent_message.delete()
            except discord.HTTPException:
                pass

    async def _execute_with_timeout(
        self,
        api_call: Callable[[], Union[Any, Awaitable[Any]]],
    ) -> Any:
        """Execute a call with timeout, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(api_call):
            return await api_call()
        return await asyncio.to_thread(api_call)

    async def execute_with_retry(
        self,
        api_call: Callable[[], Union[Any, Awaitable[Any]]],
        *,
        error_prefix: str = "요청",
        discord_message: Optional[discord.Message] = None,
        cancel_event: Optional[asyncio.Event] = None,
        log_response: Optional[Callable[[Any, int], None]] = None,
        extract_text: Optional[Callable[[Any], str]] = None,
        return_full_response: bool = False,
    ) -> Optional[Any]:
        """
        Execute API call with retries, logging, and countdown notifications.

        Args:
            api_call: The API function to call (sync or async).
            error_prefix: Prefix for error messages.
            discord_message: Discord message for error notifications.
            cancel_event: Event to check for cancellation.
            log_response: Optional callback to log responses (response, attempt).
            extract_text: Optional callback to extract text from response.
            return_full_response: If True, return full response object.

        Returns:
            The response text (or full response if return_full_response=True),
            or None if all retries fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            # Check for cancellation at the start of each retry iteration
            if cancel_event and cancel_event.is_set():
                logger.info("API call aborted due to cancellation signal at start of retry iteration %d", attempt)
                raise asyncio.CancelledError("LLM API call aborted by user")

            try:
                response = await asyncio.wait_for(
                    self._execute_with_timeout(api_call),
                    timeout=self.config.request_timeout,
                )

                if log_response:
                    log_response(response, attempt)

                if return_full_response:
                    return response
                if extract_text:
                    return extract_text(response)
                return response

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError()
                logger.warning(
                    "API 타임아웃 (%s/%s)",
                    attempt,
                    self.config.max_retries,
                )

                if attempt < self.config.max_retries:
                    logger.info("API 타임아웃, 재시도 중...")
                    continue
                break

            except Exception as e:
                last_error = e
                logger.error(
                    "API 에러 (%s/%s): %s",
                    attempt,
                    self.config.max_retries,
                    e,
                    exc_info=True,
                )

                if self._is_fatal_error(e):
                    logger.warning("Encountered fatal error. Re-throwing.")
                    raise e

                if self._is_rate_limit_error(e):
                    delay = self._extract_retry_delay(e) or self.config.rate_limit_delay
                    await self._wait_with_countdown(delay, discord_message, cancel_event)
                    continue

                if attempt >= self.config.max_retries:
                    break

                backoff = self._calculate_backoff(attempt)
                logger.info("에러 발생, %.1f초 후 재시도", backoff)

                # Check for cancellation before backoff sleep
                if cancel_event and cancel_event.is_set():
                    logger.info("Retry loop aborted due to cancellation signal during backoff")
                    raise asyncio.CancelledError("LLM API call aborted by user")

                await asyncio.sleep(backoff)

        # All retries failed - notify user
        await self._notify_final_error(last_error, discord_message)
        return None

    async def _notify_final_error(
        self,
        error: Optional[Exception],
        discord_message: Optional[discord.Message],
    ) -> None:
        """Log final error and notify user via Discord if message provided."""
        if isinstance(error, asyncio.TimeoutError):
            logger.error("❌ API 요청 시간 초과")
            msg_content = ERROR_API_TIMEOUT
        else:
            logger.error(
                "❌ 최대 재시도 횟수(%s)를 초과했습니다.",
                self.config.max_retries,
            )
            msg_content = GENERIC_ERROR_MESSAGE

        if discord_message:
            try:
                await discord_message.reply(msg_content, mention_author=False)
            except discord.HTTPException:
                pass


class GeminiRetryHandler(RetryHandler):
    """Retry handler specialized for Gemini API errors."""

    def _is_rate_limit_error(self, error: Exception) -> bool:
        error_str = str(error)
        # Avoid matching "rate" in generic error messages (e.g. "operate")
        # 400 errors should typically not be retried as rate limits unless "quota" is explicit.
        if "400" in error_str and "quota" not in error_str.lower():
            return False
        return (
            "429" in error_str
            or "quota" in error_str.lower()
            or "rate limit" in error_str.lower()
        )

    def _is_fatal_error(self, error: Exception) -> bool:
        """Check if the error is a fatal cache error."""
        error_str = str(error).lower()
        # "CachedContent not found" or "403 PERMISSION_DENIED" on a cached resource
        return "cachedcontent not found" in error_str or (
            "403" in error_str and "permission" in error_str
        )


class OpenAIRetryHandler(RetryHandler):
    """Retry handler specialized for OpenAI API errors."""

    def _is_rate_limit_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return (
            "429" in error_str
            or "rate_limit" in error_str
            or "rate limit" in error_str
            or "quota" in error_str
        )


class ZAIRetryHandler(RetryHandler):
    """Retry handler specialized for Z.AI API errors."""

    def _is_rate_limit_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return (
            "429" in error_str
            or "rate_limit" in error_str
            or "rate limit" in error_str
            or "quota" in error_str
        )

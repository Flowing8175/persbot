"""Local LLM service for SoyeBot using OpenAI-compatible API."""

import asyncio
import time
import re
import logging
from typing import Optional, Tuple, Any, List, Dict

import discord
from openai import AsyncOpenAI

from config import AppConfig
from prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT

logger = logging.getLogger(__name__)


class LocalLLMService:
    """Local LLM API with OpenAI-compatible endpoint management."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key="not-needed",  # Local endpoints don't require real API keys
            base_url=config.llm_endpoint_url,
        )
        logger.info(f"Local LLM 모델 '{config.model_name}' 로드 완료. 엔드포인트: {config.llm_endpoint_url}")

    def _is_rate_limit_error(self, error_str: str) -> bool:
        return (
            "429" in error_str
            or "quota" in error_str.lower()
            or "rate" in error_str.lower()
        )

    def _extract_retry_delay(self, error_str: str) -> Optional[float]:
        match = re.search(r"Please retry in ([0-9.]+)s", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"seconds:\s*(\d+)", error_str)
        if match:
            return float(match.group(1))
        return None

    def _log_raw_request(
        self, messages: List[Dict[str, str]], attempt: int = 1
    ) -> None:
        """Log raw API request data being sent.

        Args:
            messages: The messages list being sent
            attempt: The current attempt number
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug(
                f"[RAW API REQUEST {attempt}] Total messages: {len(messages)}"
            )
            for msg_idx, msg in enumerate(messages[-5:]):  # Log last 5 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content_preview = (
                    content[:100].replace("\n", " ")
                    if content
                    else "(empty)"
                )
                logger.debug(
                    f"[RAW API REQUEST {attempt}]   [{msg_idx}] {role}: {content_preview}... ({len(content)} chars)"
                )
            logger.debug(f"[RAW API REQUEST {attempt}] Raw request logging completed")
        except Exception as e:
            logger.error(f"[RAW API REQUEST] Error logging raw request: {e}", exc_info=True)

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw API response data for debugging.

        Args:
            response_obj: The response object from OpenAI API
            attempt: The current attempt number
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug(f"[RAW API RESPONSE {attempt}] Response type: {type(response_obj).__name__}")

            # Log choice
            if hasattr(response_obj, "choices") and response_obj.choices:
                logger.debug(f"[RAW API RESPONSE {attempt}] Choices count: {len(response_obj.choices)}")

                for choice_idx, choice in enumerate(response_obj.choices):
                    logger.debug(f"[RAW API RESPONSE {attempt}] Choice {choice_idx}:")

                    if hasattr(choice, "finish_reason"):
                        logger.debug(
                            f"[RAW API RESPONSE {attempt}]   Finish reason: {choice.finish_reason}"
                        )

                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        content = choice.message.content
                        content_preview = (
                            content[:100].replace("\n", " ") if content else "(empty)"
                        )
                        logger.debug(
                            f"[RAW API RESPONSE {attempt}]   Content: {content_preview}... ({len(content)} chars)"
                        )

            # Log usage data if available
            if hasattr(response_obj, "usage"):
                usage = response_obj.usage
                logger.debug(f"[RAW API RESPONSE {attempt}] Usage data:")
                if hasattr(usage, "prompt_tokens"):
                    logger.debug(
                        f"[RAW API RESPONSE {attempt}]   Prompt tokens: {usage.prompt_tokens}"
                    )
                if hasattr(usage, "completion_tokens"):
                    logger.debug(
                        f"[RAW API RESPONSE {attempt}]   Completion tokens: {usage.completion_tokens}"
                    )
                if hasattr(usage, "total_tokens"):
                    logger.debug(
                        f"[RAW API RESPONSE {attempt}]   Total tokens: {usage.total_tokens}"
                    )

            logger.debug(f"[RAW API RESPONSE {attempt}] Raw response logging completed")
        except Exception as e:
            logger.error(f"[RAW API RESPONSE {attempt}] Error logging raw response: {e}", exc_info=True)

    async def _api_request_with_retry(
        self,
        api_call,
        error_prefix: str = "요청",
        discord_message: Optional[discord.Message] = None,
    ) -> Optional[Any]:
        """재시도 및 에러 처리를 포함한 API 요청 래퍼

        Args:
            api_call: The async function to call
            error_prefix: Error message prefix
            discord_message: Optional Discord message for error reporting

        Returns:
            Response object or None
        """
        for attempt in range(1, self.config.api_max_retries + 1):
            try:
                logger.debug(
                    f"[API Request {attempt}/{self.config.api_max_retries}] Starting {error_prefix}..."
                )

                # Call the async function with timeout
                response = await asyncio.wait_for(
                    api_call(),
                    timeout=self.config.api_request_timeout,
                )

                logger.debug(f"[API Request {attempt}] Response received successfully")
                self._log_raw_response(response, attempt)

                return response
            except asyncio.TimeoutError:
                logger.warning(
                    f"로컬 LLM API 타임아웃 ({attempt}/{self.config.api_max_retries})"
                )
                if attempt < self.config.api_max_retries:
                    logger.info(f"API 타임아웃, 재시도 중...")
                    continue
                logger.error(f"❌ 에러: API 요청 시간 초과")
                if discord_message:
                    await discord_message.reply(
                        "❌ LLM API 요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
                        mention_author=False,
                    )
                return None
            except Exception as e:
                error_str = str(e)
                logger.error(
                    f"로컬 LLM API 에러 ({attempt}/{self.config.api_max_retries}): {e}",
                    exc_info=True,
                )

                if self._is_rate_limit_error(error_str):
                    delay = (
                        self._extract_retry_delay(error_str)
                        or self.config.api_rate_limit_retry_after
                    )
                    initial_message_content = (
                        "⏳ 소예봇 뇌 과부하! 조금만 기다려 주세요."
                    )
                    sent_message = None
                    if discord_message:
                        sent_message = await discord_message.reply(
                            initial_message_content, mention_author=False
                        )

                    for remaining in range(int(delay), 0, -1):
                        countdown_message = f"⏳ 소예봇 뇌 과부하! 조금만 기다려 주세요. ({remaining}초)"
                        if sent_message:
                            await sent_message.edit(content=countdown_message)
                        logger.info(countdown_message)
                        await asyncio.sleep(1)
                    if sent_message:
                        await sent_message.delete()
                    continue

                if attempt >= self.config.api_max_retries:
                    if discord_message:
                        await discord_message.reply(
                            f"❌ API 요청 중 오류가 발생했습니다: {error_str}. 잠시 후 다시 시도해주세요.",
                            mention_author=False,
                        )
                    break
                logger.info(f"에러 발생, 재시도 중...")
                await asyncio.sleep(2)

        logger.error(
            f"❌ 에러: 최대 재시도 횟수({self.config.api_max_retries})를 초과했습니다."
        )
        if discord_message:
            await discord_message.reply(
                f"❌ 최대 재시도 횟수({self.config.api_max_retries})를 초과하여 요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요.",
                mention_author=False,
            )
        return None

    async def summarize_text(self, text: str) -> Optional[str]:
        """Summarize text using the local LLM.

        Args:
            text: Text to summarize

        Returns:
            Summary text or None if failed
        """
        if not text.strip():
            logger.debug("Summarization requested for empty text")
            return "요약할 메시지가 없습니다."

        logger.info(f"Summarizing text ({len(text)} characters)...")

        prompt = f"Discord 대화 내용:\n{text}"
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]

        logger.debug(f"[RAW API REQUEST] Text to summarize:\n{text}")
        logger.debug(
            f"[RAW API REQUEST] Full prompt being sent ({len(prompt)} characters):\n{prompt}"
        )

        async def api_call():
            return await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.7,
            )

        response = await self._api_request_with_retry(api_call, "요약")

        if response is None:
            return None

        return response.choices[0].message.content.strip()

    async def generate_chat_response(
        self,
        chat_session: "LocalLLMChatSession",
        user_message: str,
        discord_message: discord.Message,
        tools: Optional[list] = None,
    ) -> Optional[Tuple[str, Optional[Any]]]:
        """Generate chat response with conversation history.

        Args:
            chat_session: LocalLLMChatSession object
            user_message: User message
            discord_message: Discord message object
            tools: Optional list of function calling tools (not yet supported)

        Returns:
            Tuple of (response_text, response_object) or None
        """
        logger.debug(
            f"Generating chat response - User message length: {len(user_message)}, Tools enabled: {tools is not None}"
        )

        # Add user message to history
        chat_session.add_user_message(user_message)
        messages = chat_session.get_messages_with_system_prompt()

        # Log raw request data
        self._log_raw_request(messages)

        async def api_call():
            return await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.7,
            )

        response_obj = await self._api_request_with_retry(
            api_call,
            "응답 생성",
            discord_message=discord_message,
        )

        if response_obj is None:
            # Remove the message we added since the API call failed
            chat_session.messages.pop()
            return None

        # Extract response text
        response_text = response_obj.choices[0].message.content.strip()

        # Add assistant response to history
        chat_session.add_assistant_message(response_text)

        return (response_text, response_obj)

    def parse_function_calls(self, response_obj) -> list:
        """Parse function calls from response (not yet implemented for local LLM).

        Args:
            response_obj: Response object

        Returns:
            List of function call dictionaries (empty for now)
        """
        # Function calling support can be added later
        return []


class LocalLLMChatSession:
    """Manages chat session with conversation history for local LLM."""

    def __init__(self, system_instruction: str):
        """Initialize chat session.

        Args:
            system_instruction: System prompt for the chat
        """
        self.system_instruction = system_instruction
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        """Add user message to history.

        Args:
            content: User message content
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to history.

        Args:
            content: Assistant message content
        """
        self.messages.append({"role": "assistant", "content": content})

    def get_messages_with_system_prompt(self) -> List[Dict[str, str]]:
        """Get messages with system prompt prepended.

        Returns:
            List of messages with system prompt at the beginning
        """
        return [
            {"role": "system", "content": self.system_instruction},
        ] + self.messages

    def clear(self) -> None:
        """Clear chat history."""
        self.messages = []

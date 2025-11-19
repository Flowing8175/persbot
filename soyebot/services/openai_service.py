"""OpenAI API service for SoyeBot."""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Deque, Optional, Tuple

import discord
from openai import OpenAI
from openai import RateLimitError

from config import AppConfig
from prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT
from metrics import get_metrics
from utils import GENERIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)


class AssistantChatSession:
    """Assistant API-backed chat session with a bounded context window."""

    def __init__(
        self,
        client: OpenAI,
        assistant_id: str,
        temperature: float,
        max_messages: int,
        service_tier: str,
        text_extractor,
    ):
        self._client = client
        self._assistant_id = assistant_id
        self._temperature = temperature
        self._max_messages = max_messages
        self._service_tier = service_tier
        self._text_extractor = text_extractor
        self._thread_id = self._create_thread()
        self._history: Deque[dict] = deque(maxlen=max_messages)

    def _create_thread(self) -> str:
        thread = self._client.beta.threads.create(
            extra_headers={"OpenAI-Beta": "assistants=v2,prompt-caching=1"}
        )
        return thread.id

    def get_history(self):
        return list(self._history)

    def _append_history(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def send_message(self, user_message: str):
        self._client.beta.threads.messages.create(
            thread_id=self._thread_id,
            role="user",
            content=user_message,
            extra_headers={"OpenAI-Beta": "assistants=v2,prompt-caching=1"},
        )
        self._append_history("user", user_message)

        run = self._client.beta.threads.runs.create_and_poll(
            thread_id=self._thread_id,
            assistant_id=self._assistant_id,
            temperature=self._temperature,
            truncation_strategy={"type": "last_messages", "last_messages": self._max_messages},
            extra_headers={"OpenAI-Beta": "assistants=v2,prompt-caching=1"},
            extra_body={"service_tier": self._service_tier},
        )

        messages = self._client.beta.threads.messages.list(
            thread_id=self._thread_id,
            run_id=run.id,
            order="desc",
            limit=1,
            extra_headers={"OpenAI-Beta": "assistants=v2,prompt-caching=1"},
        )

        message_content = ""
        for message in messages.data:
            if message.role == "assistant":
                message_content = self._text_extractor(message)
                self._append_history("assistant", message_content)
                break

        return message_content, run


class _AssistantModel:
    """Assistant wrapper mirroring the cached model API."""

    def __init__(
        self,
        client: OpenAI,
        assistant_id: str,
        temperature: float,
        max_messages: int,
        service_tier: str,
        text_extractor,
    ):
        self._client = client
        self._assistant_id = assistant_id
        self._temperature = temperature
        self._max_messages = max_messages
        self._service_tier = service_tier
        self._text_extractor = text_extractor

    def start_chat(self):
        return AssistantChatSession(
            self._client,
            self._assistant_id,
            self._temperature,
            self._max_messages,
            self._service_tier,
            self._text_extractor,
        )


class OpenAIService:
    """OpenAI API와의 모든 상호작용을 관리합니다."""

    def __init__(self, config: AppConfig, *, assistant_model_name: str, summary_model_name: Optional[str] = None):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self._assistant_cache: dict[int, _AssistantModel] = {}
        self._max_messages = 10
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self._assistant_id_override = getattr(config, 'openai_assistant_id', None)

        # Preload default assistant
        self.assistant_model = self._get_or_create_assistant(self._assistant_model_name, BOT_PERSONA_PROMPT)
        if self._assistant_id_override:
            logger.info("OpenAI Assistant ID '%s' 준비 완료.", self._assistant_id_override)
        else:
            logger.info("OpenAI Assistant '%s' 준비 완료.", self._assistant_model_name)

    def _get_or_create_assistant(self, model_name: str, system_instruction: str) -> _AssistantModel:
        if self._assistant_id_override:
            key = hash(("provided", self._assistant_id_override))
            if key not in self._assistant_cache:
                try:
                    self.client.beta.assistants.retrieve(
                        self._assistant_id_override,
                        extra_headers={"OpenAI-Beta": "assistants=v2,prompt-caching=1"},
                    )
                except Exception:
                    logger.exception("OPENAI_ASSISTANT_ID 확인 중 오류가 발생했습니다.")
                    raise

                self._assistant_cache[key] = _AssistantModel(
                    self.client,
                    self._assistant_id_override,
                    getattr(self.config, 'temperature', 1.0),
                    self._max_messages,
                    getattr(self.config, 'service_tier', 'flex'),
                    self._extract_text_from_message,
                )
            return self._assistant_cache[key]

        key = hash((model_name, system_instruction))
        if key not in self._assistant_cache:
            assistant = self.client.beta.assistants.create(
                model=model_name,
                instructions=system_instruction,
                temperature=getattr(self.config, 'temperature', 1.0),
                extra_headers={"OpenAI-Beta": "assistants=v2,prompt-caching=1"},
            )
            self._assistant_cache[key] = _AssistantModel(
                self.client,
                assistant.id,
                getattr(self.config, 'temperature', 1.0),
                self._max_messages,
                getattr(self.config, 'service_tier', 'flex'),
                self._extract_text_from_message,
            )
        return self._assistant_cache[key]

    def create_assistant_model(self, system_instruction: str) -> _AssistantModel:
        return self._get_or_create_assistant(self._assistant_model_name, system_instruction)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True
        error_str = str(error).lower()
        return 'rate limit' in error_str or '429' in error_str

    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug("[RAW API REQUEST] User message preview: %r", user_message[:200])
            if chat_session and hasattr(chat_session, 'get_history'):
                history = chat_session.get_history()
                formatted = []
                for msg in history[-5:]:
                    role = msg.get('role', 'unknown')
                    content = str(msg.get('content', ''))
                    truncated = content[:100].replace("\n", " ")
                    formatted.append(f"{role}: {truncated}")
                if formatted:
                    logger.debug("[RAW API REQUEST] Recent history:\n%s", "\n".join(formatted))
        except Exception:
            logger.exception("[RAW API REQUEST] Error logging raw request")

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug("[RAW API RESPONSE %s] %s", attempt, response_obj)
        except Exception:
            logger.exception("[RAW API RESPONSE %s] Error logging raw response", attempt)

    async def _api_request_with_retry(
        self,
        model_call,
        error_prefix: str = "요청",
        return_full_response: bool = False,
        discord_message: Optional[discord.Message] = None,
    ) -> Optional[Any]:
        metrics = get_metrics()
        request_start = time.perf_counter()
        metrics.increment_counter('api_requests_total')

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.api_max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(model_call):
                    response = await asyncio.wait_for(
                        model_call(), timeout=self.config.api_request_timeout
                    )
                else:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(model_call), timeout=self.config.api_request_timeout
                    )

                self._log_raw_response(response, attempt)

                duration_ms = (time.perf_counter() - request_start) * 1000
                metrics.record_latency('llm_api', duration_ms)
                metrics.increment_counter('api_requests_success')

                if return_full_response:
                    return response
                return self._extract_text_from_response(response)
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError()
                logger.warning("OpenAI API 타임아웃 (%s/%s)", attempt, self.config.api_max_retries)
                if attempt < self.config.api_max_retries:
                    logger.info("API 타임아웃, 재시도 중...")
                    continue
                break
            except Exception as e:
                last_error = e
                logger.error("OpenAI API 에러 (%s/%s): %s", attempt, self.config.api_max_retries, e, exc_info=True)

                if self._is_rate_limit_error(e):
                    delay = self.config.api_rate_limit_retry_after
                    logger.info("⏳ 레이트 제한 감지. %s초 대기 중...", int(delay))
                    sent_message = None
                    if discord_message:
                        sent_message = await discord_message.reply(
                            f"⏳ 소예봇 뇌 과부하! {int(delay)}초 기다려 주세요.",
                            mention_author=False,
                        )

                    remaining = int(delay)
                    while remaining > 0:
                        if remaining % 10 == 0 or remaining <= 3:
                            countdown = (
                                f"⏳ 소예봇 뇌 과부하! 조금만 기다려 주세요. ({remaining}초)"
                            )
                            if sent_message:
                                await sent_message.edit(content=countdown)
                            logger.info(countdown)
                        await asyncio.sleep(1)
                        remaining -= 1

                    if sent_message:
                        await sent_message.delete()
                    continue

                if attempt >= self.config.api_max_retries:
                    break

                backoff = min(
                    self.config.api_retry_backoff_base ** attempt,
                    self.config.api_retry_backoff_max,
                )
                logger.info("에러 발생, %.1f초 후 재시도", backoff)
                await asyncio.sleep(backoff)

        metrics.increment_counter('api_requests_error')
        if isinstance(last_error, asyncio.TimeoutError):
            logger.error("❌ 에러: API 요청 시간 초과")
        else:
            logger.error(
                "❌ 에러: 최대 재시도 횟수(%s)를 초과했습니다. (%s)",
                self.config.api_max_retries,
                error_prefix,
            )
        if discord_message:
            await discord_message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)
        return None

    def _extract_text_from_response(self, response_obj) -> str:
        try:
            choices = getattr(response_obj, 'choices', []) or []
            for choice in choices:
                message = getattr(choice, 'message', None)
                if message and getattr(message, 'content', None):
                    return str(message.content).strip()
            return ""
        except Exception:
            logger.exception("Failed to extract text from OpenAI response")
            return ""

    def _extract_text_from_message(self, message_obj) -> str:
        try:
            content_parts = getattr(message_obj, 'content', []) or []
            text_fragments = []
            for part in content_parts:
                text_block = getattr(part, 'text', None)
                if text_block and getattr(text_block, 'value', None):
                    text_fragments.append(str(text_block.value))
            return "\n".join(text_fragments).strip()
        except Exception:
            logger.exception("Failed to extract text from assistant message")
            return ""

    def _extract_structured_json(self, response_obj) -> Optional[dict]:
        try:
            content = self._extract_text_from_response(response_obj)
            if content:
                return json.loads(content)
        except Exception:
            logger.debug("Structured response text was not valid JSON")
        return None

    async def summarize_text(self, text: str) -> Optional[str]:
        if not text.strip():
            return "요약할 메시지가 없습니다."

        prompt = f"Discord 대화 내용:\n{text}"
        return await self._api_request_with_retry(
            lambda: self.client.chat.completions.create(
                model=self._summary_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SUMMARY_SYSTEM_INSTRUCTION,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=getattr(self.config, 'temperature', 1.0),
                service_tier=getattr(self.config, 'service_tier', 'flex'),
            ),
            "요약",
        )

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        discord_message: discord.Message,
    ) -> Optional[Tuple[str, Any]]:
        self._log_raw_request(user_message, chat_session)

        response_obj = await self._api_request_with_retry(
            lambda: chat_session.send_message(user_message),
            "응답 생성",
            return_full_response=True,
            discord_message=discord_message,
        )

        if response_obj is None:
            return None

        response_text, run = response_obj
        return response_text, run


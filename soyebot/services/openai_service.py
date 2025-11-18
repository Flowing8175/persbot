"""OpenAI API service for SoyeBot."""

import asyncio
import json
import logging
import time
from typing import Any, Optional, Tuple

import discord
from openai import OpenAI
from openai import RateLimitError

from config import AppConfig
from prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT
from metrics import get_metrics
from utils import GENERIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)


class OpenAIChatSession:
    """Lightweight chat session wrapper for OpenAI responses."""

    def __init__(self, client: OpenAI, model_name: str, system_instruction: str, temperature: float):
        self._client = client
        self._model_name = model_name
        self._temperature = temperature
        self._messages = [
            {"role": "system", "content": system_instruction or BOT_PERSONA_PROMPT}
        ]

    def send_message(self, user_message: str):
        self._messages.append({"role": "user", "content": user_message})
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=self._messages,
            temperature=self._temperature,
        )
        message_content = ""
        if response.choices and response.choices[0].message:
            message_content = response.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": message_content})
        return response

    def get_history(self):
        return self._messages


class _OpenAIModel:
    """Model wrapper to mirror Gemini's cached model API."""

    def __init__(self, client: OpenAI, model_name: str, system_instruction: str, temperature: float):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature

    def start_chat(self):
        return OpenAIChatSession(
            self._client,
            self._model_name,
            self._system_instruction,
            self._temperature,
        )


class OpenAIService:
    """OpenAI API와의 모든 상호작용을 관리합니다."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self._model_cache: dict[int, _OpenAIModel] = {}

        # Preload default models
        self.summary_model = self._get_or_create_model(self.config.model_name, SUMMARY_SYSTEM_INSTRUCTION)
        self.assistant_model = self._get_or_create_model(self.config.model_name, BOT_PERSONA_PROMPT)
        logger.info("OpenAI 모델 '%s' 로드 완료.", config.model_name)

    def _get_or_create_model(self, model_name: str, system_instruction: str) -> _OpenAIModel:
        key = hash(system_instruction)
        if key not in self._model_cache:
            self._model_cache[key] = _OpenAIModel(
                self.client,
                model_name,
                system_instruction,
                getattr(self.config, 'temperature', 1.0),
            )
        return self._model_cache[key]

    def create_assistant_model(self, system_instruction: str) -> _OpenAIModel:
        return self._get_or_create_model(self.config.model_name, system_instruction)

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
            choices = getattr(response_obj, 'choices', []) or []
            for idx, choice in enumerate(choices):
                message = getattr(choice, 'message', None)
                logger.debug(
                    "[RAW API RESPONSE %s] Choice %s finish_reason=%s", attempt, idx, getattr(choice, 'finish_reason', 'unknown')
                )
                if message and getattr(message, 'content', None):
                    logger.debug(
                        "[RAW API RESPONSE %s] Choice %s text: %s",
                        attempt,
                        idx,
                        str(message.content)[:200].replace('\n', ' '),
                    )
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
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=getattr(self.config, 'temperature', 1.0),
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

        response_text = self._extract_text_from_response(response_obj)
        return response_text, response_obj

    async def score_topic_similarity(self, text_a: str, text_b: str) -> Optional[float]:
        if not text_a.strip() or not text_b.strip():
            return 0.0

        prompt = (
            "You are scoring whether two short Discord messages belong to the same conversation topic. "
            "Return JSON that includes a similarity between 0 (unrelated) and 1 (identical topic) and a boolean same_topic. "
            "Scoring rubric: 1.0 for the same ask or paraphrase; ~0.75 for clear follow-ups on the same task; ~0.5 when the main subject is shared but the request shifts; ~0.25 for only weak lexical overlap; 0.0 for unrelated topics. "
            "Be decisive and return a single numeric similarity in the 0-1 range."
        )

        response_obj = await self._api_request_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.config.eval_model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Message A:\n{text_a}\n\nMessage B:\n{text_b}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            ),
            "세션 유사도 판정",
            return_full_response=True,
        )

        if response_obj is None:
            return None

        parsed = self._extract_structured_json(response_obj)
        if not parsed:
            return None

        similarity = parsed.get("similarity")
        if isinstance(similarity, str):
            try:
                similarity = float(similarity)
            except ValueError:
                similarity = None

        if isinstance(similarity, (int, float)):
            return max(0.0, min(1.0, float(similarity)))

        return None

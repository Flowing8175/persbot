"Gemini API service for SoyeBot."

import asyncio
import json
import time
import re
import logging
from typing import Optional, Tuple, Any

import discord
import google.genai as genai
from google.genai import types as genai_types

from config import AppConfig
from prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT
from metrics import get_metrics
from utils import GENERIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)


class _CachedModel:
    """Lightweight wrapper that mimics the old GenerativeModel interface."""

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        config: genai_types.GenerateContentConfig,
    ):
        self._client = client
        self._model_name = model_name
        self._config = config

    def generate_content(self, contents: str):
        return self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=self._config,
        )

    def start_chat(self):
        return self._client.chats.create(
            model=self._model_name,
            config=self._config,
        )


class GeminiService:
    """Gemini API와의 모든 상호작용을 관리합니다."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)

        # Cache wrapper instances keyed by system instruction hash
        self._model_cache: dict[int, _CachedModel] = {}

        # Pre-load default models using cache
        self.summary_model = self._get_or_create_model(
            config.model_name, SUMMARY_SYSTEM_INSTRUCTION
        )
        self.assistant_model = self._get_or_create_model(
            config.model_name, BOT_PERSONA_PROMPT
        )
        logger.info(f"Gemini 모델 '{config.model_name}' 로드 완료. (구성 캐시 활성화)")

    def _get_or_create_model(self, model_name: str, system_instruction: str) -> _CachedModel:
        """Get cached model instance or create new one.

        Args:
            model_name: Name of the model to use
            system_instruction: System instruction for the model

        Returns:
            Cached or newly created model wrapper instance
        """
        key = hash(system_instruction)
        if key not in self._model_cache:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=getattr(self.config, 'temperature', 1.0),
            )
            self._model_cache[key] = _CachedModel(self.client, model_name, config)
        else:
            return self._model_cache[key]

    def create_assistant_model(self, system_instruction: str) -> _CachedModel:
        """Create or retrieve a cached assistant model with custom system instruction.

        Args:
            system_instruction: Custom system instruction for the model

        Returns:
            Model wrapper instance with the custom instruction (cached if possible)
        """
        return self._get_or_create_model(self.config.model_name, system_instruction)

    def _is_rate_limit_error(self, error_str: str) -> bool:
        return ("429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower())

    def _extract_retry_delay(self, error_str: str) -> Optional[float]:
        match = re.search(r'Please retry in ([0-9.]+)s', error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'seconds:\s*(\d+)', error_str)
        if match:
            return float(match.group(1))
        return None

    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        """Log raw API request data being sent (debug level only)."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug(f"[RAW API REQUEST] User message preview: {user_message[:200]!r}")

            if chat_session and hasattr(chat_session, 'get_history'):
                history = chat_session.get_history()
                formatted_history = []
                for msg in history[-5:]:
                    role = getattr(msg, 'role', 'unknown')
                    parts = getattr(msg, 'parts', [])
                    texts = []
                    for part in parts:
                        text = getattr(part, 'text', '')
                        if text:
                            texts.append(text[:100].replace('\n', ' '))
                    formatted_history.append(f"{role}: {' '.join(texts)}")
                if formatted_history:
                    logger.debug("[RAW API REQUEST] Recent history:\n" + "\n".join(formatted_history))
        except Exception as e:
            logger.error(f"[RAW API REQUEST] Error logging raw request: {e}", exc_info=True)

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw API response data for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            if hasattr(response_obj, 'candidates') and response_obj.candidates:
                for idx, candidate in enumerate(response_obj.candidates):
                    finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                    logger.debug(f"[RAW API RESPONSE {attempt}] Candidate {idx} finish_reason={finish_reason}")
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        texts = []
                        for part in candidate.content.parts:
                            text = getattr(part, 'text', '')
                            if text:
                                texts.append(text[:200].replace('\n', ' '))
                        if texts:
                            logger.debug(f"[RAW API RESPONSE {attempt}] Candidate {idx} text: {' '.join(texts)}")

            metadata = getattr(response_obj, 'usage_metadata', None)
            if metadata:
                prompt_tokens = getattr(metadata, 'prompt_token_count', 'unknown')
                response_tokens = getattr(metadata, 'candidates_token_count', 'unknown')
                total_tokens = getattr(metadata, 'total_token_count', 'unknown')
                logger.debug(
                    f"[RAW API RESPONSE {attempt}] Token usage "
                    f"(prompt={prompt_tokens}, response={response_tokens}, total={total_tokens})"
                )
        except Exception as e:
            logger.error(f"[RAW API RESPONSE {attempt}] Error logging raw response: {e}", exc_info=True)

    async def _api_request_with_retry(
        self,
        model_call,
        error_prefix: str = "요청",
        return_full_response: bool = False,
        discord_message: Optional[discord.Message] = None,
    ) -> Optional[Any]:
        """재시도 및 에러 처리를 포함한 API 요청 래퍼

        Args:
            model_call: The function to call
            error_prefix: Error message prefix
            return_full_response: If True, return full response object; if False, return text only

        Returns:
            Response text (str) or full response object depending on return_full_response flag
        """
        metrics = get_metrics()
        request_start = time.perf_counter()
        metrics.increment_counter('api_requests_total')

        error_notified = False
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.api_max_retries + 1):
            try:

                # If the call is async, await it directly; otherwise run the blocking call off the loop
                if asyncio.iscoroutinefunction(model_call):
                    response = await asyncio.wait_for(
                        model_call(),
                        timeout=self.config.api_request_timeout,
                    )
                else:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(model_call),
                        timeout=self.config.api_request_timeout,
                    )

                self._log_raw_response(response, attempt)

                # Track successful API request
                duration_ms = (time.perf_counter() - request_start) * 1000
                metrics.record_latency('gemini_api', duration_ms)
                metrics.increment_counter('api_requests_success')

                if return_full_response:
                    return response
                else:
                    return response.text.strip()
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError()
                logger.warning(f"Gemini API 타임아웃 ({attempt}/{self.config.api_max_retries})")
                if attempt < self.config.api_max_retries:
                    logger.info(f"API 타임아웃, 재시도 중...")
                    continue
                break
            except Exception as e:
                error_str = str(e)
                logger.error(f"Gemini API 에러 ({attempt}/{self.config.api_max_retries}): {e}", exc_info=True)
                last_error = e

                if self._is_rate_limit_error(error_str):
                    delay = self._extract_retry_delay(error_str) or self.config.api_rate_limit_retry_after
                    logger.info(f"⏳ 레이트 제한 감지. {int(delay)}초 대기 중...")
                    sent_message = None
                    if discord_message:
                        sent_message = await discord_message.reply(
                            f"⏳ 소예봇 뇌 과부하! {int(delay)}초 기다려 주세요.",
                            mention_author=False
                        )

                    # Wait for the delay, updating Discord only every 10 seconds or final 3 seconds
                    remaining = int(delay)
                    while remaining > 0:
                        if remaining % 10 == 0 or remaining <= 3:
                            countdown_message = f"⏳ 소예봇 뇌 과부하! 조금만 기다려 주세요. ({remaining}초)"
                            if sent_message:
                                await sent_message.edit(content=countdown_message)
                            logger.info(countdown_message)
                        await asyncio.sleep(1)
                        remaining -= 1

                    if sent_message:
                        await sent_message.delete()  # Delete the countdown message after it finishes
                    continue

                if attempt >= self.config.api_max_retries:
                    break
                logger.info(f"에러 발생, 재시도 중...")
                await asyncio.sleep(2)

        if isinstance(last_error, asyncio.TimeoutError):
            logger.error("❌ 에러: API 요청 시간 초과")
        else:
            logger.error(f"❌ 에러: 최대 재시도 횟수({self.config.api_max_retries})를 초과했습니다.")
        metrics.increment_counter('api_requests_error')
        if discord_message and not error_notified:
            await discord_message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)
            error_notified = True
        return None

    async def summarize_text(self, text: str) -> Optional[str]:
        if not text.strip():
            return "요약할 메시지가 없습니다."
        logger.info(f"Summarizing text ({len(text)} characters)...")
        prompt = f"Discord 대화 내용:\n{text}"
        return await self._api_request_with_retry(
            lambda: self.summary_model.generate_content(prompt),
            "요약"
        )

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        discord_message: discord.Message,
    ) -> Optional[Tuple[str, Any]]:
        """Generate chat response."""

        # Log raw request data
        self._log_raw_request(user_message, chat_session)

        def api_call():
            return chat_session.send_message(user_message)

        response_obj = await self._api_request_with_retry(
            api_call,
            "응답 생성",
            return_full_response=True,
            discord_message=discord_message,
        )

        if response_obj is None:
            return None

        # Extract text from response
        response_text = self._extract_text_from_response(response_obj)
        return (response_text, response_obj)

    def _extract_text_from_response(self, response_obj) -> str:
        """Extract text content from Gemini response.

        Args:
            response_obj: Gemini response object

        Returns:
            Extracted text from response, or empty string if no text parts exist
        """
        try:
            text_parts = []
            if hasattr(response_obj, 'candidates') and response_obj.candidates:
                for candidate in response_obj.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)

            if text_parts:
                combined = ' '.join(text_parts).strip()
                return combined

            return ""

        except Exception as e:
            logger.error(f"Failed to extract text from response: {e}", exc_info=True)
            return ""

    def _extract_structured_json(self, response_obj) -> Optional[dict]:
        """Extract JSON payload from a structured Gemini response."""

        try:
            if not hasattr(response_obj, "candidates") or not response_obj.candidates:
                return None

            for candidate in response_obj.candidates:
                content = getattr(candidate, "content", None)
                if not content or not hasattr(content, "parts"):
                    continue

                for part in content.parts:
                    text = getattr(part, "text", "")
                    if text:
                        try:
                            return json.loads(text)
                        except Exception:
                            logger.debug("Structured response text was not valid JSON: %s", text)

                    func_call = getattr(part, "function_call", None)
                    if func_call:
                        args = getattr(func_call, "args", None)
                        if isinstance(args, dict):
                            return args
                        args_json = getattr(func_call, "args_json", None)
                        if isinstance(args_json, str):
                            try:
                                return json.loads(args_json)
                            except Exception:
                                logger.debug("Function call args_json not parseable: %s", args_json)

                    inline = getattr(part, "inline_data", None)
                    if inline and hasattr(inline, "data"):
                        data = getattr(inline, "data")
                        if data:
                            try:
                                return json.loads(data)
                            except Exception:
                                logger.debug("Inline data not parseable as JSON")

        except Exception:
            logger.exception("Failed to parse structured Gemini response")
        return None

    async def score_topic_similarity(self, text_a: str, text_b: str) -> Optional[float]:
        """Score semantic similarity between two short Discord messages using structured output."""

        if not text_a.strip() or not text_b.strip():
            return 0.0

        schema = genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "similarity": genai_types.Schema(
                    type=genai_types.Type.NUMBER,
                    description="Semantic similarity between 0 and 1",
                ),
                "same_topic": genai_types.Schema(
                    type=genai_types.Type.BOOLEAN,
                    description="True when both messages are about the same topic",
                ),
                "reason": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Short explanation",
                ),
            },
            required=["similarity", "same_topic"],
        )

        prompt = (
            "You are scoring whether two short Discord messages belong to the same conversation topic. "
            "Return JSON that includes a similarity between 0 (unrelated) and 1 (identical topic) and a boolean same_topic. "
            "Scoring rubric: 1.0 for the same ask or paraphrase; ~0.75 for clear follow-ups on the same task; ~0.5 when the main subject is shared but the request shifts; ~0.25 for only weak lexical overlap; 0.0 for unrelated topics. "
            "Guidelines: focus on intent, subject matter, and entities (project, ticket, feature), not tone or emoji. Scheduling vs reminder about the same release is similar; release talk vs API error is not. "
            "Examples: "
            "A: '오늘 저녁에 뭐 먹을까?' / B: '파스타 어때?' => ~0.8-0.9, same_topic true. "
            "A: '주말에 등산 갈래?' / B: '토요일 오전에 시간 괜찮아' => ~0.75-0.85, same_topic true. "
            "A: '이번 주말 영화 볼래?' / B: '다음 주에 친구 결혼식 있어' => ~0.2, same_topic false. "
            "A: '우산 챙겼어?' / B: '어제 본 드라마 재밌더라' => 0.0, same_topic false. "
            "Be decisive and return a single numeric similarity in the 0-1 range."
        )

        def api_call():
            return self.client.models.generate_content(
                model=self.config.eval_model_name,
                contents=[
                    prompt,
                    f"Message A:\n{text_a}\n\nMessage B:\n{text_b}",
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0,
                    response_schema=schema,
                    response_mime_type="application/json",
                ),
            )

        response_obj = await self._api_request_with_retry(
            api_call,
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

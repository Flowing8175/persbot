"Gemini API service for SoyeBot."

import json
import logging
import re
from typing import Any, Optional, Tuple

import discord
import google.genai as genai
from google.genai import types as genai_types

from config import AppConfig
from prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT
from services.base import BaseLLMService

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


class GeminiService(BaseLLMService):
    """Gemini API와의 모든 상호작용을 관리합니다."""

    def __init__(self, config: AppConfig, *, assistant_model_name: str, summary_model_name: Optional[str] = None):
        super().__init__(config)
        self.client = genai.Client(api_key=config.gemini_api_key)
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name

        # Cache wrapper instances keyed by system instruction hash
        self._model_cache: dict[int, _CachedModel] = {}

        # Pre-load default models using cache
        self.summary_model = self._get_or_create_model(
            self._summary_model_name, SUMMARY_SYSTEM_INSTRUCTION
        )
        self.assistant_model = self._get_or_create_model(
            self._assistant_model_name, BOT_PERSONA_PROMPT
        )
        logger.info(
            "Gemini 모델 assistant='%s', summary='%s' 로드 완료. (구성 캐시 활성화)",
            self._assistant_model_name,
            self._summary_model_name,
        )

    def _get_or_create_model(self, model_name: str, system_instruction: str) -> _CachedModel:
        """Get cached model instance or create new one."""
        key = hash((model_name, system_instruction))
        if key not in self._model_cache:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=getattr(self.config, 'temperature', 1.0),
                top_p=getattr(self.config, 'top_p', 1.0),
            )
            self._model_cache[key] = _CachedModel(self.client, model_name, config)
        return self._model_cache[key]

    def create_assistant_model(self, system_instruction: str) -> _CachedModel:
        """Create or retrieve a cached assistant model with custom system instruction."""
        return self._get_or_create_model(self._assistant_model_name, system_instruction)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        error_str = str(error)
        return "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower()

    def _extract_retry_delay(self, error: Exception) -> Optional[float]:
        error_str = str(error)
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

    def _extract_text_from_response(self, response_obj: Any) -> str:
        """Extract text content from Gemini response."""
        try:
            text_parts = []
            if hasattr(response_obj, 'candidates') and response_obj.candidates:
                for candidate in response_obj.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)

            if text_parts:
                return ' '.join(text_parts).strip()

            return ""

        except Exception as e:
            logger.error(f"Failed to extract text from response: {e}", exc_info=True)
            return ""

    async def summarize_text(self, text: str) -> Optional[str]:
        if not text.strip():
            return "요약할 메시지가 없습니다."
        logger.info(f"Summarizing text ({len(text)} characters)...")
        prompt = f"Discord 대화 내용:\n{text}"
        return await self.execute_with_retry(
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
        self._log_raw_request(user_message, chat_session)

        def api_call():
            return chat_session.send_message(user_message)

        response_obj = await self.execute_with_retry(
            api_call,
            "응답 생성",
            return_full_response=True,
            discord_message=discord_message,
        )

        if response_obj is None:
            return None

        response_text = self._extract_text_from_response(response_obj)
        return (response_text, response_obj)


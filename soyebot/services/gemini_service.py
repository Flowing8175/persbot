"Gemini API service for SoyeBot."

import datetime
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import discord
import google.genai as genai
from google.genai import types as genai_types
from google.genai.errors import ClientError

from soyebot.config import AppConfig
from soyebot.prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT
from soyebot.services.base import BaseLLMService, ChatMessage

logger = logging.getLogger(__name__)


class _ChatSession:
    """A wrapper for a Gemini chat session to manage history with author tracking."""
    def __init__(self, underlying_chat):
        self._chat = underlying_chat
        # We will manage the history manually to include author_id
        self.history: list[ChatMessage] = []

    def generate_response(self, user_message: str):
        """Generate a response WITHOUT updating history immediately."""
        # Reconstruct history for the API call
        api_history = []
        for msg in self.history:
            api_history.append({"role": msg.role, "parts": msg.parts})

        self._chat.history = api_history
        response = self._chat.send_message(user_message)
        return response

    def add_exchange(self, user_message: str, response_text: str, author_id: int, message_id: Optional[str] = None):
        """Manually add the user message and response to history."""
        # User message
        self.history.append(ChatMessage(
            role="user",
            content=user_message,
            parts=[{"text": user_message}],
            author_id=author_id,
            message_id=message_id
        ))
        # Model response
        self.history.append(ChatMessage(
            role="model",
            content=response_text,
            parts=[{"text": response_text}],
            author_id=None
        ))
        
        # Sync underlying history (for next call)
        self._chat.history = [
            {"role": msg.role, "parts": msg.parts} for msg in self.history
        ]


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
        underlying_chat = self._client.chats.create(
            model=self._model_name,
            config=self._config,
        )
        return _ChatSession(underlying_chat)


class GeminiService(BaseLLMService):
    """Gemini API와의 모든 상호작용을 관리합니다."""

    def __init__(self, config: AppConfig, *, assistant_model_name: str, summary_model_name: Optional[str] = None):
        super().__init__(config)
        self.client = genai.Client(api_key=config.gemini_api_key)
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name

        # Cache wrapper instances keyed by system instruction hash
        # Stores tuple: (model_wrapper, expiration_time: Optional[datetime.datetime])
        self._model_cache: dict[int, Tuple[_CachedModel, Optional[datetime.datetime]]] = {}

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
        now = datetime.datetime.now(datetime.timezone.utc)

        # Check existing cache validity
        if key in self._model_cache:
            model, expires_at = self._model_cache[key]
            if expires_at and now >= expires_at:
                logger.info("Cached model expired (TTL reached). Refreshing...")
                del self._model_cache[key]
            else:
                return model

        # Check for explicit caching (for large prompts)
        cached_content_name = self._ensure_cached_content(model_name, system_instruction)

        config_kwargs = {
            "temperature": getattr(self.config, 'temperature', 1.0),
            "top_p": getattr(self.config, 'top_p', 1.0),
        }

        expires_at = None
        if cached_content_name:
            # If using cache, system_instruction is already embedded in the cache resource
            config_kwargs["cached_content"] = cached_content_name

            # Set local expiration slightly earlier than the actual TTL to ensure we refresh safely
            ttl_minutes = getattr(self.config, 'gemini_cache_ttl_minutes', 60)
            # Use 90% of TTL or subtract a buffer (e.g. 5 mins) to be safe
            buffer_minutes = 5
            safe_ttl = max(1, ttl_minutes - buffer_minutes)
            expires_at = now + datetime.timedelta(minutes=safe_ttl)
            expires_at_logger = expires_at.strftime('%Y-%m-%d %H:%M:%S')
            logger.debug("Local cache set to expire at %s", expires_at_logger)
        else:
            # Standard mode: pass system_instruction directly
            config_kwargs["system_instruction"] = system_instruction

        config = genai_types.GenerateContentConfig(**config_kwargs)
        model = _CachedModel(self.client, model_name, config)

        self._model_cache[key] = (model, expires_at)
        return model

    def create_assistant_model(self, system_instruction: str) -> _CachedModel:
        """Create or retrieve a cached assistant model with custom system instruction."""
        return self._get_or_create_model(self._assistant_model_name, system_instruction)

    def reload_parameters(self) -> None:
        """Reload parameters by clearing the model cache."""
        self._model_cache.clear()
        logger.info("Gemini model cache cleared to apply new parameters.")

    def get_user_role_name(self) -> str:
        """Return the role name for user messages."""
        return "user"

    def get_assistant_role_name(self) -> str:
        """Return the role name for assistant messages."""
        return "model"

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

            if chat_session and hasattr(chat_session, 'history'):
                history = chat_session.history
                formatted_history = []
                for msg in history[-5:]:
                    role = msg.role
                    texts = [part.get('text', '') for part in msg.parts]
                    formatted_history.append(f"{role} (author: {msg.author_id}): {' '.join(texts)}")
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

    def _get_cache_key(self, content: str) -> str:
        """Generate a consistent cache key/name based on content hash."""
        # Using a fixed prefix and hash to ensure we can find it again
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"soyebot-persona-{content_hash[:10]}"

    def _ensure_cached_content(self, model_name: str, system_instruction: str) -> Optional[str]:
        """
        Check if system instruction qualifies for caching and return cache name.
        Creates the cache if it doesn't exist.
        """
        if not system_instruction:
            return None

        # 1. Check token count
        try:
            count_result = self.client.models.count_tokens(
                model=model_name,
                contents=system_instruction
            )
            token_count = count_result.total_tokens
        except Exception as e:
            logger.warning("Failed to count tokens for caching check: %s", e)
            return None

        min_tokens = getattr(self.config, 'gemini_cache_min_tokens', 32768)
        if token_count < min_tokens:
            logger.debug(
                "System instruction too short for caching (%d < %d tokens). Using standard context.",
                token_count, min_tokens
            )
            return None

        # 2. Prepare cache configuration
        cache_display_name = self._get_cache_key(system_instruction)
        ttl_minutes = getattr(self.config, 'gemini_cache_ttl_minutes', 60)
        ttl_seconds = ttl_minutes * 60

        # 3. Search for existing cache
        # We need to iterate because list() returns an iterator
        try:
            # Note: In a production environment with many caches, filtering by name is better.
            # But the SDK might not support filtering by display_name easily in list().
            # We'll iterate through recent caches or just try to create and catch error?
            # Creating with same name is not unique. Resource names are unique IDs.
            # We'll use display_name to find ours.
            for cache in self.client.caches.list():
                if cache.display_name == cache_display_name:
                    logger.info("Found existing Gemini context cache: %s (%s)", cache.name, cache.display_name)

                    # Refresh the TTL of the existing cache to match our new local session
                    try:
                        self.client.caches.update(
                            name=cache.name,
                            config=genai_types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s")
                        )
                        logger.info("Refreshed TTL for cache: %s", cache.name)
                    except Exception as update_err:
                        logger.warning("Failed to refresh TTL for cache %s: %s", cache.name, update_err)
                        # If update fails, we still return the name, risking expiry.
                        # Ideally we might want to delete and recreate, but let's assume it's transient
                        pass

                    return cache.name
        except Exception as e:
            logger.warning("Error listing caches: %s", e)

        # 4. Create new cache
        try:
            logger.info(
                "Creating new Gemini context cache '%s' (%d tokens, TTL %dm)",
                cache_display_name, token_count, ttl_minutes
            )

            # Use 'contents' list wrapper for system instruction
            contents = [genai_types.Content(parts=[genai_types.Part(text=system_instruction)])]

            cached_content = self.client.caches.create(
                model=model_name,
                config=genai_types.CreateCachedContentConfig(
                    display_name=cache_display_name,
                    contents=contents,
                    ttl=f"{ttl_seconds}s"
                )
            )
            logger.info("Successfully created cache: %s", cached_content.name)
            return cached_content.name
        except Exception as e:
            logger.error("Failed to create Gemini context cache: %s", e)
            return None

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

        # Remove these lines as they are no longer used here
        # author_id = discord_message.author.id
        # message_id = str(discord_message.id)

        def api_call():
             # Call the new generate_response, NOT send_message
            return chat_session.generate_response(user_message)

        response_obj = await self.execute_with_retry(
            api_call,
            "응답 생성",
            return_full_response=True,
            discord_message=discord_message,
        )

        if response_obj is None:
            return None
        
        # NOTE: Only update history if we successfully got a response (not None)
        # This prevents "cancelled" requests (which return None due to exception)
        # from polluting the history.
        response_text = self._extract_text_from_response(response_obj)
        
        chat_session.add_exchange(
            user_message=user_message,
            response_text=response_text,
            author_id=discord_message.author.id,
            message_id=str(discord_message.id)
        )

        return (response_text, response_obj)


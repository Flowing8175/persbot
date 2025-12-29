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


def extract_clean_text(response_obj: Any) -> str:
    """Extract text content from Gemini response, filtering out thoughts."""
    try:
        text_parts = []
        if hasattr(response_obj, 'candidates') and response_obj.candidates:
            for candidate in response_obj.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # Skip parts that are marked as thoughts
                        if getattr(part, 'thought', False):
                            continue

                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)

        if text_parts:
            return ' '.join(text_parts).strip()

        return ""

    except Exception as e:
        logger.error(f"Failed to extract text from response: {e}", exc_info=True)
        return ""


class _ChatSession:
    """A wrapper for a Gemini chat session to manage history with author tracking."""
    def __init__(self, underlying_chat):
        self._chat = underlying_chat
        # We will manage the history manually to include author_id
        self.history: list[ChatMessage] = []

    def send_message(self, user_message: str, author_id: int, author_name: Optional[str] = None, message_id: Optional[str] = None):
        # Reconstruct history for the API call
        api_history = []
        for msg in self.history:
            api_history.append({"role": msg.role, "parts": msg.parts})

        self._chat.history = api_history

        response = self._chat.send_message(user_message)

        # After sending, save the user message and response to our custom history
        self.history.append(ChatMessage(
            role="user",
            content=user_message,
            parts=[{"text": user_message}],
            author_id=author_id,
            author_name=author_name,
            message_ids=[message_id] if message_id else []
        ))
        # Assuming the response text is in response.text
        # We need to ensure we don't store thoughts in history, so we use our helper
        # to extract only the text content (which filters out thoughts).
        clean_content = extract_clean_text(response)

        self.history.append(ChatMessage(
            role="model",
            content=clean_content,
            parts=[{"text": clean_content}],
            author_id=None # Bot messages have no author
        ))

        # Keep the underlying history in sync, though we are the source of truth
        self._chat.history = [
            {"role": msg.role, "parts": msg.parts} for msg in self.history
        ]

        return response

    def sync_history(self) -> None:
        """Force sync the underlying chat history with our local history."""
        try:
            api_history = []
            for msg in self.history:
                api_history.append({"role": msg.role, "parts": msg.parts})
            self._chat.history = api_history
            logger.debug("Synced Gemini chat history. Length: %d", len(self.history))
        except Exception as e:
            logger.error("Failed to sync Gemini history: %s", e)

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

        # Check logic for caching (token count check)
        # If valid for caching, this returns the cache name (resource ID)
        cache_name, cache_expiration = self._get_gemini_cache(model_name, system_instruction)

        config_kwargs = {
            "temperature": getattr(self.config, 'temperature', 1.0),
            "top_p": getattr(self.config, 'top_p', 1.0),
        }

        if cache_name:
            # Use the cached content
            config_kwargs["cached_content"] = cache_name
            logger.debug("Using cached content: %s", cache_name)
        else:
            # Standard mode: pass system_instruction directly
            config_kwargs["system_instruction"] = system_instruction

        if getattr(self.config, 'thinking_budget', None):
            thinking_budget_val = self.config.thinking_budget
            # If set to -1 (auto), we don't pass thinking_budget,
            # effectively letting the model decide (standard behavior for dynamic budget if supported),
            # OR we pass it if the API supports -1 explicitly.
            # Based on docs: "To use dynamic budget through the API, set thinking_budget to -1."
            # However, type checking might complain if not handled carefully,
            # but usually it's an int.
            # If -1 causes issues, we might need to omit it.
            # Let's assume -1 is valid as per docs.

            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=thinking_budget_val
            )

        config = genai_types.GenerateContentConfig(**config_kwargs)
        model = _CachedModel(self.client, model_name, config)

        self._model_cache[key] = (model, cache_expiration)
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
                    content = ' '.join(texts)
                    
                    # Clean up content display if it starts with "Name: "
                    author_label = str(msg.author_name or msg.author_id or "bot")
                    display_content = content
                    if msg.author_name and content.startswith(f"{msg.author_name}:"):
                        display_content = content[len(msg.author_name)+1:].strip()
                    
                    formatted_history.append(f"{role} (author:{author_label}) {display_content}")
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

    def _get_gemini_cache(self, model_name: str, system_instruction: str) -> Tuple[Optional[str], Optional[datetime.datetime]]:
        """
        Attempts to find or create a Gemini cache for the given system instruction.
        Returns: (cache_name, local_expiration_datetime)
        """
        if not system_instruction:
            return None, None

        # 1. Check token count
        try:
            count_result = self.client.models.count_tokens(
                model=model_name,
                contents=system_instruction
            )
            token_count = count_result.total_tokens
        except Exception as e:
            logger.warning("Failed to count tokens for caching check: %s", e)
            return None, None

        min_tokens = getattr(self.config, 'gemini_cache_min_tokens', 32768)
        if token_count < min_tokens:
            logger.debug(
                "System instruction too short for caching (%d < %d tokens). Using standard context.",
                token_count, min_tokens
            )
            return None, None

        # 2. Config setup
        cache_display_name = self._get_cache_key(system_instruction)
        ttl_minutes = getattr(self.config, 'gemini_cache_ttl_minutes', 60)
        ttl_seconds = ttl_minutes * 60
        
        # Calculate local expiration (safe buffer)
        now = datetime.datetime.now(datetime.timezone.utc)
        safe_ttl_minutes = max(1, ttl_minutes - 5) # Buffer 5 minutes
        local_expiration = now + datetime.timedelta(minutes=safe_ttl_minutes)

        # 3. Search for existing cache
        try:
            # We iterate to find a cache with our unique display name
            for cache in self.client.caches.list():
                if cache.display_name == cache_display_name:
                    logger.info("Found existing Gemini context cache: %s", cache.name)
                    
                    # Refresh TTL
                    try:
                        self.client.caches.update(
                            name=cache.name,
                            config=genai_types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s")
                        )
                        logger.info("Refreshed TTL for cache: %s", cache.name)
                    except Exception as update_err:
                        logger.warning("Failed to refresh TTL: %s", update_err)
                    
                    return cache.name, local_expiration

        except Exception as e:
            logger.warning("Error listing caches: %s", e)

        # 4. Create new cache using the user's requested style
        try:
            logger.info("Creating new Gemini cache '%s' (TTL: %ds)...", cache_display_name, ttl_seconds)
            
            cache = self.client.caches.create(
                model=model_name,
                config=genai_types.CreateCachedContentConfig(
                    display_name=cache_display_name,
                    system_instruction=system_instruction,
                    ttl=f"{ttl_seconds}s",
                )
            )
            
            logger.info("Successfully created cache: %s", cache.name)
            return cache.name, local_expiration

        except Exception as e:
            logger.error("Failed to create Gemini context cache: %s", e)
            return None, None

    def _extract_text_from_response(self, response_obj: Any) -> str:
        return extract_clean_text(response_obj)

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

        author_id = discord_message.author.id
        author_name = getattr(discord_message.author, 'name', str(author_id))
        message_id = str(discord_message.id)
 
        def api_call():
            return chat_session.send_message(user_message, author_id=author_id, author_name=author_name, message_id=message_id)

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


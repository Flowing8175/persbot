"Gemini API service for SoyeBot."

import time
import datetime
import hashlib
import json
import logging
import asyncio
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Callable, Awaitable

import discord
import google.genai as genai
from google.genai import types as genai_types
from google.genai.errors import ClientError

from soyebot.config import AppConfig
from soyebot.services.base import BaseLLMService, ChatMessage
from soyebot.services.prompt_service import PromptService
from soyebot.utils import GENERIC_ERROR_MESSAGE

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
    def __init__(self, system_instruction: str, factory: '_CachedModel'):
        self._system_instruction = system_instruction
        self._factory = factory
        # We will manage the history manually to include author_id
        self.history: list[ChatMessage] = []

    def _get_api_history(self) -> list[dict]:
        """Convert local history to API format."""
        api_history = []
        for msg in self.history:
            api_history.append({"role": msg.role, "parts": msg.parts})
        return api_history

    def send_message(self, user_message: str, author_id: int, author_name: Optional[str] = None, message_ids: Optional[list[str]] = None):
        # 1. Build the full content list for this turn (History + Current Message)
        contents = self._get_api_history()
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        # 2. Call generate_content directly (Stateless)
        response = self._factory.generate_content(contents=contents)

        # 3. Create ChatMessage objects but do NOT append to self.history yet.
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            parts=[{"text": user_message}],
            author_id=author_id,
            author_name=author_name,
            message_ids=message_ids or []
        )

        clean_content = extract_clean_text(response)
        model_msg = ChatMessage(
            role="model",
            content=clean_content,
            parts=[{"text": clean_content}],
            author_id=None # Bot messages have no author
        )

        # Return the new messages and the raw response
        return user_msg, model_msg, response

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

    def generate_content(self, contents: Union[str, list]):
        return self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=self._config,
        )

    def start_chat(self, system_instruction: str):
        # No underlying chat needed
        return _ChatSession(system_instruction, self)


class GeminiService(BaseLLMService):
    """Gemini API와의 모든 상호작용을 관리합니다."""

    def __init__(
        self,
        config: AppConfig,
        *,
        assistant_model_name: str,
        summary_model_name: Optional[str] = None,
        prompt_service: PromptService
    ):
        super().__init__(config)
        self.client = genai.Client(api_key=config.gemini_api_key)
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

        # Cache wrapper instances keyed by system instruction hash
        # Stores tuple: (model_wrapper, expiration_time: Optional[datetime.datetime])
        self._model_cache: dict[int, Tuple[_CachedModel, Optional[datetime.datetime]]] = {}

        # Pre-load default models using cache
        self.summary_model = self._get_or_create_model(
            self._summary_model_name, self.prompt_service.get_summary_prompt()
        )
        self.assistant_model = self._get_or_create_model(
            self._assistant_model_name, self.prompt_service.get_active_assistant_prompt()
        )
        logger.info(
            "Gemini 모델 assistant='%s', summary='%s' 로드 완료. (구성 캐시 활성화)",
            self._assistant_model_name,
            self._summary_model_name,
        )

    def _get_or_create_model(self, model_name: str, system_instruction: str, use_cache: bool = True) -> _CachedModel:
        """Get cached model instance or create new one."""
        key = hash((model_name, system_instruction, use_cache))
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
        cache_name = None
        cache_expiration = None
        if use_cache:
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

        # Enable Google Search Grounding for the assistant model
        if model_name == self._assistant_model_name:
            grounding_tool = genai_types.Tool(
                google_search=genai_types.GoogleSearch()
            )
            config_kwargs["tools"] = [grounding_tool]

        config = genai_types.GenerateContentConfig(**config_kwargs)
        model = _CachedModel(self.client, model_name, config)

        self._model_cache[key] = (model, cache_expiration)
        return model

    def create_assistant_model(self, system_instruction: str, use_cache: bool = True) -> _CachedModel:
        """Create or retrieve a cached assistant model with custom system instruction."""
        return self._get_or_create_model(self._assistant_model_name, system_instruction, use_cache=use_cache)

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
        # Avoid matching "rate" in generic error messages (e.g. "operate")
        # 400 errors should typically not be retried as rate limits unless "quota" is explicit.
        if "400" in error_str and "quota" not in error_str.lower():
            return False
        return "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower()

    def _is_fatal_error(self, error: Exception) -> bool:
        """Check if the error is a fatal cache error."""
        error_str = str(error).lower()
        # "CachedContent not found" or "403 PERMISSION_DENIED" on a cached resource
        return "cachedcontent not found" in error_str or ("403" in error_str and "permission" in error_str)

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

    def _get_cache_key(self, model_name: str, content: str) -> str:
        """Generate a consistent cache key/name based on model and content hash."""
        # Clean model name for use in display_name
        safe_model = re.sub(r'[^a-zA-Z0-9-]', '-', model_name)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"soyebot-{safe_model}-{content_hash[:10]}"

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
            logger.warning(
                "System instruction too short for caching (%d < %d tokens). Using standard context.",
                token_count, min_tokens
            )
            return None, None

        # 2. Config setup
        cache_display_name = self._get_cache_key(model_name, system_instruction)
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
                    
                    # Check if it has expired (though list() shouldn't show it if it is)
                    # But the API key might have changed or permissions issue.
                    
                    # Refresh TTL
                    try:
                        # Ensure ttl_seconds is a string with 's' or just an int if required
                        # The new SDK often accepts a string like '3600s' or an int.
                        self.client.caches.update(
                            name=cache.name,
                            config=genai_types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s")
                        )
                        logger.info("Refreshed TTL for cache: %s (New TTL: %ds)", cache.name, ttl_seconds)
                        # Re-calculate local expiration based on actual state
                        return cache.name, local_expiration
                    except Exception as update_err:
                        logger.warning("Failed to refresh TTL for %s: %s", cache.name, update_err)
                        if "permission" in str(update_err).lower() or "not found" in str(update_err).lower():
                            continue # Try next or create new
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

    def _is_cache_error(self, error: Exception) -> bool:
        """Check if the error is a 403 CachedContent not found error."""
        if isinstance(error, ClientError):
            if error.code == 403 and "CachedContent not found" in str(error.message):
                return True
        return False

    async def summarize_text(self, text: str) -> Optional[str]:
        if not text.strip():
            return "요약할 메시지가 없습니다."
        logger.info(f"Summarizing text ({len(text)} characters)...")
        prompt = f"Discord 대화 내용:\n{text}"

        async def _refresh_summary_model():
            self._model_cache.clear()
            self.summary_model = self._get_or_create_model(
                self._summary_model_name, self.prompt_service.get_summary_prompt()
            )
            logger.info("Refreshed summary model after cache invalidation.")

        # Use _gemini_retry to handle cache errors
        return await self._gemini_retry(
            lambda: self.summary_model.generate_content(prompt),
            on_cache_error=_refresh_summary_model
        )

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        discord_message: Union[discord.Message, list[discord.Message]],
    ) -> Optional[Tuple[str, Any]]:
        """Generate chat response."""
        self._log_raw_request(user_message, chat_session)

        if isinstance(discord_message, list):
            primary_msg = discord_message[0]
            message_ids = [str(m.id) for m in discord_message]
        else:
            primary_msg = discord_message
            message_ids = [str(discord_message.id)]

        author_id = primary_msg.author.id
        author_name = getattr(primary_msg.author, 'name', str(author_id))
 
        async def _refresh_chat_session():
            logger.warning("Refreshing chat session due to 403 Cache Error...")
            self._model_cache.clear()

            # Create a fresh model
            # Use the system instruction from the existing session to ensure continuity
            # Fallback to active prompt if somehow missing
            system_instruction = getattr(chat_session, '_system_instruction', None) or self.prompt_service.get_active_assistant_prompt()

            fresh_model = self._get_or_create_model(
                self._assistant_model_name, system_instruction
            )

            # Update the wrapper's factory reference
            chat_session._factory = fresh_model

            logger.info("Chat session successfully refreshed with new model/cache.")

        try:
            # First attempt: normal flow with retry for cache errors
            result = await self._gemini_retry(
                lambda: chat_session.send_message(user_message, author_id=author_id, author_name=author_name, message_ids=message_ids),
                on_cache_error=_refresh_chat_session,
                discord_message=primary_msg
            )

            if result is None:
                return None

            # Unpack the result from send_message
            user_msg, model_msg, response_obj = result

            # Safely update history now that we are back in the main thread (and not cancelled)
            chat_session.history.append(user_msg)
            chat_session.history.append(model_msg)

            response_text = self._extract_text_from_response(response_obj)
            return (response_text, response_obj)

        except Exception as e:
            if self._is_fatal_error(e) and hasattr(chat_session, '_system_instruction'):
                logger.warning("Cache missing in generate_chat_response. Refreshing model and retrying...")
                self._model_cache.clear()
                
                # Re-create underlying chat session for this specific ChatSession
                system_instr = chat_session._system_instruction
                new_model = self.create_assistant_model(system_instr)

                # Update the wrapper's factory reference
                chat_session._factory = new_model
                
                # Retry the call
                try:
                    result = await self.execute_with_retry(
                        lambda: chat_session.send_message(user_message, author_id=author_id, author_name=author_name, message_ids=message_ids),
                        "응답 생성 (재시도)",
                        return_full_response=True,
                        discord_message=primary_msg,
                    )
                    if result:
                        user_msg, model_msg, response_obj = result
                        chat_session.history.append(user_msg)
                        chat_session.history.append(model_msg)

                        response_text = self._extract_text_from_response(response_obj)
                        return (response_text, response_obj)
                except Exception as retry_e:
                    logger.error(f"Generate chat response retry failed after cache refresh: {retry_e}", exc_info=True)
            else:
                logger.error(f"Generate chat response failed with non-fatal error: {e}", exc_info=True)
            
            # Re-raise or return None if still failing
            return None

    async def _gemini_retry(
        self,
        model_call: Callable[[], Union[Any, Any]],
        on_cache_error: Callable[[], Awaitable[None]],
        discord_message: Optional[discord.Message] = None,
    ) -> Optional[Any]:
        """Custom retry logic for Gemini to handle 403 Cache Errors."""
        last_error = None

        # We assume model_call is a lambda that uses the CURRENT state of objects.
        # If on_cache_error updates those objects, the next call to model_call will use them.

        for attempt in range(1, self.config.api_max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._execute_model_call(model_call),
                    timeout=self.config.api_request_timeout,
                )
                self._log_raw_response(response, attempt)

                # GeminiService methods usually expect the raw response object here,
                # or text depending on what model_call returns.
                # Here model_call returns the response object (generate_content or send_message result).
                return response

            except Exception as e:
                last_error = e

                if self._is_cache_error(e):
                    logger.warning("Gemini 403 Cache Error (Attempt %s). Refreshing session...", attempt)
                    try:
                        await on_cache_error()
                        # After refreshing, we immediately retry in the NEXT loop iteration.
                        # We do NOT decrement attempt, so this consumes a retry slot.
                        # This avoids infinite loops.
                        continue
                    except Exception as refresh_err:
                        logger.error("Failed to refresh session during cache error handling: %s", refresh_err)
                        # If refresh fails, we probably can't recover.
                        break

                if self._is_rate_limit_error(e):
                    logger.warning("Gemini Rate Limit (Attempt %s). Waiting...", attempt)
                    delay = self._extract_retry_delay(e) or self.config.api_rate_limit_retry_after
                    await self._wait_with_countdown(delay, discord_message)
                    continue

                logger.error("Gemini API Error (Attempt %s): %s", attempt, e)

                if attempt >= self.config.api_max_retries:
                    break

                backoff = min(
                    self.config.api_retry_backoff_base ** attempt,
                    self.config.api_retry_backoff_max,
                )
                await asyncio.sleep(backoff)

        if isinstance(last_error, asyncio.TimeoutError):
             logger.error("❌ Gemini API Timeout")
        else:
            logger.error("❌ Gemini API Failed after retries: %s", last_error)

        if discord_message:
            try:
                await discord_message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)
            except discord.HTTPException:
                pass

        return None

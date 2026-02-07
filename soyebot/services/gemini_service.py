"Gemini API service for SoyeBot."

import time
import datetime
import hashlib
import json
import logging
import asyncio
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Callable, Awaitable, List, Dict

import discord
import google.genai as genai
from google.genai import types as genai_types
from google.genai.errors import ClientError

from soyebot.config import AppConfig
from soyebot.services.base import BaseLLMService, ChatMessage
from soyebot.services.prompt_service import PromptService
from soyebot.utils import GENERIC_ERROR_MESSAGE, get_mime_type
from soyebot.tools.adapters.gemini_adapter import GeminiToolAdapter

logger = logging.getLogger(__name__)

# Configuration Constants
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_CACHE_MIN_TOKENS = 32768
DEFAULT_CACHE_TTL_MINUTES = 60
DEFAULT_MAX_HISTORY = 50
CACHE_REFRESH_BUFFER_MIN_MINUTES = 1
CACHE_REFRESH_BUFFER_MAX_MINUTES = 5

# Display Constants
REQUEST_PREVIEW_LENGTH = 200
RESPONSE_TEXT_PREVIEW_LENGTH = 200
HISTORY_DISPLAY_LIMIT = 5


def extract_clean_text(response_obj: Any) -> str:
    """Extract text content from Gemini response, filtering out thoughts."""
    try:
        text_parts = []
        if hasattr(response_obj, "candidates") and response_obj.candidates:
            for candidate in response_obj.candidates:
                if hasattr(candidate, "content") and hasattr(
                    candidate.content, "parts"
                ):
                    for part in candidate.content.parts:
                        # Skip parts that are marked as thoughts
                        if getattr(part, "thought", False):
                            continue

                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)

        if text_parts:
            return " ".join(text_parts).strip()

        return ""

    except Exception as e:
        logger.error(f"Failed to extract text from response: {e}", exc_info=True)
        return ""


class _ChatSession:
    """A wrapper for a Gemini chat session to manage history with author tracking."""

    def __init__(self, system_instruction: str, factory: "_CachedModel"):
        self._system_instruction = system_instruction
        self._factory = factory
        # We will manage the history manually to include author_id
        # Use deque with maxlen for automatic memory management
        max_history = 50  # Default max history size
        self.history: deque[ChatMessage] = deque(maxlen=max_history)

    def _get_api_history(self) -> list[dict]:
        """Convert local history to API format."""
        api_history = []
        for msg in self.history:
            final_parts = []

            # Add existing text/content parts
            if msg.parts:
                for p in msg.parts:
                    if isinstance(p, dict) and "text" in p:
                        final_parts.append(p)
                    elif hasattr(p, "text") and p.text:  # It's a Part object
                        final_parts.append(p)

            # Reconstruct image parts from stored bytes
            if hasattr(msg, "images") and msg.images:
                for img_data in msg.images:
                    mime_type = get_mime_type(img_data)
                    final_parts.append(
                        genai_types.Part.from_bytes(data=img_data, mime_type=mime_type)
                    )

            api_history.append({"role": msg.role, "parts": final_parts})
        return api_history

    def send_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[list[str]] = None,
        images: list[bytes] = None,
        tools: Optional[list] = None,
    ):
        # 1. Build the full content list for this turn (History + Current Message)
        contents = self._get_api_history()

        current_parts = []
        if user_message:
            current_parts.append({"text": user_message})

        if images:
            for img_data in images:
                mime_type = get_mime_type(img_data)
                current_parts.append(
                    genai_types.Part.from_bytes(data=img_data, mime_type=mime_type)
                )

        contents.append({"role": "user", "parts": current_parts})

        # 2. Call generate_content directly (Stateless)
        # Pass tools to generate_content to enable function calling
        response = self._factory.generate_content(contents=contents, tools=tools)

        # 3. Create ChatMessage objects but do NOT append to self.history yet.
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            parts=[
                {"text": user_message}
            ],  # We store text part only in parts for compatibility/simplicity?
            # Or we should store the text part. Images are stored in 'images' field.
            images=images or [],
            author_id=author_id,
            author_name=author_name,
            message_ids=message_ids or [],
        )

        clean_content = extract_clean_text(response)
        model_msg = ChatMessage(
            role="model",
            content=clean_content,
            parts=[{"text": clean_content}],
            author_id=None,  # Bot messages have no author
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

    def generate_content(
        self, contents: Union[str, list], tools: Optional[list] = None
    ):
        """Generate content with optional tools override.

        Args:
            contents: The content to generate.
            tools: Optional override for tools configuration.

        Returns:
            The API response.
        """
        if tools is not None:
            # Rebuild config with new tools while preserving other settings
            config_kwargs = {
                "temperature": getattr(self._config, "temperature", 1.0),
                "top_p": getattr(self._config, "top_p", 1.0),
                "cached_content": getattr(self._config, "cached_content", None),
                "system_instruction": getattr(self._config, "system_instruction", None),
                "tools": tools,
            }
            # Add thinking config if present
            if (
                hasattr(self._config, "thinking_config")
                and self._config.thinking_config
            ):
                config_kwargs["thinking_config"] = self._config.thinking_config

            config = genai_types.GenerateContentConfig(**config_kwargs)
        else:
            config = self._config

        return self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
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
        prompt_service: PromptService,
    ):
        super().__init__(config)
        self.client = genai.Client(api_key=config.gemini_api_key)
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

        # Cache wrapper instances keyed by system instruction hash
        # Stores tuple: (model_wrapper, expiration_time: Optional[datetime.datetime])
        self._model_cache: dict[
            int, Tuple[_CachedModel, Optional[datetime.datetime]]
        ] = {}

        # Pre-load default models using cache
        self.summary_model = self._get_or_create_model(
            self._summary_model_name, self.prompt_service.get_summary_prompt()
        )
        self.assistant_model = self._get_or_create_model(
            self._assistant_model_name,
            self.prompt_service.get_active_assistant_prompt(),
        )
        logger.info(
            "Gemini 모델 assistant='%s', summary='%s' 로드 완료. (구성 캐시 활성화)",
            self._assistant_model_name,
            self._summary_model_name,
        )

        # Start periodic cache cleanup task
        if config.gemini_cache_ttl_minutes > 0:
            asyncio.create_task(self._periodic_cache_cleanup())
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

        # Cache wrapper instances keyed by system instruction hash
        # Stores tuple: (model_wrapper, expiration_time: Optional[datetime.datetime])
        self._model_cache: dict[
            int, Tuple[_CachedModel, Optional[datetime.datetime]]
        ] = {}

        # Pre-load default models using cache
        self.summary_model = self._get_or_create_model(
            self._summary_model_name, self.prompt_service.get_summary_prompt()
        )
        self.assistant_model = self._get_or_create_model(
            self._assistant_model_name,
            self.prompt_service.get_active_assistant_prompt(),
        )
        logger.info(
            "Gemini 모델 assistant='%s', summary='%s' 로드 완료. (구성 캐시 활성화)",
            self._assistant_model_name,
            self._summary_model_name,
        )

    def _get_or_create_model(
        self,
        model_name: str,
        system_instruction: str,
        use_cache: bool = True,
        tools: Optional[list] = None,
    ) -> _CachedModel:
        """Get cached model instance or create new one."""
        key = hash((model_name, system_instruction, use_cache))
        now = datetime.datetime.now(datetime.timezone.utc)

        # Check for valid cached model
        cached = self._check_model_cache_validity(key, now)
        if cached:
            return cached

        # Configure tools and caching
        # Use provided tools or fall back to search tools
        effective_tools = (
            tools if tools is not None else self._get_search_tools(model_name)
        )
        cache_name, cache_expiration = self._resolve_gemini_cache(
            model_name, system_instruction, effective_tools, use_cache
        )

        # Build and create model
        config = self._build_generation_config(
            cache_name, system_instruction, effective_tools
        )
        model = _CachedModel(self.client, model_name, config)
        self._model_cache[key] = (model, cache_expiration)

        return model

    def _check_model_cache_validity(
        self, cache_key: int, now: datetime.datetime
    ) -> Optional[_CachedModel]:
        """Check if cached model exists and is still valid."""
        if cache_key not in self._model_cache:
            return None

        model, expires_at = self._model_cache[cache_key]
        if expires_at and now >= expires_at:
            logger.info("Cached model expired (TTL reached). Refreshing...")
            del self._model_cache[cache_key]
            return None

        return model

    def _get_search_tools(self, model_name: str) -> Optional[list]:
        """Get Google Search tools for assistant model only."""
        if model_name != self._assistant_model_name:
            return None
        return [genai_types.Tool(google_search=genai_types.GoogleSearch())]

    def _resolve_gemini_cache(
        self,
        model_name: str,
        system_instruction: str,
        tools: Optional[list],
        use_cache: bool,
    ) -> Tuple[Optional[str], Optional[datetime.datetime]]:
        """Resolve Gemini cache and log status."""
        if not use_cache:
            return None, None

        cache_name, cache_expiration = self._get_gemini_cache(
            model_name, system_instruction, tools=tools
        )

        # Log cache status
        if cache_name:
            logger.info("Gemini Model initialized with CachedContent: %s", cache_name)
        else:
            logger.debug("Gemini Model will use standard context (no cache).")

        return cache_name, cache_expiration

    def _build_generation_config(
        self, cache_name: Optional[str], system_instruction: str, tools: Optional[list]
    ) -> genai_types.GenerateContentConfig:
        """Build GenerateContentConfig with appropriate settings."""
        config_kwargs = {
            "temperature": getattr(self.config, "temperature", DEFAULT_TEMPERATURE),
            "top_p": getattr(self.config, "top_p", DEFAULT_TOP_P),
        }

        if cache_name:
            config_kwargs["cached_content"] = cache_name
        else:
            config_kwargs["system_instruction"] = system_instruction
            if tools:
                config_kwargs["tools"] = tools

        # Add thinking config if enabled
        thinking_budget = getattr(self.config, "thinking_budget", None)
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                include_thoughts=True, thinking_budget=thinking_budget
            )

        return genai_types.GenerateContentConfig(**config_kwargs)

    def create_assistant_model(
        self, system_instruction: str, use_cache: bool = True
    ) -> _CachedModel:
        """Create or retrieve a cached assistant model with custom system instruction."""
        return self._get_or_create_model(
            self._assistant_model_name, system_instruction, use_cache=use_cache
        )

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

    def _extract_retry_delay(self, error: Exception) -> Optional[float]:
        error_str = str(error)
        match = re.search(r"Please retry in ([0-9.]+)s", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"seconds:\s*(\d+)", error_str)
        if match:
            return float(match.group(1))
        return None

    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        """Log raw API request data being sent (debug level only)."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug(
                f"[RAW API REQUEST] User message preview: {user_message[:REQUEST_PREVIEW_LENGTH]!r}"
            )

            if chat_session and hasattr(chat_session, "history"):
                history = chat_session.history
                formatted_history = []
                for msg in history[-HISTORY_DISPLAY_LIMIT:]:
                    role = msg.role
                    texts = [part.get("text", "") for part in msg.parts]
                    content = " ".join(texts)

                    # Clean up content display if it starts with "Name: "
                    author_label = str(msg.author_name or msg.author_id or "bot")
                    display_content = content
                    if msg.author_name and content.startswith(f"{msg.author_name}:"):
                        display_content = content[len(msg.author_name) + 1 :].strip()

                    formatted_history.append(
                        f"{role} (author:{author_label}) {display_content}"
                    )
                if formatted_history:
                    logger.debug(
                        "[RAW API REQUEST] Recent history:\n"
                        + "\n".join(formatted_history)
                    )
        except Exception as e:
            logger.error(
                f"[RAW API REQUEST] Error logging raw request: {e}", exc_info=True
            )

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw API response data for debugging."""
        # Unpack if tuple (from chat session: user_msg, model_msg, response)
        actual_response = response_obj
        if isinstance(response_obj, tuple) and len(response_obj) >= 3:
            actual_response = response_obj[2]
        try:
            metadata = getattr(actual_response, "usage_metadata", None)
            if metadata:
                prompt_tokens = getattr(metadata, "prompt_token_count", "unknown")
                response_tokens = getattr(metadata, "candidates_token_count", "unknown")
                cached_tokens = getattr(metadata, "cached_content_token_count", 0)
                total_tokens = getattr(metadata, "total_token_count", "unknown")
                logger.info(
                    f"(prompt={prompt_tokens}, cached={cached_tokens}, response={response_tokens}, total={total_tokens})"
                )
        except Exception as e:
            logger.error(
                f"[RAW API RESPONSE {attempt}] Error logging token counts: {e}",
                exc_info=True,
            )

        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            if hasattr(actual_response, "candidates") and actual_response.candidates:
                for idx, candidate in enumerate(actual_response.candidates):
                    finish_reason = getattr(candidate, "finish_reason", "unknown")
                    logger.debug(
                        f"[RAW API RESPONSE {attempt}] Candidate {idx} finish_reason={finish_reason}"
                    )
                    if hasattr(candidate, "content") and hasattr(
                        candidate.content, "parts"
                    ):
                        texts = []
                        for part in candidate.content.parts:
                            text = getattr(part, "text", "")
                            if text:
                                texts.append(
                                    text[:RESPONSE_TEXT_PREVIEW_LENGTH].replace(
                                        "\n", " "
                                    )
                                )
                        if texts:
                            logger.debug(
                                f"[RAW API RESPONSE {attempt}] Candidate {idx} text: {' '.join(texts)}"
                            )
        except Exception as e:
            logger.error(
                f"[RAW API RESPONSE {attempt}] Error logging raw response: {e}",
                exc_info=True,
            )

    def _get_cache_key(
        self, model_name: str, content: str, tools: Optional[list] = None
    ) -> str:
        """Generate a consistent cache key/name based on model and content hash."""
        # Clean model name for use in display_name
        safe_model = re.sub(r"[^a-zA-Z0-9-]", "-", model_name)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        tool_suffix = "-tools" if tools else ""
        return f"soyebot-{safe_model}-{content_hash[:10]}{tool_suffix}"

    def _get_gemini_cache(
        self, model_name: str, system_instruction: str, tools: Optional[list] = None
    ) -> Tuple[Optional[str], Optional[datetime.datetime]]:
        """
        Attempts to find or create a Gemini cache for the given system instruction.
        Returns: (cache_name, local_expiration_datetime)
        """
        if not system_instruction:
            return None, None

        # 1. Check token count
        try:
            # We must count tokens including tools and system instruction to be accurate.
            count_result = self.client.models.count_tokens(
                model=model_name,
                contents=[system_instruction],
            )
            token_count = count_result.total_tokens
        except Exception as e:
            logger.warning(
                "Failed to count tokens for caching check: %s. Using standard context.",
                e,
            )
            return None, None

        min_tokens = getattr(
            self.config, "gemini_cache_min_tokens", DEFAULT_CACHE_MIN_TOKENS
        )
        if token_count < min_tokens:
            logger.info(
                "Gemini Context Caching skipped: Token count (%d) < min_tokens (%d). Using standard context.",
                token_count,
                min_tokens,
            )
            return None, None

        logger.info(
            "Token count (%d) meets requirement for Gemini caching.", token_count
        )

        # 2. Config setup
        cache_display_name = self._get_cache_key(model_name, system_instruction, tools)
        ttl_minutes = getattr(
            self.config, "gemini_cache_ttl_minutes", DEFAULT_CACHE_TTL_MINUTES
        )
        ttl_seconds = ttl_minutes * 60

        now = datetime.datetime.now(datetime.timezone.utc)
        # local_expiration: trigger refresh halfway through TTL window or with a buffer
        refresh_buffer_minutes = min(
            CACHE_REFRESH_BUFFER_MAX_MINUTES,
            max(CACHE_REFRESH_BUFFER_MIN_MINUTES, ttl_minutes // 2),
        )
        local_expiration = now + datetime.timedelta(
            minutes=ttl_minutes - refresh_buffer_minutes
        )

        # 3. Search for existing cache
        try:
            # We iterate to find a cache with our unique display name
            for cache in self.client.caches.list():
                if cache.display_name == cache_display_name:
                    logger.info("Found existing Gemini context cache: %s", cache.name)

                    # Refresh TTL to prevent expiration
                    try:
                        self.client.caches.update(
                            name=cache.name,
                            config=genai_types.UpdateCachedContentConfig(
                                ttl=f"{ttl_seconds}s"
                            ),
                        )
                        logger.info(
                            "Successfully refreshed TTL for %s to %ds.",
                            cache.name,
                            ttl_seconds,
                        )
                        return cache.name, local_expiration
                    except Exception as update_err:
                        logger.warning(
                            "Failed to refresh TTL for %s: %s. Will attempt re-creation.",
                            cache.name,
                            update_err,
                        )
                        # If update fails, DISCARD this cache and continue searching or create new
                        continue

        except Exception as e:
            logger.error("Error listing Gemini caches: %s", e)

        # 4. Create new cache using the user's requested style
        try:
            logger.info(
                "Creating new Gemini cache '%s' (TTL: %ds)...",
                cache_display_name,
                ttl_seconds,
            )

            cache = self.client.caches.create(
                model=model_name,
                config=genai_types.CreateCachedContentConfig(
                    display_name=cache_display_name,
                    system_instruction=system_instruction,
                    tools=tools,
                    ttl=f"{ttl_seconds}s",
                ),
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
        response = await self._gemini_retry(
            lambda: self.summary_model.generate_content(prompt),
            on_cache_error=_refresh_summary_model,
        )
        return self._extract_text_from_response(response) if response else None

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        discord_message: Union[discord.Message, list[discord.Message]],
        model_name: Optional[str] = None,
        tools: Optional[Any] = None,
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
        author_name = getattr(primary_msg.author, "name", str(author_id))

        # Extract images from the primary message (and potentially others if batched?)
        # BaseLLMService supports single message extraction.
        # If batched, we might want to extract from all.
        images = []
        if isinstance(discord_message, list):
            for msg in discord_message:
                imgs = await self._extract_images_from_message(msg)
                images.extend(imgs)
        else:
            images = await self._extract_images_from_message(discord_message)

        async def _refresh_chat_session():
            logger.warning("Refreshing chat session due to 403 Cache Error...")
            self._model_cache.clear()

            # Create a fresh model
            # Use the system instruction from the existing session to ensure continuity
            # Fallback to active prompt if somehow missing
            system_instruction = (
                getattr(chat_session, "_system_instruction", None)
                or self.prompt_service.get_active_assistant_prompt()
            )

            # We use the currently requested model name for refresh, or fallback to default assistant model
            target_model = model_name or self._assistant_model_name

            fresh_model = self._get_or_create_model(target_model, system_instruction)

            # Update the wrapper's factory reference
            chat_session._factory = fresh_model

            logger.info("Chat session successfully refreshed with new model/cache.")

        try:
            # Convert custom tools to Gemini format if provided
            custom_tools = []
            if tools:
                custom_tools = GeminiToolAdapter.convert_tools(tools)

            # Combine custom tools with search tools (for assistant model only)
            final_tools = []
            if custom_tools:
                final_tools.extend(custom_tools)

            # Add search tools for assistant model if not already added
            search_tools = self._get_search_tools(
                model_name or self._assistant_model_name
            )
            if search_tools:
                final_tools.extend(search_tools)

            # Check if we need to switch model for this session
            current_model_name = getattr(chat_session._factory, "_model_name", None)
            if model_name and current_model_name != model_name:
                logger.info(
                    "Switching chat session model from %s to %s",
                    current_model_name,
                    model_name,
                )
                system_instr = (
                    getattr(chat_session, "_system_instruction", None)
                    or self.prompt_service.get_active_assistant_prompt()
                )
                new_model = self._get_or_create_model(
                    model_name, system_instr, tools=final_tools
                )
                chat_session._factory = new_model
            elif final_tools:
                # Update tools for the current model if they changed
                logger.debug(
                    f"Using {len(final_tools)} tools for model {current_model_name}"
                )

            # First attempt: normal flow with retry for cache errors
            result = await self._gemini_retry(
                lambda: chat_session.send_message(
                    user_message,
                    author_id=author_id,
                    author_name=author_name,
                    message_ids=message_ids,
                    images=images,
                    tools=final_tools,
                ),
                on_cache_error=_refresh_chat_session,
                discord_message=primary_msg,
            )

            if result is None:
                return None

            # Unpack the result from send_message
            user_msg, model_msg, response_obj = result

            # Safely update history now that we are back in the main thread (and not cancelled)
            # Use deque.append (automatically handles maxlen)
            chat_session.history.append(user_msg)
            chat_session.history.append(model_msg)

            # Note: deque automatically truncates to maxlen, no need for manual check

            response_text = self._extract_text_from_response(response_obj)
            return (response_text, response_obj)

        except Exception as e:
            if self._is_fatal_error(e) and hasattr(chat_session, "_system_instruction"):
                logger.warning(
                    "Cache missing in generate_chat_response. Refreshing model and retrying..."
                )
                self._model_cache.clear()

                # Re-create underlying chat session for this specific ChatSession
                system_instr = chat_session._system_instruction
                target_model = model_name or self._assistant_model_name
                new_model = self._get_or_create_model(target_model, system_instr)

                # Update the wrapper's factory reference
                chat_session._factory = new_model

                # Retry the call
                try:
                    result = await self.execute_with_retry(
                        lambda: chat_session.send_message(
                            user_message,
                            author_id=author_id,
                            author_name=author_name,
                            message_ids=message_ids,
                        ),
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
                    logger.error(
                        f"Generate chat response retry failed after cache refresh: {retry_e}",
                        exc_info=True,
                    )
            else:
                logger.error(
                    f"Generate chat response failed with non-fatal error: {e}",
                    exc_info=True,
                )

            # Re-raise or return None if still failing
            return None

    async def _gemini_retry(
        self,
        model_call: Callable[[], Union[Any, Any]],
        on_cache_error: Callable[[], Awaitable[None]],
        discord_message: Optional[discord.Message] = None,
    ) -> Optional[Any]:
        """Custom retry logic for Gemini to handle various error types."""
        last_error = None

        for attempt in range(1, self.config.api_max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._execute_model_call(model_call),
                    timeout=self.config.api_request_timeout,
                )
                self._log_raw_response(response, attempt)
                return response

            except Exception as e:
                last_error = e
                should_continue = await self._handle_retry_error(
                    e, attempt, on_cache_error, discord_message
                )
                if not should_continue:
                    break

        await self._notify_final_error(last_error, discord_message)
        return None

    async def _handle_retry_error(
        self,
        error: Exception,
        attempt: int,
        on_cache_error: Callable[[], Awaitable[None]],
        discord_message: Optional[discord.Message],
    ) -> bool:
        """Handle retry error and return True if should continue retrying."""
        # Handle cache errors
        if self._is_cache_error(error):
            return await self._handle_cache_error_retry(attempt, on_cache_error)

        # Handle rate limit errors
        if self._is_rate_limit_error(error):
            await self._handle_rate_limit_retry(error, attempt, discord_message)
            return True

        # Handle generic errors
        logger.error("Gemini API Error (Attempt %s): %s", attempt, error)

        if attempt >= self.config.api_max_retries:
            return False

        await asyncio.sleep(self._calculate_backoff(attempt))
        return True

    async def _handle_cache_error_retry(
        self, attempt: int, on_cache_error: Callable[[], Awaitable[None]]
    ) -> bool:
        """Handle cache error by refreshing session. Returns True if should continue."""
        logger.warning(
            "Gemini 403 Cache Error (Attempt %s). Refreshing session...", attempt
        )
        try:
            await on_cache_error()
            return True
        except Exception as refresh_err:
            logger.error(
                "Failed to refresh session during cache error handling: %s", refresh_err
            )
            return False

    async def _handle_rate_limit_retry(
        self, error: Exception, attempt: int, discord_message: Optional[discord.Message]
    ) -> None:
        """Handle rate limit error with countdown wait."""
        logger.warning("Gemini Rate Limit (Attempt %s). Waiting...", attempt)
        delay = (
            self._extract_retry_delay(error) or self.config.api_rate_limit_retry_after
        )
        await self._wait_with_countdown(delay, discord_message)

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return min(
            self.config.api_retry_backoff_base**attempt,
            self.config.api_retry_backoff_max,
        )

    async def _notify_final_error(
        self, error: Optional[Exception], discord_message: Optional[discord.Message]
    ) -> None:
        """Log final error and notify user via Discord."""
        if isinstance(error, asyncio.TimeoutError):
            logger.error("❌ Gemini API Timeout")
        else:
            logger.error("❌ Gemini API Failed after retries: %s", error)

        if discord_message:
            try:
                await discord_message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)
            except discord.HTTPException:
                pass

    # Tool support methods
    def get_tools_for_provider(self, tools: List[Any]) -> Any:
        """Convert tool definitions to Gemini format.

        Args:
            tools: List of ToolDefinition objects to convert.

        Returns:
            List of genai_types.Tool objects.
        """
        return GeminiToolAdapter.convert_tools(tools)

    def extract_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from Gemini response.

        Args:
            response: Gemini response object.

        Returns:
            List of function call dictionaries with 'name' and 'parameters'.
        """
        return GeminiToolAdapter.extract_function_calls(response)

    def format_function_results(self, results: List[Dict[str, Any]]) -> Any:
        """Format function results for sending back to Gemini.

        Args:
            results: List of dicts with 'name', 'result', and optionally 'error'.

        Returns:
            List of genai_types.Part objects with function responses.
        """
        return GeminiToolAdapter.create_function_response_parts(results)

    async def _periodic_cache_cleanup(self):
        """Periodically clean up expired cache entries."""
        ttl_minutes = getattr(self.config, "gemini_cache_ttl_minutes", 60)
        if ttl_minutes <= 0:
            return

        # Cleanup interval: half of TTL to prevent too frequent refreshes
        cleanup_interval = ttl_minutes * 30  # seconds

        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                # Refresh expired cache entries to prevent expiration
                await self._refresh_expired_cache()
            except asyncio.CancelledError:
                logger.info("Cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}", exc_info=True)

    async def _refresh_expired_cache(self):
        """Refresh expired cache entries to prevent expiration."""
        try:
            ttl_minutes = getattr(self.config, "gemini_cache_ttl_minutes", 60)
            cache_name = f"soyebot-{self._assistant_model_name[:20]}"
            # Use a timeout to avoid blocking too long
            timeout = 5.0
            response = await asyncio.wait_for(
                asyncio.to_thread(self.client.caches.list), timeout=timeout
            )

            # Process a reasonable number of caches to avoid API overload
            cached_items = list(response)[:100]  # Limit to 100 caches

            for cache in cached_items:
                try:
                    # Refresh TTL for frequently used caches
                    ttl_seconds = ttl_minutes * 60
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.caches.update(
                                name=cache.name,
                                config=genai_types.UpdateCachedContentConfig(
                                    ttl=f"{ttl_seconds}s"
                                ),
                            )
                        ),
                        timeout=2.0,
                    )
                except Exception as e:
                    # Log but don't fail on individual cache refresh
                    if "403 PERMISSION_DENIED" not in str(e):
                        logger.debug(f"Failed to refresh cache {cache.name}: {e}")
        except asyncio.TimeoutError:
            logger.debug("Cache list operation timed out")
        except Exception as e:
            logger.error(f"Cache cleanup task error: {e}", exc_info=True)

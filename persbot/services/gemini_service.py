"Gemini API service for SoyeBot."

import asyncio
import datetime
import hashlib
import logging
import re
from functools import lru_cache
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import discord
import google.genai as genai
from google.genai import types as genai_types
from google.genai.errors import ClientError

from persbot.config import AppConfig
from persbot.constants import (
    CacheConfig,
    DisplayConfig,
    LLMDefaults,
)
from persbot.services.base import BaseLLMServiceCore, ChatMessage
from persbot.services.model_wrappers.gemini_model import GeminiCachedModel
from persbot.services.prompt_service import PromptService
from persbot.services.retry_handler import (
    GeminiRetryHandler,
    RetryHandler,
)
from persbot.services.session_wrappers.gemini_session import extract_clean_text
from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter
from persbot.utils import GENERIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)

# Maximum size for in-memory cache name lookup
CACHE_NAME_LRU_SIZE = 64


class GeminiService(BaseLLMServiceCore):
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
        self._model_cache: dict[int, Tuple[GeminiCachedModel, Optional[datetime.datetime]]] = {}

        # In-memory LRU cache for cache name lookups to avoid repeated caches.list() calls
        # Maps: cache_display_name -> (cache_name, expiration_datetime)
        self._cache_name_cache: Dict[str, Tuple[str, datetime.datetime]] = {}

        # Models are created lazily on first use to avoid blocking startup
        # Use properties _assistant_model and _summary_model for lazy access
        self._assistant_model_instance: Optional[GeminiCachedModel] = None
        self._summary_model_instance: Optional[GeminiCachedModel] = None
        self._models_initialized = False
        self._cache_warmup_task: Optional[asyncio.Task] = None

        # Start periodic cache cleanup task
        if config.gemini_cache_ttl_minutes > 0:
            asyncio.create_task(self._periodic_cache_cleanup())

    @property
    def assistant_model(self) -> GeminiCachedModel:
        """Lazily get or create the assistant model."""
        if self._assistant_model_instance is None:
            self._assistant_model_instance = self._get_or_create_model(
                self._assistant_model_name, self.prompt_service.get_active_assistant_prompt()
            )
        return self._assistant_model_instance

    @assistant_model.setter
    def assistant_model(self, value: GeminiCachedModel) -> None:
        """Set the assistant model instance (useful for testing)."""
        self._assistant_model_instance = value

    @property
    def summary_model(self) -> GeminiCachedModel:
        """Lazily get or create the summary model."""
        if self._summary_model_instance is None:
            self._summary_model_instance = self._get_or_create_model(
                self._summary_model_name, self.prompt_service.get_summary_prompt()
            )
        return self._summary_model_instance

    @summary_model.setter
    def summary_model(self, value: GeminiCachedModel) -> None:
        """Set the summary model instance (useful for testing)."""
        self._summary_model_instance = value

    async def warmup_caches(self) -> None:
        """Pre-create commonly used caches in background to reduce first-message latency.

        This method should be called during bot startup, not on first message.
        """
        if self._models_initialized:
            return

        try:
            # Create models with caching enabled (async to avoid blocking)
            async def _create_model_async(model_name: str, system_prompt: str) -> GeminiCachedModel:
                return await asyncio.to_thread(
                    self._get_or_create_model, model_name, system_prompt
                )

            # Warm up both models in parallel
            self._assistant_model_instance, self._summary_model_instance = await asyncio.gather(
                _create_model_async(
                    self._assistant_model_name,
                    self.prompt_service.get_active_assistant_prompt()
                ),
                _create_model_async(
                    self._summary_model_name,
                    self.prompt_service.get_summary_prompt()
                ),
            )

            self._models_initialized = True
        except Exception as e:
            logger.warning("Cache warmup failed (will use lazy loading): %s", e)
            # Models will be created lazily on first use

    def start_background_warmup(self) -> None:
        """Start cache warmup as a background task (non-blocking)."""
        if self._cache_warmup_task is None or self._cache_warmup_task.done():
            self._cache_warmup_task = asyncio.create_task(self.warmup_caches())

    def _get_or_create_model(
        self,
        model_name: str,
        system_instruction: str,
        use_cache: bool = True,
        tools: Optional[list] = None,
    ) -> GeminiCachedModel:
        """Get cached model instance or create new one."""
        # Include tools in cache key so tool-enabled and non-tool models use different caches
        # Generate a stable hash based on tool content, not memory addresses
        tools_hash = 0
        if tools:
            tool_names = []
            for tool in tools:
                if hasattr(tool, 'function_declarations') and tool.function_declarations:
                    for fd in tool.function_declarations:
                        if hasattr(fd, 'name'):
                            tool_names.append(fd.name)
                elif hasattr(tool, 'google_search') and tool.google_search:
                    tool_names.append('google_search')
                elif hasattr(tool, 'name'):  # ToolDefinition objects
                    tool_names.append(tool.name)
            tool_names.sort()
            tools_hash = hash(tuple(tool_names))

        key = hash((model_name, system_instruction, use_cache, tools_hash))
        now = datetime.datetime.now(datetime.timezone.utc)

        # Check for valid cached model
        cached = self._check_model_cache_validity(key, now)
        if cached:
            return cached

        # Configure tools and caching
        # Combine provided tools with search tools using the helper
        # NOTE: tools may already be converted genai_types.Tool objects (from generate_chat_response_stream)
        # or may be ToolDefinition objects. Handle both cases.
        search_tools = self._get_search_tools(model_name)
        if tools:
            # Check if tools are already genai_types.Tool objects (already converted)
            # genai_types.Tool has function_declarations, google_search, etc. but not 'name'
            if hasattr(tools[0], 'function_declarations') or hasattr(tools[0], 'google_search'):
                # Already converted, use as-is
                effective_tools = self._combine_tools(tools, search_tools)
            else:
                # Need to convert from ToolDefinition
                custom_tools = GeminiToolAdapter.convert_tools(tools)
                effective_tools = self._combine_tools(custom_tools, search_tools)
        else:
            effective_tools = self._combine_tools(None, search_tools)

        cache_name, cache_expiration = self._resolve_gemini_cache(
            model_name, system_instruction, effective_tools, use_cache
        )

        # Build and create model
        config = self._build_generation_config(cache_name, system_instruction, effective_tools)
        model = GeminiCachedModel(self.client, model_name, config)
        self._model_cache[key] = (model, cache_expiration)

        return model

    def _check_model_cache_validity(
        self, cache_key: int, now: datetime.datetime
    ) -> Optional[GeminiCachedModel]:
        """Check if cached model exists and is still valid."""
        if cache_key not in self._model_cache:
            return None

        model, expires_at = self._model_cache[cache_key]
        if expires_at and now >= expires_at:
            del self._model_cache[cache_key]
            return None

        return model

    # Models that don't support function calling / tool use
    # These models will return 400 Bad Request if tools are passed
    MODELS_WITHOUT_TOOL_SUPPORT = frozenset([
        "gemini-3-flash-preview",
        "gemini-3-flash",
    ])

    def _model_supports_tools(self, model_name: Optional[str]) -> bool:
        """Check if a model supports function calling / tool use.

        Args:
            model_name: The model name to check.

        Returns:
            True if the model supports tools, False otherwise.
        """
        if not model_name:
            return True  # Assume supported if no model specified
        return model_name not in self.MODELS_WITHOUT_TOOL_SUPPORT

    def _get_search_tools(self, model_name: str) -> Optional[list]:
        """Get Google Search tools for Gemini models.

        Note: Some preview models (e.g., gemini-3-flash-preview) don't support
        tool use yet and will return 400 Bad Request if tools are passed.
        """
        # Check if this model supports tools
        if not self._model_supports_tools(model_name):
            return None

        return [genai_types.Tool(google_search=genai_types.GoogleSearch())]

    def _combine_tools(
        self,
        custom_tools: Optional[list],
        search_tools: Optional[list],
    ) -> Optional[list]:
        """Combine custom function tools with Google Search into a single Tool object.

        Gemini API requires all tools to be combined into a single Tool object
        when using both function_declarations and google_search together.
        """
        if not custom_tools and not search_tools:
            return None

        # Extract all function declarations from custom tools
        all_function_declarations = []
        for tool in custom_tools or []:
            if hasattr(tool, "function_declarations") and tool.function_declarations:
                all_function_declarations.extend(tool.function_declarations)

        # Check if we have search tools
        has_google_search = any(
            hasattr(tool, "google_search") and tool.google_search is not None
            for tool in (search_tools or [])
        )

        if all_function_declarations or has_google_search:
            # Combine into a single Tool object
            return [
                genai_types.Tool(
                    function_declarations=all_function_declarations if all_function_declarations else None,
                    google_search=genai_types.GoogleSearch() if has_google_search else None,
                )
            ]

        return None

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

        # Gemini's cached content doesn't support function calling.
        # When tools are needed, skip caching and use standard context.
        if tools:
            return None, None

        cache_name, cache_expiration = self._get_gemini_cache(
            model_name, system_instruction, tools=None
        )

        return cache_name, cache_expiration

    def _build_generation_config(
        self, cache_name: Optional[str], system_instruction: str, tools: Optional[list]
    ) -> genai_types.GenerateContentConfig:
        """Build GenerateContentConfig with appropriate settings."""
        config_kwargs = {
            "temperature": getattr(self.config, "temperature", LLMDefaults.TEMPERATURE),
            "top_p": getattr(self.config, "top_p", LLMDefaults.TOP_P),
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
    ) -> GeminiCachedModel:
        """Create or retrieve a cached assistant model with custom system instruction."""
        return self._get_or_create_model(
            self._assistant_model_name, system_instruction, use_cache=use_cache
        )

    def reload_parameters(self) -> None:
        """Reload parameters by clearing the model cache."""
        self._model_cache.clear()
        self._cache_name_cache.clear()
        self._assistant_model_instance = None
        self._summary_model_instance = None
        self._models_initialized = False

    def _create_retry_handler(self) -> RetryHandler:
        """Create Gemini-specific retry handler."""
        config = self._create_retry_config_core()
        return GeminiRetryHandler(config)

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
            "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower()
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
        """Log raw API request data being sent."""
        pass  # Request logging removed

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw API response data for debugging."""
        pass  # Response logging removed

    def _get_cache_key(self, model_name: str, content: str, tools: Optional[list] = None) -> str:
        """Generate a consistent cache key/name based on model and content hash."""
        # Clean model name for use in display_name
        safe_model = re.sub(r"[^a-zA-Z0-9-]", "-", model_name)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Generate a stable hash for tools based on their content, not memory addresses
        tool_suffix = ""
        if tools:
            # Extract tool names for hashing (more stable than full serialization)
            tool_names = []
            for tool in tools:
                if hasattr(tool, 'function_declarations') and tool.function_declarations:
                    for fd in tool.function_declarations:
                        if hasattr(fd, 'name'):
                            tool_names.append(fd.name)
                elif hasattr(tool, 'google_search') and tool.google_search:
                    tool_names.append('google_search')
            tool_names.sort()  # Sort for consistency
            if tool_names:
                tool_hash = hashlib.sha256(','.join(tool_names).encode()).hexdigest()[:6]
                tool_suffix = f"-tools-{tool_hash}"

        return f"persbot-{safe_model}-{content_hash[:10]}{tool_suffix}"

    def _get_gemini_cache(
        self, model_name: str, system_instruction: str, tools: Optional[list] = None
    ) -> Tuple[Optional[str], Optional[datetime.datetime]]:
        """
        Attempts to find or create a Gemini cache for the given system instruction.
        Uses in-memory cache for lookups to avoid repeated caches.list() calls.
        Returns: (cache_name, local_expiration_datetime)

        Note: This method is synchronous but uses cached data where possible.
        For new cache creation, it will make blocking API calls.
        """
        if not system_instruction:
            return None, None

        # Generate cache display name
        cache_display_name = self._get_cache_key(model_name, system_instruction, tools)
        ttl_minutes = getattr(self.config, "gemini_cache_ttl_minutes", CacheConfig.TTL_MINUTES)
        ttl_seconds = ttl_minutes * 60
        now = datetime.datetime.now(datetime.timezone.utc)

        # Calculate local expiration
        refresh_buffer_minutes = min(
            CacheConfig.REFRESH_BUFFER_MAX,
            max(CacheConfig.REFRESH_BUFFER_MIN, ttl_minutes // 2),
        )
        local_expiration = now + datetime.timedelta(minutes=ttl_minutes - refresh_buffer_minutes)

        # FAST PATH: Check in-memory cache first to avoid API calls
        if cache_display_name in self._cache_name_cache:
            cached_name, cached_expiration = self._cache_name_cache[cache_display_name]
            # Check if cached entry is still valid
            if cached_expiration and now < cached_expiration:
                return cached_name, cached_expiration
            else:
                # Expired, remove from cache
                del self._cache_name_cache[cache_display_name]

        # SLOW PATH: Check token count and create/list caches
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

        min_tokens = getattr(self.config, "gemini_cache_min_tokens", CacheConfig.MIN_TOKENS)
        if token_count < min_tokens:
            return None, None

        # Search for existing cache
        try:
            # We iterate to find a cache with our unique display name
            for cache in self.client.caches.list():
                if cache.display_name == cache_display_name:
                    # Refresh TTL to prevent expiration
                    try:
                        self.client.caches.update(
                            name=cache.name,
                            config=genai_types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s"),
                        )
                        # Cache the result in memory for future lookups
                        self._cache_name_cache[cache_display_name] = (cache.name, local_expiration)
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

        # Create new cache
        try:
            cache = self.client.caches.create(
                model=model_name,
                config=genai_types.CreateCachedContentConfig(
                    display_name=cache_display_name,
                    system_instruction=system_instruction,
                    tools=tools,
                    ttl=f"{ttl_seconds}s",
                ),
            )

            # Cache the result in memory for future lookups
            self._cache_name_cache[cache_display_name] = (cache.name, local_expiration)
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
        prompt = f"Discord 대화 내용:\n{text}"

        async def _refresh_summary_model():
            self._model_cache.clear()
            self.summary_model = self._get_or_create_model(
                self._summary_model_name, self.prompt_service.get_summary_prompt()
            )

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
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Generate chat response."""
        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("LLM API call aborted by user")

        self._log_raw_request(user_message, chat_session)

        if isinstance(discord_message, list):
            primary_msg = discord_message[0]
            message_ids = [str(m.id) for m in discord_message]
        else:
            primary_msg = discord_message
            message_ids = [str(discord_message.id)]

        author_id = primary_msg.author.id
        author_name = getattr(primary_msg.author, "name", str(author_id))

        # Extract images from message(s) - supports both single and list of messages
        images = await self._extract_images_from_messages(discord_message)

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

        try:
            effective_model = model_name or self._assistant_model_name

            # Only process tools if the model supports them
            if self._model_supports_tools(effective_model):
                # Convert custom tools to Gemini format if provided
                custom_tools = GeminiToolAdapter.convert_tools(tools) if tools else None

                # Combine custom tools with search tools using the helper
                search_tools = self._get_search_tools(effective_model)
                final_tools = self._combine_tools(custom_tools, search_tools)
            else:
                final_tools = None

            # Check if we need to switch model for this session
            current_model_name = getattr(chat_session._factory, "_model_name", None)
            if model_name and current_model_name != model_name:
                system_instr = (
                    getattr(chat_session, "_system_instruction", None)
                    or self.prompt_service.get_active_assistant_prompt()
                )
                new_model = self._get_or_create_model(model_name, system_instr, tools=final_tools)
                chat_session._factory = new_model
            elif final_tools:
                # When using cached content, tools are baked into the cache and cannot be overridden.
                # We need to get/create a model that has these tools in its cache.
                system_instr = (
                    getattr(chat_session, "_system_instruction", None)
                    or self.prompt_service.get_active_assistant_prompt()
                )
                new_model = self._get_or_create_model(
                    current_model_name or self._assistant_model_name,
                    system_instr,
                    tools=final_tools
                )
                chat_session._factory = new_model

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
                cancel_event=cancel_event,
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
                        cancel_event=cancel_event,
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

    async def generate_chat_response_stream(
        self,
        chat_session,
        user_message: str,
        discord_message: Union[discord.Message, list[discord.Message]],
        model_name: Optional[str] = None,
        tools: Optional[Any] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response from Gemini.

        Yields text chunks as they arrive from the API.
        Chunks are yielded after each line break for faster initial response.

        Args:
            chat_session: The Gemini chat session.
            user_message: The user's message.
            discord_message: The Discord message(s) for context.
            model_name: Optional specific model to use.
            tools: Optional tools for function calling.
            cancel_event: Optional event to check for cancellation.

        Yields:
            Text chunks as they are generated.
        """
        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("LLM API call aborted by user")

        self._log_raw_request(user_message, chat_session)

        if isinstance(discord_message, list):
            primary_msg = discord_message[0]
            message_ids = [str(m.id) for m in discord_message]
        else:
            primary_msg = discord_message
            message_ids = [str(discord_message.id)]

        author_id = primary_msg.author.id
        author_name = getattr(primary_msg.author, "name", str(author_id))

        # Extract images from messages first
        images = await self._extract_images_from_messages(discord_message)

        effective_model = model_name or self._assistant_model_name

        # Only process tools if the model supports them
        if self._model_supports_tools(effective_model):
            # Convert custom tools to Gemini format if provided
            custom_tools = GeminiToolAdapter.convert_tools(tools) if tools else None

            # Combine custom tools with search tools using the helper
            search_tools = self._get_search_tools(effective_model)
            final_tools = self._combine_tools(custom_tools, search_tools)
        else:
            final_tools = None

        # Check if we need to switch model for this session
        current_model_name = getattr(chat_session._factory, "_model_name", None)
        if model_name and current_model_name != model_name:
            system_instr = (
                getattr(chat_session, "_system_instruction", None)
                or self.prompt_service.get_active_assistant_prompt()
            )
            new_model = self._get_or_create_model(model_name, system_instr, tools=final_tools)
            chat_session._factory = new_model
        elif final_tools:
            # When using cached content, tools are baked into the cache and cannot be overridden.
            # We need to get/create a model that has these tools in its cache.
            system_instr = (
                getattr(chat_session, "_system_instruction", None)
                or self.prompt_service.get_active_assistant_prompt()
            )
            new_model = self._get_or_create_model(
                current_model_name or self._assistant_model_name,
                system_instr,
                tools=final_tools
            )
            chat_session._factory = new_model

        # Helper to refresh session on cache error
        async def _refresh_streaming_session():
            logger.warning("Refreshing streaming session due to 403 Cache Error...")
            self._model_cache.clear()
            self._cache_name_cache.clear()

            system_instruction = (
                getattr(chat_session, "_system_instruction", None)
                or self.prompt_service.get_active_assistant_prompt()
            )
            target_model = effective_model
            fresh_model = self._get_or_create_model(target_model, system_instruction)
            chat_session._factory = fresh_model

        # Start streaming - get user message and stream iterator
        # This may fail with cache error, so we wrap in try-except for retry
        max_retries = 1  # Only retry once for cache errors

        for attempt in range(max_retries + 1):
            try:
                user_msg, stream = await chat_session.send_message_stream(
                    user_message,
                    author_id=author_id,
                    author_name=author_name,
                    message_ids=message_ids,
                    images=images,
                    tools=final_tools,
                )
                break  # Success, exit retry loop
            except Exception as e:
                if self._is_cache_error(e) and attempt < max_retries:
                    await _refresh_streaming_session()
                    continue
                raise  # Re-raise non-cache errors or final retry failure

        # Buffer for streaming - yield on newline for natural line breaks
        buffer = ""
        full_content = ""
        final_chunk = None  # Track final chunk for usage metadata

        try:
            async for chunk in stream:
                # Check for cancellation
                if cancel_event and cancel_event.is_set():
                    # Close the stream to stop server-side generation
                    if hasattr(stream, 'aclose'):
                        await stream.aclose()
                    elif hasattr(stream, 'close'):
                        stream.close()
                    raise asyncio.CancelledError("LLM streaming aborted by user")

                # Save reference to final chunk for metadata extraction
                final_chunk = chunk

                # Extract text from chunk
                text = self._extract_text_from_stream_chunk(chunk)
                if text:
                    buffer += text
                    full_content += text

                    # Yield when we see a line break
                    if "\n" in buffer:
                        lines = buffer.split("\n")
                        # Yield all complete lines
                        for line in lines[:-1]:
                            if line:  # Skip empty lines
                                yield line + "\n"
                        # Keep the last incomplete line in buffer
                        buffer = lines[-1]

            # Yield any remaining content in buffer
            if buffer:
                yield buffer

            # Log usage metadata from final chunk
            if final_chunk:
                self._log_raw_response(final_chunk, 1)

        except asyncio.CancelledError:
            # Ensure stream is closed on cancellation to stop server-side generation
            try:
                if hasattr(stream, 'aclose'):
                    await stream.aclose()
                elif hasattr(stream, 'close'):
                    stream.close()
            except BaseException:
                pass
            raise
        except Exception:
            # Close stream on any exception to prevent resource leaks and stop server generation
            try:
                if hasattr(stream, 'aclose'):
                    await stream.aclose()
                elif hasattr(stream, 'close'):
                    stream.close()
            except BaseException:
                pass
            raise

        # Update history with the full conversation
        model_msg = ChatMessage(
            role="model",
            content=full_content,
            parts=[{"text": full_content}],
        )
        chat_session.history.append(user_msg)
        chat_session.history.append(model_msg)

    def _extract_text_from_stream_chunk(self, chunk: Any) -> str:
        """Extract text from a streaming chunk, filtering out thoughts."""
        try:
            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            # Skip parts that are marked as thoughts
                            if getattr(part, "thought", False):
                                continue

                            if hasattr(part, "text") and part.text:
                                return part.text
        except Exception:
            pass
        return ""

    async def send_tool_results(
        self,
        chat_session,
        tool_rounds,
        tools=None,
        discord_message=None,
        cancel_event: Optional[asyncio.Event] = None,
    ):
        """Send tool results back to model and get continuation response.

        Args:
            chat_session: The Gemini chat session.
            tool_rounds: List of (response_obj, tool_results) tuples.
            tools: Original tool definitions (will be converted to Gemini format).
            discord_message: Discord message for error notifications.
            cancel_event: Optional event to check for abort signals.

        Returns:
            Tuple of (response_text, response_obj) or None.
        """
        # Check cancellation event before starting API call
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("LLM API call aborted by user")

        model_name = getattr(chat_session._factory, "_model_name", self._assistant_model_name)

        # Only process tools if the model supports them
        if self._model_supports_tools(model_name):
            # Convert tools to Gemini format and combine with search tools
            custom_tools = GeminiToolAdapter.convert_tools(tools) if tools else None
            search_tools = self._get_search_tools(model_name)
            final_tools = self._combine_tools(custom_tools, search_tools)
        else:
            final_tools = None

        result = await self.execute_with_retry(
            lambda: chat_session.send_tool_results(tool_rounds, tools=final_tools or None),
            "tool 결과 전송",
            return_full_response=True,
            discord_message=discord_message,
            cancel_event=cancel_event,
        )

        if result is None:
            return None

        model_msg, response_obj = result

        # Update the last model entry in history with the final response
        if chat_session.history and chat_session.history[-1].role == "model":
            chat_session.history[-1] = model_msg

        response_text = self._extract_text_from_response(response_obj)
        return (response_text, response_obj)

    async def _gemini_retry(
        self,
        model_call: Callable[[], Union[Any, Any]],
        on_cache_error: Callable[[], Awaitable[None]],
        discord_message: Optional[discord.Message] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Any]:
        """Custom retry logic for Gemini to handle various error types."""
        last_error = None

        for attempt in range(1, self.config.api_max_retries + 1):
            try:
                # Check for cancellation before attempting API call
                if cancel_event and cancel_event.is_set():
                    raise asyncio.CancelledError("LLM API call aborted by user")

                response = await asyncio.wait_for(
                    self._execute_model_call(model_call),
                    timeout=self.config.api_request_timeout,
                )
                # Check for cancellation AFTER API call returns
                # The underlying thread can't be cancelled mid-flight, so we check here
                if cancel_event and cancel_event.is_set():
                    raise asyncio.CancelledError("LLM API call aborted by user")
                self._log_raw_response(response, attempt)
                return response

            except Exception as e:
                last_error = e
                should_continue = await self._handle_retry_error(
                    e, attempt, on_cache_error, discord_message, cancel_event
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
        cancel_event: Optional[asyncio.Event] = None,
    ) -> bool:
        """Handle retry error and return True if should continue retrying."""
        # Handle cache errors
        if self._is_cache_error(error):
            return await self._handle_cache_error_retry(attempt, on_cache_error)

        # Handle rate limit errors
        if self._is_rate_limit_error(error):
            await self._handle_rate_limit_retry(error, attempt, discord_message, cancel_event)
            return True

        # Handle generic errors
        logger.error("Gemini API Error (Attempt %s): %s", attempt, error)

        if attempt >= self.config.api_max_retries:
            return False

        # Check for cancellation before backoff sleep
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("LLM API call aborted by user")

        await asyncio.sleep(self._calculate_backoff(attempt))
        return True

    async def _handle_cache_error_retry(
        self, attempt: int, on_cache_error: Callable[[], Awaitable[None]]
    ) -> bool:
        """Handle cache error by refreshing session. Returns True if should continue."""
        logger.warning("Gemini 403 Cache Error (Attempt %s). Refreshing session...", attempt)
        try:
            await on_cache_error()
            return True
        except Exception as refresh_err:
            logger.error("Failed to refresh session during cache error handling: %s", refresh_err)
            return False

    async def _handle_rate_limit_retry(
        self,
        error: Exception,
        attempt: int,
        discord_message: Optional[discord.Message],
        cancel_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Handle rate limit error with countdown wait."""
        logger.warning("Gemini Rate Limit (Attempt %s). Waiting...", attempt)
        delay = self._extract_retry_delay(error) or self.config.api_rate_limit_retry_after
        await self._wait_with_countdown(delay, discord_message, cancel_event)

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
        return GeminiToolAdapter.format_results(results)

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
                break
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}", exc_info=True)

    async def _refresh_expired_cache(self):
        """Refresh expired cache entries to prevent expiration and update in-memory cache."""
        try:
            ttl_minutes = getattr(self.config, "gemini_cache_ttl_minutes", 60)
            ttl_seconds = ttl_minutes * 60

            # Calculate local expiration for cached entries
            refresh_buffer_minutes = min(
                CacheConfig.REFRESH_BUFFER_MAX,
                max(CacheConfig.REFRESH_BUFFER_MIN, ttl_minutes // 2),
            )
            local_expiration = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
                minutes=ttl_minutes - refresh_buffer_minutes
            )

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
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.caches.update,
                            cache.name,
                            genai_types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s"),
                        ),
                        timeout=2.0,
                    )

                    # Update in-memory cache for faster lookups
                    if cache.display_name:
                        self._cache_name_cache[cache.display_name] = (cache.name, local_expiration)

                except Exception as e:
                    # Log but don't fail on individual cache refresh
                    if "403 PERMISSION_DENIED" not in str(e):
                        pass
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Cache cleanup task error: {e}", exc_info=True)

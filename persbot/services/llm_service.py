"""LLM service selector for SoyeBot.

This module provides a unified interface for accessing different LLM providers.
It acts as a factory and registry for provider backends.
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple


from persbot.config import AppConfig
from persbot.constants import META_PROMPT, Provider, QUESTION_GENERATION_PROMPT
from persbot.exceptions import ProviderUnavailableException
from persbot.services.base import BaseLLMService
from persbot.services.gemini_service import GeminiService
from persbot.services.model_usage_service import ModelUsageService
from persbot.services.openai_service import OpenAIService
from persbot.services.prompt_service import PromptService
from persbot.services.usage_service import ImageUsageService
from persbot.services.zai_service import ZAIService
from persbot.utils import extract_message_metadata

logger = logging.getLogger(__name__)

# Sentinel value to distinguish "no change" from "set to None"
_UNSET = object()


class ProviderRegistry:
    """Registry for managing LLM provider backends."""

    def __init__(self, config: AppConfig, prompt_service: PromptService):
        """Initialize the provider registry.

        Args:
            config: Application configuration.
            prompt_service: Service for managing prompts.
        """
        self.config = config
        self.prompt_service = prompt_service
        self._backends: Dict[str, BaseLLMService] = {}

    def get_provider(self, provider_name: str) -> Optional[BaseLLMService]:
        """Get a provider backend by name.

        Args:
            provider_name: The provider name ('gemini', 'openai', 'zai').

        Returns:
            The provider backend, or None if not available.
        """
        return self._backends.get(provider_name)

    def register(self, provider_name: str, backend: BaseLLMService) -> None:
        """Register a provider backend.

        Args:
            provider_name: The provider name.
            backend: The backend service.
        """
        self._backends[provider_name] = backend

    def get_or_create_provider(
        self, provider_name: str, model_name: Optional[str] = None
    ) -> Optional[BaseLLMService]:
        """Get existing provider or create new one.

        Args:
            provider_name: The provider name.
            model_name: Optional specific model name.

        Returns:
            The provider backend, or None if unavailable.
        """
        if provider_name in self._backends:
            return self._backends[provider_name]

        # Create new backend
        backend = self._create_provider(provider_name, model_name)
        if backend:
            self._backends[provider_name] = backend
        return backend

    def _create_provider(
        self, provider_name: str, model_name: Optional[str] = None
    ) -> Optional[BaseLLMService]:
        """Create a new provider backend.

        Args:
            provider_name: The provider name.
            model_name: Optional specific model name.

        Returns:
            The created backend, or None if creation failed.
        """
        provider_str = provider_name.lower()
        effective_model = model_name or self._get_default_model_for_provider(provider_str)

        try:
            if provider_str == Provider.OPENAI:
                if not self.config.openai_api_key:
                    logger.warning("OpenAI API key missing")
                    return None
                return OpenAIService(
                    self.config,
                    assistant_model_name=effective_model,
                    prompt_service=self.prompt_service,
                )
            elif provider_str == Provider.ZAI:
                if not self.config.zai_api_key:
                    logger.warning("Z.AI API key missing")
                    return None
                return ZAIService(
                    self.config,
                    assistant_model_name=effective_model,
                    prompt_service=self.prompt_service,
                )
            else:  # Gemini (default)
                if not self.config.gemini_api_key:
                    logger.warning("Gemini API key missing")
                    return None
                return GeminiService(
                    self.config,
                    assistant_model_name=effective_model,
                    prompt_service=self.prompt_service,
                )
        except Exception:
            logger.exception("Failed to create %s provider", provider_name)
            return None

    def _get_default_model_for_provider(self, provider: str) -> str:
        """Get default model name for a provider.

        Args:
            provider: The provider name.

        Returns:
            Default model name.
        """
        # Use hardcoded default model names matching config.py defaults
        # AppConfig only stores assistant_model_name and summarizer_model_name
        # which are already resolved based on the provider during load_config()
        defaults = {
            Provider.OPENAI: "gpt-5-mini",  # DEFAULT_OPENAI_ASSISTANT_MODEL
            Provider.ZAI: "glm-4.7",  # DEFAULT_ZAI_ASSISTANT_MODEL
            Provider.GEMINI: "gemini-2.5-flash",  # DEFAULT_GEMINI_ASSISTANT_MODEL
        }
        return defaults.get(provider, self.config.assistant_model_name)


class LLMService:
    """Factory-like wrapper that selects the configured LLM provider.

    This service manages provider backends and provides a unified interface
    for LLM operations across different providers.
    """

    def __init__(self, config: AppConfig):
        """Initialize the LLM service.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.prompt_service = PromptService()
        self.image_usage_service = ImageUsageService()
        # Get provider before creating ModelUsageService to pass as default_provider
        assistant_provider = (config.assistant_llm_provider or Provider.GEMINI).lower()
        self.model_usage_service = ModelUsageService(default_provider=assistant_provider)

        # Initialize provider registry
        self._registry = ProviderRegistry(config, self.prompt_service)

        # Create primary backends
        summarizer_provider = (config.summarizer_llm_provider or assistant_provider).lower()

        # Create assistant backend
        self.assistant_backend = self._registry.get_or_create_provider(
            assistant_provider, config.assistant_model_name
        )
        if not self.assistant_backend:
            raise ProviderUnavailableException(
                f"Cannot create assistant provider: {assistant_provider}"
            )

        # Create or share summarizer backend
        if assistant_provider == summarizer_provider:
            self.summarizer_backend = self.assistant_backend
        else:
            self.summarizer_backend = self._registry.get_or_create_provider(
                summarizer_provider, config.summarizer_model_name
            )
            if not self.summarizer_backend:
                raise ProviderUnavailableException(
                    f"Cannot create summarizer provider: {summarizer_provider}"
                )

        # Provider label for display
        self.provider_label = self._get_provider_label(assistant_provider)

    def _get_provider_label(self, provider: str) -> str:
        """Get display label for a provider.

        Args:
            provider: The provider name.

        Returns:
            Display label for the provider.
        """
        labels = {
            Provider.OPENAI: "OpenAI",
            Provider.ZAI: "Z.AI",
            Provider.GEMINI: "Gemini",
        }
        return labels.get(provider.lower(), provider)

    def get_backend_for_model(self, model_alias: str) -> Optional[BaseLLMService]:
        """Retrieve the appropriate backend service for a given model alias.

        This method lazy-loads the provider if not already initialized.

        Args:
            model_alias: The model alias to get the backend for.

        Returns:
            The provider backend, or None if unavailable.
        """
        # Resolve target provider from model alias
        target_def = self.model_usage_service.MODEL_DEFINITIONS.get(model_alias)
        target_provider = target_def.provider if target_def else Provider.GEMINI

        # Check if current assistant backend matches
        if self._is_provider_type(self.assistant_backend, target_provider):
            return self.assistant_backend

        # Get or create the target provider
        api_model_name = self.model_usage_service.get_api_model_name(model_alias)
        return self._registry.get_or_create_provider(target_provider, api_model_name)

    def _is_provider_type(self, backend: BaseLLMService, provider: str) -> bool:
        """Check if a backend is of a specific provider type.

        Args:
            backend: The backend to check.
            provider: The provider name to match.

        Returns:
            True if backend matches the provider type.
        """
        if provider == Provider.OPENAI:
            return isinstance(backend, OpenAIService)
        elif provider == Provider.ZAI:
            return isinstance(backend, ZAIService)
        else:  # Gemini
            return isinstance(backend, GeminiService)

    def create_chat_session_for_alias(self, model_alias: str, system_instruction: str) -> Any:
        """Create a chat session for the given model alias.

        Args:
            model_alias: The model alias to create a session for.
            system_instruction: The system instruction for the session.

        Returns:
            A chat session object.

        Raises:
            ProviderUnavailableException: If the backend is unavailable.
        """
        backend = self.get_backend_for_model(model_alias)
        if not backend:
            logger.warning(
                f"Backend unavailable for {model_alias}, using default assistant backend"
            )
            backend = self.assistant_backend

        api_model_name = self.model_usage_service.get_api_model_name(model_alias)

        # Use provider-specific session creation
        if isinstance(backend, GeminiService):
            model = backend._get_or_create_model(api_model_name, system_instruction)
            return model.start_chat(system_instruction)
        elif isinstance(backend, OpenAIService):
            model = backend._get_or_create_assistant(api_model_name, system_instruction)
            return model.start_chat(system_instruction)

        # Fallback for other providers
        model = backend.create_assistant_model(system_instruction)
        if hasattr(model, "start_chat"):
            return model.start_chat(system_instruction)
        return model

    def create_assistant_model(self, system_instruction: str, use_cache: bool = True) -> Any:
        """Create an assistant model with the default backend.

        Args:
            system_instruction: The system instruction.
            use_cache: Whether to use context caching.

        Returns:
            A model instance.
        """
        return self.assistant_backend.create_assistant_model(
            system_instruction, use_cache=use_cache
        )

    async def summarize_text(self, text: str) -> Optional[str]:
        """Summarize the given text.

        Args:
            text: The text to summarize.

        Returns:
            The summarized text, or None if summarization failed.
        """
        return await self.summarizer_backend.summarize_text(text)

    async def generate_prompt_from_concept(self, concept: str) -> Optional[str]:
        """Generate a detailed system prompt from a simple concept using Meta Prompt."""
        # Different backends have different APIs for generating content
        # Use the appropriate method based on the backend type

        # For OpenAI and ZAI backends, use direct client API calls
        from persbot.services.zai_service import ZAIService
        from persbot.services.openai_service import OpenAIService

        if isinstance(self.summarizer_backend, (OpenAIService, ZAIService)):
            # Use client.chat.completions.create directly with META_PROMPT as system instruction
            return await self.summarizer_backend.execute_with_retry(
                lambda: self.summarizer_backend.client.chat.completions.create(
                    model=self.summarizer_backend._summary_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": META_PROMPT,
                        },
                        {"role": "user", "content": concept},
                    ],
                    temperature=getattr(self.config, "temperature", 1.0),
                    top_p=getattr(self.config, "top_p", 1.0),
                ),
                "프롬프트 생성",
                timeout=60.0,
            )

        # For Gemini backend, use the existing pattern with generate_content
        meta_model = self.summarizer_backend.create_assistant_model(META_PROMPT, use_cache=False)
        result = await self.summarizer_backend.execute_with_retry(
            lambda: meta_model.generate_content(concept),
            "프롬프트 생성",
            timeout=60.0,
        )
        return result

    async def generate_questions_from_concept(self, concept: str) -> Optional[str]:
        """Generate clarifying questions with sample answers from a persona concept.

        Returns a JSON string containing questions and sample answers.
        """

        from persbot.services.zai_service import ZAIService
        from persbot.services.openai_service import OpenAIService

        if isinstance(self.summarizer_backend, (OpenAIService, ZAIService)):
            raw_response = await self.summarizer_backend.execute_with_retry(
                lambda: self.summarizer_backend.client.chat.completions.create(
                    model=self.summarizer_backend._summary_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": QUESTION_GENERATION_PROMPT,
                        },
                        {"role": "user", "content": concept},
                    ],
                    temperature=0.7,
                ),
                "질문 생성",
                timeout=60.0,
            )

            # Extract content from response
            if hasattr(raw_response, "choices") and raw_response.choices:
                return raw_response.choices[0].message.content
            return raw_response

        # For Gemini backend
        question_model = self.summarizer_backend.create_assistant_model(
            QUESTION_GENERATION_PROMPT, use_cache=False
        )
        result = await self.summarizer_backend.execute_with_retry(
            lambda: question_model.generate_content(concept),
            "질문 생성",
            timeout=60.0,
        )

        # For Gemini, extract text from response
        if hasattr(result, "text"):
            return result.text
        return result

    async def generate_prompt_from_concept_with_answers(
        self, concept: str, questions_and_answers: str
    ) -> Optional[str]:
        """Generate a detailed system prompt from concept and user answers.

        Args:
            concept: Original persona concept
            questions_and_answers: Formatted string of Q&A pairs

        Returns:
            Generated system prompt
        """
        from persbot.services.zai_service import ZAIService
        from persbot.services.openai_service import OpenAIService

        # Enhanced concept with answers
        enhanced_concept = (
            f"**Original Concept:** {concept}\n\n"
            f"**Additional Details from Interview:**\n{questions_and_answers}\n\n"
            f"Use these interview answers to create a more personalized and detailed persona."
        )

        if isinstance(self.summarizer_backend, (OpenAIService, ZAIService)):
            return await self.summarizer_backend.execute_with_retry(
                lambda: self.summarizer_backend.client.chat.completions.create(
                    model=self.summarizer_backend._summary_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": META_PROMPT,
                        },
                        {"role": "user", "content": enhanced_concept},
                    ],
                    temperature=getattr(self.config, "temperature", 1.0),
                    top_p=getattr(self.config, "top_p", 1.0),
                ),
                "프롬프트 생성",
                timeout=60.0,
            )

        # For Gemini backend
        meta_model = self.summarizer_backend.create_assistant_model(META_PROMPT, use_cache=False)
        result = await self.summarizer_backend.execute_with_retry(
            lambda: meta_model.generate_content(enhanced_concept),
            "프롬프트 생성",
            timeout=60.0,
        )
        return result

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        discord_message,
        use_summarizer_backend: bool = False,
        tools: Optional[List[Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
        images: Optional[List[bytes]] = None,
    ) -> Optional[Tuple[Optional[str], Any]]:
        """Generate a chat response.

        Args:
            chat_session: The chat session.
            user_message: The user's message.
            discord_message: The Discord message(s).
            use_summarizer_backend: Whether to use summarizer backend.
            tools: Optional list of tools for function calling.
            cancel_event: Optional cancellation event.
            images: Optional pre-extracted images (e.g., for retry operations).

        Returns:
            Tuple of (response_text, response_obj) or None.
        """
        # Extract metadata
        model_alias = getattr(
            chat_session, "model_alias", self.model_usage_service.DEFAULT_MODEL_ALIAS
        )
        user_id, channel_id, guild_id, primary_author = self._extract_message_metadata(
            discord_message
        )

        # Check usage limits
        (
            is_allowed,
            final_alias,
            notification,
        ) = await self.model_usage_service.check_and_increment_usage(guild_id, model_alias)
        if final_alias != model_alias:
            # Check if provider changed - need to recreate session if so
            original_def = self.model_usage_service.MODEL_DEFINITIONS.get(model_alias)
            final_def = self.model_usage_service.MODEL_DEFINITIONS.get(final_alias)
            original_provider = original_def.provider if original_def else None
            final_provider = final_def.provider if final_def else None

            if original_provider != final_provider:
                # Provider changed - create new session for the new provider
                system_instruction = getattr(chat_session, '_system_instruction', '')
                if not system_instruction and hasattr(chat_session, '_history'):
                    # Try to extract from first system message in history
                    for msg in chat_session._history:
                        if hasattr(msg, 'role') and msg.role == 'system':
                            system_instruction = getattr(msg, 'content', '')
                            break

                new_session = self.create_chat_session_for_alias(final_alias, system_instruction)
                # Copy history if possible
                if hasattr(chat_session, 'history') and hasattr(new_session, 'history'):
                    new_session.history = chat_session.history
                chat_session = new_session
            else:
                chat_session.model_alias = final_alias

        if not is_allowed:
            return (notification or "❌ 사용 한도를 초과했습니다.", None)

        # Get backend
        api_model_name = self.model_usage_service.get_api_model_name(final_alias)
        active_backend = (
            self.summarizer_backend
            if use_summarizer_backend
            else self.get_backend_for_model(final_alias)
        )

        if not active_backend:
            return ("❌ 선택한 모델을 사용할 수 없습니다.", None)

        # Generate response
        response = await active_backend.generate_chat_response(
            chat_session,
            user_message,
            discord_message,
            model_name=api_model_name,
            tools=tools,
            cancel_event=cancel_event,
            images=images,
        )

        return self._prepare_response_with_notification(response, notification)

    async def generate_chat_response_stream(
        self,
        chat_session,
        user_message: str,
        discord_message,
        use_summarizer_backend: bool = False,
        tools: Optional[List[Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response.

        Yields text chunks as they arrive from the API.
        Chunks are yielded after each line break for faster initial response.

        Args:
            chat_session: The chat session.
            user_message: The user's message.
            discord_message: The Discord message(s).
            use_summarizer_backend: Whether to use summarizer backend.
            tools: Optional list of tools for function calling.
            cancel_event: Optional cancellation event.

        Yields:
            Text chunks as they are generated.
        """
        # Extract metadata
        model_alias = getattr(
            chat_session, "model_alias", self.model_usage_service.DEFAULT_MODEL_ALIAS
        )
        user_id, channel_id, guild_id, primary_author = self._extract_message_metadata(
            discord_message
        )

        # Check usage limits
        (
            is_allowed,
            final_alias,
            notification,
        ) = await self.model_usage_service.check_and_increment_usage(guild_id, model_alias)
        if final_alias != model_alias:
            # Check if provider changed - need to recreate session if so
            original_def = self.model_usage_service.MODEL_DEFINITIONS.get(model_alias)
            final_def = self.model_usage_service.MODEL_DEFINITIONS.get(final_alias)
            original_provider = original_def.provider if original_def else None
            final_provider = final_def.provider if final_def else None

            if original_provider != final_provider:
                # Provider changed - create new session for the new provider
                system_instruction = getattr(chat_session, '_system_instruction', '')
                if not system_instruction and hasattr(chat_session, '_history'):
                    # Try to extract from first system message in history
                    for msg in chat_session._history:
                        if hasattr(msg, 'role') and msg.role == 'system':
                            system_instruction = getattr(msg, 'content', '')
                            break

                new_session = self.create_chat_session_for_alias(final_alias, system_instruction)
                # Copy history if possible
                if hasattr(chat_session, 'history') and hasattr(new_session, 'history'):
                    new_session.history = chat_session.history
                chat_session = new_session
            else:
                chat_session.model_alias = final_alias

        if not is_allowed:
            yield notification or "❌ 사용 한도를 초과했습니다."
            return

        # Get backend
        api_model_name = self.model_usage_service.get_api_model_name(final_alias)
        active_backend = (
            self.summarizer_backend
            if use_summarizer_backend
            else self.get_backend_for_model(final_alias)
        )

        if not active_backend:
            yield "❌ 선택한 모델을 사용할 수 없습니다."
            return

        # Check if backend supports streaming
        if not hasattr(active_backend, "generate_chat_response_stream"):
            # Fall back to non-streaming
            logger.warning(
                "Backend %s does not support streaming, falling back", type(active_backend).__name__
            )
            result = await active_backend.generate_chat_response(
                chat_session,
                user_message,
                discord_message,
                model_name=api_model_name,
                tools=tools,
                cancel_event=cancel_event,
            )
            if result:
                text, _ = result
                if notification:
                    yield f"📢 {notification}\n\n{text}"
                else:
                    yield text
            return

        # Prepend notification if exists
        if notification:
            yield f"📢 {notification}\n\n"

        # Stream response
        async for chunk in active_backend.generate_chat_response_stream(
            chat_session,
            user_message,
            discord_message,
            model_name=api_model_name,
            tools=tools,
            cancel_event=cancel_event,
        ):
            yield chunk

    def get_active_backend(
        self, chat_session, use_summarizer_backend: bool = False
    ) -> Optional[BaseLLMService]:
        """Get the active backend for a chat session."""
        if use_summarizer_backend:
            return self.summarizer_backend
        model_alias = getattr(
            chat_session, "model_alias", self.model_usage_service.DEFAULT_MODEL_ALIAS
        )
        return self.get_backend_for_model(model_alias) or self.assistant_backend

    async def send_tool_results(
        self,
        chat_session,
        tool_rounds,
        tools=None,
        use_summarizer_backend: bool = False,
        discord_message=None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[Optional[str], Any]]:
        """Send tool results back to the model and get continuation response.

        Args:
            chat_session: The chat session.
            tool_rounds: List of (response_obj, tool_results) tuples.
            tools: Tool definitions for the API call.
            use_summarizer_backend: Whether to use summarizer backend.
            discord_message: Discord message for error notifications.

        Returns:
            Tuple of (response_text, response_obj) or None.
        """
        active_backend = self.get_active_backend(chat_session, use_summarizer_backend)

        if not active_backend or not hasattr(active_backend, "send_tool_results"):
            logger.warning("Active backend does not support send_tool_results")
            return None

        return await active_backend.send_tool_results(
            chat_session,
            tool_rounds,
            tools=tools,
            discord_message=discord_message,
            cancel_event=cancel_event,
        )

    def extract_function_calls_from_response(
        self, backend: BaseLLMService, response: Any
    ) -> List[Dict[str, Any]]:
        """Extract function calls from a backend response.

        Args:
            backend: The backend service that generated the response.
            response: The response object.

        Returns:
            List of function call dictionaries.
        """
        return backend.extract_function_calls(response)

    def _extract_message_metadata(self, discord_message) -> tuple:
        """Extract user_id, channel_id, guild_id, and primary_author from message(s).

        Args:
            discord_message: Single message or list of messages.

        Returns:
            Tuple of (user_id, channel_id, guild_id, primary_author).
        """
        return extract_message_metadata(discord_message)

    def _prepare_response_with_notification(
        self, response, notification: Optional[str]
    ) -> Optional[Tuple[Optional[str], Any]]:
        """Prepend notification to response if exists.

        Args:
            response: The response tuple (text, obj).
            notification: Optional notification message.

        Returns:
            Response tuple with notification prepended.
        """
        if response and notification:
            text, obj = response
            return (f"📢 {notification}\n\n{text}", obj)
        return response

    def get_user_role_name(self) -> str:
        """Pass through to the active assistant backend."""
        return self.assistant_backend.get_user_role_name()

    def get_assistant_role_name(self) -> str:
        """Pass through to the active assistant backend."""
        return self.assistant_backend.get_assistant_role_name()

    def update_parameters(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        thinking_budget: Optional[int] = _UNSET,
    ) -> None:
        """Update model parameters and reload backends.

        Args:
            temperature: Optional temperature value (0.0-2.0).
            top_p: Optional top-p value (0.0-1.0).
            thinking_budget: Optional thinking budget in tokens. Pass None to disable.
        """
        if temperature is not None:
            self.config.temperature = temperature
        if top_p is not None:
            self.config.top_p = top_p
        if thinking_budget is not _UNSET:
            self.config.thinking_budget = thinking_budget

        # Reload all registered backends
        backends_to_reload = [self.assistant_backend]
        if self.summarizer_backend is not self.assistant_backend:
            backends_to_reload.append(self.summarizer_backend)

        # Also reload any auxiliary backends
        for backend in self._registry._backends.values():
            if backend not in backends_to_reload:
                backends_to_reload.append(backend)

        for backend in backends_to_reload:
            if hasattr(backend, "reload_parameters"):
                backend.reload_parameters()


    def start_background_cache_warmup(self) -> None:
        """Start background cache warmup for Gemini backends to reduce first-message latency.

        This should be called after bot startup to pre-create commonly used caches.
        """
        # Warm up assistant backend if it's a Gemini service
        if hasattr(self.assistant_backend, "start_background_warmup"):
            self.assistant_backend.start_background_warmup()

        # Warm up summarizer backend if it's different and is a Gemini service
        if (
            self.summarizer_backend is not self.assistant_backend
            and hasattr(self.summarizer_backend, "start_background_warmup")
        ):
            self.summarizer_backend.start_background_warmup()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from the assistant backend.

        Returns:
            Dict with cache metrics (hits, misses, hit_rate, tokens_saved)
            or empty dict if caching not supported.
        """
        if hasattr(self.assistant_backend, "get_cache_stats"):
            return self.assistant_backend.get_cache_stats()
        return {}

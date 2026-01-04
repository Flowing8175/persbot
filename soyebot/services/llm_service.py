"""LLM service selector for SoyeBot."""

import logging
from typing import Optional

from soyebot.config import AppConfig
from soyebot.services.base import BaseLLMService
from soyebot.services.gemini_service import GeminiService
from soyebot.services.openai_service import OpenAIService
from soyebot.services.prompt_service import PromptService
from soyebot.services.usage_service import ImageUsageService
from soyebot.services.model_usage_service import ModelUsageService
from soyebot.prompts import META_PROMPT
import discord

logger = logging.getLogger(__name__)


class LLMService:
    """Factory-like wrapper that selects the configured LLM provider."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.prompt_service = PromptService()
        self.image_usage_service = ImageUsageService()
        self.model_usage_service = ModelUsageService()

        # Cache for lazy-loaded auxiliary backends (e.g. OpenAI when Gemini is default)
        self._aux_backends = {}

        assistant_provider = (config.assistant_llm_provider or 'gemini').lower()
        summarizer_provider = (config.summarizer_llm_provider or assistant_provider).lower()

        self.assistant_backend = self._create_backend(
            assistant_provider,
            assistant_model_name=config.assistant_model_name,
            summary_model_name=config.summarizer_model_name if assistant_provider == summarizer_provider else None,
        )

        if assistant_provider == summarizer_provider:
            self.summarizer_backend = self.assistant_backend
        else:
            self.summarizer_backend = self._create_backend(
                summarizer_provider,
                assistant_model_name=config.summarizer_model_name,
                summary_model_name=config.summarizer_model_name,
            )

        provider_label = 'OpenAI' if assistant_provider == 'openai' else 'Gemini'
        self.provider_label = provider_label
        logger.info(
            "LLM provider 설정: assistant=%s, summarizer=%s", assistant_provider, summarizer_provider
        )

    def _create_backend(
        self,
        provider: str,
        *,
        assistant_model_name: str,
        summary_model_name: Optional[str] = None,
    ):
        if provider == 'openai':
            return OpenAIService(
                self.config,
                assistant_model_name=assistant_model_name,
                summary_model_name=summary_model_name,
                prompt_service=self.prompt_service
            )
        return GeminiService(
            self.config,
            assistant_model_name=assistant_model_name,
            summary_model_name=summary_model_name,
            prompt_service=self.prompt_service
        )

    def get_backend_for_model(self, model_alias: str) -> Optional[BaseLLMService]:
        """
        Retrieve the appropriate backend service for a given model alias.
        Lazy loads the service if not already initialized.
        """
        # Resolve target provider
        target_def = self.model_usage_service.MODEL_DEFINITIONS.get(model_alias)
        target_provider = target_def.provider if target_def else 'gemini'

        # Check current assistant backend
        current_provider = 'openai' if isinstance(self.assistant_backend, OpenAIService) else 'gemini'

        if target_provider == current_provider:
            return self.assistant_backend

        # Check cached aux backends
        if target_provider in self._aux_backends:
            return self._aux_backends[target_provider]

        # Create new backend if needed
        # We initialize with the specific model requested, though services should handle dynamic models
        api_model_name = self.model_usage_service.get_api_model_name(model_alias)

        if target_provider == 'openai':
            if not self.config.openai_api_key:
                logger.warning("OpenAI API key missing, cannot switch to OpenAI model.")
                return None
            service = OpenAIService(self.config, assistant_model_name=api_model_name, prompt_service=self.prompt_service)
            self._aux_backends['openai'] = service
            return service

        elif target_provider == 'gemini':
            if not self.config.gemini_api_key:
                logger.warning("Gemini API key missing, cannot switch to Gemini model.")
                return None
            service = GeminiService(self.config, assistant_model_name=api_model_name, prompt_service=self.prompt_service)
            self._aux_backends['gemini'] = service
            return service

        return None

    def create_chat_session_for_alias(self, model_alias: str, system_instruction: str):
        """Create a chat session (model wrapper) appropriate for the given model alias."""
        backend = self.get_backend_for_model(model_alias)
        if not backend:
            # Fallback to default if backend unavailable
            logger.warning(f"Backend unavailable for alias {model_alias}. Falling back to default assistant backend.")
            backend = self.assistant_backend

        # We need to make sure the backend uses the correct model name for session creation
        # Backend.create_assistant_model uses its internal _assistant_model_name.
        # But we want a specific model.
        # We can update the service's model name? No, dangerous if shared.
        # We need `create_model(model_name, system_instruction)`.
        # Existing services have `create_assistant_model` which uses default.
        # `GeminiService` has `_get_or_create_model`.
        # `OpenAIService` has `_get_or_create_assistant`.
        # Both are internal/protected.
        # But `LLMService` can't access protected members easily (it can but it's dirty).

        # Let's inspect BaseLLMService? No common method for creating arbitrary model session.
        # However, `create_assistant_model` is public.
        # If we just use `backend.create_assistant_model`, it uses the backend's configured default.
        # This is WRONG if the backend was cached with a different default.

        # SOLUTION:
        # If the backend supports creating a model with specific name, use it.
        # `GeminiService`: `_get_or_create_model(name, instr)`
        # `OpenAIService`: `_get_or_create_assistant(name, instr)`

        # I will access them directly as we are in the same package scope context effectively, or rely on python access.

        api_model_name = self.model_usage_service.get_api_model_name(model_alias)

        if isinstance(backend, GeminiService):
            model = backend._get_or_create_model(api_model_name, system_instruction)
            return model.start_chat(system_instruction)
        elif isinstance(backend, OpenAIService):
            model = backend._get_or_create_assistant(api_model_name, system_instruction)
            return model.start_chat(system_instruction)

        model = backend.create_assistant_model(system_instruction)
        if hasattr(model, 'start_chat'):
            return model.start_chat(system_instruction)
        return model

    def create_assistant_model(self, system_instruction: str, use_cache: bool = True):
        # Legacy method delegating to default backend
        return self.assistant_backend.create_assistant_model(system_instruction, use_cache=use_cache)

    async def summarize_text(self, text: str):
        return await self.summarizer_backend.summarize_text(text)

    async def generate_prompt_from_concept(self, concept: str) -> Optional[str]:
        """Generate a detailed system prompt from a simple concept using Meta Prompt."""
        # Use a powerful model for this task (usually the summarizer or assistant model)
        # We'll create a temporary model instance with the META_PROMPT
        # Disable caching for the meta prompt generation itself
        meta_model = self.summarizer_backend.create_assistant_model(META_PROMPT, use_cache=False)
        
        if hasattr(self.summarizer_backend, 'assistant_model'):
             # Create meta model
             result = await self.summarizer_backend.execute_with_retry(
                 lambda: meta_model.generate_content(concept),
                 "프롬프트 생성",
                 timeout=60.0
             )
             return result
        return None

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        discord_message,
        use_summarizer_backend: bool = False,
    ):
        # 0. Check usage limit
        model_alias = getattr(chat_session, 'model_alias', self.model_usage_service.DEFAULT_MODEL_ALIAS)

        # Determine user_id, channel_id, and guild_id
        user_id = 0
        channel_id = 0
        guild_id = 0
        primary_author = None

        if isinstance(discord_message, list):
             if discord_message:
                primary_author = discord_message[0].author
                user_id = primary_author.id
                channel_id = discord_message[0].channel.id
                if discord_message[0].guild:
                    guild_id = discord_message[0].guild.id
                else:
                    guild_id = user_id
        else:
             primary_author = discord_message.author
             user_id = primary_author.id
             channel_id = discord_message.channel.id
             if discord_message.guild:
                 guild_id = discord_message.guild.id
             else:
                 guild_id = user_id

        is_allowed, final_alias, notification = await self.model_usage_service.check_and_increment_usage(
            guild_id, model_alias
        )

        # Update session if model changed due to fallback
        if final_alias != model_alias:
             chat_session.model_alias = final_alias

        if not is_allowed:
             return (notification or "❌ 사용 한도를 초과했습니다.", None)

        # Resolve API model name
        api_model_name = self.model_usage_service.get_api_model_name(final_alias)

        # Get appropriate backend
        active_backend = self.get_backend_for_model(final_alias)
        if not active_backend:
             return ("❌ 선택한 모델을 사용할 수 없습니다 (Provider 설정 오류).", None)

        if use_summarizer_backend:
            # Summarizer overrides
            active_backend = self.summarizer_backend

        # 1. Rate Limiting Check for Images
        image_count = 0

        if isinstance(discord_message, list):
             for msg in discord_message:
                 image_count += len([a for a in msg.attachments if a.content_type and a.content_type.startswith('image/')])
        else:
             image_count = len([a for a in discord_message.attachments if a.content_type and a.content_type.startswith('image/')])

        if image_count > 0 and primary_author:
            # Check permissions (Member only has guild_permissions)
            is_admin = False
            if isinstance(primary_author, discord.Member):
                 is_admin = primary_author.guild_permissions.manage_guild

            if not is_admin:
                if not self.image_usage_service.check_can_upload(primary_author.id, image_count, limit=3):
                    return ("❌ 이미지는 하루에 최대 3개 업로드하실 수 있습니다.", None)

        # 2. Proceed with generation
        # We pass `model_name` argument to generate_chat_response to ensure dynamic switching inside session works
        response = await active_backend.generate_chat_response(
            chat_session,
            user_message,
            discord_message,
            model_name=api_model_name
        )

        # 3. If successful and images were sent, record usage
        if response is not None and image_count > 0 and primary_author:
             is_admin = False
             if isinstance(primary_author, discord.Member):
                 is_admin = primary_author.guild_permissions.manage_guild

             if not is_admin:
                 await self.image_usage_service.record_upload(primary_author.id, image_count)

        # Prepend notification if exists
        if response and notification:
             text, obj = response
             new_text = f"📢 {notification}\n\n{text}"
             return (new_text, obj)

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
        thinking_budget: Optional[Optional[int]] = -1 # Special default for "not provided"
    ) -> None:
        """Update model parameters and reload backends."""
        if temperature is not None:
            self.config.temperature = temperature
        if top_p is not None:
            self.config.top_p = top_p
        if thinking_budget != -1:
            self.config.thinking_budget = thinking_budget

        # Reload backends to pick up new config
        if hasattr(self.assistant_backend, 'reload_parameters'):
            self.assistant_backend.reload_parameters()

        # Only reload summarizer if it's a different instance (though reload is safe either way)
        if (self.summarizer_backend is not self.assistant_backend and
            hasattr(self.summarizer_backend, 'reload_parameters')):
            self.summarizer_backend.reload_parameters()

        # Also reload auxiliary services if they exist
        for key, backend in self._aux_backends.items():
            if hasattr(backend, 'reload_parameters'):
                backend.reload_parameters()

        logger.info("Updated parameters: temperature=%s, top_p=%s, thinking_budget=%s", 
                    self.config.temperature, self.config.top_p, self.config.thinking_budget)

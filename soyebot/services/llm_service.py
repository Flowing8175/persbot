"""LLM service selector for SoyeBot."""

import logging
from typing import Optional

from soyebot.config import AppConfig
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

    def create_assistant_model(self, system_instruction: str, use_cache: bool = True):
        return self.assistant_backend.create_assistant_model(system_instruction, use_cache=use_cache)

    async def summarize_text(self, text: str):
        return await self.summarizer_backend.summarize_text(text)

    async def generate_prompt_from_concept(self, concept: str) -> Optional[str]:
        """Generate a detailed system prompt from a simple concept using Meta Prompt."""
        # Use a powerful model for this task (usually the summarizer or assistant model)
        # We'll create a temporary model instance with the META_PROMPT
        # Disable caching for the meta prompt generation itself
        meta_model = self.summarizer_backend.create_assistant_model(META_PROMPT, use_cache=False)
        
        # We manually call a generation method. BaseLLMService might not have one, 
        # so we'll ensure backend has a clean way.
        # For now, we can use hypothesize a 'generate_text' on backend or use existing ones.
        # summarizer_backend.summarize_text(text) does: model.generate_content(prompt)
        # Let's add 'generate_response_with_system_instruction' to backends.
        
        # Actually, let's just make it simple if the backend supports it.
        # summarizer_backend.summary_model is a _CachedModel with SUMMARY_SYSTEM_INSTRUCTION.
        # We want one with META_PROMPT.
        
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

        # Determine user_id and channel_id
        user_id = 0
        channel_id = 0
        primary_author = None

        if isinstance(discord_message, list):
             if discord_message:
                primary_author = discord_message[0].author
                user_id = primary_author.id
                channel_id = discord_message[0].channel.id
        else:
             primary_author = discord_message.author
             user_id = primary_author.id
             channel_id = discord_message.channel.id

        is_allowed, final_alias, notification = await self.model_usage_service.check_and_increment_usage(
            user_id, channel_id, model_alias
        )

        # Update session if model changed
        if final_alias != model_alias:
             chat_session.model_alias = final_alias

        if not is_allowed:
             return (notification or "❌ 사용 한도를 초과했습니다.", None)

        # Resolve API model name
        api_model_name = self.model_usage_service.get_api_model_name(final_alias)

        # Check provider mismatch? For now we assume aliases map to current provider logic or we might need multi-provider switch.
        # The MODEL_DEFINITIONS has 'provider'. We should check if current backend matches.
        # But `LLMService` initializes one backend. Swapping backend on the fly is complex.
        # The user request implies selecting models, some are Gemini, some GPT.
        # If user selects GPT model but backend is Gemini, we need to switch backend?
        # Or `LLMService` should just instantiate both backends?
        # For simplicity, we assume `assistant_backend` can handle the request if it's the right provider,
        # OR we need to dynamically pick the backend.

        # Current architecture has `self.assistant_backend` set at init.
        # To support switching providers per request, we need both services available.
        # Let's lazy load or keep both if needed.

        target_def = self.model_usage_service.MODEL_DEFINITIONS.get(final_alias)
        target_provider = target_def.provider if target_def else 'gemini'

        active_backend = None
        if target_provider == 'openai':
             # We need an OpenAI backend.
             # If `self.assistant_backend` is OpenAI, use it.
             # Else we need to instantiate/get one.
             if isinstance(self.assistant_backend, OpenAIService):
                 active_backend = self.assistant_backend
             else:
                 # Check if we have API key
                 if self.config.openai_api_key:
                     # Create temporary or cached service
                     # For efficiency, we really should have initialized both if keys exist.
                     # But let's create on fly for now or assume user config matches provider.
                     # Actually, `!model` allows selecting ANY. So we MUST support switching.
                     # Let's create a cached OpenAI service instance in LLMService.
                     if not hasattr(self, '_openai_service'):
                          self._openai_service = OpenAIService(self.config, assistant_model_name=api_model_name, prompt_service=self.prompt_service)
                     active_backend = self._openai_service
                 else:
                     return ("❌ OpenAI API 키가 설정되지 않아 해당 모델을 사용할 수 없습니다.", None)
        else:
             # Gemini
             if isinstance(self.assistant_backend, GeminiService):
                 active_backend = self.assistant_backend
             else:
                 if self.config.gemini_api_key:
                     if not hasattr(self, '_gemini_service'):
                          self._gemini_service = GeminiService(self.config, assistant_model_name=api_model_name, prompt_service=self.prompt_service)
                     active_backend = self._gemini_service
                 else:
                      return ("❌ Gemini API 키가 설정되지 않아 해당 모델을 사용할 수 없습니다.", None)

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
        # We pass `model_name` argument to generate_chat_response
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
        if hasattr(self, '_openai_service'):
            self._openai_service.reload_parameters()
        if hasattr(self, '_gemini_service'):
            self._gemini_service.reload_parameters()

        logger.info("Updated parameters: temperature=%s, top_p=%s, thinking_budget=%s", 
                    self.config.temperature, self.config.top_p, self.config.thinking_budget)

"""LLM service selector for SoyeBot."""

import logging
from typing import Optional

from soyebot.config import AppConfig
from soyebot.metrics import get_metrics
from soyebot.services.gemini_service import GeminiService
from soyebot.services.openai_service import OpenAIService
from soyebot.services.prompt_service import PromptService
from soyebot.prompts import META_PROMPT

logger = logging.getLogger(__name__)


class LLMService:
    """Factory-like wrapper that selects the configured LLM provider."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.prompt_service = PromptService()

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
        get_metrics().set_llm_provider(provider_label)
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
                 "프롬프트 생성"
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
        backend = self.summarizer_backend if use_summarizer_backend else self.assistant_backend
        return await backend.generate_chat_response(chat_session, user_message, discord_message)

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

        logger.info("Updated parameters: temperature=%s, top_p=%s, thinking_budget=%s", 
                    self.config.temperature, self.config.top_p, self.config.thinking_budget)

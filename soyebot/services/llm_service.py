"""LLM service selector for SoyeBot."""

import logging
from typing import Optional

from config import AppConfig
from metrics import get_metrics
from services.gemini_service import GeminiService
from services.openai_service import OpenAIService

logger = logging.getLogger(__name__)


class LLMService:
    """Factory-like wrapper that selects the configured LLM provider."""

    def __init__(self, config: AppConfig):
        self.config = config
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
            )
        return GeminiService(
            self.config,
            assistant_model_name=assistant_model_name,
            summary_model_name=summary_model_name,
        )

    def create_assistant_model(self, system_instruction: str):
        return self.assistant_backend.create_assistant_model(system_instruction)

    async def summarize_text(self, text: str):
        return await self.summarizer_backend.summarize_text(text)

    async def generate_chat_response(self, chat_session, user_message: str, discord_message):
        return await self.assistant_backend.generate_chat_response(chat_session, user_message, discord_message)


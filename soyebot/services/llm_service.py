"""LLM service selector for SoyeBot."""

import logging

from config import AppConfig
from metrics import get_metrics
from services.gemini_service import GeminiService
from services.openai_service import OpenAIService

logger = logging.getLogger(__name__)


class LLMService:
    """Factory-like wrapper that selects the configured LLM provider."""

    def __init__(self, config: AppConfig):
        provider = (config.llm_provider or 'gemini').lower()
        self.provider = provider

        if provider == 'openai':
            self.backend = OpenAIService(config)
            provider_label = 'OpenAI'
        else:
            self.backend = GeminiService(config)
            provider_label = 'Gemini'

        self.provider_label = provider_label
        get_metrics().set_llm_provider(provider_label)
        logger.info("LLM provider 설정: %s", provider_label)

    def create_assistant_model(self, system_instruction: str):
        return self.backend.create_assistant_model(system_instruction)

    async def summarize_text(self, text: str):
        return await self.backend.summarize_text(text)

    async def generate_chat_response(self, chat_session, user_message: str, discord_message):
        return await self.backend.generate_chat_response(chat_session, user_message, discord_message)


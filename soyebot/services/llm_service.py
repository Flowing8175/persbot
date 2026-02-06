"""LLM service selector for SoyeBot."""

import logging
from typing import Optional, List, Dict, Any

from soyebot.config import AppConfig
from soyebot.services.base import BaseLLMService
from soyebot.services.gemini_service import GeminiService
from soyebot.services.openai_service import OpenAIService
from soyebot.services.zai_service import ZAIService
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

        assistant_provider = (config.assistant_llm_provider or "gemini").lower()
        summarizer_provider = (
            config.summarizer_llm_provider or assistant_provider
        ).lower()

        self.assistant_backend = self._create_backend(
            assistant_provider,
            assistant_model_name=config.assistant_model_name,
            summary_model_name=config.summarizer_model_name
            if assistant_provider == summarizer_provider
            else None,
        )

        if assistant_provider == summarizer_provider:
            self.summarizer_backend = self.assistant_backend
        else:
            self.summarizer_backend = self._create_backend(
                summarizer_provider,
                assistant_model_name=config.summarizer_model_name,
                summary_model_name=config.summarizer_model_name,
            )

        provider_label = (
            "OpenAI"
            if assistant_provider == "openai"
            else "Z.AI"
            if assistant_provider == "zai"
            else "Gemini"
        )
        self.provider_label = provider_label
        logger.info(
            "LLM provider 설정: assistant=%s, summarizer=%s",
            assistant_provider,
            summarizer_provider,
        )

    def _create_backend(
        self,
        provider: str,
        *,
        assistant_model_name: str,
        summary_model_name: Optional[str] = None,
    ):
        if provider == "openai":
            return OpenAIService(
                self.config,
                assistant_model_name=assistant_model_name,
                summary_model_name=summary_model_name,
                prompt_service=self.prompt_service,
            )
        if provider == "zai":
            return ZAIService(
                self.config,
                assistant_model_name=assistant_model_name,
                summary_model_name=summary_model_name,
                prompt_service=self.prompt_service,
            )
        return GeminiService(
            self.config,
            assistant_model_name=assistant_model_name,
            summary_model_name=summary_model_name,
            prompt_service=self.prompt_service,
        )
        return GeminiService(
            self.config,
            assistant_model_name=assistant_model_name,
            summary_model_name=summary_model_name,
            prompt_service=self.prompt_service,
        )

    def get_backend_for_model(self, model_alias: str) -> Optional[BaseLLMService]:
        """
        Retrieve the appropriate backend service for a given model alias.
        Lazy loads the service if not already initialized.
        """
        # Resolve target provider
        target_def = self.model_usage_service.MODEL_DEFINITIONS.get(model_alias)
        target_provider = target_def.provider if target_def else "gemini"

        # Check current assistant backend
        current_provider = (
            "openai"
            if isinstance(self.assistant_backend, OpenAIService)
            else "zai"
            if isinstance(self.assistant_backend, ZAIService)
            else "gemini"
        )

        if target_provider == current_provider:
            return self.assistant_backend

        # Check cached aux backends
        if target_provider in self._aux_backends:
            return self._aux_backends[target_provider]

        # Create new backend if needed
        # We initialize with the specific model requested, though services should handle dynamic models
        api_model_name = self.model_usage_service.get_api_model_name(model_alias)

        if target_provider == "openai":
            if not self.config.openai_api_key:
                logger.warning("OpenAI API key missing, cannot switch to OpenAI model.")
                return None
            service = OpenAIService(
                self.config,
                assistant_model_name=api_model_name,
                prompt_service=self.prompt_service,
            )
            self._aux_backends["openai"] = service
            return service

        elif target_provider == "gemini":
            if not self.config.gemini_api_key:
                logger.warning("Gemini API key missing, cannot switch to Gemini model.")
                return None
            service = GeminiService(
                self.config,
                assistant_model_name=api_model_name,
                prompt_service=self.prompt_service,
            )
            self._aux_backends["gemini"] = service
            return service

        elif target_provider == "zai":
            if not self.config.zai_api_key:
                logger.warning("Z.AI API key missing, cannot switch to Z.AI model.")
                return None
            service = ZAIService(
                self.config,
                assistant_model_name=api_model_name,
                prompt_service=self.prompt_service,
            )
            self._aux_backends["zai"] = service
            return service

        return None

    def create_chat_session_for_alias(self, model_alias: str, system_instruction: str):
        """Create a chat session (model wrapper) appropriate for the given model alias."""
        backend = self.get_backend_for_model(model_alias)
        if not backend:
            # Fallback to default if backend unavailable
            logger.warning(
                f"Backend unavailable for alias {model_alias}. Falling back to default assistant backend."
            )
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
        if hasattr(model, "start_chat"):
            return model.start_chat(system_instruction)
        return model

    def create_assistant_model(self, system_instruction: str, use_cache: bool = True):
        # Legacy method delegating to default backend
        return self.assistant_backend.create_assistant_model(
            system_instruction, use_cache=use_cache
        )

    async def summarize_text(self, text: str):
        return await self.summarizer_backend.summarize_text(text)

    async def generate_prompt_from_concept(self, concept: str) -> Optional[str]:
        """Generate a detailed system prompt from a simple concept using Meta Prompt."""
        # Use a powerful model for this task (usually the summarizer or assistant model)
        # We'll create a temporary model instance with the META_PROMPT
        # Disable caching for the meta prompt generation itself
        meta_model = self.summarizer_backend.create_assistant_model(
            META_PROMPT, use_cache=False
        )

        if hasattr(self.summarizer_backend, "assistant_model"):
            # Create meta model
            result = await self.summarizer_backend.execute_with_retry(
                lambda: meta_model.generate_content(concept),
                "프롬프트 생성",
                timeout=60.0,
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
        # Extract message metadata
        model_alias = getattr(
            chat_session, "model_alias", self.model_usage_service.DEFAULT_MODEL_ALIAS
        )
        user_id, channel_id, guild_id, primary_author = self._extract_message_metadata(
            discord_message
        )

        # Check and update usage
        (
            is_allowed,
            final_alias,
            notification,
        ) = await self.model_usage_service.check_and_increment_usage(
            guild_id, model_alias
        )
        if final_alias != model_alias:
            chat_session.model_alias = final_alias

        if not is_allowed:
            return (notification or "❌ 사용 한도를 초과했습니다.", None)

        # Get backend and model name
        api_model_name = self.model_usage_service.get_api_model_name(final_alias)
        active_backend = (
            self.summarizer_backend
            if use_summarizer_backend
            else self.get_backend_for_model(final_alias)
        )

        if not active_backend:
            return ("❌ 선택한 모델을 사용할 수 없습니다 (Provider 설정 오류).", None)

        # Check image usage limits
        image_count = self._count_images_in_message(discord_message)
        if image_count > 0 and primary_author:
            limit_error = self._check_image_usage_limit(primary_author, image_count)
            if limit_error:
                return limit_error

        # Generate response
        response = await active_backend.generate_chat_response(
            chat_session, user_message, discord_message, model_name=api_model_name
        )

        # Record image usage after successful generation
        if response is not None and image_count > 0 and primary_author:
            await self._record_image_usage_if_needed(primary_author, image_count)

        return self._prepare_response_with_notification(response, notification)

    def get_tools_for_backend(self, backend: BaseLLMService, tools: List[Any]) -> Any:
        """Get tools in the format required by a specific backend.

        Args:
            backend: The backend service to format tools for.
            tools: List of tool definitions.

        Returns:
            Provider-specific tool format.
        """
        return backend.get_tools_for_provider(tools)

    def extract_function_calls_from_response(
        self,
        backend: BaseLLMService,
        response: Any
    ) -> List[Dict[str, Any]]:
        """Extract function calls from a backend response.

        Args:
            backend: The backend service that generated the response.
            response: The response object.

        Returns:
            List of function call dictionaries.
        """
        return backend.extract_function_calls(response)

    def format_function_results_for_backend(
        self,
        backend: BaseLLMService,
        results: List[Dict[str, Any]]
    ) -> Any:
        """Format function results for a specific backend.

        Args:
            backend: The backend service to format results for.
            results: List of tool execution results.

        Returns:
            Provider-specific formatted results.
        """
        return backend.format_function_results(results)

    def _extract_message_metadata(self, discord_message) -> tuple:
        """Extract user_id, channel_id, guild_id, and primary_author from message(s)."""
        if isinstance(discord_message, list) and discord_message:
            primary = discord_message[0]
        else:
            primary = discord_message

        primary_author = primary.author
        user_id = primary_author.id
        channel_id = primary.channel.id
        guild_id = primary.guild.id if primary.guild else user_id

        return user_id, channel_id, guild_id, primary_author

    def _count_images_in_message(self, discord_message) -> int:
        """Count image attachments in message(s)."""

        def count_in_msg(msg):
            return len(
                [
                    a
                    for a in msg.attachments
                    if a.content_type and a.content_type.startswith("image/")
                ]
            )

        if isinstance(discord_message, list):
            return sum(count_in_msg(msg) for msg in discord_message)
        return count_in_msg(discord_message)

    def _check_image_usage_limit(self, author, image_count: int) -> Optional[tuple]:
        """Check if user can upload images. Returns error tuple or None if allowed."""
        is_admin = (
            isinstance(author, discord.Member) and author.guild_permissions.manage_guild
        )
        # Bypass permission check if NO_CHECK_PERMISSION is set
        if self.config.no_check_permission:
            is_admin = True
        if is_admin:
            return None
        if not self.image_usage_service.check_can_upload(
            author.id, image_count, limit=3
        ):
            return ("❌ 이미지는 하루에 최대 3개 업로드하실 수 있습니다.", None)
        return None

    async def _record_image_usage_if_needed(self, author, image_count: int) -> None:
        """Record image usage for non-admin users."""
        is_admin = (
            isinstance(author, discord.Member) and author.guild_permissions.manage_guild
        )
        # Bypass permission check if NO_CHECK_PERMISSION is set
        if self.config.no_check_permission:
            is_admin = True
        if not is_admin:
            await self.image_usage_service.record_upload(author.id, image_count)

    def _prepare_response_with_notification(
        self, response, notification: Optional[str]
    ):
        """Prepend notification to response if exists."""
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
        thinking_budget: Optional[
            Optional[int]
        ] = -1,  # Special default for "not provided"
    ) -> None:
        """Update model parameters and reload backends."""
        if temperature is not None:
            self.config.temperature = temperature
        if top_p is not None:
            self.config.top_p = top_p
        if thinking_budget != -1:
            self.config.thinking_budget = thinking_budget

        # Reload backends to pick up new config
        if hasattr(self.assistant_backend, "reload_parameters"):
            self.assistant_backend.reload_parameters()

        # Only reload summarizer if it's a different instance (though reload is safe either way)
        if self.summarizer_backend is not self.assistant_backend and hasattr(
            self.summarizer_backend, "reload_parameters"
        ):
            self.summarizer_backend.reload_parameters()

        # Also reload auxiliary services if they exist
        for key, backend in self._aux_backends.items():
            if hasattr(backend, "reload_parameters"):
                backend.reload_parameters()

        logger.info(
            "Updated parameters: temperature=%s, top_p=%s, thinking_budget=%s",
            self.config.temperature,
            self.config.top_p,
            self.config.thinking_budget,
        )

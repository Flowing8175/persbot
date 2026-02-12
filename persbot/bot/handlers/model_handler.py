"""Model command handler.

This module handles model-related commands including switching models,
viewing model info, and parameter adjustments.
"""

import logging
from typing import Any, Optional

import discord
from discord.ext import commands

from persbot.bot.handlers.base_handler import BaseHandler
from persbot.config import AppConfig
from persbot.services.llm_service import LLMService
from persbot.services.model_usage_service import ModelUsageService

logger = logging.getLogger(__name__)


class ModelCommandHandler(BaseHandler):
    """Handler for model management commands."""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        model_usage_service: ModelUsageService,
    ):
        """Initialize the model handler.

        Args:
            bot: The Discord bot instance.
            config: Application configuration.
            llm_service: LLM service.
            model_usage_service: Model usage tracking service.
        """
        super().__init__(bot, config)
        self.llm_service = llm_service
        self.model_usage_service = model_usage_service

    async def handle_list(
        self,
        ctx: commands.Context,
    ) -> Optional[discord.Message]:
        """Handle listing all available models.

        Args:
            ctx: The command context.

        Returns:
            The response message.
        """
        models = self.model_usage_service.get_available_models()

        if not models:
            return await self.send_response(
                ctx,
                "사용 가능한 모델이 없습니다.",
            )

        embed = discord.Embed(
            title="사용 가능한 모델",
            description="",
            color=discord.Color.blue(),
        )

        # Group by provider
        by_provider: dict[str, list[tuple[str, str]]] = {}
        for alias, definition in models.items():
            provider = definition.provider
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append((alias, definition.display_name))

        for provider, model_list in by_provider.items():
            models_text = "\n".join(
                f"• **{alias}**: {display_name}" for alias, display_name in model_list
            )
            embed.add_field(
                name=provider.upper(),
                value=models_text,
                inline=False,
            )

        return await self.send_response(ctx, embed=embed)

    async def handle_switch(
        self,
        ctx: commands.Context,
        model_alias: str,
    ) -> Optional[discord.Message]:
        """Handle switching the active model.

        Args:
            ctx: The command context.
            model_alias: The model alias to switch to.

        Returns:
            The response message.
        """
        # Check if model exists
        model_def = self.model_usage_service.MODEL_DEFINITIONS.get(model_alias)

        if not model_def:
            return await self.send_error(
                ctx,
                f"모델 '{model_alias}'을 찾을 수 없습니다.",
            )

        # Get the display name
        display_name = model_def.display_name

        return await self.send_success(
            ctx,
            f"모델을 **{display_name}** ({model_alias})로 설정했습니다.",
        )

    async def handle_info(
        self,
        ctx: commands.Context,
        model_alias: Optional[str] = None,
    ) -> Optional[discord.Message]:
        """Handle showing model information.

        Args:
            ctx: The command context.
            model_alias: Optional model alias to show info for.

        Returns:
            The response message.
        """
        if model_alias:
            # Show specific model info
            model_def = self.model_usage_service.MODEL_DEFINITIONS.get(model_alias)

            if not model_def:
                return await self.send_error(
                    ctx,
                    f"모델 '{model_alias}'을 찾을 수 없습니다.",
                )

            embed = discord.Embed(
                title=f"모델 정보: {model_def.display_name}",
                color=discord.Color.blue(),
            )
            embed.add_field(name="Alias", value=model_alias, inline=True)
            embed.add_field(name="Provider", value=model_def.provider.upper(), inline=True)
            embed.add_field(name="API Model", value=model_def.api_model, inline=True)

            return await self.send_response(ctx, embed=embed)
        else:
            # Show current model info
            current = self.llm_service.provider_label
            return await self.send_response(
                ctx,
                f"현재 사용 중인 공급자: **{current}**",
            )

    async def handle_set_parameters(
        self,
        ctx: commands.Context,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        thinking_budget: Optional[int] = None,
    ) -> Optional[discord.Message]:
        """Handle setting model parameters.

        Args:
            ctx: The command context.
            temperature: Optional temperature (0.0-2.0).
            top_p: Optional top-p (0.0-1.0).
            thinking_budget: Optional thinking budget in tokens.

        Returns:
            The response message.
        """
        # Validate parameters
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            return await self.send_error(ctx, "temperature는 0.0에서 2.0 사이여야 합니다.")

        if top_p is not None and not (0.0 <= top_p <= 1.0):
            return await self.send_error(ctx, "top_p는 0.0에서 1.0 사이여야 합니다.")

        # Update parameters
        self.llm_service.update_parameters(
            temperature=temperature,
            top_p=top_p,
            thinking_budget=thinking_budget,
        )

        # Build response
        changes = []
        if temperature is not None:
            changes.append(f"temperature={temperature}")
        if top_p is not None:
            changes.append(f"top_p={top_p}")
        if thinking_budget is not None:
            changes.append(f"thinking_budget={thinking_budget}")

        return await self.send_success(
            ctx,
            f"파라미터를 업데이트했습니다: {', '.join(changes)}",
        )

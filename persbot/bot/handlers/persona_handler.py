"""Persona command handler.

This module handles persona-related commands including creation,
editing, and switching personas.
"""

import logging
from typing import Optional

import discord
from discord.ext import commands

from persbot.bot.handlers.base_handler import BaseHandler
from persbot.config import AppConfig
from persbot.services.llm_service import LLMService
from persbot.services.prompt_service import PromptService
from persbot.use_cases.prompt_use_case import PromptUseCase

logger = logging.getLogger(__name__)


class PersonaCommandHandler(BaseHandler):
    """Handler for persona management commands."""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        prompt_service: PromptService,
    ):
        """Initialize the persona handler.

        Args:
            bot: The Discord bot instance.
            config: Application configuration.
            llm_service: LLM service for prompt generation.
            prompt_service: Service for managing prompts.
        """
        super().__init__(bot, config)
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.prompt_use_case = PromptUseCase(config, llm_service)

    async def handle_create_from_concept(
        self,
        ctx: commands.Context,
        concept: str,
    ) -> Optional[discord.Message]:
        """Handle persona creation from a simple concept.

        Args:
            ctx: The command context.
            concept: The persona concept.

        Returns:
            The response message.
        """
        await self.defer_if_needed(ctx)

        # Generate prompt from concept
        response = await self.prompt_use_case.generate_prompt_from_concept(
            PromptUseCase.PromptGenerationRequest(concept=concept)
        )

        if not response.success:
            return await self.send_error(ctx, response.error or "페르소나 생성에 실패했습니다.")

        # Save the generated prompt as a new persona
        persona_name = f"Generated: {concept[:30]}"
        self.prompt_service.save_persona(persona_name, response.system_prompt)

        return await self.send_success(
            ctx,
            f"페르소나를 생성했습니다: **{persona_name}**\n\n{response.system_prompt[:200]}...",
        )

    async def handle_create_interview(
        self,
        ctx: commands.Context,
        concept: str,
    ) -> Optional[discord.Message]:
        """Handle persona creation with interview mode.

        Args:
            ctx: The command context.
            concept: The persona concept.

        Returns:
            The response message.
        """
        # Generate questions first
        response = await self.prompt_use_case.generate_questions(
            PromptUseCase.QuestionGenerationRequest(concept=concept)
        )

        if not response.success:
            return await self.send_error(ctx, response.error or "질문 생성에 실패했습니다.")

        # Create UI for interview
        # This is a simplified version - full implementation would use modals
        questions = response.questions
        return await self.send_response(
            ctx,
            f"질문을 생성했습니다 ({len(questions)}개). "
            "인터뷰를 진행하려면 각 질문에 답변해주세요.",
        )

    async def handle_list(
        self,
        ctx: commands.Context,
    ) -> Optional[discord.Message]:
        """Handle listing all available personas.

        Args:
            ctx: The command context.

        Returns:
            The response message.
        """
        personas = self.prompt_service.list_personas()

        if not personas:
            return await self.send_response(
                ctx,
                "사용 가능한 페르소나가 없습니다.",
            )

        # Format persona list
        embed = discord.Embed(
            title="사용 가능한 페르소나",
            description="",
            color=discord.Color.blue(),
        )

        for i, (name, description) in enumerate(personas, 1):
            embed.add_field(
                name=f"{i}. {name}",
                value=description or "설명 없음",
                inline=False,
            )

        return await self.send_response(ctx, embed=embed)

    async def handle_set(
        self,
        ctx: commands.Context,
        persona_name: str,
    ) -> Optional[discord.Message]:
        """Handle setting the active persona.

        Args:
            ctx: The command context.
            persona_name: The name of the persona to set.

        Returns:
            The response message.
        """
        prompt = self.prompt_service.get_persona(persona_name)

        if not prompt:
            return await self.send_error(
                ctx,
                f"페르소나 '{persona_name}'을 찾을 수 없습니다.",
            )

        # Set as active
        self.prompt_service.set_active_persona(persona_name)

        return await self.send_success(
            ctx,
            f"페르소나를 **{persona_name}**로 설정했습니다.",
        )

    async def handle_show(
        self,
        ctx: commands.Context,
    ) -> Optional[discord.Message]:
        """Handle showing the current active persona.

        Args:
            ctx: The command context.

        Returns:
            The response message.
        """
        active = self.prompt_service.get_active_persona()

        if not active:
            return await self.send_response(
                ctx,
                "활성화된 페르소나가 없습니다.",
            )

        persona_name, prompt = active

        embed = discord.Embed(
            title=f"활성 페르소나: {persona_name}",
            description=prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
            color=discord.Color.green(),
        )

        return await self.send_response(ctx, embed=embed)

"""Model Selector Cog for SoyeBot."""

import logging
from typing import Optional

import discord
from discord.ext import commands

from persbot.bot.session import SessionManager
from persbot.config import load_config
from persbot.services.image_model_service import (
    get_available_image_models,
    get_channel_image_model,
    set_channel_image_model,
)
from persbot.services.model_usage_service import ModelUsageService
from persbot.utils import send_discord_message

logger = logging.getLogger(__name__)


class ModelSelectorView(discord.ui.View):
    """View containing the model selection dropdown."""

    def __init__(
        self,
        session_manager: SessionManager,
        current_model: str,
        original_message: Optional[discord.Message] = None,
    ):
        super().__init__(timeout=60)
        self.session_manager = session_manager
        self.original_message = original_message

        # Populate options from ModelUsageService definitions
        options = []

        # Ensure we have access to the initialized definitions
        # Assuming the dict is populated since services are initialized in main.py
        definitions = ModelUsageService.MODEL_DEFINITIONS

        for alias, definition in definitions.items():
            # Add description if needed (e.g., daily limit)
            # Scope updated to '서버' (Guild)
            desc = f"1일 한도: {definition.daily_limit}회 (서버 공통)"

            options.append(
                discord.SelectOption(
                    label=alias,
                    description=desc,
                    default=(alias == current_model),
                    emoji=(
                        "🤖"
                        if definition.provider == "gemini"
                        else "⚡" if definition.provider == "zai" else "🧠"
                    ),
                )
            )

        self.add_item(ModelSelect(options))


class ModelSelect(discord.ui.Select):
    """Dropdown for selecting a model."""

    def __init__(self, options):
        super().__init__(
            placeholder="모델을 선택하세요...",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction):
        view: ModelSelectorView = self.view
        selected_alias = self.values[0]

        # Defer to allow time and prevent interaction failure if slow, though this op is fast.
        # But mostly to allow us to send a reply to the *original* message comfortably.
        await interaction.response.defer()

        # Update the session model
        # We use the channel ID from interaction
        view.session_manager.set_session_model(interaction.channel_id, selected_alias)

        confirmation_text = f"✅ 모델이 **{selected_alias}**로 변경되었습니다."

        # Logic change: Reply to the original !model command message, then delete the embed
        if view.original_message:
            try:
                await send_discord_message(
                    view.original_message, confirmation_text, mention_author=False
                )
            except (discord.NotFound, discord.HTTPException):
                # Fallback if original deleted: reply to interaction (as followup since deferred)
                await send_discord_message(interaction, confirmation_text)
        else:
            await send_discord_message(interaction, confirmation_text)

        # Delete the interaction message (the embed with dropdown)
        try:
            if interaction.message:
                await interaction.message.delete()
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass


class ModelSelectorCog(commands.Cog):
    """Cog for managing model selection."""

    def __init__(self, bot: commands.Bot, session_manager: SessionManager):
        self.bot = bot
        self.session_manager = session_manager
        # Load config for image model info
        self.config = load_config()

    @commands.hybrid_group(
        name="model",
        aliases=["모델"],
        description="사용할 AI 모델을 선택합니다.",
        invoke_without_command=True,
    )
    async def model_command(self, ctx: commands.Context):
        """사용할 AI 모델을 선택합니다 (기본: LLM 모델)."""
        # When invoked without subcommand, show LLM model selection
        await self.llm_subcommand(ctx)

    @model_command.command(
        name="llm", description="사용할 LLM 모델을 선택합니다."
    )
    async def llm_subcommand(self, ctx: commands.Context):
        """사용할 LLM 모델을 선택합니다."""

        # Determine current model for this channel
        session_key = f"channel:{ctx.channel.id}"
        current_alias = ModelUsageService.DEFAULT_MODEL_ALIAS

        if session_key in self.session_manager.session_contexts:
            ctx_alias = self.session_manager.session_contexts[session_key].model_alias
            if ctx_alias:
                current_alias = ctx_alias
        elif ctx.channel.id in self.session_manager.channel_model_preferences:
            current_alias = self.session_manager.channel_model_preferences[ctx.channel.id]

        # Pass ctx.message as original_message
        view = ModelSelectorView(self.session_manager, current_alias, original_message=ctx.message)
        await send_discord_message(
            ctx,
            f"현재 LLM 모델: **{current_alias}**\n변경할 모델을 선택해 주세요.",
            view=view,
            mention_author=False,
        )

    @model_command.command(
        name="image", aliases=["이미지"], description="사용할 이미지 생성 모델을 선택합니다."
    )
    async def image_subcommand(self, ctx: commands.Context):
        """사용할 이미지 생성 모델을 선택합니다."""

        # Get current image model for this channel from the service
        current_api_model = get_channel_image_model(ctx.channel.id)

        # Build the image model selection view
        view = ImageModelSelectorView(
            self, current_api_model, original_message=ctx.message
        )

        # Get display name for current model
        current_model_def = get_available_image_models()
        current_display_name = current_api_model
        for model in current_model_def:
            if model.api_model_name == current_api_model:
                current_display_name = model.display_name
                break

        await send_discord_message(
            ctx,
            f"현재 이미지 모델: **{current_display_name}** (`{current_api_model}`)\n변경할 모델을 선택해 주세요.",
            view=view,
            mention_author=False,
        )


class ImageModelSelectorView(discord.ui.View):
    """View containing the image model selection dropdown."""

    def __init__(
        self,
        cog: ModelSelectorCog,
        current_model: str,
        original_message: Optional[discord.Message] = None,
    ):
        super().__init__(timeout=60)
        self.cog = cog
        self.original_message = original_message

        # Populate options from get_available_image_models()
        options = []
        for model in get_available_image_models():
            options.append(
                discord.SelectOption(
                    label=model.display_name,
                    description=model.description,
                    default=(model.api_model_name == current_model),
                    emoji="🎨",
                )
            )

        self.add_item(ImageModelSelect(options))


class ImageModelSelect(discord.ui.Select):
    """Dropdown for selecting an image model."""

    def __init__(self, options):
        super().__init__(
            placeholder="이미지 모델을 선택하세요...",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction):
        view: ImageModelSelectorView = self.view
        selected_display_name = self.values[0]

        # Defer to allow time
        await interaction.response.defer()

        # Find the selected model's api_model_name
        selected_model = None
        for model in get_available_image_models():
            if model.display_name == selected_display_name:
                selected_model = model
                break

        if not selected_model:
            await send_discord_message(
                interaction, "❌ 선택한 모델을 찾을 수 없습니다."
            )
            return

        # Update the channel's image model preference using the service
        set_channel_image_model(interaction.channel_id, selected_model.api_model_name)

        confirmation_text = (
            f"✅ 이미지 모델이 **{selected_display_name}** (`{selected_model.api_model_name}`)로 변경되었습니다."
        )

        # Logic change: Reply to the original !model image command message, then delete the embed
        if view.original_message:
            try:
                await send_discord_message(
                    view.original_message, confirmation_text, mention_author=False
                )
            except (discord.NotFound, discord.HTTPException):
                # Fallback if original deleted
                await send_discord_message(interaction, confirmation_text)
        else:
            await send_discord_message(interaction, confirmation_text)

        # Delete the interaction message (the embed with dropdown)
        try:
            if interaction.message:
                await interaction.message.delete()
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass

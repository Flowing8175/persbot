"""Model Selector Cog for SoyeBot."""

import logging
from typing import Optional

import discord
from discord.ext import commands

from soyebot.bot.session import SessionManager
from soyebot.services.model_usage_service import ModelUsageService

logger = logging.getLogger(__name__)


class ModelSelectorView(discord.ui.View):
    """View containing the model selection dropdown."""

    def __init__(self, session_manager: SessionManager, current_model: str, original_message: Optional[discord.Message] = None):
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

            options.append(discord.SelectOption(
                label=alias,
                description=desc,
                default=(alias == current_model),
                emoji="🤖" if definition.provider == "gemini" else "🧠"
            ))

        self.add_item(ModelSelect(options))


class ModelSelect(discord.ui.Select):
    """Dropdown for selecting a model."""

    def __init__(self, options):
        super().__init__(
            placeholder="모델을 선택하세요...",
            min_values=1,
            max_values=1,
            options=options
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
                await view.original_message.reply(confirmation_text, mention_author=False)
            except (discord.NotFound, discord.HTTPException):
                # Fallback if original deleted: reply to interaction (as followup since deferred)
                await interaction.followup.send(confirmation_text)
        else:
             await interaction.followup.send(confirmation_text)

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

    @commands.hybrid_command(name='model', aliases=['모델'], description="사용할 AI 모델을 선택합니다.")
    async def model_command(self, ctx: commands.Context):
        """사용할 AI 모델을 선택합니다."""

        # Determine current model for this channel
        # We need to peek into session or context
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
        await ctx.reply(
            f"현재 모델: **{current_alias}**\n변경할 모델을 선택해 주세요.",
            view=view,
            mention_author=False
        )

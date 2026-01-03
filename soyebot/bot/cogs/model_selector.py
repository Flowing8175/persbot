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

    def __init__(self, session_manager: SessionManager, current_model: str):
        super().__init__(timeout=60)
        self.session_manager = session_manager

        # Populate options from ModelUsageService definitions
        options = []
        for alias, definition in ModelUsageService.MODEL_DEFINITIONS.items():
            # Add description if needed (e.g., daily limit)
            desc = f"1일 한도: {definition.daily_limit}회 ({'채널' if definition.scope == 'channel' else '유저'} 공통)"

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

        # Update the session model
        # We use the channel ID from interaction
        view.session_manager.set_session_model(interaction.channel_id, selected_alias)

        await interaction.response.send_message(
            f"✅ 모델이 **{selected_alias}**로 변경되었습니다.",
            ephemeral=False
        )
        # Disable the view after selection
        self.disabled = True
        # Update original message to remove dropdown or disable it?
        # Interaction response is a new message.
        # We can edit the original message if we want, but 'interaction.message' might be null if slash command?
        # This is triggered from a !command, so there is an original message.
        if interaction.message:
            await interaction.message.edit(view=None)


class ModelSelectorCog(commands.Cog):
    """Cog for managing model selection."""

    def __init__(self, bot: commands.Bot, session_manager: SessionManager):
        self.bot = bot
        self.session_manager = session_manager

    @commands.command(name='model', aliases=['모델'])
    async def model_command(self, ctx: commands.Context):
        """현재 채널의 LLM 모델을 선택합니다."""

        # Determine current model for this channel
        # We need to peek into session or context
        session_key = f"channel:{ctx.channel.id}"
        current_alias = ModelUsageService.DEFAULT_MODEL_ALIAS

        if session_key in self.session_manager.session_contexts:
            ctx_alias = self.session_manager.session_contexts[session_key].model_alias
            if ctx_alias:
                current_alias = ctx_alias

        view = ModelSelectorView(self.session_manager, current_alias)
        await ctx.reply(
            f"현재 모델: **{current_alias}**\n변경할 모델을 선택해 주세요.",
            view=view,
            mention_author=False
        )

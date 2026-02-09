"""Main AssistantCog class - coordinates commands, events, and utilities."""

from typing import Optional

import discord
from discord.ext import commands

from soyebot.bot.cogs.base import BaseChatCog
from soyebot.bot.session import SessionManager
from soyebot.config import AppConfig
from soyebot.services.llm_service import LLMService
from soyebot.services.prompt_service import PromptService
from soyebot.tools.manager import ToolManager

from . import commands as cmd_module
from . import events as evt_module
from . import utils


class AssistantCog(BaseChatCog):
    """@mention을 통한 AI 어시스턴트 기능을 처리하는 Cog"""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        session_manager: SessionManager,
        prompt_service: PromptService,
        tool_manager: Optional["ToolManager"] = None,
    ):
        super().__init__(bot, config, llm_service, session_manager, tool_manager)
        self.prompt_service = prompt_service

        # Register commands and events from modules
        cmd_module.register_commands(self)
        evt_module.register_events(self)

    async def _send_response(self, message: discord.Message, reply):
        """Send the generated reply to Discord."""
        await utils.send_response(
            message, reply, self.config, self.session_manager, self._handle_break_cut_sending
        )

    async def _prepare_batch_context(self, messages: list[discord.Message]) -> str:
        """Prepare the text content for the LLM, including context from previous messages."""
        return await utils.prepare_batch_context(messages)

    async def _handle_error(self, message: discord.Message, error: Exception):
        """Handle errors during processing."""
        await utils.handle_error(message, error)

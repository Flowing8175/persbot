"""Main AssistantCog class - coordinates commands, events, and utilities."""

from typing import Optional

import discord
from discord.ext import commands

from persbot.bot.cogs.base import BaseChatCog
from persbot.bot.session import SessionManager
from persbot.config import AppConfig
from persbot.services.llm_service import LLMService
from persbot.services.prompt_service import PromptService
from persbot.tools.manager import ToolManager

from . import commands as cmd_module
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

        # Register commands from module
        cmd_module.register_commands(self)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle incoming messages and process them if bot is mentioned."""
        if utils.should_ignore_message(message, self.bot.user, self.config):
            return

        messages_to_prepend = self._cancel_active_tasks(message.channel.id, message.author.name)

        await self.message_buffer.add_message(message.channel.id, message, self._process_batch)

        if messages_to_prepend:
            # Ensure the list exists before prepending
            if message.channel.id in self.message_buffer.buffers:
                self.message_buffer.buffers[message.channel.id][0:0] = messages_to_prepend

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

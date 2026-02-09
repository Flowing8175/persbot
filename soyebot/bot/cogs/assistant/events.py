"""Event handlers for Assistant Cog."""

import discord
from discord.ext import commands

from soyebot.config import AppConfig

from . import utils


def register_events(cog):
    """Register all event handlers to the given cog instance."""
    cog.on_message = on_message


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

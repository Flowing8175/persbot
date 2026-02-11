"""Base handler for command operations.

This module provides the base class for all command handlers,
ensuring consistent behavior across different command types.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import discord
from discord.ext import commands

from persbot.config import AppConfig


class BaseHandler(ABC):
    """Base class for command handlers.

    Handlers encapsulate the logic for responding to Discord commands,
    keeping cogs focused on event handling and state management.
    """

    def __init__(self, bot: commands.Bot, config: AppConfig):
        """Initialize the handler.

        Args:
            bot: The Discord bot instance.
            config: Application configuration.
        """
        self.bot = bot
        self.config = config

    @abstractmethod
    async def handle(
        self,
        ctx: commands.Context,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[discord.Message]:
        """Handle the command.

        Args:
            ctx: The command context.
            *args: Command arguments.
            **kwargs: Keyword arguments.

        Returns:
            The response message, or None if no response was sent.
        """
        pass

    async def defer_if_needed(self, ctx: commands.Context) -> None:
        """Defer the interaction if it hasn't been responded to yet.

        Args:
            ctx: The command context.
        """
        if ctx.interaction and not ctx.interaction.response.is_done():
            await ctx.defer()

    async def send_response(
        self,
        ctx: commands.Context,
        content: str,
        **kwargs: Any,
    ) -> discord.Message:
        """Send a response, handling both slash and prefix commands.

        Args:
            ctx: The command context.
            content: The message content.
            **kwargs: Additional send arguments.

        Returns:
            The sent message.
        """
        if ctx.interaction and ctx.interaction.response.is_done():
            # Interaction already deferred, use followup
            return await ctx.followup.send(content, **kwargs)

        if ctx.interaction:
            # Interaction not yet responded to
            await ctx.response.send_message(content, **kwargs)
            return await ctx.original_response()

        # Prefix command
        return await ctx.send(content, **kwargs)

    async def send_error(
        self,
        ctx: commands.Context,
        message: str,
    ) -> discord.Message:
        """Send an error message.

        Args:
            ctx: The command context.
            message: The error message.

        Returns:
            The sent message.
        """
        return await self.send_response(
            ctx,
            f"❌ {message}",
        )

    async def send_success(
        self,
        ctx: commands.Context,
        message: str,
    ) -> discord.Message:
        """Send a success message.

        Args:
            ctx: The command context.
            message: The success message.

        Returns:
            The sent message.
        """
        return await self.send_response(
            ctx,
            f"✅ {message}",
        )


class HandlerResponse:
    """Standard response from command handlers."""

    def __init__(
        self,
        content: str = "",
        embed: Optional[discord.Embed] = None,
        view: Optional[discord.ui.View] = None,
        ephemeral: bool = False,
    ):
        """Initialize the handler response.

        Args:
            content: The response content.
            embed: Optional embed.
            view: Optional view with components.
            ephemeral: Whether the response should be ephemeral.
        """
        self.content = content
        self.embed = embed
        self.view = view
        self.ephemeral = ephemeral

    def to_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for send/response.

        Returns:
            Dictionary of keyword arguments.
        """
        kwargs = {}
        if self.content:
            kwargs["content"] = self.content
        if self.embed:
            kwargs["embed"] = self.embed
        if self.view:
            kwargs["view"] = self.view
        if self.ephemeral:
            kwargs["ephemeral"] = True
        return kwargs

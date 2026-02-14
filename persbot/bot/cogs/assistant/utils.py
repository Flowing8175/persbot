"""Helper functions and utilities for Assistant Cog."""

import asyncio
import logging
from typing import Optional

import discord
from discord.ext import commands

from persbot.bot.chat_handler import ChatReply
from persbot.bot.session import SessionManager
from persbot.config import AppConfig
from persbot.services.llm_service import LLMService
from persbot.utils import GENERIC_ERROR_MESSAGE, extract_message_content

logger = logging.getLogger(__name__)


def should_ignore_message(
    message: discord.Message,
    bot_user: Optional[discord.ClientUser],
    config: AppConfig,
) -> bool:
    """Return True when the bot should not process the message."""
    if message.author.bot:
        return True
    # If this channel is handled by AutoChannelCog, let it handle the response
    # to avoid duplicate replies (one plain, one reply).
    if message.channel.id in config.auto_reply_channel_ids:
        return True
    # Ignore @everyone/@here mentions
    if message.mention_everyone:
        return True
    if not bot_user or not bot_user.mentioned_in(message):
        return True
    return False


async def prepare_batch_context(messages: list[discord.Message]) -> str:
    """Prepare the text content for the LLM, including context from previous messages."""
    # 1. Fetch recent context (10 messages before the primary message)
    primary_message = messages[0]
    context_messages = [
        msg async for msg in primary_message.channel.history(limit=10, before=primary_message)
    ]
    context_messages.reverse()  # Chronological order

    context_text = ""
    if context_messages:
        context_lines = []
        for msg in context_messages:
            c_content = extract_message_content(msg)
            if c_content:
                context_lines.append(f"{msg.author.id}: {c_content}")

        if context_lines:
            context_text = (
                "=== 이전 대화 문맥 (참고용) ===\n"
                + "\n".join(context_lines)
                + "\n=== 현재 메시지 ===\n"
            )

    # 2. Combine current batch contents
    combined_content = []
    for msg in messages:
        content = extract_message_content(msg)
        if content:
            if len(messages) > 1 and msg.author.id:
                combined_content.append(f"{msg.author.id}: {content}")
            else:
                combined_content.append(content)

    current_text = "\n".join(combined_content)

    if not current_text:
        return ""

    # Prepend context to the full text
    return context_text + current_text


async def send_response(
    message: discord.Message,
    reply: ChatReply,
    config: AppConfig,
    session_manager: SessionManager,
    handle_break_cut_sending_func,
) -> None:
    """Send the generated reply to Discord, handling break-cut mode if enabled."""
    if not reply.text and not reply.images:
        logger.debug("LLM returned no text response or images for the mention.")
        return

    # Use cog's _send_response method (handles streaming/non-streaming based on break_cut_mode)
    await handle_break_cut_sending_func(message.channel.id, message.channel, reply)


async def handle_error(message: discord.Message, error: Exception) -> None:
    """Handle errors during processing."""
    await message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)


async def process_removed_messages(
    ctx: commands.Context, removed_messages: list, llm_service: LLMService
) -> str:
    """Process removed messages: delete assistant messages and return user content."""
    user_role = llm_service.get_user_role_name()
    assistant_role = llm_service.get_assistant_role_name()
    user_content = ""

    for msg in removed_messages:
        if msg.role == user_role:
            user_content = msg.content
        elif msg.role == assistant_role:
            await delete_assistant_messages(ctx.channel, msg)

    return user_content


async def delete_assistant_messages(channel, msg) -> None:
    """Delete assistant messages from Discord."""
    if not hasattr(msg, "message_ids") or not msg.message_ids:
        return
    for mid in msg.message_ids:
        try:
            old_msg = await channel.fetch_message(int(mid))
            await old_msg.delete()
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass


async def regenerate_response(
    ctx: commands.Context,
    session_key: str,
    user_content: str,
    bot: commands.Bot,
    llm_service: LLMService,
    session_manager: SessionManager,
    tool_manager,
    send_response_func,
    config: AppConfig,
) -> None:
    """Regenerate LLM response and send it."""
    from persbot.bot.chat_handler import create_chat_reply
    from persbot.bot.session import ResolvedSession

    async with ctx.channel.typing():
        resolution = ResolvedSession(session_key, user_content)
        reply = await create_chat_reply(
            ctx.message,
            resolution=resolution,
            llm_service=llm_service,
            session_manager=session_manager,
            tool_manager=tool_manager,
        )

        if reply and reply.text:
            await send_response_func(ctx.message, reply)
            # Clean up deferred interaction in break-cut mode
            if config.break_cut_mode and ctx.interaction:
                try:
                    await ctx.interaction.delete_original_response()
                except (discord.Forbidden, discord.HTTPException):
                    pass
        else:
            await ctx.send(GENERIC_ERROR_MESSAGE)

    # Clean up command message
    try:
        await ctx.message.delete()
    except (
        discord.Forbidden,
        discord.HTTPException,
        discord.NotFound,
        AttributeError,
    ):
        pass


def cancel_channel_tasks(
    channel_id: int,
    processing_tasks: dict[int, asyncio.Task],
    sending_tasks: dict[int, asyncio.Task],
    channel_name: str = "",
    reason: str = "",
    cancellation_signals: Optional[dict[int, asyncio.Event]] = None,
) -> bool:
    """Cancel active processing and sending tasks for a channel. Returns True if any cancelled."""
    cancelled = False

    # Trigger cancellation signal to abort LLM API calls
    if cancellation_signals and channel_id in cancellation_signals:
        logger.debug("%s triggered abort signal for channel #%s", reason, channel_name)
        cancellation_signals[channel_id].set()

    if channel_id in processing_tasks:
        task = processing_tasks[channel_id]
        if not task.done():
            logger.debug("%s interrupted active processing in #%s", reason, channel_name)
            task.cancel()
            cancelled = True

    if channel_id in sending_tasks:
        task = sending_tasks[channel_id]
        if not task.done():
            logger.debug("%s interrupted active sending in #%s", reason, channel_name)
            task.cancel()
            cancelled = True

    return cancelled


async def send_abort_success(ctx: commands.Context) -> None:
    """Send success response for abort command."""
    if ctx.interaction:
        await ctx.reply("🛑 중단되었습니다.", ephemeral=False)
    else:
        await ctx.message.add_reaction("🛑")


async def send_abort_no_tasks(ctx: commands.Context) -> None:
    """Send no-tasks response for abort command."""
    if ctx.interaction:
        await ctx.reply("❓ 중단할 작업이 없습니다.", ephemeral=True)
    else:
        await ctx.message.add_reaction("❓")

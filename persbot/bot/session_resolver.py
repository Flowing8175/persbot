"""Session resolution for Discord bot interactions.

This module handles resolving which chat session to use for a given message.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import discord

from persbot.bot.chat_models import SessionContext
from persbot.bot.session import ResolvedSession, SessionManager

logger = logging.getLogger(__name__)


async def resolve_session_for_message(
    message: discord.Message,
    content: str,
    *,
    session_manager: SessionManager,
    cancel_event: Optional[object] = None,
) -> Optional[ResolvedSession]:
    """Resolve the logical chat session for a Discord message.

    This function determines which chat session should handle a given message,
    taking into account replies, threads, and other context factors.

    Args:
        message: The Discord message.
        content: The cleaned message content.
        session_manager: The session manager instance.
        cancel_event: Optional cancellation event.

    Returns:
        The resolved session, or None if the message should be ignored.
    """
    is_reply_to_summary = False

    # Handle replies by adding context to the message content
    if message.reference and message.reference.message_id:
        ref_msg = message.reference.resolved

        # If not resolved or deleted, try to fetch it
        if ref_msg is None or isinstance(ref_msg, discord.DeletedReferencedMessage):
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
            except (discord.NotFound, discord.HTTPException):
                ref_msg = None

        if ref_msg:
            # Add reply context to the content
            ref_text = ref_msg.clean_content
            reply_context = f'(답장 대상: {ref_msg.author.id}, 내용: "{ref_text}")\n'
            content = reply_context + content

            # Check if this is a reply to a summary message
            if ref_msg.author.bot and "요약:**" in ref_text:
                is_reply_to_summary = True

    resolution = await session_manager.resolve_session(
        channel_id=message.channel.id,
        author_id=message.author.id,
        username=message.author.name,
        message_id=str(message.id),
        message_content=content,
        reference_message_id=None,
        created_at=message.created_at,
        cancel_event=cancel_event,
    )

    if resolution:
        resolution.is_reply_to_summary = is_reply_to_summary

    return resolution if resolution and resolution.cleaned_message else None


def extract_session_context(
    message: Union[discord.Message, list[discord.Message]],
) -> SessionContext:
    """Extract session context from a Discord message or list.

    Args:
        message: A Discord message or list of messages.

    Returns:
        The session context.
    """
    if isinstance(message, list):
        primary = message[0]
    else:
        primary = message

    return SessionContext(
        channel_id=primary.channel.id,
        user_id=primary.author.id,
        username=primary.author.name,
        message_id=str(primary.id),
        created_at=primary.created_at,
    )

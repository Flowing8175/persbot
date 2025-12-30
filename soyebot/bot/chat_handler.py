"""Shared helpers for Discord chat-driven cogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import discord

import asyncio
import logging
from soyebot.bot.session import SessionManager, ResolvedSession
from soyebot.services.llm_service import LLMService

logger = logging.getLogger(__name__)

__all__ = ["ChatReply", "resolve_session_for_message", "create_chat_reply", "send_split_response"]


@dataclass(frozen=True)
class ChatReply:
    """Container for an LLM response tied to a session."""

    text: str
    session_key: str
    response: object


async def resolve_session_for_message(
    message: discord.Message,
    content: str,
    *,
    session_manager: SessionManager,
) -> Optional[ResolvedSession]:
    """Resolve the logical chat session for a Discord message."""

    is_reply_to_summary = False
    # Handle replies by adding context to the message content
    # We no longer branch sessions for replies; instead, we just provide context to the LLM.
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
            # Using clean_content to get readable names instead of mention IDs
            ref_text = ref_msg.clean_content
            reply_context = f"(답장 대상: {ref_msg.author.display_name}, 내용: \"{ref_text}\")\n"
            content = reply_context + content

            # Check if this is a reply to a summary message
            # Summary messages typically start with "**... 요약:**"
            # We also check if it's from the bot itself
            if ref_msg.author.bot and "요약:**" in ref_text:
                is_reply_to_summary = True

    resolution = await session_manager.resolve_session(
        channel_id=message.channel.id,
        author_id=message.author.id,
        username=message.author.name,
        message_id=str(message.id),
        message_content=content,
        # Pass None for reference_message_id to ensure we stick to the channel session
        reference_message_id=None,
        created_at=message.created_at,
    )
    if resolution:
        resolution.is_reply_to_summary = is_reply_to_summary
    return resolution if resolution.cleaned_message else None


async def create_chat_reply(
    message: discord.Message,
    *,
    resolution: ResolvedSession,
    llm_service: LLMService,
    session_manager: SessionManager,
) -> Optional[ChatReply]:
    """Create or reuse a chat session and fetch an LLM reply."""

    chat_session, session_key = await session_manager.get_or_create(
        user_id=message.author.id,
        username=message.author.name,
        session_key=resolution.session_key,
        channel_id=message.channel.id,
        message_content=resolution.cleaned_message,
        message_ts=message.created_at,
        message_id=str(message.id),
    )

    response_result = await llm_service.generate_chat_response(
        chat_session,
        resolution.cleaned_message,
        message,
        use_summarizer_backend=resolution.is_reply_to_summary,
    )

    if not response_result:
        return None

    response_text, response_obj = response_result
    return ChatReply(text=response_text or "", session_key=session_key, response=response_obj)


async def send_split_response(
    channel: discord.abc.Messageable, 
    reply: ChatReply, 
    session_manager: SessionManager
):
    """
    Shared utility to split and send a response line by line.
    Handles cancellation by undoing the last exchange in session history.
    """
    try:
        lines = reply.text.split('\n')
        for line in lines:
            if not line.strip():
                continue

            # Calculate delay: proportional to length, clamped 0.5s - 1.7s
            delay = max(0.5, min(1.7, len(line) * 0.05))

            # Send typing status while waiting
            async with channel.typing():
                await asyncio.sleep(delay)
                sent_msg = await channel.send(line)
                
                # Link message to session
                session_manager.link_message_to_session(str(sent_msg.id), reply.session_key)

    except asyncio.CancelledError:
        logger.info(f"Sending interrupted for channel {channel.id}.")
        raise  # Re-raise to signal cancellation

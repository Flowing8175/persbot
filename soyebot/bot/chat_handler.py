"""Shared helpers for Discord chat-driven cogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import discord

from bot.session import SessionManager, ResolvedSession
from services.llm_service import LLMService

__all__ = ["ChatReply", "resolve_session_for_message", "create_chat_reply"]


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

    reference_id = (
        str(message.reference.message_id)
        if message.reference and message.reference.message_id
        else None
    )

    resolution = await session_manager.resolve_session(
        channel_id=message.channel.id,
        author_id=message.author.id,
        username=message.author.name,
        message_id=str(message.id),
        message_content=content,
        reference_message_id=reference_id,
        created_at=message.created_at,
    )
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

    session_manager.link_message_to_session(str(message.id), session_key)

    response_result = await llm_service.generate_chat_response(
        chat_session,
        resolution.cleaned_message,
        message,
    )

    if not response_result:
        return None

    response_text, response_obj = response_result
    return ChatReply(text=response_text or "", session_key=session_key, response=response_obj)

"""Chat session management for SoyeBot."""

import logging
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from soyebot.config import AppConfig
from soyebot.services.llm_service import LLMService
from soyebot.prompts import BOT_PERSONA_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Represents a very short-lived LLM chat tied to a session key."""
    chat: object
    user_id: str
    session_id: str
    last_activity_at: datetime
    last_message_id: Optional[str] = None


@dataclass
class SessionContext:
    """Lightweight metadata for picking and reusing sessions."""

    session_id: str
    channel_id: int
    user_id: str
    username: str
    started_at: datetime
    last_activity_at: datetime
    last_message_preview: str = ""
    title: Optional[str] = None


@dataclass
class ResolvedSession:
    """Result of session resolution for a new message."""

    session_key: str
    cleaned_message: str
    is_reply_to_summary: bool = False


class SessionManager:
    """Manages chat sessions, optionally caching them indefinitely for deep threads."""

    def __init__(
        self,
        config: AppConfig,
        llm_service: LLMService,
    ):
        self.config = config
        self.llm_service = llm_service
        self.sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self.session_contexts: OrderedDict[str, SessionContext] = OrderedDict()
        self.channel_prompts: Dict[int, str] = {} # channel_id -> prompt_content override

    def _evict_if_needed(self) -> None:
        """Ensure the session cache does not grow without bounds."""
        while len(self.sessions) > self.config.session_cache_limit:
            evicted_key, _ = self.sessions.popitem(last=False)
            logger.debug("Evicted chat session %s due to cache limit", evicted_key)
        while len(self.session_contexts) > self.config.session_cache_limit:
            evicted_key, _ = self.session_contexts.popitem(last=False)
            logger.debug("Evicted session context %s due to cache limit", evicted_key)

    def _record_session_context(
        self,
        session_key: str,
        channel_id: int,
        user_id: str,
        username: str,
        message_content: str,
        message_ts: Optional[datetime],
    ) -> None:
        now = message_ts or datetime.now(timezone.utc)
        preview = message_content.strip()
        existing = self.session_contexts.get(session_key)

        if existing:
            existing.last_activity_at = now
            existing.last_message_preview = preview or existing.last_message_preview
            self.session_contexts.move_to_end(session_key)
        else:
            self.session_contexts[session_key] = SessionContext(
                session_id=session_key,
                channel_id=channel_id,
                user_id=user_id,
                username=username,
                started_at=now,
                last_activity_at=now,
                last_message_preview=preview,
            )

        self._evict_if_needed()

    async def get_or_create(
        self,
        user_id: str,
        username: str,
        session_key: str,
        channel_id: int,
        message_content: str,
        message_ts: Optional[datetime] = None,
        message_id: Optional[str] = None,
    ) -> Tuple[object, str]:
        """Retrieve a cached chat or start a new one without evicting prior sessions."""
        user_id = str(user_id)
        existing_session = self.sessions.get(session_key)

        if existing_session:
            existing_session.last_activity_at = datetime.now(timezone.utc)
            existing_session.last_message_id = message_id
            self.sessions.move_to_end(session_key)
            self._record_session_context(session_key, channel_id, user_id, username, message_content, message_ts)
            return existing_session.chat, session_key

        logger.info(f"Creating new session {session_key} for user {user_id}")

        system_prompt = self.channel_prompts.get(channel_id, BOT_PERSONA_PROMPT)

        assistant_model = self.llm_service.create_assistant_model(system_prompt)
        chat = assistant_model.start_chat(system_prompt)

        self.sessions[session_key] = ChatSession(
            chat=chat,
            user_id=user_id,
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            last_message_id=message_id,
        )

        self._record_session_context(session_key, channel_id, user_id, username, message_content, message_ts)
        self._evict_if_needed()

        return chat, session_key

    def set_channel_prompt(self, channel_id: int, prompt_content: Optional[str]) -> None:
        """Set a custom system prompt for a specific channel."""
        if prompt_content:
            self.channel_prompts[channel_id] = prompt_content
        else:
            self.channel_prompts.pop(channel_id, None)
        
        # Reset current session for this channel to apply the new prompt
        self.reset_session_by_channel(channel_id)

    def reset_session_by_channel(self, channel_id: int) -> bool:
        """Clear cached session and metadata for a channel, if present."""
        session_key = f"channel:{channel_id}"
        removed = False

        if session_key in self.sessions:
            del self.sessions[session_key]
            removed = True

        if session_key in self.session_contexts:
            del self.session_contexts[session_key]
            removed = True

        if removed:
            logger.info("Session %s reset for channel %s", session_key, channel_id)

        return removed

    async def resolve_session(
        self,
        *,
        channel_id: int,
        author_id: int,
        username: str,
        message_id: str,
        message_content: str,
        reference_message_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> ResolvedSession:
        """Decide which session should be used for an incoming message."""

        cleaned_message = message_content.strip()

        # Note: reference_message_id is now ignored as we treat replies as context
        # within the main channel session rather than branching or joining threads.

        session_key = f"channel:{channel_id}"
        return ResolvedSession(session_key, cleaned_message)

    def link_message_to_session(self, message_id: str, session_key: str) -> None:
        """Links a Discord message ID to the last message in the session history by appending to message_ids."""
        session = self.sessions.get(session_key)
        if session and hasattr(session.chat, 'history') and session.chat.history:
            last_msg = session.chat.history[-1]
            if not hasattr(last_msg, 'message_ids'):
                 last_msg.message_ids = []
            last_msg.message_ids.append(message_id)

    def undo_last_exchanges(self, session_key: str, num_to_undo: int) -> list:
        """Remove the last N user/assistant exchanges from a session's history."""
        session = self.sessions.get(session_key)
        if not session or not hasattr(session.chat, 'history'):
            return []

        try:
            assistant_role = self.llm_service.get_assistant_role_name()
            user_role = self.llm_service.get_user_role_name()

            # Find indices of the last N assistant messages
            assistant_indices = [
                i for i, msg in enumerate(session.chat.history) if msg.role == assistant_role
            ]
            if not assistant_indices:
                return []

            indices_to_remove = set()
            num_to_undo = min(num_to_undo, len(assistant_indices))

            # We want the last `num_to_undo` exchanges
            assistant_messages_to_remove = assistant_indices[-num_to_undo:]

            for assistant_index in assistant_messages_to_remove:
                indices_to_remove.add(assistant_index)
                # Mark preceding user messages for removal
                user_message_index = assistant_index - 1
                while user_message_index >= 0 and session.chat.history[user_message_index].role == user_role:
                    indices_to_remove.add(user_message_index)
                    user_message_index -= 1

            if not indices_to_remove:
                return []

            # Separate the history into kept and removed messages
            new_history = []
            removed_messages = []
            for i, msg in enumerate(session.chat.history):
                if i in indices_to_remove:
                    removed_messages.append(msg)
                else:
                    new_history.append(msg)

            session.chat.history = new_history

            logger.info(
                "Undid last %d exchanges from session %s. New history length: %d",
                num_to_undo,
                session_key,
                len(new_history)
            )
            return removed_messages

        except Exception as e:
            logger.error(
                "Error undoing exchanges in session %s: %s", session_key, e, exc_info=True
            )
            return []

"""Chat session management for SoyeBot."""

import logging
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from config import AppConfig
from services.llm_service import LLMService
from services.database_service import DatabaseService
from prompts import BOT_PERSONA_PROMPT
from metrics import get_metrics

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


class SessionManager:
    """Manages chat sessions, optionally caching them indefinitely for deep threads."""

    def __init__(
        self,
        config: AppConfig,
        llm_service: LLMService,
        db_service: DatabaseService,
    ):
        self.config = config
        self.llm_service = llm_service
        self.db_service = db_service
        self.sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self.message_sessions: OrderedDict[str, str] = OrderedDict()
        self.session_contexts: OrderedDict[str, SessionContext] = OrderedDict()

    def _evict_if_needed(self) -> None:
        """Ensure the session cache does not grow without bounds."""
        while len(self.sessions) > self.config.session_cache_limit:
            evicted_key, _ = self.sessions.popitem(last=False)
            logger.debug("Evicted chat session %s due to cache limit", evicted_key)
        while len(self.message_sessions) > self.config.session_cache_limit:
            evicted_key, _ = self.message_sessions.popitem(last=False)
            logger.debug("Evicted message link %s due to cache limit", evicted_key)
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

        await asyncio.to_thread(self.db_service.get_or_create_user, user_id, username)

        system_prompt = BOT_PERSONA_PROMPT

        assistant_model = self.llm_service.create_assistant_model(system_prompt)
        chat = assistant_model.start_chat()

        self.sessions[session_key] = ChatSession(
            chat=chat,
            user_id=user_id,
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            last_message_id=message_id,
        )

        get_metrics().increment_counter('sessions_created')

        self._record_session_context(session_key, channel_id, user_id, username, message_content, message_ts)
        self._evict_if_needed()

        return chat, session_key

    def link_message_to_session(self, message_id: Optional[str], session_key: str) -> None:
        """Remember which Discord message belongs to which session for reply chains."""
        if not message_id or not session_key:
            return
        self.message_sessions[str(message_id)] = session_key

    def get_session_for_message(self, message_id: Optional[str]) -> Optional[str]:
        """Return the session key for a referenced message if we have seen it."""
        if not message_id:
            return None
        return self.message_sessions.get(str(message_id))

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

        message_ids_to_remove = [
            message_id
            for message_id, mapped_session_key in self.message_sessions.items()
            if mapped_session_key == session_key
        ]

        for message_id in message_ids_to_remove:
            del self.message_sessions[message_id]
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

        if reference_message_id:
            existing_session = self.get_session_for_message(reference_message_id)
            if existing_session:
                return ResolvedSession(existing_session, cleaned_message)

        session_key = f"channel:{channel_id}"
        return ResolvedSession(session_key, cleaned_message)

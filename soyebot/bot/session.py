"""Chat session management for SoyeBot."""

import logging
import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Optional, Tuple

from config import AppConfig
from services.gemini_service import GeminiService
from services.database_service import DatabaseService
from prompts import BOT_PERSONA_PROMPT
from metrics import get_metrics

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Represents a very short-lived Gemini chat tied to a session key."""
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
        gemini_service: GeminiService,
        db_service: DatabaseService,
    ):
        self.config = config
        self.gemini_service = gemini_service
        self.db_service = db_service
        self.sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self.message_sessions: OrderedDict[str, str] = OrderedDict()
        self.session_contexts: OrderedDict[str, SessionContext] = OrderedDict()
        self.channel_recent_sessions: OrderedDict[int, str] = OrderedDict()

    def _evict_if_needed(self) -> None:
        """Ensure the session cache does not grow without bounds."""
        while len(self.sessions) > self.config.session_cache_limit:
            evicted_key, _ = self.sessions.popitem(last=False)
            logger.debug("Evicted chat session %s due to cache limit", evicted_key)
        while len(self.session_contexts) > self.config.session_cache_limit:
            evicted_key, _ = self.session_contexts.popitem(last=False)
            logger.debug("Evicted session context %s due to cache limit", evicted_key)

    def _slugify_topic(self, topic: str) -> str:
        slug = re.sub(r"\s+", "-", topic.strip().lower())
        slug = re.sub(r"[^a-z0-9\-]+", "", slug)
        return slug or "untitled"

    def _extract_explicit_topic(self, message: str) -> Tuple[Optional[str], str]:
        """Return explicit topic name and cleaned message if present."""

        stripped = message.strip()
        lowered = stripped.lower()
        for prefix in ("/topic", "!topic"):
            if lowered.startswith(prefix):
                parts = stripped.split(maxsplit=2)
                if len(parts) >= 2:
                    topic_name = parts[1]
                    remainder = parts[2] if len(parts) > 2 else ""
                    return topic_name, remainder.strip() or topic_name
        return None, message

    def _is_new_session_trigger(self, message: str) -> bool:
        lowered = message.lower()
        trigger_phrases = ["새 주제", "새로 시작"]
        return any(trigger in lowered for trigger in trigger_phrases)

    def _generate_session_id(self, channel_id: int) -> str:
        return f"{channel_id}-{int(time.time() * 1000)}"

    def _get_recent_session(self, channel_id: int) -> Optional[SessionContext]:
        session_key = self.channel_recent_sessions.get(channel_id)
        if session_key:
            return self.session_contexts.get(session_key)
        return None

    def _is_stale(self, context: SessionContext, at: Optional[datetime]) -> bool:
        reference_time = at or datetime.now(timezone.utc)
        return reference_time - context.last_activity_at > timedelta(minutes=self.config.session_inactive_minutes)

    async def _similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0

        try:
            score = await self.gemini_service.score_topic_similarity(a, b)
            if score is not None:
                return score
        except Exception:
            logger.exception("Gemini similarity scoring failed; falling back to fuzzy match")

        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

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

        self.channel_recent_sessions[channel_id] = session_key
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

        assistant_model = self.gemini_service.create_assistant_model(system_prompt)
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

        topic_name, cleaned_after_topic = self._extract_explicit_topic(cleaned_message)
        if topic_name:
            topic_slug = self._slugify_topic(topic_name)
            session_key = f"topic:{channel_id}:{topic_slug}"
            return ResolvedSession(session_key, cleaned_after_topic)

        if self._is_new_session_trigger(cleaned_message):
            return ResolvedSession(self._generate_session_id(channel_id), cleaned_message)

        recent = self._get_recent_session(channel_id)
        if recent and not self._is_stale(recent, created_at):
            similarity = await self._similarity(cleaned_message, recent.last_message_preview)
            if similarity >= self.config.session_similarity_threshold:
                return ResolvedSession(recent.session_id, cleaned_message)
            logger.debug(
                "Similarity %.3f below threshold %.2f for channel %s; creating new session.",
                similarity,
                self.config.session_similarity_threshold,
                channel_id,
            )

        return ResolvedSession(self._generate_session_id(channel_id), cleaned_message)

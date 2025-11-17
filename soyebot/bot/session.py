"""Chat session management for SoyeBot."""

import logging
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
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


class SessionManager:
    """Keeps at most `max_session_records` active sessions to cap memory usage."""

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
        self.message_session_capacity = getattr(config, 'max_tracked_message_ids', 800)

    async def get_or_create(
        self,
        user_id: str,
        username: str,
        message_id: Optional[str] = None,
    ) -> Tuple[object, str]:
        """Retrieve a cached chat or start a new one while dropping old sessions."""
        user_id = str(user_id)
        session_key = message_id or user_id
        existing_session = self.sessions.get(session_key)

        if existing_session:
            existing_session.last_activity_at = datetime.now(timezone.utc)
            existing_session.last_message_id = message_id
            self.sessions.move_to_end(session_key)
            return existing_session.chat, user_id

        logger.info(f"Creating new session {session_key} for user {user_id}")

        await asyncio.to_thread(self.db_service.get_or_create_user, user_id, username)

        system_prompt = BOT_PERSONA_PROMPT

        assistant_model = self.gemini_service.create_assistant_model(system_prompt)
        chat = assistant_model.start_chat()

        self._enforce_capacity()

        self.sessions[session_key] = ChatSession(
            chat=chat,
            user_id=user_id,
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            last_message_id=message_id,
        )

        get_metrics().increment_counter('sessions_created')

        return chat, session_key

    def link_message_to_session(self, message_id: Optional[str], session_key: str) -> None:
        """Remember which Discord message belongs to which session for reply chains."""
        if not message_id or not session_key:
            return
        self.message_sessions[str(message_id)] = session_key
        self.message_sessions.move_to_end(str(message_id))
        while len(self.message_sessions) > self.message_session_capacity:
            evicted_message, _ = self.message_sessions.popitem(last=False)
            logger.debug(f"Dropped message {evicted_message} from session lookup (capacity {self.message_session_capacity})")

    def get_session_for_message(self, message_id: Optional[str]) -> Optional[str]:
        """Return the session key for a referenced message if we have seen it."""
        if not message_id:
            return None
        return self.message_sessions.get(str(message_id))

    def _enforce_capacity(self):
        """Evict oldest sessions to keep memory bounded."""
        limit = max(1, getattr(self.config, 'max_session_records', 2))
        while len(self.sessions) >= limit:
            evicted_key, evicted_session = self.sessions.popitem(last=False)
            logger.info(f"Evicting session {evicted_key} (user {evicted_session.user_id}) to honor cache limit")
            get_metrics().increment_counter('sessions_cleaned')

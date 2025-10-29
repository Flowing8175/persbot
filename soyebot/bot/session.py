"""Chat session management for SoyeBot."""

import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from config import AppConfig
from services.gemini_service import GeminiService
from services.database_service import DatabaseService
from services.memory_service import MemoryService
from prompts import build_system_prompt_with_memory

logger = logging.getLogger(__name__)

@dataclass
class ChatSession:
    """사용자별 채팅 세션 - now user-based instead of message-based"""
    chat: object
    user_id: str
    last_activity_at: datetime
    last_message_id: Optional[str] = None  # For threading

    def is_expired(self, ttl_minutes: int) -> bool:
        expiry_time = self.last_activity_at + timedelta(minutes=ttl_minutes)
        return datetime.now(timezone.utc) > expiry_time

class SessionManager:
    """사용자 기반 채팅 세션을 관리합니다 - Manages user-based sessions with memory integration."""
    def __init__(
        self,
        config: AppConfig,
        gemini_service: GeminiService,
        db_service: DatabaseService,
        memory_service: MemoryService,
    ):
        self.config = config
        self.gemini_service = gemini_service
        self.db_service = db_service
        self.memory_service = memory_service
        self.sessions: dict[str, ChatSession] = {}  # user_id -> ChatSession
        self.last_cleanup_time = time.time()

    def get_or_create(
        self,
        user_id: str,
        username: str,
        message_id: Optional[str] = None,
    ) -> Tuple[object, str]:
        """Get or create user session.

        Args:
            user_id: Discord user ID
            username: Discord username
            message_id: Current message ID for threading

        Returns:
            Tuple of (chat_object, session_id)
        """
        user_id = str(user_id)

        # Check for existing session
        if user_id in self.sessions:
            session = self.sessions[user_id]
            if not session.is_expired(self.config.session_ttl_minutes):
                session.last_activity_at = datetime.now(timezone.utc)
                if message_id:
                    session.last_message_id = message_id
                logger.debug(f"세션 재사용: {user_id}")
                return session.chat, user_id

            logger.info(f"만료된 세션 삭제: {user_id}")
            del self.sessions[user_id]

        # Create new session
        logger.info(f"새 세션 생성: {user_id}")

        # Ensure user exists in database
        self.db_service.get_or_create_user(user_id, username)

        # Load context (unified memories only - few-shot examples are now in system prompt)
        memory_context = self.memory_service.get_memories_context()
        logger.debug(f"Memory context length: {len(memory_context)} characters")
        if memory_context:
            logger.debug(f"Memory context preview: {memory_context[:200]}...")

        system_prompt = build_system_prompt_with_memory(memory_context)
        logger.debug(f"System prompt length: {len(system_prompt)} characters")
        logger.debug(f"[RAW SESSION REQUEST] System prompt:\n{system_prompt}")

        # Create new model with enhanced system prompt (including memory context + few-shot examples)
        assistant_model = self.gemini_service.create_assistant_model(system_prompt)

        # Create new chat without separate history (few-shot examples are embedded in system prompt)
        chat = assistant_model.start_chat()

        # Store session
        self.sessions[user_id] = ChatSession(
            chat=chat,
            user_id=user_id,
            last_activity_at=datetime.now(timezone.utc),
            last_message_id=message_id,
        )

        return chat, user_id

    def cleanup_expired(self):
        """Clean up expired sessions."""
        now = time.time()
        if now - self.last_cleanup_time < self.config.session_cleanup_interval:
            return

        expired_ids = [
            uid for uid, session in self.sessions.items()
            if session.is_expired(self.config.session_ttl_minutes)
        ]
        for uid in expired_ids:
            logger.info(f"만료된 세션 정리: {uid}")
            del self.sessions[uid]

        if expired_ids:
            logger.info(f"정리 완료: {len(expired_ids)}개 삭제, 남은 세션: {len(self.sessions)}개")
        self.last_cleanup_time = now

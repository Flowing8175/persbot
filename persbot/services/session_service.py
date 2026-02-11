"""Session service for managing chat sessions.

This module provides a service layer for session management,
separating session logic from the bot layer.
"""

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from persbot.config import AppConfig
from persbot.constants import SessionConfig
from persbot.domain import SessionKey, UserId
from persbot.exceptions import (
    SessionConflictException,
    SessionExpiredException,
    SessionNotFoundException,
)
from persbot.services.llm_service import LLMService
from persbot.services.model_usage_service import ModelUsageService

logger = logging.getLogger(__name__)


@dataclass
class ChatSessionData:
    """Data stored for a chat session.

    Attributes:
        session_key: Unique session identifier.
        user_id: The user's Discord ID.
        channel_id: The channel's Discord ID.
        chat_object: The LLM chat object.
        created_at: When the session was created.
        last_activity: Last activity timestamp.
        model_alias: The model being used.
        message_count: Number of messages in session.
        history: Session message history.
    """

    session_key: str
    user_id: int
    channel_id: int
    chat_object: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_alias: Optional[str] = None
    message_count: int = 0
    history: list = field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        """Check if session is stale based on inactivity."""
        if not self.last_activity:
            return False
        age = datetime.now(timezone.utc) - self.last_activity
        return age.total_seconds() > (SessionConfig.INACTIVE_MINUTES * 60)

    @property
    def age_seconds(self) -> float:
        """Get session age in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


@dataclass
class SessionResolution:
    """Result of session resolution.

    Attributes:
        session_key: The resolved session key.
        cleaned_message: The processed message content.
        is_new: Whether a new session was created.
        is_reply_to_summary: Whether this is a reply to a summary.
        model_alias: The model alias to use.
    """

    session_key: str
    cleaned_message: str
    is_new: bool = False
    is_reply_to_summary: bool = False
    model_alias: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class SessionService:
    """Service for managing chat sessions.

    This service handles session creation, retrieval, cleanup, and
    provides a clean API for session management across the bot.
    """

    def __init__(
        self,
        config: AppConfig,
        llm_service: LLMService,
        model_usage_service: ModelUsageService,
    ) -> None:
        """Initialize the session service.

        Args:
            config: Application configuration.
            llm_service: The LLM service.
            model_usage_service: Model usage tracking service.
        """
        self.config = config
        self.llm_service = llm_service
        self.model_usage_service = model_usage_service

        # Session storage
        self._sessions: Dict[str, ChatSessionData] = OrderedDict()
        self._channel_prompts: Dict[int, str] = {}
        self._channel_models: Dict[int, str] = {}

        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        if config.session_inactive_minutes > 0:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    def get_session(self, session_key: str) -> Optional[ChatSessionData]:
        """Get a session by key.

        Args:
            session_key: The session key.

        Returns:
            The session data, or None if not found.
        """
        return self._sessions.get(session_key)

    def get_or_create_session(
        self,
        session_key: str,
        user_id: int,
        channel_id: int,
        chat_factory: Callable[[], Any],
        model_alias: Optional[str] = None,
    ) -> Tuple[Any, bool]:
        """Get existing session or create new one.

        Args:
            session_key: The session key.
            user_id: The user's Discord ID.
            channel_id: The channel's Discord ID.
            chat_factory: Factory function to create new chat object.
            model_alias: Optional model alias to use.

        Returns:
            Tuple of (chat_object, is_new).
        """
        session = self._sessions.get(session_key)

        if session and not session.is_stale:
            # Update existing session
            session.last_activity = datetime.now(timezone.utc)
            session.message_count += 1
            return session.chat_object, False

        # Create new session
        chat_object = chat_factory()
        new_session = ChatSessionData(
            session_key=session_key,
            user_id=user_id,
            channel_id=channel_id,
            chat_object=chat_object,
            model_alias=model_alias or self.model_usage_service.DEFAULT_MODEL_ALIAS,
        )

        self._sessions[session_key] = new_session
        self._evict_if_needed()

        logger.info(f"Created new session: {session_key}")
        return chat_object, True

    def remove_session(self, session_key: str) -> bool:
        """Remove a session from storage.

        Args:
            session_key: The session key.

        Returns:
            True if session was removed, False if not found.
        """
        session = self._sessions.pop(session_key, None)
        if session:
            logger.info(f"Removed session: {session_key}")
            return True
        return False

    def cleanup_stale_sessions(self) -> int:
        """Remove all stale sessions.

        Returns:
            Number of sessions removed.
        """
        to_remove = [key for key, session in self._sessions.items() if session.is_stale]

        for key in to_remove:
            self._sessions.pop(key)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} stale sessions")

        return len(to_remove)

    def get_channel_prompt(self, channel_id: int) -> Optional[str]:
        """Get custom prompt for a channel.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            The custom prompt, or None if not set.
        """
        return self._channel_prompts.get(channel_id)

    def set_channel_prompt(self, channel_id: int, prompt: str) -> None:
        """Set custom prompt for a channel.

        Args:
            channel_id: The Discord channel ID.
            prompt: The prompt content.
        """
        self._channel_prompts[channel_id] = prompt
        logger.info(f"Set custom prompt for channel {channel_id}")

    def get_channel_model(self, channel_id: int) -> Optional[str]:
        """Get preferred model for a channel.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            The model alias, or None if not set.
        """
        return self._channel_models.get(channel_id)

    def set_channel_model(self, channel_id: int, model_alias: str) -> None:
        """Set preferred model for a channel.

        Args:
            channel_id: The Discord channel ID.
            model_alias: The model alias.
        """
        self._channel_models[channel_id] = model_alias
        logger.info(f"Set model {model_alias} for channel {channel_id}")

    def get_all_sessions(self) -> List[ChatSessionData]:
        """Get all active sessions.

        Returns:
            List of all session data.
        """
        return list(self._sessions.values())

    def get_session_count(self) -> int:
        """Get the number of active sessions.

        Returns:
            Number of sessions.
        """
        return len(self._sessions)

    def _evict_if_needed(self) -> None:
        """Evict oldest sessions if cache limit exceeded."""
        while len(self._sessions) > SessionConfig.CACHE_LIMIT:
            evicted_key, _ = self._sessions.popitem(last=False)
            logger.debug(f"Evicted session {evicted_key} due to cache limit")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up stale sessions."""
        interval = SessionConfig.INACTIVE_MINUTES * 30  # seconds

        while True:
            try:
                await asyncio.sleep(interval)
                removed = self.cleanup_stale_sessions()
                if removed > 0:
                    logger.info(f"Periodic cleanup: removed {removed} sessions")
            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}", exc_info=True)

    async def shutdown(self) -> None:
        """Gracefully shutdown the session service.

        Cancels the cleanup task and clears all sessions.
        """
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        session_count = len(self._sessions)
        self._sessions.clear()
        logger.info(f"Session service shutdown (cleared {session_count} sessions)")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get session statistics.

        Returns:
            Dictionary with session stats.
        """
        return {
            "total_sessions": len(self._sessions),
            "custom_prompts": len(self._channel_prompts),
            "custom_models": len(self._channel_models),
        }

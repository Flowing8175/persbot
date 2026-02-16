"""Chat session management for SoyeBot."""

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from persbot.config import AppConfig
from persbot.prompts import BOT_PERSONA_PROMPT
from persbot.services.llm_service import LLMService
from persbot.services.model_usage_service import ModelUsageService

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Represents a very short-lived LLM chat tied to a session key."""

    chat: object
    user_id: str
    session_id: str
    last_activity_at: datetime
    last_message_id: Optional[str] = None
    model_alias: Optional[str] = None


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
    model_alias: Optional[str] = None


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
        self.channel_prompts: Dict[int, str] = {}  # channel_id -> prompt_content override
        self.channel_model_preferences: Dict[int, str] = {}  # channel_id -> model_alias override
        self._cleanup_task: Optional[asyncio.Task] = (
            None  # Track cleanup task for graceful shutdown
        )
        self._sessions_lock = asyncio.Lock()  # Lock for thread-safe session operations

        # Start periodic session cleanup task
        if config.session_inactive_minutes > 0:
            self._cleanup_task = asyncio.create_task(self._periodic_session_cleanup())

    def _evict_if_needed(self) -> None:
        """Ensure the session cache does not grow without bounds."""
        while len(self.sessions) > self.config.session_cache_limit:
            evicted_key, _ = self.sessions.popitem(last=False)
        while len(self.session_contexts) > self.config.session_cache_limit:
            evicted_key, _ = self.session_contexts.popitem(last=False)

    def _record_session_context(
        self,
        session_key: str,
        channel_id: int,
        user_id: str,
        username: str,
        message_content: str,
        message_ts: Optional[datetime],
        model_alias: Optional[str] = None,
    ) -> None:
        now = message_ts or datetime.now(timezone.utc)
        preview = message_content.strip()
        existing = self.session_contexts.get(session_key)

        if existing:
            existing.last_activity_at = now
            existing.last_message_preview = preview or existing.last_message_preview
            if model_alias:
                existing.model_alias = model_alias
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
                title=None,
                model_alias=model_alias,
            )

        self._evict_if_needed()

    def set_session_model(self, channel_id: int, model_alias: str) -> None:
        """Set the model alias for the current session associated with the channel."""
        session_key = f"channel:{channel_id}"

        # Store preference permanently for this runtime (handles cases where session doesn't exist yet)
        self.channel_model_preferences[channel_id] = model_alias

        # We only update the context. This forces get_or_create to detect a mismatch
        # with the active session (if any) and trigger a recreation/model switch
        # on the next interaction.
        if session_key in self.session_contexts:
            self.session_contexts[session_key].model_alias = model_alias

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
        """Retrieve a cached chat or start a new one."""
        user_id = str(user_id)
        target_model_alias = self._resolve_target_model_alias(session_key, channel_id)

        async with self._sessions_lock:
            existing_session = self.sessions.get(session_key)
            if existing_session:
                if self._check_session_model_compatibility(
                    existing_session, session_key, target_model_alias
                ):
                    # Compatible - update and return existing session
                    self._update_existing_session(
                        existing_session,
                        session_key,
                        channel_id,
                        user_id,
                        username,
                        message_content,
                        message_ts,
                        message_id,
                    )
                    return existing_session.chat, session_key
                # Incompatible - will create new below
                del self.sessions[session_key]

            return await self._create_new_session(
                session_key,
                channel_id,
                user_id,
                username,
                message_content,
                message_ts,
                message_id,
                target_model_alias,
            )

    def _resolve_target_model_alias(self, session_key: str, channel_id: int) -> str:
        """Determine the target model alias for a session."""
        if session_key in self.session_contexts:
            ctx_alias = self.session_contexts[session_key].model_alias
            if ctx_alias:
                return ctx_alias
        if channel_id in self.channel_model_preferences:
            return self.channel_model_preferences[channel_id]

        # Derive default from config by finding the alias that matches the configured model
        config_model = self.config.assistant_model_name
        for alias, definition in self.llm_service.model_usage_service.MODEL_DEFINITIONS.items():
            if definition.api_model_name == config_model:
                return alias

        # Fallback to class default if no match found
        return ModelUsageService.DEFAULT_MODEL_ALIAS

    def _check_session_model_compatibility(
        self, session: ChatSession, session_key: str, target_alias: str
    ) -> bool:
        """Check if existing session is compatible with target model. Returns False if needs reset."""
        if session.model_alias != target_alias:
            return False
        return True

    def _update_existing_session(
        self,
        session: ChatSession,
        session_key: str,
        channel_id: int,
        user_id: str,
        username: str,
        message_content: str,
        message_ts: Optional[datetime],
        message_id: Optional[str],
    ) -> None:
        """Update existing session with new activity."""
        session.last_activity_at = datetime.now(timezone.utc)
        session.last_message_id = message_id
        self.sessions.move_to_end(session_key)
        self._record_session_context(
            session_key, channel_id, user_id, username, message_content, message_ts
        )

    async def _create_new_session(
        self,
        session_key: str,
        channel_id: int,
        user_id: str,
        username: str,
        message_content: str,
        message_ts: Optional[datetime],
        message_id: Optional[str],
        model_alias: str,
    ) -> Tuple[object, str]:
        """Create a new chat session."""
        system_prompt = self.channel_prompts.get(channel_id, BOT_PERSONA_PROMPT)
        chat = self.llm_service.create_chat_session_for_alias(model_alias, system_prompt)
        chat.model_alias = model_alias

        self.sessions[session_key] = ChatSession(
            chat=chat,
            user_id=user_id,
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            last_message_id=message_id,
            model_alias=model_alias,
        )

        self._record_session_context(
            session_key,
            channel_id,
            user_id,
            username,
            message_content,
            message_ts,
            model_alias=model_alias,
        )
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
        cancel_event: Optional[asyncio.Event] = None,
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
        if not session:
            return

        # Get the history container - handle both OpenAI/ZAI (_history) and Gemini (history)
        # OpenAI/ZAI use _history internally with a history property that returns a copy
        # Gemini uses history directly as a deque
        if hasattr(session.chat, "_history") and session.chat._history:
            history_container = session.chat._history
        elif hasattr(session.chat, "history") and session.chat.history:
            history_container = session.chat.history
        else:
            return

        last_msg = history_container[-1]
        if not hasattr(last_msg, "message_ids"):
            last_msg.message_ids = []
        last_msg.message_ids.append(message_id)

    def undo_last_exchanges(self, session_key: str, num_to_undo: int) -> list:
        """Remove the last N user/assistant exchanges from a session's history."""
        session = self.sessions.get(session_key)
        if not session or not hasattr(session.chat, "history"):
            return []

        try:
            history = session.chat.history
            assistant_role = self.llm_service.get_assistant_role_name()
            user_role = self.llm_service.get_user_role_name()

            indices_to_remove = self._find_exchange_indices_to_remove(
                history, assistant_role, user_role, num_to_undo
            )
            if not indices_to_remove:
                return []

            new_history, removed = self._split_history_by_indices(history, indices_to_remove)
            session.chat.history = new_history

            return removed

        except Exception as e:
            logger.error(
                "Error undoing exchanges in session %s: %s",
                session_key,
                e,
                exc_info=True,
            )
            return []

    def _find_exchange_indices_to_remove(
        self, history: list, assistant_role: str, user_role: str, num_to_undo: int
    ) -> set:
        """Find indices of messages to remove for undo operation (single-pass optimization)."""
        indices_to_remove = set()

        # Iterate backwards to find user/assistant pairs efficiently
        # Start from the end and work backwards
        i = len(history) - 1
        pairs_found = 0

        while i >= 0 and pairs_found < num_to_undo:
            msg = history[i]
            if msg.role == assistant_role:
                # Found an assistant message - include it
                indices_to_remove.add(i)
                pairs_found += 1

                # Include preceding user messages in this pair
                prev_idx = i - 1
                while prev_idx >= 0 and history[prev_idx].role == user_role:
                    indices_to_remove.add(prev_idx)
                    prev_idx -= 1

            i -= 1

        return indices_to_remove

    def _split_history_by_indices(self, history: list, indices_to_remove: set) -> tuple:
        """Split history into kept and removed messages."""
        new_history = []
        removed = []
        for i, msg in enumerate(history):
            if i in indices_to_remove:
                removed.append(msg)
            else:
                new_history.append(msg)
        return new_history, removed

    async def _periodic_session_cleanup(self):
        """Periodically clean up inactive sessions."""
        inactive_minutes = getattr(self.config, "session_inactive_minutes", 30)
        if inactive_minutes <= 0:
            return

        # Cleanup interval: half of inactive time
        cleanup_interval = inactive_minutes * 30  # seconds

        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._cleanup_inactive_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during session cleanup: {e}", exc_info=True)

    async def _cleanup_inactive_sessions(self):
        """Remove sessions that haven't been active for configured time."""
        try:
            inactive_threshold = datetime.now(timezone.utc) - timedelta(
                minutes=getattr(self.config, "session_inactive_minutes", 30)
            )
            cleaned_count = 0

            # Clean up old session contexts
            keys_to_remove = []
            for key, context in list(self.session_contexts.items()):
                if context.last_activity_at < inactive_threshold:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.session_contexts[key]
                cleaned_count += 1

        except Exception as e:
            logger.error(f"Error during session cleanup: {e}", exc_info=True)

    async def cleanup(self) -> None:
        """Cancel the periodic cleanup task and clean up resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.sessions.clear()
        self.session_contexts.clear()

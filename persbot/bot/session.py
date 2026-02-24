"""Chat session management for SoyeBot."""

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from persbot.config import AppConfig
from persbot.prompts import BOT_PERSONA_PROMPT
from persbot.services.channel_prompt_service import ChannelPromptService
from persbot.services.llm_service import LLMService
from persbot.services.model_usage_service import ModelUsageService
from persbot.services.summarization_service import SummarizationConfig, SummarizationService

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
    # Pre-extracted images for retry operations (bypasses discord message extraction)
    images: list[bytes] = field(default_factory=list)


class SessionManager:
    """Manages chat sessions, optionally caching them indefinitely for deep threads."""

    def __init__(
        self,
        config: AppConfig,
        llm_service: LLMService,
        channel_prompt_service: Optional[ChannelPromptService] = None,
    ):
        self.config = config
        self.llm_service = llm_service
        self.channel_prompt_service = channel_prompt_service or ChannelPromptService()
        self.sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self.session_contexts: OrderedDict[str, SessionContext] = OrderedDict()
        self.channel_model_preferences: Dict[int, str] = {}  # channel_id -> model_alias override
        self._cleanup_task: Optional[asyncio.Task] = (
            None  # Track cleanup task for graceful shutdown
        )
        self._sessions_lock = asyncio.Lock()  # Lock for thread-safe session operations

        # Initialize summarization service
        self._summarization_service = SummarizationService(
            SummarizationConfig(
                threshold=config.summarization_threshold,
                keep_recent_messages=config.summarization_keep_recent,
                summarization_model=config.summarization_model,
                max_summary_length=config.summarization_max_tokens,
            )
        )

        # Start periodic session cleanup task
        if config.session_inactive_minutes > 0:
            self._cleanup_task = asyncio.create_task(self._periodic_session_cleanup())
            self._cleanup_task.add_done_callback(self._log_cleanup_error)

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
                # Incompatible - extract history before deletion
                history_to_transfer = self._extract_history(existing_session.chat)
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
                    history_to_transfer=history_to_transfer,
                )

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
        history_to_transfer: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[object, str]:
        """Create a new chat session, optionally with transferred history."""
        system_prompt = self.channel_prompt_service.get_effective_prompt(channel_id)
        chat = self.llm_service.create_chat_session_for_alias(model_alias, system_prompt)
        chat.model_alias = model_alias

        # Apply transferred history if provided
        if history_to_transfer and hasattr(chat, 'history'):
            self._apply_history(chat, history_to_transfer)
            logger.info(
                "Transferred %d messages to new session %s with model %s",
                len(history_to_transfer),
                session_key,
                model_alias,
            )

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

    def _extract_history(self, chat: object) -> List[Dict[str, str]]:
        """Extract history from a chat object into a normalized format.

        Works with both OpenAI/ZAI (_history list) and Gemini (history deque) formats.
        Returns a list of {role, content} dicts, excluding system messages.
        """
        history_container = None

        # Handle OpenAI/ZAI format (uses _history internally)
        if hasattr(chat, "_history") and chat._history:
            history_container = chat._history
        # Handle Gemini format (uses history directly)
        elif hasattr(chat, "history") and chat.history:
            history_container = chat.history

        if not history_container:
            return []

        # Ensure history_container is iterable (handles Mock objects gracefully)
        try:
            iter(history_container)
        except TypeError:
            return []

        normalized = []
        for msg in history_container:
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)

            # Skip system messages (new session gets fresh system prompt)
            # Skip messages without proper role/content
            if role and content and role != 'system':
                normalized.append({'role': role, 'content': content})

        return normalized

    def _apply_history(self, chat: object, history: List[Dict[str, str]]) -> None:
        """Apply normalized history to a chat object.

        Appends messages to the chat's history in a format-compatible way.
        Works with both OpenAI/ZAI and Gemini chat objects.
        """
        if not history or not hasattr(chat, 'history'):
            return

        # Get the message class used by this chat implementation
        # Most chat objects have their messages as objects with role/content attributes
        # We'll create simple message objects that the chat can work with

        # Try to get the history container (handle both _history and history)
        if hasattr(chat, "_history"):
            history_container = chat._history
        else:
            history_container = chat.history

        # Determine the message class to use based on existing history
        if history_container:
            # Use the same type as existing messages
            sample_msg = history_container[0] if len(history_container) > 0 else None
            msg_class = type(sample_msg) if sample_msg else None
        else:
            msg_class = None

        for msg_dict in history:
            if msg_class:
                # Create message of the same type
                try:
                    new_msg = msg_class(
                        role=msg_dict['role'],
                        content=msg_dict['content']
                    )
                    history_container.append(new_msg)
                    continue
                except Exception:
                    pass  # Fall through to dict fallback

            # Fallback: try to create a simple object with role/content
            class SimpleMessage:
                def __init__(self, role, content):
                    self.role = role
                    self.content = content

            history_container.append(SimpleMessage(
                role=msg_dict['role'],
                content=msg_dict['content']
            ))

    @property
    def channel_prompts(self) -> Dict[int, str]:
        """Backward-compatible property returning all channel prompts."""
        return self.channel_prompt_service.get_all_channel_prompts()

    async def set_channel_prompt(self, channel_id: int, prompt_content: Optional[str]) -> None:
        """Set and persist a custom system prompt for a specific channel."""
        await self.channel_prompt_service.set_channel_prompt(channel_id, prompt_content)

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

    def _log_cleanup_error(self, task: asyncio.Task) -> None:
        """Log errors from background session cleanup task."""
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("Session cleanup task was cancelled")
        except Exception as e:
            logger.error("Session cleanup task failed: %s", e, exc_info=True)

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

    def check_and_summarize_history(self, session_key: str) -> bool:
        """Check if history needs summarization and trigger if threshold exceeded.

        This method should be called after messages are added to history.
        If the history exceeds the threshold, older messages are summarized
        and replaced with a summary message.

        Args:
            session_key: The session key to check.

        Returns:
            True if summarization was performed, False otherwise.
        """
        session = self.sessions.get(session_key)
        if not session or not hasattr(session.chat, "history"):
            return False

        history = session.chat.history
        if not self._summarization_service.should_summarize(history):
            return False

        # Get messages to summarize and keep
        to_summarize, to_keep = self._summarization_service.get_messages_to_summarize(history)

        if not to_summarize:
            return False

        # Estimate tokens saved
        tokens_before = self._summarization_service.estimate_tokens(to_summarize)
        estimated_summary_tokens = self._summarization_service.config.max_summary_length
        tokens_saved = max(0, tokens_before - estimated_summary_tokens)

        # Format and create summary prompt
        formatted = self._summarization_service.format_history_for_summary(to_summarize)
        summary_prompt = self._summarization_service.create_summary_prompt(formatted)

        # Store the summarization request for async processing
        # The actual summarization will be done by the LLM service
        session._pending_summarization = {
            "prompt": summary_prompt,
            "messages_to_summarize": to_summarize,
            "messages_to_keep": to_keep,
            "tokens_saved_estimate": tokens_saved,
        }

        logger.info(
            "Session %s history exceeds threshold (%d messages). Summarization queued.",
            session_key,
            len(history),
        )

        return True

    def apply_summarization(self, session_key: str, summary_text: str) -> bool:
        """Apply summarization by replacing old messages with summary.

        Args:
            session_key: The session key to update.
            summary_text: The generated summary text.

        Returns:
            True if summarization was applied, False otherwise.
        """
        session = self.sessions.get(session_key)
        if not session or not hasattr(session.chat, "history"):
            return False

        pending = getattr(session, "_pending_summarization", None)
        if not pending:
            return False

        # Create summary message
        summary_msg = self._summarization_service.create_summary_message(summary_text)

        # Replace history: summary + kept messages
        new_history = [summary_msg] + pending["messages_to_keep"]
        session.chat.history = new_history

        # Record metrics
        self._summarization_service.record_summarization(
            len(pending["messages_to_summarize"]),
            pending["tokens_saved_estimate"],
        )

        # Clear pending summarization
        session._pending_summarization = None

        logger.info(
            "Summarization applied to session %s. New history length: %d",
            session_key,
            len(new_history),
        )

        return True

    def get_summarization_stats(self) -> dict:
        """Get summarization statistics.

        Returns:
            Dict with summarization metrics.
        """
        return self._summarization_service.get_stats()

    def get_cache_stats(self) -> dict:
        """Get cache statistics from the LLM service.

        Returns:
            Dict with cache metrics if available, empty dict otherwise.
        """
        if hasattr(self.llm_service, "get_cache_stats"):
            return self.llm_service.get_cache_stats()
        return {}

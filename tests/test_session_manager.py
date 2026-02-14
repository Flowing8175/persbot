"""Feature tests for SessionManager.

Tests focus on behavior:
- ChatSession, SessionContext, ResolvedSession dataclasses
- Session creation and retrieval via get_or_create()
- Model preferences and session compatibility
- Channel prompts and resets
- History manipulation with undo_last_exchanges()
- Cache eviction
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch

from persbot.bot.session import (
    ChatSession,
    SessionContext,
    ResolvedSession,
    SessionManager,
)


# ============================================================================
# Test Dataclasses
# ============================================================================


class TestChatSession:
    """Tests for ChatSession dataclass."""

    def test_creates_with_required_fields(self):
        """ChatSession can be created with required fields."""
        mock_chat = Mock()
        session = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id="channel:123",
            last_activity_at=datetime.now(timezone.utc),
        )
        assert session.chat == mock_chat
        assert session.user_id == "user123"
        assert session.session_id == "channel:123"
        assert session.last_activity_at is not None

    def test_default_last_message_id_is_none(self):
        """last_message_id defaults to None."""
        session = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id="channel:123",
            last_activity_at=datetime.now(timezone.utc),
        )
        assert session.last_message_id is None

    def test_default_model_alias_is_none(self):
        """model_alias defaults to None."""
        session = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id="channel:123",
            last_activity_at=datetime.now(timezone.utc),
        )
        assert session.model_alias is None

    def test_can_set_all_fields(self):
        """ChatSession can have all fields set."""
        now = datetime.now(timezone.utc)
        session = ChatSession(
            chat=Mock(),
            user_id="user456",
            session_id="channel:456",
            last_activity_at=now,
            last_message_id="msg789",
            model_alias="Gemini 2.5 Flash",
        )
        assert session.user_id == "user456"
        assert session.session_id == "channel:456"
        assert session.last_activity_at == now
        assert session.last_message_id == "msg789"
        assert session.model_alias == "Gemini 2.5 Flash"


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_creates_with_required_fields(self):
        """SessionContext can be created with required fields."""
        now = datetime.now(timezone.utc)
        ctx = SessionContext(
            session_id="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            started_at=now,
            last_activity_at=now,
        )
        assert ctx.session_id == "channel:123"
        assert ctx.channel_id == 123
        assert ctx.user_id == "user123"
        assert ctx.username == "testuser"
        assert ctx.started_at == now
        assert ctx.last_activity_at == now

    def test_default_last_message_preview_is_empty(self):
        """last_message_preview defaults to empty string."""
        now = datetime.now(timezone.utc)
        ctx = SessionContext(
            session_id="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            started_at=now,
            last_activity_at=now,
        )
        assert ctx.last_message_preview == ""

    def test_default_title_is_none(self):
        """title defaults to None."""
        now = datetime.now(timezone.utc)
        ctx = SessionContext(
            session_id="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            started_at=now,
            last_activity_at=now,
        )
        assert ctx.title is None

    def test_default_model_alias_is_none(self):
        """model_alias defaults to None."""
        now = datetime.now(timezone.utc)
        ctx = SessionContext(
            session_id="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            started_at=now,
            last_activity_at=now,
        )
        assert ctx.model_alias is None

    def test_can_set_all_fields(self):
        """SessionContext can have all fields set."""
        now = datetime.now(timezone.utc)
        ctx = SessionContext(
            session_id="channel:456",
            channel_id=456,
            user_id="user456",
            username="anotheruser",
            started_at=now,
            last_activity_at=now,
            last_message_preview="Hello world",
            title="My Session",
            model_alias="GPT-4o",
        )
        assert ctx.last_message_preview == "Hello world"
        assert ctx.title == "My Session"
        assert ctx.model_alias == "GPT-4o"


class TestResolvedSession:
    """Tests for ResolvedSession dataclass."""

    def test_creates_with_required_fields(self):
        """ResolvedSession can be created with required fields."""
        resolved = ResolvedSession(
            session_key="channel:123",
            cleaned_message="Hello bot",
        )
        assert resolved.session_key == "channel:123"
        assert resolved.cleaned_message == "Hello bot"

    def test_default_is_reply_to_summary_is_false(self):
        """is_reply_to_summary defaults to False."""
        resolved = ResolvedSession(
            session_key="channel:123",
            cleaned_message="Hello",
        )
        assert resolved.is_reply_to_summary is False

    def test_can_set_is_reply_to_summary(self):
        """is_reply_to_summary can be set to True."""
        resolved = ResolvedSession(
            session_key="channel:123",
            cleaned_message="Reply",
            is_reply_to_summary=True,
        )
        assert resolved.is_reply_to_summary is True


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Create a mock config with cleanup disabled."""
    config = Mock()
    config.session_inactive_minutes = 0  # Disable cleanup task
    config.session_cache_limit = 200
    config.assistant_model_name = "gemini-2.5-flash"
    return config


@pytest.fixture
def mock_config_with_eviction():
    """Create a mock config with small cache limit for eviction tests."""
    config = Mock()
    config.session_inactive_minutes = 0  # Disable cleanup task
    config.session_cache_limit = 2  # Small limit for testing eviction
    config.assistant_model_name = "gemini-2.5-flash"
    return config


@pytest.fixture
def mock_model_usage_service():
    """Create a mock model usage service with MODEL_DEFINITIONS."""
    service = Mock()
    service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash"

    # Create mock model definitions
    gemini_def = Mock()
    gemini_def.api_model_name = "gemini-2.5-flash"
    gemini_def.provider = "gemini"

    gpt_def = Mock()
    gpt_def.api_model_name = "gpt-5-mini"
    gpt_def.provider = "openai"

    service.MODEL_DEFINITIONS = {
        "Gemini 2.5 flash": gemini_def,
        "GPT 5 mini": gpt_def,
    }

    return service


@pytest.fixture
def mock_llm_service(mock_model_usage_service):
    """Create a mock LLM service."""
    service = Mock()
    service.model_usage_service = mock_model_usage_service
    service.get_assistant_role_name = Mock(return_value="assistant")
    service.get_user_role_name = Mock(return_value="user")
    return service


@pytest.fixture
def manager(mock_config, mock_llm_service):
    """Create a SessionManager with cleanup disabled."""
    return SessionManager(
        config=mock_config,
        llm_service=mock_llm_service,
    )


@pytest.fixture
def manager_with_eviction(mock_config_with_eviction, mock_llm_service):
    """Create a SessionManager with small cache limit for eviction tests."""
    return SessionManager(
        config=mock_config_with_eviction,
        llm_service=mock_llm_service,
    )


# ============================================================================
# Test SessionManager Initialization
# ============================================================================


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_creates_manager(self, mock_config, mock_llm_service):
        """SessionManager can be created."""
        manager = SessionManager(
            config=mock_config,
            llm_service=mock_llm_service,
        )
        assert manager.config == mock_config
        assert manager.llm_service == mock_llm_service
        assert manager.sessions == {}
        assert manager.session_contexts == {}
        assert manager.channel_prompts == {}
        assert manager.channel_model_preferences == {}

    def test_no_cleanup_task_when_disabled(self, mock_config, mock_llm_service):
        """No cleanup task is created when session_inactive_minutes is 0."""
        manager = SessionManager(
            config=mock_config,
            llm_service=mock_llm_service,
        )
        assert manager._cleanup_task is None


# ============================================================================
# Test get_or_create()
# ============================================================================


class TestGetOrCreate:
    """Tests for get_or_create() method."""

    @pytest.mark.asyncio
    async def test_creates_new_session_when_not_exists(self, manager, mock_llm_service):
        """get_or_create creates new session when not found."""
        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        chat, session_key = await manager.get_or_create(
            user_id="user123",
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="Hello",
        )

        assert chat == mock_chat
        assert session_key == "channel:123"
        assert "channel:123" in manager.sessions
        mock_llm_service.create_chat_session_for_alias.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_existing_session_when_compatible(self, manager, mock_llm_service):
        """get_or_create returns existing session when model is compatible."""
        # Create initial session
        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        await manager.get_or_create(
            user_id="user123",
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="First message",
        )

        # Get existing session (should not create new one)
        chat2, session_key = await manager.get_or_create(
            user_id="user123",
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="Second message",
        )

        # Should return same chat, only called once
        assert chat2 == mock_chat
        assert session_key == "channel:123"
        mock_llm_service.create_chat_session_for_alias.assert_called_once()

    @pytest.mark.asyncio
    async def test_resets_session_when_model_changes(self, manager, mock_llm_service):
        """get_or_create creates new session when model changes."""
        # Create initial session with default model
        mock_chat1 = Mock()
        mock_chat1.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat1)

        await manager.get_or_create(
            user_id="user123",
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="First message",
        )

        # Change model preference for this channel
        manager.set_session_model(123, "GPT 5 mini")

        # Next call should create new session with different model
        mock_chat2 = Mock()
        mock_chat2.model_alias = "GPT 5 mini"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat2)

        chat2, _ = await manager.get_or_create(
            user_id="user123",
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="Second message",
        )

        # Should have created new chat (mock was reassigned, so call_count is 1)
        assert chat2 == mock_chat2
        assert mock_llm_service.create_chat_session_for_alias.call_count == 1

    @pytest.mark.asyncio
    async def test_records_session_context(self, manager, mock_llm_service):
        """get_or_create records session context."""
        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        await manager.get_or_create(
            user_id="user123",
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="Hello world",
        )

        assert "channel:123" in manager.session_contexts
        ctx = manager.session_contexts["channel:123"]
        assert ctx.channel_id == 123
        assert ctx.user_id == "user123"
        assert ctx.username == "testuser"
        assert ctx.last_message_preview == "Hello world"

    @pytest.mark.asyncio
    async def test_updates_existing_session_activity(self, manager, mock_llm_service):
        """get_or_create updates activity timestamp for existing session."""
        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        # Create session
        await manager.get_or_create(
            user_id="user123",
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="First",
        )

        first_activity = manager.sessions["channel:123"].last_activity_at

        # Small delay to ensure timestamp difference
        import asyncio
        await asyncio.sleep(0.01)

        # Get existing session
        await manager.get_or_create(
            user_id="user123",
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="Second",
        )

        second_activity = manager.sessions["channel:123"].last_activity_at
        assert second_activity > first_activity

    @pytest.mark.asyncio
    async def test_converts_user_id_to_string(self, manager, mock_llm_service):
        """get_or_create converts integer user_id to string."""
        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        await manager.get_or_create(
            user_id=12345,  # Integer
            username="testuser",
            session_key="channel:123",
            channel_id=123,
            message_content="Hello",
        )

        assert manager.sessions["channel:123"].user_id == "12345"


# ============================================================================
# Test set_session_model()
# ============================================================================


class TestSetSessionModel:
    """Tests for set_session_model() method."""

    def test_updates_channel_model_preferences(self, manager):
        """set_session_model updates channel_model_preferences."""
        manager.set_session_model(123, "GPT 5 mini")

        assert manager.channel_model_preferences[123] == "GPT 5 mini"

    def test_updates_session_context_model_alias(self, manager, mock_llm_service):
        """set_session_model updates model_alias in session context."""
        # Create a session context first
        manager._record_session_context(
            session_key="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            message_content="Hello",
            message_ts=datetime.now(timezone.utc),
        )

        manager.set_session_model(123, "GPT 5 mini")

        assert manager.session_contexts["channel:123"].model_alias == "GPT 5 mini"

    def test_updates_preference_without_existing_context(self, manager):
        """set_session_model stores preference even without existing context."""
        manager.set_session_model(999, "GPT 5 mini")

        assert manager.channel_model_preferences[999] == "GPT 5 mini"


# ============================================================================
# Test set_channel_prompt() and reset_session_by_channel()
# ============================================================================


class TestChannelPromptAndReset:
    """Tests for set_channel_prompt() and reset_session_by_channel() methods."""

    def test_set_channel_prompt_stores_prompt(self, manager):
        """set_channel_prompt stores the prompt for a channel."""
        manager.set_channel_prompt(123, "Custom system prompt")

        assert manager.channel_prompts[123] == "Custom system prompt"

    def test_set_channel_prompt_removes_prompt_when_none(self, manager):
        """set_channel_prompt removes prompt when None is passed."""
        manager.channel_prompts[123] = "Existing prompt"

        manager.set_channel_prompt(123, None)

        assert 123 not in manager.channel_prompts

    def test_set_channel_prompt_resets_session(self, manager, mock_llm_service):
        """set_channel_prompt resets the session for the channel."""
        # Create a session
        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        manager.sessions["channel:123"] = Mock()
        manager.session_contexts["channel:123"] = Mock()

        manager.set_channel_prompt(123, "New prompt")

        assert "channel:123" not in manager.sessions
        assert "channel:123" not in manager.session_contexts

    def test_reset_session_by_channel_returns_true_when_exists(self, manager):
        """reset_session_by_channel returns True when session exists."""
        manager.sessions["channel:123"] = Mock()
        manager.session_contexts["channel:123"] = Mock()

        result = manager.reset_session_by_channel(123)

        assert result is True

    def test_reset_session_by_channel_returns_false_when_not_found(self, manager):
        """reset_session_by_channel returns False when session not found."""
        result = manager.reset_session_by_channel(999)

        assert result is False

    def test_reset_session_by_channel_removes_session(self, manager):
        """reset_session_by_channel removes session and context."""
        manager.sessions["channel:123"] = Mock()
        manager.session_contexts["channel:123"] = Mock()

        manager.reset_session_by_channel(123)

        assert "channel:123" not in manager.sessions
        assert "channel:123" not in manager.session_contexts


# ============================================================================
# Test resolve_session()
# ============================================================================


class TestResolveSession:
    """Tests for resolve_session() method."""

    @pytest.mark.asyncio
    async def test_returns_resolved_session_with_channel_key(self, manager):
        """resolve_session returns ResolvedSession with channel key."""
        result = await manager.resolve_session(
            channel_id=123,
            author_id=456,
            username="testuser",
            message_id="msg789",
            message_content="  Hello bot!  ",
        )

        assert isinstance(result, ResolvedSession)
        assert result.session_key == "channel:123"
        assert result.cleaned_message == "Hello bot!"
        assert result.is_reply_to_summary is False

    @pytest.mark.asyncio
    async def test_strips_message_content(self, manager):
        """resolve_session strips whitespace from message content."""
        result = await manager.resolve_session(
            channel_id=123,
            author_id=456,
            username="testuser",
            message_id="msg789",
            message_content="   Spaced message   ",
        )

        assert result.cleaned_message == "Spaced message"


# ============================================================================
# Test link_message_to_session()
# ============================================================================


class TestLinkMessageToSession:
    """Tests for link_message_to_session() method."""

    def test_links_message_when_chat_has_history(self, manager):
        """link_message_to_session appends message_id to last history item."""
        # Create mock chat with _history (accessed directly, not through property)
        mock_chat = Mock()
        last_msg = Mock()
        last_msg.message_ids = []
        mock_chat._history = [Mock(), last_msg]

        manager.sessions["channel:123"] = Mock(chat=mock_chat)

        manager.link_message_to_session("discord_msg_456", "channel:123")

        assert "discord_msg_456" in last_msg.message_ids

    def test_creates_message_ids_list_if_not_exists(self, manager):
        """link_message_to_session creates message_ids list if not present."""
        # Create mock chat with _history but no message_ids
        mock_chat = Mock()
        last_msg = Mock(spec=[])  # No message_ids attribute
        mock_chat._history = [last_msg]

        manager.sessions["channel:123"] = Mock(chat=mock_chat)

        manager.link_message_to_session("discord_msg_456", "channel:123")

        assert hasattr(last_msg, "message_ids")
        assert "discord_msg_456" in last_msg.message_ids

    def test_does_nothing_when_session_not_found(self, manager):
        """link_message_to_session does nothing when session not found."""
        # Should not raise
        manager.link_message_to_session("discord_msg_456", "channel:nonexistent")

    def test_does_nothing_when_chat_has_no_history(self, manager):
        """link_message_to_session does nothing when chat has no history."""
        mock_chat = Mock()
        mock_chat._history = []

        manager.sessions["channel:123"] = Mock(chat=mock_chat)

        # Should not raise
        manager.link_message_to_session("discord_msg_456", "channel:123")


# ============================================================================
# Test undo_last_exchanges()
# ============================================================================


class TestUndoLastExchanges:
    """Tests for undo_last_exchanges() method."""

    def test_removes_user_assistant_pairs(self, manager, mock_llm_service):
        """undo_last_exchanges removes user/assistant pairs from history."""
        # Create history with user/assistant pairs
        msg1 = Mock(role="user", content="Q1")
        msg2 = Mock(role="assistant", content="A1")
        msg3 = Mock(role="user", content="Q2")
        msg4 = Mock(role="assistant", content="A2")

        mock_chat = Mock()
        mock_chat.history = [msg1, msg2, msg3, msg4]

        manager.sessions["channel:123"] = Mock(chat=mock_chat)
        mock_llm_service.get_assistant_role_name = Mock(return_value="assistant")
        mock_llm_service.get_user_role_name = Mock(return_value="user")

        removed = manager.undo_last_exchanges("channel:123", 1)

        assert len(removed) == 2  # One user + one assistant
        assert msg3 in removed  # Last user
        assert msg4 in removed  # Last assistant
        assert len(mock_chat.history) == 2  # Q1, A1 remain

    def test_removes_multiple_pairs(self, manager, mock_llm_service):
        """undo_last_exchanges removes multiple user/assistant pairs."""
        msg1 = Mock(role="user", content="Q1")
        msg2 = Mock(role="assistant", content="A1")
        msg3 = Mock(role="user", content="Q2")
        msg4 = Mock(role="assistant", content="A2")

        mock_chat = Mock()
        mock_chat.history = [msg1, msg2, msg3, msg4]

        manager.sessions["channel:123"] = Mock(chat=mock_chat)
        mock_llm_service.get_assistant_role_name = Mock(return_value="assistant")
        mock_llm_service.get_user_role_name = Mock(return_value="user")

        removed = manager.undo_last_exchanges("channel:123", 2)

        assert len(removed) == 4  # Two pairs
        assert len(mock_chat.history) == 0

    def test_returns_empty_list_when_no_session(self, manager):
        """undo_last_exchanges returns empty list when session not found."""
        result = manager.undo_last_exchanges("channel:nonexistent", 1)
        assert result == []

    def test_returns_empty_list_when_no_history(self, manager):
        """undo_last_exchanges returns empty list when no history."""
        mock_chat = Mock()
        mock_chat.history = []

        manager.sessions["channel:123"] = Mock(chat=mock_chat)

        result = manager.undo_last_exchanges("channel:123", 1)
        assert result == []

    def test_handles_multiple_user_before_assistant(self, manager, mock_llm_service):
        """undo_last_exchanges handles multiple user messages before assistant."""
        msg1 = Mock(role="user", content="Q1")
        msg2 = Mock(role="user", content="Q1-followup")
        msg3 = Mock(role="assistant", content="A1")
        msg4 = Mock(role="user", content="Q2")

        mock_chat = Mock()
        mock_chat.history = [msg1, msg2, msg3, msg4]

        manager.sessions["channel:123"] = Mock(chat=mock_chat)
        mock_llm_service.get_assistant_role_name = Mock(return_value="assistant")
        mock_llm_service.get_user_role_name = Mock(return_value="user")

        removed = manager.undo_last_exchanges("channel:123", 1)

        # Should remove the assistant and all preceding user messages
        assert msg3 in removed  # Assistant
        assert len(mock_chat.history) == 1  # Only Q2 remains


# ============================================================================
# Test _evict_if_needed()
# ============================================================================


class TestEvictIfNeeded:
    """Tests for _evict_if_needed() method."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_sessions(self, manager_with_eviction, mock_llm_service):
        """_evict_if_needed removes oldest sessions when limit exceeded."""
        manager = manager_with_eviction
        manager.config.session_cache_limit = 2

        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        # Create 3 sessions (limit is 2)
        await manager.get_or_create("user1", "u1", "channel:1", 1, "msg1")
        await manager.get_or_create("user2", "u2", "channel:2", 2, "msg2")
        await manager.get_or_create("user3", "u3", "channel:3", 3, "msg3")

        # First session should be evicted
        assert "channel:1" not in manager.sessions
        assert "channel:2" in manager.sessions
        assert "channel:3" in manager.sessions

    @pytest.mark.asyncio
    async def test_evicts_session_contexts_too(self, manager_with_eviction, mock_llm_service):
        """_evict_if_needed removes session contexts when limit exceeded."""
        manager = manager_with_eviction
        manager.config.session_cache_limit = 2

        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        # Create 3 sessions
        await manager.get_or_create("user1", "u1", "channel:1", 1, "msg1")
        await manager.get_or_create("user2", "u2", "channel:2", 2, "msg2")
        await manager.get_or_create("user3", "u3", "channel:3", 3, "msg3")

        # First session context should be evicted too
        assert "channel:1" not in manager.session_contexts
        assert "channel:2" in manager.session_contexts
        assert "channel:3" in manager.session_contexts


# ============================================================================
# Test _record_session_context()
# ============================================================================


class TestRecordSessionContext:
    """Tests for _record_session_context() method."""

    def test_creates_new_context(self, manager):
        """_record_session_context creates new context when not exists."""
        now = datetime.now(timezone.utc)

        manager._record_session_context(
            session_key="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            message_content="Hello",
            message_ts=now,
            model_alias="Gemini 2.5 flash",
        )

        assert "channel:123" in manager.session_contexts
        ctx = manager.session_contexts["channel:123"]
        assert ctx.channel_id == 123
        assert ctx.model_alias == "Gemini 2.5 flash"

    def test_updates_existing_context(self, manager):
        """_record_session_context updates existing context."""
        now = datetime.now(timezone.utc)

        # Create initial context
        manager._record_session_context(
            session_key="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            message_content="First",
            message_ts=now,
        )

        first_activity = manager.session_contexts["channel:123"].last_activity_at

        # Update context
        import asyncio
        import time
        time.sleep(0.01)  # Small delay

        manager._record_session_context(
            session_key="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            message_content="Second",
            message_ts=datetime.now(timezone.utc),
            model_alias="GPT 5 mini",
        )

        ctx = manager.session_contexts["channel:123"]
        assert ctx.last_activity_at > first_activity
        assert ctx.last_message_preview == "Second"
        assert ctx.model_alias == "GPT 5 mini"

    def test_preserves_existing_preview_on_empty(self, manager):
        """_record_session_context preserves existing preview when new is empty."""
        now = datetime.now(timezone.utc)

        manager._record_session_context(
            session_key="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            message_content="Original preview",
            message_ts=now,
        )

        manager._record_session_context(
            session_key="channel:123",
            channel_id=123,
            user_id="user123",
            username="testuser",
            message_content="   ",  # Whitespace only
            message_ts=datetime.now(timezone.utc),
        )

        ctx = manager.session_contexts["channel:123"]
        assert ctx.last_message_preview == "Original preview"


# ============================================================================
# Test _resolve_target_model_alias()
# ============================================================================


class TestResolveTargetModelAlias:
    """Tests for _resolve_target_model_alias() method."""

    def test_returns_context_model_alias_if_set(self, manager, mock_llm_service):
        """Returns model_alias from session context if set."""
        manager.session_contexts["channel:123"] = Mock(model_alias="GPT 5 mini")

        result = manager._resolve_target_model_alias("channel:123", 123)

        assert result == "GPT 5 mini"

    def test_returns_channel_preference_if_no_context(self, manager):
        """Returns channel_model_preferences if no context model_alias."""
        manager.channel_model_preferences[123] = "GPT 5 mini"

        result = manager._resolve_target_model_alias("channel:123", 123)

        assert result == "GPT 5 mini"

    def test_returns_default_from_config(self, manager, mock_llm_service):
        """Returns default model derived from config when no overrides."""
        # The manager should derive the alias from config.assistant_model_name
        result = manager._resolve_target_model_alias("channel:123", 123)

        # Should return an alias (either from MODEL_DEFINITIONS or default)
        assert result is not None


# ============================================================================
# Test _periodic_session_cleanup() - indirect tests via cleanup()
# ============================================================================


class TestCleanup:
    """Tests for cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleanup_clears_sessions(self, manager, mock_llm_service):
        """cleanup clears all sessions and contexts."""
        mock_chat = Mock()
        mock_chat.model_alias = "Gemini 2.5 flash"
        mock_llm_service.create_chat_session_for_alias = Mock(return_value=mock_chat)

        # Create some sessions
        await manager.get_or_create("user1", "u1", "channel:1", 1, "msg1")
        await manager.get_or_create("user2", "u2", "channel:2", 2, "msg2")

        assert len(manager.sessions) == 2

        await manager.cleanup()

        assert len(manager.sessions) == 0
        assert len(manager.session_contexts) == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_no_cleanup_task(self, manager):
        """cleanup handles case where no cleanup task exists."""
        # Manager with cleanup disabled has no task
        assert manager._cleanup_task is None

        # Should not raise
        await manager.cleanup()

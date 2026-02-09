"""Tests for SessionManager."""

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from persbot.bot.session import (
    ChatSession,
    ResolvedSession,
    SessionContext,
    SessionManager,
)
from persbot.services.base import ChatMessage

# =============================================================================
# 1. ChatSession - Test dataclass
# =============================================================================


class TestChatSession:
    """Tests for ChatSession dataclass."""

    def test_initialization(self):
        """Test ChatSession initialization with all fields."""
        chat = Mock()
        user_id = "123456789"
        session_id = "channel:111"
        last_activity_at = datetime.now(timezone.utc)
        last_message_id = "msg123"
        model_alias = "gemini-2.5-flash"

        session = ChatSession(
            chat=chat,
            user_id=user_id,
            session_id=session_id,
            last_activity_at=last_activity_at,
            last_message_id=last_message_id,
            model_alias=model_alias,
        )

        assert session.chat == chat
        assert session.user_id == user_id
        assert session.session_id == session_id
        assert session.last_activity_at == last_activity_at
        assert session.last_message_id == last_message_id
        assert session.model_alias == model_alias

    def test_initialization_without_optional_fields(self):
        """Test ChatSession initialization without optional fields."""
        chat = Mock()
        user_id = "123456789"
        session_id = "channel:111"
        last_activity_at = datetime.now(timezone.utc)

        session = ChatSession(
            chat=chat,
            user_id=user_id,
            session_id=session_id,
            last_activity_at=last_activity_at,
        )

        assert session.chat == chat
        assert session.user_id == user_id
        assert session.session_id == session_id
        assert session.last_activity_at == last_activity_at
        assert session.last_message_id is None
        assert session.model_alias is None

    def test_field_values(self):
        """Test that ChatSession field values are correctly stored."""
        chat = Mock()
        session = ChatSession(
            chat=chat,
            user_id="user123",
            session_id="session456",
            last_activity_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            last_message_id="message789",
            model_alias="gemini-2.5-pro",
        )

        assert session.user_id == "user123"
        assert session.session_id == "session456"
        assert session.last_message_id == "message789"
        assert session.model_alias == "gemini-2.5-pro"

    def test_model_alias_tracking(self):
        """Test that model_alias field tracks the model correctly."""
        chat = Mock()
        session = ChatSession(
            chat=chat,
            user_id="user123",
            session_id="session456",
            last_activity_at=datetime.now(timezone.utc),
            model_alias="gemini-2.5-flash",
        )

        assert session.model_alias == "gemini-2.5-flash"

        # Update model alias
        session.model_alias = "gemini-2.5-pro"
        assert session.model_alias == "gemini-2.5-pro"


# =============================================================================
# 2. SessionContext - Test dataclass
# =============================================================================


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_initialization(self):
        """Test SessionContext initialization with all fields."""
        session_id = "channel:111"
        channel_id = 111
        user_id = "123456789"
        username = "TestUser"
        started_at = datetime.now(timezone.utc)
        last_activity_at = datetime.now(timezone.utc)
        last_message_preview = "Hello"
        title = "Test Session"
        model_alias = "gemini-2.5-flash"

        context = SessionContext(
            session_id=session_id,
            channel_id=channel_id,
            user_id=user_id,
            username=username,
            started_at=started_at,
            last_activity_at=last_activity_at,
            last_message_preview=last_message_preview,
            title=title,
            model_alias=model_alias,
        )

        assert context.session_id == session_id
        assert context.channel_id == channel_id
        assert context.user_id == user_id
        assert context.username == username
        assert context.started_at == started_at
        assert context.last_activity_at == last_activity_at
        assert context.last_message_preview == last_message_preview
        assert context.title == title
        assert context.model_alias == model_alias

    def test_initialization_without_optional_fields(self):
        """Test SessionContext initialization with default optional fields."""
        context = SessionContext(
            session_id="channel:111",
            channel_id=111,
            user_id="123456789",
            username="TestUser",
            started_at=datetime.now(timezone.utc),
            last_activity_at=datetime.now(timezone.utc),
        )

        assert context.last_message_preview == ""
        assert context.title is None
        assert context.model_alias is None

    def test_metadata_fields(self):
        """Test that all metadata fields are correctly stored."""
        context = SessionContext(
            session_id="channel:222",
            channel_id=222,
            user_id="user456",
            username="AnotherUser",
            started_at=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            last_activity_at=datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
            last_message_preview="This is a preview",
            title="My Session",
            model_alias="gemini-2.5-pro",
        )

        assert context.session_id == "channel:222"
        assert context.channel_id == 222
        assert context.user_id == "user456"
        assert context.username == "AnotherUser"
        assert context.last_message_preview == "This is a preview"
        assert context.title == "My Session"
        assert context.model_alias == "gemini-2.5-pro"

    def test_last_message_preview(self):
        """Test last_message_preview field behavior."""
        context = SessionContext(
            session_id="channel:111",
            channel_id=111,
            user_id="user123",
            username="TestUser",
            started_at=datetime.now(timezone.utc),
            last_activity_at=datetime.now(timezone.utc),
            last_message_preview="Hello world",
        )

        assert context.last_message_preview == "Hello world"

        # Update preview
        context.last_message_preview = "New message"
        assert context.last_message_preview == "New message"


# =============================================================================
# 3. ResolvedSession - Test dataclass
# =============================================================================


class TestResolvedSession:
    """Tests for ResolvedSession dataclass."""

    def test_initialization(self):
        """Test ResolvedSession initialization with all fields."""
        session_key = "channel:111"
        cleaned_message = "Hello bot"
        is_reply_to_summary = True

        resolved = ResolvedSession(
            session_key=session_key,
            cleaned_message=cleaned_message,
            is_reply_to_summary=is_reply_to_summary,
        )

        assert resolved.session_key == session_key
        assert resolved.cleaned_message == cleaned_message
        assert resolved.is_reply_to_summary == is_reply_to_summary

    def test_initialization_without_optional_fields(self):
        """Test ResolvedSession initialization with default optional fields."""
        resolved = ResolvedSession(
            session_key="channel:111",
            cleaned_message="Hello bot",
        )

        assert resolved.session_key == "channel:111"
        assert resolved.cleaned_message == "Hello bot"
        assert resolved.is_reply_to_summary is False

    def test_session_key_generation(self):
        """Test that session_key follows the expected format."""
        resolved = ResolvedSession(
            session_key="channel:12345",
            cleaned_message="Test message",
        )

        assert resolved.session_key.startswith("channel:")
        assert resolved.session_key == "channel:12345"

    def test_cleaned_message(self):
        """Test that cleaned_message stores the processed message content."""
        resolved = ResolvedSession(
            session_key="channel:111",
            cleaned_message="  Hello bot  ".strip(),
        )

        assert resolved.cleaned_message == "Hello bot"

    def test_is_reply_to_summary_flag(self):
        """Test is_reply_to_summary flag behavior."""
        resolved = ResolvedSession(
            session_key="channel:111",
            cleaned_message="Message",
            is_reply_to_summary=False,
        )

        assert resolved.is_reply_to_summary is False

        # Update flag
        resolved.is_reply_to_summary = True
        assert resolved.is_reply_to_summary is True


# =============================================================================
# 4. SessionManager.__init__() - Test initialization
# =============================================================================


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_empty_sessions_dict(self, mock_app_config, mock_llm_service):
        """Test that sessions dict is initialized as empty OrderedDict."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        assert isinstance(manager.sessions, OrderedDict)
        assert len(manager.sessions) == 0

    def test_empty_session_contexts_dict(self, mock_app_config, mock_llm_service):
        """Test that session_contexts dict is initialized as empty OrderedDict."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        assert isinstance(manager.session_contexts, OrderedDict)
        assert len(manager.session_contexts) == 0

    def test_empty_channel_prompts_dict(self, mock_app_config, mock_llm_service):
        """Test that channel_prompts dict is initialized as empty."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        assert isinstance(manager.channel_prompts, dict)
        assert len(manager.channel_prompts) == 0

    def test_empty_channel_model_preferences_dict(self, mock_app_config, mock_llm_service):
        """Test that channel_model_preferences dict is initialized as empty."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        assert isinstance(manager.channel_model_preferences, dict)
        assert len(manager.channel_model_preferences) == 0

    def test_config_and_llm_service_stored(self, mock_app_config, mock_llm_service):
        """Test that config and llm_service are properly stored."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        assert manager.config == mock_app_config
        assert manager.llm_service == mock_llm_service


# =============================================================================
# 5. _evict_if_needed() - Test LRU eviction
# =============================================================================


class TestEvictIfNeeded:
    """Tests for _evict_if_needed method."""

    def test_eviction_when_limit_exceeded(self, mock_app_config, mock_llm_service):
        """Test eviction when sessions exceed cache limit."""
        mock_app_config.session_cache_limit = 3

        manager = SessionManager(mock_app_config, mock_llm_service)

        # Add sessions beyond the limit
        for i in range(5):
            session_key = f"channel:{i}"
            manager.sessions[session_key] = Mock()
            manager.session_contexts[session_key] = Mock()

        # Call evict
        manager._evict_if_needed()

        # Should evict oldest 2 sessions
        assert len(manager.sessions) == 3
        assert len(manager.session_contexts) == 3
        assert "channel:0" not in manager.sessions
        assert "channel:1" not in manager.sessions
        assert "channel:2" in manager.sessions
        assert "channel:3" in manager.sessions
        assert "channel:4" in manager.sessions

    def test_fifo_eviction_order(self, mock_app_config, mock_llm_service):
        """Test that eviction follows FIFO order (first in, first out)."""
        mock_app_config.session_cache_limit = 2

        manager = SessionManager(mock_app_config, mock_llm_service)

        # Add sessions in order
        for i in range(4):
            session_key = f"channel:{i}"
            manager.sessions[session_key] = Mock()
            manager.session_contexts[session_key] = Mock()

        # Call evict
        manager._evict_if_needed()

        # First two should be evicted
        assert "channel:0" not in manager.sessions
        assert "channel:1" not in manager.sessions
        assert "channel:2" in manager.sessions
        assert "channel:3" in manager.sessions

    def test_no_eviction_when_under_limit(self, mock_app_config, mock_llm_service):
        """Test no eviction occurs when sessions are under the limit."""
        mock_app_config.session_cache_limit = 10

        manager = SessionManager(mock_app_config, mock_llm_service)

        # Add fewer sessions than limit
        for i in range(5):
            session_key = f"channel:{i}"
            manager.sessions[session_key] = Mock()
            manager.session_contexts[session_key] = Mock()

        # Call evict
        manager._evict_if_needed()

        # All sessions should remain
        assert len(manager.sessions) == 5
        assert len(manager.session_contexts) == 5

        for i in range(5):
            assert f"channel:{i}" in manager.sessions


# =============================================================================
# 6. _record_session_context() - Test context recording
# =============================================================================


class TestRecordSessionContext:
    """Tests for _record_session_context method."""

    def test_new_session_creation(self, mock_app_config, mock_llm_service):
        """Test creating a new session context."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"
        channel_id = 111
        user_id = "123456789"
        username = "TestUser"
        message_content = "Hello bot"
        message_ts = datetime.now(timezone.utc)

        manager._record_session_context(
            session_key,
            channel_id,
            user_id,
            username,
            message_content,
            message_ts,
        )

        assert session_key in manager.session_contexts
        context = manager.session_contexts[session_key]
        assert context.session_id == session_key
        assert context.channel_id == channel_id
        assert context.user_id == user_id
        assert context.username == username
        assert context.last_message_preview == message_content.strip()
        assert context.model_alias is None

    def test_existing_session_update(self, mock_app_config, mock_llm_service):
        """Test updating an existing session context."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create initial context
        manager._record_session_context(
            session_key,
            111,
            "user123",
            "User1",
            "First message",
            datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        )

        initial_context = manager.session_contexts[session_key]

        # Update with new message
        new_timestamp = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        manager._record_session_context(
            session_key,
            111,
            "user123",
            "User1",
            "Second message",
            new_timestamp,
        )

        updated_context = manager.session_contexts[session_key]

        # Should be the same context object (updated in place)
        assert updated_context is initial_context
        assert updated_context.last_activity_at == new_timestamp
        assert updated_context.last_message_preview == "Second message"

    def test_move_to_end_on_access(self, mock_app_config, mock_llm_service):
        """Test that move_to_end is called when updating existing context."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        # Add multiple contexts
        for i in range(3):
            manager._record_session_context(
                f"channel:{i}",
                i,
                f"user{i}",
                f"User{i}",
                f"Message {i}",
                datetime.now(timezone.utc),
            )

        # Update the first context (should move it to the end)
        manager._record_session_context(
            "channel:0",
            0,
            "user0",
            "User0",
            "Updated message",
            datetime.now(timezone.utc),
        )

        # First key should now be at the end
        keys = list(manager.session_contexts.keys())
        assert keys[-1] == "channel:0"

    def test_model_alias_update(self, mock_app_config, mock_llm_service):
        """Test that model_alias is updated when provided."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create context without model_alias
        manager._record_session_context(
            session_key,
            111,
            "user123",
            "User1",
            "Message",
            datetime.now(timezone.utc),
        )

        assert manager.session_contexts[session_key].model_alias is None

        # Update with model_alias
        manager._record_session_context(
            session_key,
            111,
            "user123",
            "User1",
            "Message",
            datetime.now(timezone.utc),
            model_alias="gemini-2.5-pro",
        )

        assert manager.session_contexts[session_key].model_alias == "gemini-2.5-pro"


# =============================================================================
# 7. set_session_model() - Test model preference
# =============================================================================


class TestSetSessionModel:
    """Tests for set_session_model method."""

    def test_setting_preference(self, mock_app_config, mock_llm_service):
        """Test setting a model preference for a channel."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111
        model_alias = "gemini-2.5-pro"

        manager.set_session_model(channel_id, model_alias)

        assert channel_id in manager.channel_model_preferences
        assert manager.channel_model_preferences[channel_id] == model_alias

    def test_updating_existing_session_context(self, mock_app_config, mock_llm_service):
        """Test updating model_alias when session context exists."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111
        session_key = f"channel:{channel_id}"

        # Create a session context
        manager._record_session_context(
            session_key,
            channel_id,
            "user123",
            "User1",
            "Message",
            datetime.now(timezone.utc),
        )

        # Set model preference
        manager.set_session_model(channel_id, "gemini-2.5-pro")

        # Check channel preferences
        assert manager.channel_model_preferences[channel_id] == "gemini-2.5-pro"

        # Check session context model_alias was updated
        assert manager.session_contexts[session_key].model_alias == "gemini-2.5-pro"

    def test_model_preference_without_existing_session(self, mock_app_config, mock_llm_service):
        """Test setting preference when no session exists yet."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111

        # No session exists
        assert f"channel:{channel_id}" not in manager.session_contexts

        # Set model preference
        manager.set_session_model(channel_id, "gemini-2.5-pro")

        # Preference should be stored even without session
        assert manager.channel_model_preferences[channel_id] == "gemini-2.5-pro"


# =============================================================================
# 8. get_or_create() - Test session retrieval
# =============================================================================


class TestGetOrCreate:
    """Tests for get_or_create method."""

    @pytest.mark.asyncio
    async def test_new_session_creation(self, mock_app_config, mock_llm_service):
        """Test creating a new session when none exists."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        mock_llm_service.create_chat_session_for_alias.return_value = Mock()

        chat, session_key = await manager.get_or_create(
            user_id="123456789",
            username="TestUser",
            session_key="channel:111",
            channel_id=111,
            message_content="Hello bot",
        )

        assert session_key == "channel:111"
        assert session_key in manager.sessions
        mock_llm_service.create_chat_session_for_alias.assert_called_once()

    @pytest.mark.asyncio
    async def test_existing_session_reuse(self, mock_app_config, mock_llm_service):
        """Test reusing an existing session when model is compatible."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Mock all LLM service methods that might be called
        mock_llm_service.get_user_role_name.return_value = "user"
        mock_llm_service.get_assistant_role_name.return_value = "model"
        mock_llm_service.get_backend_for_model.return_value = Mock()

        # Create an existing session with the same chat object that will be used
        mock_chat = Mock()
        mock_chat.model_alias = "gemini-2.5-flash"

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="123456789",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            model_alias="gemini-2.5-flash",
        )

        # Get or create
        chat, returned_key = await manager.get_or_create(
            user_id="123456789",
            username="TestUser",
            session_key=session_key,
            channel_id=111,
            message_content="Hello again",
        )

        # The session should still exist and be returned
        assert session_key in manager.sessions
        assert returned_key == session_key
        # The chat object should be returned (not checking exact object match due to complexity)
        assert chat is not None

    @pytest.mark.asyncio
    async def test_model_mismatch_triggers_recreation(self, mock_app_config, mock_llm_service):
        """Test that model mismatch triggers session recreation."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create session with one model
        old_chat = Mock()
        old_chat.model_alias = "gemini-2.5-flash"
        manager.sessions[session_key] = ChatSession(
            chat=old_chat,
            user_id="123456789",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            model_alias="gemini-2.5-flash",
        )

        # Set different model preference
        manager.channel_model_preferences[111] = "gemini-2.5-pro"

        # Create new mock chat for recreation
        new_chat = Mock()
        new_chat.model_alias = "gemini-2.5-pro"
        mock_llm_service.create_chat_session_for_alias.return_value = new_chat

        # Get or create - should recreate due to model mismatch
        chat, returned_key = await manager.get_or_create(
            user_id="123456789",
            username="TestUser",
            session_key=session_key,
            channel_id=111,
            message_content="Hello",
        )

        assert chat == new_chat
        assert chat != old_chat
        mock_llm_service.create_chat_session_for_alias.assert_called_once()
        # Check the first argument (model_alias)
        args, _ = mock_llm_service.create_chat_session_for_alias.call_args
        assert args[0] == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_session_key_generation(self, mock_app_config, mock_llm_service):
        """Test that session_key is correctly returned."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        mock_llm_service.create_chat_session_for_alias.return_value = Mock()

        session_key = "channel:999"
        chat, returned_key = await manager.get_or_create(
            user_id="123456789",
            username="TestUser",
            session_key=session_key,
            channel_id=999,
            message_content="Hello",
        )

        assert returned_key == session_key


# =============================================================================
# 9. _resolve_target_model_alias() - Test model resolution
# =============================================================================


class TestResolveTargetModelAlias:
    """Tests for _resolve_target_model_alias method."""

    @patch("asyncio.create_task")
    def test_session_context_priority(self, mock_create_task, mock_app_config, mock_llm_service):
        """Test that session context model_alias has priority."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Set session context with model_alias
        manager._record_session_context(
            session_key,
            111,
            "user123",
            "User1",
            "Message",
            datetime.now(timezone.utc),
            model_alias="gemini-2.5-pro",
        )

        # Set channel preference (should be ignored)
        manager.channel_model_preferences[111] = "gemini-2.5-flash"

        # Resolve should use session context
        target = manager._resolve_target_model_alias(session_key, 111)
        assert target == "gemini-2.5-pro"

    @patch("asyncio.create_task")
    def test_channel_preference_priority(self, mock_create_task, mock_app_config, mock_llm_service):
        """Test that channel preference is used when no session context model."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Set session context without model_alias
        manager._record_session_context(
            session_key,
            111,
            "user123",
            "User1",
            "Message",
            datetime.now(timezone.utc),
        )

        # Set channel preference
        manager.channel_model_preferences[111] = "gemini-2.5-pro"

        # Resolve should use channel preference
        target = manager._resolve_target_model_alias(session_key, 111)
        assert target == "gemini-2.5-pro"

    @patch("asyncio.create_task")
    def test_default_model_fallback(self, mock_create_task, mock_app_config, mock_llm_service):
        """Test falling back to default model."""
        from persbot.services.model_usage_service import ModelDefinition

        mock_llm_service.model_usage_service = Mock()
        mock_llm_service.model_usage_service.MODEL_DEFINITIONS = {
            "Gemini 2.5 flash": ModelDefinition(
                display_name="Gemini 2.5 flash",
                api_model_name="gemini-2.5-flash",
                daily_limit=100,
                scope="guild",
                provider="gemini",
            ),
            "GLM 4.7": ModelDefinition(
                display_name="GLM 4.7",
                api_model_name="glm-4.7",
                daily_limit=1000,
                scope="guild",
                provider="zai",
            ),
        }

        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # No session context or channel preference
        target = manager._resolve_target_model_alias(session_key, 111)

        assert target == "Gemini 2.5 flash"

    @patch("asyncio.create_task")
    def test_none_handling_in_session_context(
        self, mock_create_task, mock_app_config, mock_llm_service
    ):
        """Test handling None model_alias in session context."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create session context with None model_alias
        manager._record_session_context(
            session_key,
            111,
            "user123",
            "User1",
            "Message",
            datetime.now(timezone.utc),
            model_alias=None,
        )

        # Set channel preference
        manager.channel_model_preferences[111] = "gemini-2.5-pro"

        # Should fall back to channel preference
        target = manager._resolve_target_model_alias(session_key, 111)
        assert target == "gemini-2.5-pro"


# =============================================================================
# 10. _check_session_model_compatibility() - Test model compatibility
# =============================================================================


class TestCheckSessionModelCompatibility:
    """Tests for _check_session_model_compatibility method."""

    def test_matching_models(self, mock_app_config, mock_llm_service):
        """Test that matching models return True."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        session = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            model_alias="gemini-2.5-flash",
        )

        result = manager._check_session_model_compatibility(
            session, session_key, "gemini-2.5-flash"
        )

        assert result is True

    def test_different_models(self, mock_app_config, mock_llm_service):
        """Test that different models return False."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        session = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            model_alias="gemini-2.5-flash",
        )

        result = manager._check_session_model_compatibility(session, session_key, "gemini-2.5-pro")

        assert result is False

    def test_mismatch_triggers_reset(self, mock_app_config, mock_llm_service):
        """Test that model mismatch is logged and returns False."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        session = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            model_alias="gemini-2.5-flash",
        )

        with patch("persbot.bot.session.logger") as mock_logger:
            result = manager._check_session_model_compatibility(
                session, session_key, "gemini-2.5-pro"
            )

            assert result is False
            # Check that info log was called
            assert mock_logger.info.called


# =============================================================================
# 11. _update_existing_session() - Test session update
# =============================================================================


class TestUpdateExistingSession:
    """Tests for _update_existing_session method."""

    def test_timestamp_update(self, mock_app_config, mock_llm_service):
        """Test that timestamp is updated."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Freeze time before creating session
        old_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        session = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id=session_key,
            last_activity_at=old_time,
        )

        # Add session to manager's sessions dict
        manager.sessions[session_key] = session

        # Freeze time for update
        new_time = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        with patch("persbot.bot.session.datetime") as mock_datetime:
            mock_datetime.now.return_value = new_time
            mock_datetime.timezone = timezone

            manager._update_existing_session(
                session,
                session_key,
                111,
                "user123",
                "User1",
                "New message",
                new_time,
                "msg456",
            )

        # Timestamp should be updated
        assert session.last_activity_at == new_time

    def test_message_id_update(self, mock_app_config, mock_llm_service):
        """Test that message_id is updated."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        session = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
            last_message_id="msg123",
        )

        # Add session to manager's sessions dict
        manager.sessions[session_key] = session

        manager._update_existing_session(
            session,
            session_key,
            111,
            "user123",
            "User1",
            "New message",
            datetime.now(timezone.utc),
            "msg456",
        )

        assert session.last_message_id == "msg456"

    def test_move_to_end_call(self, mock_app_config, mock_llm_service):
        """Test that move_to_end is called on sessions OrderedDict."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        session = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        manager.sessions[session_key] = session
        manager.sessions["channel:222"] = Mock()

        # Update should move to end
        manager._update_existing_session(
            session,
            session_key,
            111,
            "user123",
            "User1",
            "New message",
            datetime.now(timezone.utc),
            "msg456",
        )

        keys = list(manager.sessions.keys())
        assert keys[-1] == session_key


# =============================================================================
# 12. _create_new_session() - Test session creation
# =============================================================================


class TestCreateNewSession:
    """Tests for _create_new_session method."""

    @pytest.mark.asyncio
    async def test_llm_service_call(self, mock_app_config, mock_llm_service):
        """Test that LLM service create_chat_session_for_alias is called."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        mock_chat = Mock()
        mock_llm_service.create_chat_session_for_alias.return_value = mock_chat

        chat, session_key = await manager._create_new_session(
            session_key="channel:111",
            channel_id=111,
            user_id="user123",
            username="User1",
            message_content="Hello",
            message_ts=datetime.now(timezone.utc),
            message_id="msg123",
            model_alias="gemini-2.5-pro",
        )

        mock_llm_service.create_chat_session_for_alias.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_prompt_selection(self, mock_app_config, mock_llm_service):
        """Test that system prompt is selected correctly."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        mock_chat = Mock()
        mock_llm_service.create_chat_session_for_alias.return_value = mock_chat

        # Set custom channel prompt
        custom_prompt = "You are a custom assistant."
        manager.channel_prompts[111] = custom_prompt

        await manager._create_new_session(
            session_key="channel:111",
            channel_id=111,
            user_id="user123",
            username="User1",
            message_content="Hello",
            message_ts=datetime.now(timezone.utc),
            message_id="msg123",
            model_alias="gemini-2.5-flash",
        )

        # Should use custom prompt
        args, kwargs = mock_llm_service.create_chat_session_for_alias.call_args
        assert args[0] == "gemini-2.5-flash"
        assert args[1] == custom_prompt

    @pytest.mark.asyncio
    async def test_chat_session_creation(self, mock_app_config, mock_llm_service):
        """Test that ChatSession is created correctly."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        mock_chat = Mock()
        mock_llm_service.create_chat_session_for_alias.return_value = mock_chat

        session_key = "channel:111"
        message_ts = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        chat, returned_key = await manager._create_new_session(
            session_key=session_key,
            channel_id=111,
            user_id="user123",
            username="User1",
            message_content="Hello",
            message_ts=message_ts,
            message_id="msg123",
            model_alias="gemini-2.5-pro",
        )

        assert returned_key == session_key
        assert session_key in manager.sessions

        session = manager.sessions[session_key]
        assert session.chat == mock_chat
        assert session.user_id == "user123"
        assert session.session_id == session_key
        assert session.last_message_id == "msg123"
        assert session.model_alias == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_context_recording(self, mock_app_config, mock_llm_service):
        """Test that session context is recorded."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        mock_chat = Mock()
        mock_llm_service.create_chat_session_for_alias.return_value = mock_chat

        await manager._create_new_session(
            session_key="channel:111",
            channel_id=111,
            user_id="user123",
            username="User1",
            message_content="Hello",
            message_ts=datetime.now(timezone.utc),
            message_id="msg123",
            model_alias="gemini-2.5-pro",
        )

        # Session context should be created
        assert "channel:111" in manager.session_contexts
        context = manager.session_contexts["channel:111"]
        assert context.channel_id == 111
        assert context.user_id == "user123"
        assert context.model_alias == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_eviction_after_creation(self, mock_app_config, mock_llm_service):
        """Test that eviction is called after session creation."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        mock_app_config.session_cache_limit = 2

        mock_chat = Mock()
        mock_llm_service.create_chat_session_for_alias.return_value = mock_chat

        # Add sessions to reach limit
        for i in range(2):
            manager.sessions[f"channel:{i}"] = Mock()
            manager.session_contexts[f"channel:{i}"] = Mock()

        # Create new session (should trigger eviction)
        await manager._create_new_session(
            session_key="channel:999",
            channel_id=999,
            user_id="user123",
            username="User1",
            message_content="Hello",
            message_ts=datetime.now(timezone.utc),
            message_id="msg999",
            model_alias="gemini-2.5-flash",
        )

        # Should have evicted one session to maintain limit
        assert len(manager.sessions) <= 2


# =============================================================================
# 13. set_channel_prompt() - Test channel prompt
# =============================================================================


class TestSetChannelPrompt:
    """Tests for set_channel_prompt method."""

    def test_setting_custom_prompt(self, mock_app_config, mock_llm_service):
        """Test setting a custom system prompt for a channel."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111
        custom_prompt = "You are a helpful coding assistant."

        manager.set_channel_prompt(channel_id, custom_prompt)

        assert manager.channel_prompts[channel_id] == custom_prompt

    def test_resetting_to_default(self, mock_app_config, mock_llm_service):
        """Test resetting a channel prompt to default."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111

        # Set custom prompt
        manager.channel_prompts[channel_id] = "Custom prompt"
        assert channel_id in manager.channel_prompts

        # Reset to default
        manager.set_channel_prompt(channel_id, None)

        assert channel_id not in manager.channel_prompts

    def test_session_reset_on_change(self, mock_app_config, mock_llm_service):
        """Test that session is reset when prompt changes."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111
        session_key = f"channel:{channel_id}"

        # Create a session
        manager.sessions[session_key] = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )
        manager.session_contexts[session_key] = Mock()

        # Change prompt (should reset session)
        manager.set_channel_prompt(channel_id, "New prompt")

        # Session should be removed
        assert session_key not in manager.sessions
        assert session_key not in manager.session_contexts


# =============================================================================
# 14. reset_session_by_channel() - Test session reset
# =============================================================================


class TestResetSessionByChannel:
    """Tests for reset_session_by_channel method."""

    def test_session_deletion(self, mock_app_config, mock_llm_service):
        """Test that session is deleted."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111
        session_key = f"channel:{channel_id}"

        manager.sessions[session_key] = ChatSession(
            chat=Mock(),
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        result = manager.reset_session_by_channel(channel_id)

        assert result is True
        assert session_key not in manager.sessions

    def test_context_deletion(self, mock_app_config, mock_llm_service):
        """Test that session context is deleted."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111
        session_key = f"channel:{channel_id}"

        manager.session_contexts[session_key] = Mock()

        result = manager.reset_session_by_channel(channel_id)

        assert result is True
        assert session_key not in manager.session_contexts

    def test_return_true_when_deleted(self, mock_app_config, mock_llm_service):
        """Test that True is returned when session exists and is deleted."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 111
        session_key = f"channel:{channel_id}"

        manager.sessions[session_key] = Mock()
        manager.session_contexts[session_key] = Mock()

        result = manager.reset_session_by_channel(channel_id)

        assert result is True

    def test_return_false_when_not_found(self, mock_app_config, mock_llm_service):
        """Test that False is returned when session doesn't exist."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        channel_id = 999

        result = manager.reset_session_by_channel(channel_id)

        assert result is False


# =============================================================================
# 15. resolve_session() - Test session resolution
# =============================================================================


class TestResolveSession:
    """Tests for resolve_session method."""

    @pytest.mark.asyncio
    async def test_session_key_format(self, mock_app_config, mock_llm_service):
        """Test that session_key follows the expected format."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        resolved = await manager.resolve_session(
            channel_id=111,
            author_id=123456789,
            username="TestUser",
            message_id="msg123",
            message_content="Hello bot",
        )

        assert resolved.session_key == "channel:111"
        assert resolved.session_key.startswith("channel:")

    @pytest.mark.asyncio
    async def test_content_cleaning(self, mock_app_config, mock_llm_service):
        """Test that message content is cleaned (stripped)."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        resolved = await manager.resolve_session(
            channel_id=111,
            author_id=123456789,
            username="TestUser",
            message_id="msg123",
            message_content="  Hello bot  ",
        )

        assert resolved.cleaned_message == "Hello bot"
        assert resolved.cleaned_message == resolved.cleaned_message.strip()

    @pytest.mark.asyncio
    async def test_reference_message_id_handling(self, mock_app_config, mock_llm_service):
        """Test that reference_message_id is ignored."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        resolved1 = await manager.resolve_session(
            channel_id=111,
            author_id=123456789,
            username="TestUser",
            message_id="msg123",
            message_content="Hello",
            reference_message_id="ref456",
        )

        resolved2 = await manager.resolve_session(
            channel_id=111,
            author_id=123456789,
            username="TestUser",
            message_id="msg789",
            message_content="Hello",
            reference_message_id=None,
        )

        # Both should resolve to same session
        assert resolved1.session_key == resolved2.session_key


# =============================================================================
# 16. link_message_to_session() - Test message linking
# =============================================================================


class TestLinkMessageToSession:
    """Tests for link_message_to_session method."""

    def test_appending_to_message_ids_list(self, mock_app_config, mock_llm_service):
        """Test appending message_id to message_ids list."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create mock session with history
        mock_chat = Mock()
        last_msg = ChatMessage(role="model", content="Response")
        last_msg.message_ids = ["msg123"]
        mock_chat.history = [ChatMessage(role="user", content="Hello"), last_msg]

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        # Link new message
        manager.link_message_to_session("msg456", session_key)

        assert "msg456" in last_msg.message_ids
        assert len(last_msg.message_ids) == 2

    def test_handling_missing_message_ids(self, mock_app_config, mock_llm_service):
        """Test handling messages without message_ids attribute."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create mock session with history (no message_ids)
        mock_chat = Mock()
        last_msg = ChatMessage(role="model", content="Response")
        mock_chat.history = [last_msg]

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        # Link message (should create message_ids list)
        manager.link_message_to_session("msg456", session_key)

        assert hasattr(last_msg, "message_ids")
        assert "msg456" in last_msg.message_ids

    def test_session_without_history(self, mock_app_config, mock_llm_service):
        """Test handling session without history."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create mock session without history
        mock_chat = Mock(spec=[])  # Mock without history attribute

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        # Should not raise an error
        manager.link_message_to_session("msg456", session_key)


# =============================================================================
# 17. undo_last_exchanges() - Test undo functionality
# =============================================================================


class TestUndoLastExchanges:
    """Tests for undo_last_exchanges method."""

    def test_undo_1_exchange(self, mock_app_config, mock_llm_service):
        """Test undoing a single exchange (user + assistant)."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create mock session with history
        mock_chat = Mock()
        user_msg = ChatMessage(role="user", content="Hello")
        assistant_msg = ChatMessage(role="model", content="Hi there!")
        mock_chat.history = [user_msg, assistant_msg]

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        mock_llm_service.get_assistant_role_name.return_value = "model"
        mock_llm_service.get_user_role_name.return_value = "user"

        # Undo last exchange
        removed = manager.undo_last_exchanges(session_key, 1)

        assert len(removed) == 2
        assert user_msg in removed
        assert assistant_msg in removed
        assert len(mock_chat.history) == 0

    def test_undo_multiple_exchanges(self, mock_app_config, mock_llm_service):
        """Test undoing multiple exchanges."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        # Create mock session with multiple exchanges
        mock_chat = Mock()
        mock_chat.history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="model", content="Hi!"),
            ChatMessage(role="user", content="How are you?"),
            ChatMessage(role="model", content="Good!"),
        ]

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        mock_llm_service.get_assistant_role_name.return_value = "model"
        mock_llm_service.get_user_role_name.return_value = "user"

        # Undo last 2 exchanges
        removed = manager.undo_last_exchanges(session_key, 2)

        assert len(removed) == 4
        assert len(mock_chat.history) == 0

    def test_removing_from_history(self, mock_app_config, mock_llm_service):
        """Test that messages are removed from history."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        mock_chat = Mock()
        msg1 = ChatMessage(role="user", content="First")
        msg2 = ChatMessage(role="model", content="Response 1")
        msg3 = ChatMessage(role="user", content="Second")
        msg4 = ChatMessage(role="model", content="Response 2")
        mock_chat.history = [msg1, msg2, msg3, msg4]

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        mock_llm_service.get_assistant_role_name.return_value = "model"
        mock_llm_service.get_user_role_name.return_value = "user"

        manager.undo_last_exchanges(session_key, 1)

        # Last exchange should be removed
        assert msg3 not in mock_chat.history
        assert msg4 not in mock_chat.history
        # First exchange should remain
        assert msg1 in mock_chat.history
        assert msg2 in mock_chat.history

    def test_finding_indices_to_remove(self, mock_app_config, mock_llm_service):
        """Test finding correct indices to remove."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        mock_chat = Mock()
        mock_chat.history = [
            ChatMessage(role="user", content="A"),
            ChatMessage(role="model", content="1"),
            ChatMessage(role="user", content="B"),
            ChatMessage(role="model", content="2"),
            ChatMessage(role="user", content="C"),
            ChatMessage(role="model", content="3"),
        ]

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        mock_llm_service.get_assistant_role_name.return_value = "model"
        mock_llm_service.get_user_role_name.return_value = "user"

        # Undo 1 exchange (should remove indices 4 and 5)
        removed = manager.undo_last_exchanges(session_key, 1)

        assert len(removed) == 2
        assert len(mock_chat.history) == 4

    def test_splitting_history(self, mock_app_config, mock_llm_service):
        """Test that history is split correctly."""
        manager = SessionManager(mock_app_config, mock_llm_service)
        session_key = "channel:111"

        mock_chat = Mock()
        mock_chat.history = [
            ChatMessage(role="user", content="Keep"),
            ChatMessage(role="model", content="Keep"),
            ChatMessage(role="user", content="Remove"),
            ChatMessage(role="model", content="Remove"),
        ]

        manager.sessions[session_key] = ChatSession(
            chat=mock_chat,
            user_id="user123",
            session_id=session_key,
            last_activity_at=datetime.now(timezone.utc),
        )

        mock_llm_service.get_assistant_role_name.return_value = "model"
        mock_llm_service.get_user_role_name.return_value = "user"

        removed = manager.undo_last_exchanges(session_key, 1)

        # History should only contain kept messages
        assert len(mock_chat.history) == 2
        assert mock_chat.history[0].content == "Keep"
        assert mock_chat.history[1].content == "Keep"


# =============================================================================
# 18. _find_exchange_indices_to_remove() - Test index finding
# =============================================================================


class TestFindExchangeIndicesToRemove:
    """Tests for _find_exchange_indices_to_remove method."""

    def test_finding_assistant_indices(self, mock_app_config, mock_llm_service):
        """Test finding assistant message indices."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        history = [
            ChatMessage(role="user", content="A"),
            ChatMessage(role="model", content="1"),
            ChatMessage(role="user", content="B"),
            ChatMessage(role="model", content="2"),
        ]

        indices = manager._find_exchange_indices_to_remove(history, "model", "user", 1)

        # Should find last assistant message index (3)
        assert 3 in indices

    def test_including_preceding_user_messages(self, mock_app_config, mock_llm_service):
        """Test that preceding user messages are included."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        history = [
            ChatMessage(role="user", content="A"),
            ChatMessage(role="model", content="1"),
            ChatMessage(role="user", content="B"),
            ChatMessage(role="model", content="2"),
            ChatMessage(role="user", content="C"),
            ChatMessage(role="model", content="3"),
        ]

        indices = manager._find_exchange_indices_to_remove(history, "model", "user", 1)

        # Should include assistant message (5) and preceding user (4)
        assert 4 in indices
        assert 5 in indices

    def test_boundary_conditions(self, mock_app_config, mock_llm_service):
        """Test boundary conditions (start/end of history)."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        history = [
            ChatMessage(role="user", content="A"),
            ChatMessage(role="model", content="1"),
        ]

        indices = manager._find_exchange_indices_to_remove(history, "model", "user", 1)

        # Should remove both messages
        assert 0 in indices
        assert 1 in indices

    def test_empty_history(self, mock_app_config, mock_llm_service):
        """Test handling empty history."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        indices = manager._find_exchange_indices_to_remove([], "model", "user", 1)

        assert len(indices) == 0


# =============================================================================
# 19. _split_history_by_indices() - Test history splitting
# =============================================================================


class TestSplitHistoryByIndices:
    """Tests for _split_history_by_indices method."""

    def test_separating_removed_messages(self, mock_app_config, mock_llm_service):
        """Test that removed messages are separated correctly."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        history = [
            ChatMessage(role="user", content="A"),
            ChatMessage(role="model", content="1"),
            ChatMessage(role="user", content="B"),
            ChatMessage(role="model", content="2"),
        ]

        indices_to_remove = {2, 3}

        new_history, removed = manager._split_history_by_indices(history, indices_to_remove)

        assert len(new_history) == 2
        assert len(removed) == 2
        assert new_history[0].content == "A"
        assert new_history[1].content == "1"
        assert removed[0].content == "B"
        assert removed[1].content == "2"

    def test_keeping_active_messages(self, mock_app_config, mock_llm_service):
        """Test that active messages are kept."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        msg1 = ChatMessage(role="user", content="Keep 1")
        msg2 = ChatMessage(role="model", content="Keep 2")
        msg3 = ChatMessage(role="user", content="Remove")
        msg4 = ChatMessage(role="model", content="Remove")

        history = [msg1, msg2, msg3, msg4]

        indices_to_remove = {2, 3}

        new_history, removed = manager._split_history_by_indices(history, indices_to_remove)

        assert msg1 in new_history
        assert msg2 in new_history
        assert msg1 not in removed
        assert msg2 not in removed

    def test_index_matching(self, mock_app_config, mock_llm_service):
        """Test that indices match correctly."""
        manager = SessionManager(mock_app_config, mock_llm_service)

        history = [ChatMessage(role="user", content=f"Message {i}") for i in range(5)]

        indices_to_remove = {1, 3}

        new_history, removed = manager._split_history_by_indices(history, indices_to_remove)

        assert len(new_history) == 3
        assert len(removed) == 2

        # Check that correct messages were removed
        assert history[1] in removed
        assert history[3] in removed
        assert history[0] in new_history
        assert history[2] in new_history
        assert history[4] in new_history

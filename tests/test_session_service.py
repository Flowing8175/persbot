"""Feature tests for SessionService.

Tests focus on behavior:
- Session creation and retrieval
- Stale session detection
- Channel prompts and models
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch

from persbot.services.session_service import (
    SessionService,
    ChatSessionData,
    SessionResolution,
)


class TestChatSessionData:
    """Tests for ChatSessionData dataclass."""

    def test_creates_with_required_fields(self):
        """ChatSessionData can be created with required fields."""
        session = ChatSessionData(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_object=Mock()
        )
        assert session.session_key == "channel:123"
        assert session.user_id == 1
        assert session.channel_id == 123

    def test_default_message_count_is_zero(self):
        """message_count defaults to 0."""
        session = ChatSessionData(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_object=Mock()
        )
        assert session.message_count == 0

    def test_default_history_is_empty_list(self):
        """history defaults to empty list."""
        session = ChatSessionData(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_object=Mock()
        )
        assert session.history == []

    def test_default_model_alias_is_none(self):
        """model_alias defaults to None."""
        session = ChatSessionData(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_object=Mock()
        )
        assert session.model_alias is None

    def test_is_stale_returns_false_for_new_session(self):
        """is_stale returns False for newly created session."""
        session = ChatSessionData(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_object=Mock()
        )
        assert session.is_stale is False

    def test_is_stale_returns_true_for_old_session(self):
        """is_stale returns True for old session."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        session = ChatSessionData(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_object=Mock(),
            last_activity=old_time
        )
        assert session.is_stale is True

    def test_age_seconds_returns_positive_value(self):
        """age_seconds returns a positive value."""
        session = ChatSessionData(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_object=Mock()
        )
        assert session.age_seconds >= 0

    def test_age_seconds_increases_over_time(self):
        """age_seconds reflects actual age."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        session = ChatSessionData(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_object=Mock(),
            created_at=old_time
        )
        assert session.age_seconds >= 10


class TestSessionResolution:
    """Tests for SessionResolution dataclass."""

    def test_creates_with_required_fields(self):
        """SessionResolution can be created with required fields."""
        resolution = SessionResolution(
            session_key="channel:123",
            cleaned_message="Hello"
        )
        assert resolution.session_key == "channel:123"
        assert resolution.cleaned_message == "Hello"

    def test_default_is_new_is_false(self):
        """is_new defaults to False."""
        resolution = SessionResolution(
            session_key="channel:123",
            cleaned_message="Hello"
        )
        assert resolution.is_new is False

    def test_default_is_reply_to_summary_is_false(self):
        """is_reply_to_summary defaults to False."""
        resolution = SessionResolution(
            session_key="channel:123",
            cleaned_message="Hello"
        )
        assert resolution.is_reply_to_summary is False

    def test_default_model_alias_is_none(self):
        """model_alias defaults to None."""
        resolution = SessionResolution(
            session_key="channel:123",
            cleaned_message="Hello"
        )
        assert resolution.model_alias is None

    def test_default_context_is_empty_dict(self):
        """context defaults to empty dict."""
        resolution = SessionResolution(
            session_key="channel:123",
            cleaned_message="Hello"
        )
        assert resolution.context == {}


class TestSessionService:
    """Tests for SessionService class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with cleanup disabled."""
        config = Mock()
        config.session_inactive_minutes = 0  # Disable cleanup task
        config.session_cache_limit = 200
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        return Mock()

    @pytest.fixture
    def mock_model_usage_service(self):
        """Create a mock model usage service."""
        service = Mock()
        service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 Flash"
        return service

    @pytest.fixture
    def service(self, mock_config, mock_llm_service, mock_model_usage_service):
        """Create a SessionService with cleanup disabled."""
        return SessionService(
            config=mock_config,
            llm_service=mock_llm_service,
            model_usage_service=mock_model_usage_service
        )

    def test_creates_service(self, mock_config, mock_llm_service, mock_model_usage_service):
        """SessionService can be created."""
        service = SessionService(
            config=mock_config,
            llm_service=mock_llm_service,
            model_usage_service=mock_model_usage_service
        )
        assert service.config == mock_config

    def test_get_session_returns_none_for_nonexistent(self, service):
        """get_session returns None for non-existent session."""
        result = service.get_session("nonexistent")
        assert result is None

    def test_get_or_create_session_creates_new(self, service):
        """get_or_create_session creates new session when not found."""
        chat_object = Mock()
        chat_factory = lambda: chat_object

        result, is_new = service.get_or_create_session(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_factory=chat_factory
        )

        assert is_new is True
        assert result == chat_object

    def test_get_or_create_session_returns_existing(self, service):
        """get_or_create_session returns existing session when not stale."""
        chat_object = Mock()
        chat_factory = lambda: chat_object

        # Create session
        service.get_or_create_session(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_factory=chat_factory
        )

        # Get existing session
        new_chat = Mock()
        new_factory = lambda: new_chat

        result, is_new = service.get_or_create_session(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_factory=new_factory
        )

        assert is_new is False
        assert result == chat_object  # Original, not new

    def test_remove_session_returns_true_when_exists(self, service):
        """remove_session returns True when session exists."""
        chat_object = Mock()
        chat_factory = lambda: chat_object

        service.get_or_create_session(
            session_key="channel:123",
            user_id=1,
            channel_id=123,
            chat_factory=chat_factory
        )

        result = service.remove_session("channel:123")
        assert result is True

    def test_remove_session_returns_false_when_not_found(self, service):
        """remove_session returns False when session not found."""
        result = service.remove_session("nonexistent")
        assert result is False

    def test_cleanup_stale_sessions_removes_old(self, service):
        """cleanup_stale_sessions removes stale sessions."""
        # Create a stale session manually
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        stale_session = ChatSessionData(
            session_key="channel:stale",
            user_id=1,
            channel_id=999,
            chat_object=Mock(),
            last_activity=old_time
        )
        service._sessions["channel:stale"] = stale_session

        removed = service.cleanup_stale_sessions()
        assert removed == 1
        assert "channel:stale" not in service._sessions

    def test_get_channel_prompt_returns_none_when_not_set(self, service):
        """get_channel_prompt returns None when not set."""
        result = service.get_channel_prompt(123)
        assert result is None

    def test_set_and_get_channel_prompt(self, service):
        """set_channel_prompt and get_channel_prompt work together."""
        service.set_channel_prompt(123, "Custom prompt")
        result = service.get_channel_prompt(123)
        assert result == "Custom prompt"

    def test_get_channel_model_returns_none_when_not_set(self, service):
        """get_channel_model returns None when not set."""
        result = service.get_channel_model(123)
        assert result is None

    def test_set_and_get_channel_model(self, service):
        """set_channel_model and get_channel_model work together."""
        service.set_channel_model(123, "GPT-4o")
        result = service.get_channel_model(123)
        assert result == "GPT-4o"

    def test_get_session_count_returns_correct_number(self, service):
        """get_session_count returns correct number of sessions."""
        assert service.get_session_count() == 0

        chat_factory = lambda: Mock()
        service.get_or_create_session("channel:1", 1, 1, chat_factory)
        service.get_or_create_session("channel:2", 2, 2, chat_factory)

        assert service.get_session_count() == 2

    def test_stats_returns_correct_info(self, service):
        """stats property returns correct statistics."""
        stats = service.stats
        assert "total_sessions" in stats
        assert "custom_prompts" in stats
        assert "custom_models" in stats

    def test_get_all_sessions_returns_list(self, service):
        """get_all_sessions returns list of all sessions."""
        chat_factory = lambda: Mock()
        service.get_or_create_session("channel:1", 1, 1, chat_factory)
        service.get_or_create_session("channel:2", 2, 2, chat_factory)

        sessions = service.get_all_sessions()
        assert len(sessions) == 2


class TestSessionServiceWithCleanup:
    """Tests for SessionService with cleanup task enabled."""

    @pytest.fixture
    def mock_config_with_cleanup(self):
        """Create a mock config with cleanup enabled."""
        config = Mock()
        config.session_inactive_minutes = 30  # Enable cleanup task
        config.session_cache_limit = 200
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        return Mock()

    @pytest.fixture
    def mock_model_usage_service(self):
        """Create a mock model usage service."""
        service = Mock()
        service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 Flash"
        return service

    @pytest.mark.asyncio
    async def test_shutdown_cancels_cleanup_task(
        self, mock_config_with_cleanup, mock_llm_service, mock_model_usage_service
    ):
        """shutdown cancels the cleanup task."""
        service = SessionService(
            config=mock_config_with_cleanup,
            llm_service=mock_llm_service,
            model_usage_service=mock_model_usage_service
        )

        # Verify cleanup task was created
        assert service._cleanup_task is not None

        # Shutdown should cancel it
        await service.shutdown()

        assert service._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_clears_sessions(
        self, mock_config_with_cleanup, mock_llm_service, mock_model_usage_service
    ):
        """shutdown clears all sessions."""
        service = SessionService(
            config=mock_config_with_cleanup,
            llm_service=mock_llm_service,
            model_usage_service=mock_model_usage_service
        )

        # Add some sessions
        chat_factory = lambda: Mock()
        service.get_or_create_session("channel:1", 1, 1, chat_factory)
        service.get_or_create_session("channel:2", 2, 2, chat_factory)

        assert service.get_session_count() == 2

        await service.shutdown()

        assert service.get_session_count() == 0

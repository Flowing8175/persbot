"""Feature tests for session module.

Tests focus on behavior:
- ChatSession: represents a short-lived LLM chat
- SessionContext: lightweight metadata for sessions
- ResolvedSession: result of session resolution
- SessionManager: manages chat sessions
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest


# Mock ddgs module before any imports that depend on it
_mock_ddgs = MagicMock()
_mock_ddgs.DDGS = MagicMock
_mock_ddgs.exceptions = MagicMock()
_mock_ddgs.exceptions.RatelimitException = Exception
_mock_ddgs.exceptions.DDGSException = Exception
sys.modules['ddgs'] = _mock_ddgs
sys.modules['ddgs.exceptions'] = _mock_ddgs.exceptions

# Mock bs4 module before any imports that depend on it
_mock_bs4 = MagicMock()
sys.modules['bs4'] = _mock_bs4


class TestChatSession:
    """Tests for ChatSession dataclass."""

    def test_chat_session_exists(self):
        """ChatSession class exists."""
        from persbot.bot.session import ChatSession
        assert ChatSession is not None

    def test_creates_with_required_fields(self):
        """ChatSession creates with required fields."""
        from persbot.bot.session import ChatSession

        chat = MagicMock()
        session = ChatSession(
            chat=chat,
            user_id="user123",
            session_id="session456",
            last_activity_at=datetime.now(timezone.utc),
        )

        assert session.chat == chat
        assert session.user_id == "user123"
        assert session.session_id == "session456"

    def test_default_last_message_id_is_none(self):
        """last_message_id defaults to None."""
        from persbot.bot.session import ChatSession

        session = ChatSession(
            chat=MagicMock(),
            user_id="user123",
            session_id="session456",
            last_activity_at=datetime.now(timezone.utc),
        )

        assert session.last_message_id is None

    def test_default_model_alias_is_none(self):
        """model_alias defaults to None."""
        from persbot.bot.session import ChatSession

        session = ChatSession(
            chat=MagicMock(),
            user_id="user123",
            session_id="session456",
            last_activity_at=datetime.now(timezone.utc),
        )

        assert session.model_alias is None


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_session_context_exists(self):
        """SessionContext class exists."""
        from persbot.bot.session import SessionContext
        assert SessionContext is not None

    def test_creates_with_required_fields(self):
        """SessionContext creates with required fields."""
        from persbot.bot.session import SessionContext

        now = datetime.now(timezone.utc)
        context = SessionContext(
            session_id="session123",
            channel_id=456,
            user_id="user789",
            username="TestUser",
            started_at=now,
            last_activity_at=now,
        )

        assert context.session_id == "session123"
        assert context.channel_id == 456
        assert context.user_id == "user789"
        assert context.username == "TestUser"

    def test_default_last_message_preview_is_empty(self):
        """last_message_preview defaults to empty string."""
        from persbot.bot.session import SessionContext

        now = datetime.now(timezone.utc)
        context = SessionContext(
            session_id="session123",
            channel_id=456,
            user_id="user789",
            username="TestUser",
            started_at=now,
            last_activity_at=now,
        )

        assert context.last_message_preview == ""

    def test_default_title_is_none(self):
        """title defaults to None."""
        from persbot.bot.session import SessionContext

        now = datetime.now(timezone.utc)
        context = SessionContext(
            session_id="session123",
            channel_id=456,
            user_id="user789",
            username="TestUser",
            started_at=now,
            last_activity_at=now,
        )

        assert context.title is None

    def test_default_model_alias_is_none(self):
        """model_alias defaults to None."""
        from persbot.bot.session import SessionContext

        now = datetime.now(timezone.utc)
        context = SessionContext(
            session_id="session123",
            channel_id=456,
            user_id="user789",
            username="TestUser",
            started_at=now,
            last_activity_at=now,
        )

        assert context.model_alias is None


class TestResolvedSession:
    """Tests for ResolvedSession dataclass."""

    def test_resolved_session_exists(self):
        """ResolvedSession class exists."""
        from persbot.bot.session import ResolvedSession
        assert ResolvedSession is not None

    def test_creates_with_required_fields(self):
        """ResolvedSession creates with required fields."""
        from persbot.bot.session import ResolvedSession

        resolved = ResolvedSession(
            session_key="channel:123",
            cleaned_message="Hello world",
        )

        assert resolved.session_key == "channel:123"
        assert resolved.cleaned_message == "Hello world"

    def test_default_is_reply_to_summary_is_false(self):
        """is_reply_to_summary defaults to False."""
        from persbot.bot.session import ResolvedSession

        resolved = ResolvedSession(
            session_key="channel:123",
            cleaned_message="Hello",
        )

        assert resolved.is_reply_to_summary is False


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with cleanup disabled to avoid async issues."""
        config = Mock()
        config.session_inactive_minutes = 0  # Disable periodic cleanup
        config.session_cache_limit = 100
        config.assistant_model_name = "gemini-2.5-flash"
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        service.model_usage_service = Mock()
        service.model_usage_service.MODEL_DEFINITIONS = {}
        service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-flash"
        service.create_chat_session_for_alias = Mock(return_value=MagicMock())
        service.get_assistant_role_name = Mock(return_value="assistant")
        service.get_user_role_name = Mock(return_value="user")
        return service

    def test_session_manager_exists(self):
        """SessionManager class exists."""
        from persbot.bot.session import SessionManager
        assert SessionManager is not None

    def test_creates_with_config_and_llm_service(self, mock_config, mock_llm_service):
        """SessionManager creates with config and LLM service."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        assert manager.config == mock_config
        assert manager.llm_service == mock_llm_service

    def test_sessions_dict_is_empty_initially(self, mock_config, mock_llm_service):
        """Sessions dict is empty initially."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        assert len(manager.sessions) == 0

    def test_session_contexts_dict_is_empty_initially(self, mock_config, mock_llm_service):
        """Session contexts dict is empty initially."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        assert len(manager.session_contexts) == 0

    def test_channel_prompts_dict_is_empty_initially(self, mock_config, mock_llm_service):
        """Channel prompts dict is empty initially."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        assert len(manager.channel_prompts) == 0

    def test_set_channel_prompt_stores_prompt(self, mock_config, mock_llm_service):
        """set_channel_prompt stores prompt."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        manager.set_channel_prompt(123, "Custom prompt")

        assert manager.channel_prompts[123] == "Custom prompt"

    def test_set_channel_prompt_removes_prompt_when_none(self, mock_config, mock_llm_service):
        """set_channel_prompt removes prompt when None."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        manager.channel_prompts[123] = "Existing prompt"
        manager.set_channel_prompt(123, None)

        assert 123 not in manager.channel_prompts

    def test_reset_session_by_channel_removes_session(self, mock_config, mock_llm_service):
        """reset_session_by_channel removes session."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        manager.sessions["channel:123"] = MagicMock()
        manager.session_contexts["channel:123"] = MagicMock()

        result = manager.reset_session_by_channel(123)

        assert result is True
        assert "channel:123" not in manager.sessions
        assert "channel:123" not in manager.session_contexts

    def test_reset_session_by_channel_returns_false_when_not_found(self, mock_config, mock_llm_service):
        """reset_session_by_channel returns False when session not found."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)

        result = manager.reset_session_by_channel(999)

        assert result is False

    def test_set_session_model_stores_preference(self, mock_config, mock_llm_service):
        """set_session_model stores model preference."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        manager.set_session_model(123, "gpt-4")

        assert manager.channel_model_preferences[123] == "gpt-4"

    @pytest.mark.asyncio
    async def test_resolve_session_returns_resolved_session(self, mock_config, mock_llm_service):
        """resolve_session returns ResolvedSession."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)

        result = await manager.resolve_session(
            channel_id=123,
            author_id=456,
            username="TestUser",
            message_id="789",
            message_content="Hello",
        )

        assert result.session_key == "channel:123"
        assert result.cleaned_message == "Hello"

    def test_link_message_to_session_links_message(self, mock_config, mock_llm_service):
        """link_message_to_session links message to session."""
        from persbot.bot.session import SessionManager, ChatSession

        manager = SessionManager(mock_config, mock_llm_service)

        # Create a mock chat with history
        # Note: Use _history (not history) since the code accesses the internal attribute
        # directly to avoid the copy that the history property returns
        mock_chat = MagicMock()
        mock_msg = MagicMock()
        mock_msg.message_ids = []
        mock_chat._history = [mock_msg]

        manager.sessions["channel:123"] = ChatSession(
            chat=mock_chat,
            user_id="user1",
            session_id="session1",
            last_activity_at=datetime.now(timezone.utc),
        )

        manager.link_message_to_session("msg456", "channel:123")

        assert "msg456" in mock_msg.message_ids

    @pytest.mark.asyncio
    async def test_cleanup_cancels_cleanup_task(self, mock_config, mock_llm_service):
        """cleanup cancels cleanup task."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)

        # Add a session
        manager.sessions["channel:123"] = MagicMock()
        manager.session_contexts["channel:123"] = MagicMock()

        await manager.cleanup()

        assert len(manager.sessions) == 0
        assert len(manager.session_contexts) == 0


class TestSessionManagerUndo:
    """Tests for SessionManager undo functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.session_inactive_minutes = 0  # Disable periodic cleanup
        config.session_cache_limit = 100
        config.assistant_model_name = "gemini-2.5-flash"
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        service.model_usage_service = Mock()
        service.model_usage_service.MODEL_DEFINITIONS = {}
        service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-flash"
        service.create_chat_session_for_alias = Mock(return_value=MagicMock())
        service.get_assistant_role_name = Mock(return_value="assistant")
        service.get_user_role_name = Mock(return_value="user")
        return service

    def test_undo_last_exchanges_returns_empty_when_no_session(self, mock_config, mock_llm_service):
        """undo_last_exchanges returns empty list when no session."""
        from persbot.bot.session import SessionManager

        manager = SessionManager(mock_config, mock_llm_service)
        result = manager.undo_last_exchanges("nonexistent", 1)

        assert result == []

    def test_undo_last_exchanges_returns_empty_when_no_history(self, mock_config, mock_llm_service):
        """undo_last_exchanges returns empty list when no history."""
        from persbot.bot.session import SessionManager, ChatSession

        manager = SessionManager(mock_config, mock_llm_service)

        # Create session without history
        mock_chat = MagicMock()
        del mock_chat.history  # No history attribute

        manager.sessions["channel:123"] = ChatSession(
            chat=mock_chat,
            user_id="user1",
            session_id="session1",
            last_activity_at=datetime.now(timezone.utc),
        )

        result = manager.undo_last_exchanges("channel:123", 1)

        assert result == []

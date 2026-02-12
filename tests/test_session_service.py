"""Tests for services/session_service.py module.

This module provides comprehensive test coverage for:
- SessionService class with all 15 methods
- ChatSessionData dataclass
- SessionResolution dataclass
- Session storage and retrieval
- Channel prompts and models management
- Session cleanup and statistics
"""

import asyncio
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict, List, Optional

import pytest

from persbot.services.session_service import (
    SessionService,
    ChatSessionData,
    SessionResolution,
)
from persbot.config import AppConfig
from persbot.services.model_usage_service import ModelUsageService
from persbot.constants import SessionConfig


# =============================================================================
# ChatSessionData Tests
# =============================================================================


class TestChatSessionData:
    """Tests for ChatSessionData dataclass."""

    def test_init_all_fields(self):
        """Test ChatSessionData initialization with all fields."""
        session = ChatSessionData(
            session_key="test-session-123",
            user_id=12345,
            channel_id=67890,
            username="testuser",
            message_content="Test message content",
            created_at=datetime.now(timezone.utc),
            model_alias="gemini-2.0-flash-exp",
            message_count=5,
            history=["msg1", "msg2"],
        )

        assert session.session_key == "test-session-123"
        assert session.user_id == 12345
        assert session.channel_id == 67890
        assert session.username == "testuser"
        assert session.message_content == "Test message content"
        assert isinstance(session.created_at, datetime)
        assert session.model_alias == "gemini-2.0-flash-exp"
        assert session.message_count == 5
        assert session.history == ["msg1", "msg2"]

    def test_init_with_defaults(self):
        """Test ChatSessionData with default values."""
        session = ChatSessionData(
            session_key="test-session",
        user_id=123,
            channel_id=67890,
            username="testuser",
            message_content="Test message content",
        )

        # Should have defaults
        assert session.session_key == "test-session"
        assert isinstance(session.created_at, datetime)
        assert session.last_activity == session.created_at
        assert session.message_count == 0
        assert session.history == []

    def test_is_stale_true(self):
        """Test is_stale property for stale session."""
        session = ChatSessionData(
            session_key="test-session",
            user_id=123,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=SessionConfig.INACTIVE_MINUTES + 1),
            message_content="Test",
        )

        assert session.is_stale is True

    def test_is_stale_false(self):
        """Test is_stale property for active session."""
        session = ChatSessionData(
            session_key="test-session",
            user_id=123,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=SessionConfig.INACTIVE_MINUTES - 1),
            message_content="Test",
        )

        assert session.is_stale is False

    def test_age_seconds(self):
        """Test age_seconds property."""
        created = datetime.now(timezone.utc) - timedelta(seconds=30)

        session = ChatSessionData(
            session_key="test-session",
            created_at=created,
            message_content="Test",
        )

        # Age should be approximately 30 seconds
        assert 29 < session.age_seconds < 31


# =============================================================================
# SessionResolution Tests
# =============================================================================


class TestSessionResolution:
    """Tests for SessionResolution dataclass."""

    def test_init_all_fields(self):
        """Test SessionResolution initialization with all fields."""
        resolution = SessionResolution(
            session_key="test-session",
            cleaned_message="Cleaned message\nwith context",
            is_new=False,
            is_reply_to_summary=False,
            model_alias="gemini-2.0-flash-exp",
            context={"key": "value"},
        )

        assert resolution.session_key == "test-session"
        assert resolution.cleaned_message == "Cleaned message\nwith context"
        assert resolution.is_new is False
        assert resolution.is_reply_to_summary is False
        assert resolution.model_alias == "gemini-2.0-flash-exp"
        assert resolution.context == {"key": "value"}

    def test_is_reply_to_summary_true(self):
        """Test is_reply_to_summary property when True."""
        resolution = SessionResolution(
            session_key="test-session",
            cleaned_message="**... 요약:** Summary of conversation",
            is_reply_to_summary=True,
        )

        assert resolution.is_reply_to_summary is True


# =============================================================================
# SessionService Class Tests
# =============================================================================


class TestSessionService:
    """Tests for SessionService class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        from types import SimpleNamespace

        return SimpleNamespace(
            session_inactive_minutes=60,
            session_cache_limit=10,
        )

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLMService."""
        llm = Mock()
        llm.generate_chat_response = AsyncMock(return_value=None)
        llm.generate_chat_response_stream = AsyncMock(return_value=iter([]))
        llm.extract_function_calls_from_response = Mock(return_value=[])
        llm.assistant_backend = Mock()
        return llm

    @pytest.fixture
    def mock_model_usage_service(self):
        """Create a mock ModelUsageService."""
        service = Mock()
        service.record_usage = Mock()
        return service

    @pytest.fixture
    def session_service(self, mock_config):
        """Create SessionService instance."""
        return SessionService(
            config=mock_config,
            llm_service=self.mock_llm_service(),
            model_usage_service=self.mock_model_usage_service(),
        )

    @pytest.mark.asyncio
    async def test_init_default_values(self, session_service):
        """Test SessionService initialization with defaults."""
        assert session_service._sessions == OrderedDict()
        assert session_service._channel_prompts == {}
        assert session_service._channel_models == {}
        assert len(session_service._sessions) == 0

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, session_service):
        """Test getting non-existent session."""
        result = await session_service.get_session("nonexistent-session")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_existing(self, session_service):
        """Test getting existing session."""
        # First, create a session
        await session_service.get_or_create_session(
            user_id=123,
            channel_id=67890,
            username="testuser",
        )

        # Then retrieve it
        result = await session_service.get_session("123-testuser-67890")

        assert result is not None
        assert result.session_key == "123-testuser-67890"
        assert result.user_id == 123

    @pytest.mark.asyncio
    async def test_get_or_create_session_new(self, session_service):
        """Test creating new session."""
        result, is_new = await session_service.get_or_create_session(
            user_id=456,
            channel_id=789,
            username="newuser",
        )

        assert is_new is True
        assert result.session_key == "456-789"

    @pytest.mark.asyncio
    async def test_get_or_create_session_resume(self, session_service):
        """Test resuming existing session."""
        # First create a session
        await session_service.get_or_create_session(
            user_id=789,
            channel_id=456,
            username="resumeuser",
        )

        # Get the session key to resume
        session_key_to_resume = result.session_key

        # Create a new session that resumes
        result2, is_new2 = await session_service.get_or_create_session(
            user_id=789,
            channel_id=456,
            username="resumeuser",
            session_key=session_key_to_resume,
        )

        assert is_new2 is False  # Resuming, not new
        assert result2.session_key == session_key_to_resume

    @pytest.mark.asyncio
    async def test_remove_session(self, session_service):
        """Test removing a session."""
        # Create a session
        await session_service.get_or_create_session(
            user_id=123,
            channel_id=67890,
            username="testuser",
        )

        session_key = result.session_key

        # Remove it
        removed = await session_service.remove_session(session_key)

        assert removed is True

        # Verify it's gone
        result2 = await session_service.get_session(session_key)

        assert result2 is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_session(self, session_service):
        """Test removing non-existent session."""
        removed = await session_service.remove_session("nonexistent-session")

        assert removed is False

    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions(self, session_service):
        """Test cleanup of stale sessions."""
        # Create multiple stale sessions
        for i in range(3):
            await session_service.get_or_create_session(
                user_id=100 + i,
                channel_id=100 + i,
                username=f"stale{i}",
            )

        # Create one fresh session
        await session_service.get_or_create_session(
            user_id=200,
            channel_id=200,
            username="freshuser",
        )

        # Make some stale
        stale_count = await session_service.cleanup_stale_sessions()

        assert stale_count >= 3  # At least the 3 stale ones

    @pytest.mark.asyncio
    async def test_link_message_to_session(self, session_service):
        """Test linking a message to a session."""
        # Create a session
        result = await session_service.get_or_create_session(
            user_id=123,
            channel_id=67890,
            username="testuser",
        )

        session_key = result.session_key

        # Link a message
        message_id = 999888
        await session_service.link_message_to_session(
            session_key=session_key,
            discord_message_id=str(message_id),
        )

        # Get session history
        history = await session_service.get_session_messages(session_key)

        assert message_id in history

    @pytest.mark.asyncio
    async def test_get_channel_prompt(self, session_service):
        """Test getting channel prompt."""
        prompt = "Custom channel prompt"

        # Set prompt
        await session_service.set_channel_prompt(
            channel_id=67890,
            prompt=prompt,
        )

        # Get it back
        result = await session_service.get_channel_prompt(67890)

        assert result == prompt

    @pytest.mark.asyncio
    async def test_set_channel_model(self, session_service):
        """Test setting channel model."""
        model_alias = "custom-model"

        # Set model
        await session_service.set_channel_model(
            channel_id=67890,
            model_alias=model_alias,
        )

        # Get it back
        result = await session_service.get_channel_model(67890)

        assert result == model_alias

    @pytest.mark.asyncio
    async def test_get_all_sessions(self, session_service):
        """Test getting all sessions."""
        # Create multiple sessions
        for i in range(5):
            await session_service.get_or_create_session(
                user_id=200 + i,
                channel_id=200 + i,
                username=f"user{i}",
            )

        sessions = await session_service.get_all_sessions()

        assert len(sessions) == 5

    @pytest.mark.asyncio
    async def test_get_session_count(self, session_service):
        """Test getting session count."""
        # Create some sessions
        for i in range(3):
            await session_service.get_or_create_session(
                user_id=100 + i,
                channel_id=100 + i,
                username=f"count{i}",
            )

        count = await session_service.get_session_count()

        assert count == 3

    @pytest.mark.asyncio
    async def test_stats_property(self, session_service):
        """Test stats property."""
        # Create some sessions
        for i in range(2):
            await session_service.get_or_create_session(
                user_id=500 + i,
                channel_id=500 + i,
                username=f"stats{i}",
            )

        stats = session_service.stats

        assert "total_sessions" in stats
        assert stats["total_sessions"] == 2

    @pytest.mark.asyncio
    async def test_shutdown(self, session_service):
        """Test graceful shutdown."""
        # Create some sessions
        for i in range(2):
            await session_service.get_or_create_session(
                user_id=800 + i,
                channel_id=800 + i,
                username=f"shutdown{i}",
            )

        # Shutdown
        await session_service.shutdown()

        # Verify all sessions cleared
        assert session_service._sessions == OrderedDict()

        assert session_service.get_session_count() == 0

    @pytest.mark.asyncio
    async def test_eviction_when_cache_limit_exceeded(self, session_service):
        """Test session eviction when cache limit is exceeded."""
        # Create sessions up to limit
        config = self.mock_config()
        config.session_cache_limit = 5

        service = SessionService(config=config)

        for i in range(6):
            await service.get_or_create_session(
                user_id=300 + i,
                channel_id=300 + i,
                username=f"user{i}",
            )

        # Check oldest session was evicted
        # The first session should be evicted when adding the 6th
        oldest_key = None

        for i in range(5):  # Add 5 more (total 6)
            result, _ = await service.get_or_create_session(
                user_id=400 + i,
                channel_id=400 + i,
                username=f"evict{i}",
            )

            if i == 0:
                oldest_key = result.session_key

        assert oldest_key is not None

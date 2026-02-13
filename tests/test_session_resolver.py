"""Feature tests for session_resolver module.

Tests cover:
- resolve_session_for_message: reply handling, summary detection, session resolution
- extract_session_context: single message, list of messages
"""

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# Mock discord module before importing anything else
@pytest.fixture(autouse=True)
def mock_discord():
    """Mock discord module to avoid import issues."""
    mock_discord_module = MagicMock()
    mock_discord_module.Message = MagicMock
    mock_discord_module.DeletedReferencedMessage = type('DeletedReferencedMessage', (), {})

    # Create proper exception classes that inherit from BaseException
    class MockNotFound(Exception):
        pass

    class MockHTTPException(Exception):
        pass

    mock_discord_module.NotFound = MockNotFound
    mock_discord_module.HTTPException = MockHTTPException

    sys.modules['discord'] = mock_discord_module
    yield mock_discord_module
    if 'discord' in sys.modules:
        del sys.modules['discord']


# ============================================================================
# Mock Classes
# ============================================================================

class MockMessageReference:
    """Mock discord.MessageReference for testing."""

    def __init__(self, message_id=None, resolved=None):
        self.message_id = message_id
        self.resolved = resolved


class MockAuthor:
    """Mock discord User/Member for testing."""

    def __init__(self, id=123, name="testuser", bot=False):
        self.id = id
        self.name = name
        self.bot = bot


class MockChannel:
    """Mock discord TextChannel for testing."""

    def __init__(self, id=456):
        self.id = id

    async def fetch_message(self, message_id):
        """Fetch message by ID - override in tests."""
        raise NotImplementedError("Override in test")


class MockMessage:
    """Mock discord.Message for testing."""

    def __init__(
        self,
        id=1,
        content="Test message",
        author=None,
        channel=None,
        reference=None,
        created_at=None,
    ):
        self.id = id
        self.clean_content = content
        self.author = author or MockAuthor()
        self.channel = channel or MockChannel()
        self.reference = reference
        self.created_at = created_at or datetime.now(timezone.utc)


class MockResolvedSession:
    """Mock ResolvedSession for testing."""

    def __init__(self, session_key="channel:456", cleaned_message="Test", is_reply_to_summary=False):
        self.session_key = session_key
        self.cleaned_message = cleaned_message
        self.is_reply_to_summary = is_reply_to_summary


class MockSessionManager:
    """Mock SessionManager for testing."""

    def __init__(self):
        self.resolve_session_called = False
        self.last_resolve_args = None
        self._return_value = MockResolvedSession()

    async def resolve_session(
        self,
        *,
        channel_id,
        author_id,
        username,
        message_id,
        message_content,
        reference_message_id,
        created_at,
        cancel_event,
    ):
        self.resolve_session_called = True
        self.last_resolve_args = {
            'channel_id': channel_id,
            'author_id': author_id,
            'username': username,
            'message_id': message_id,
            'message_content': message_content,
            'reference_message_id': reference_message_id,
            'created_at': created_at,
            'cancel_event': cancel_event,
        }
        return self._return_value


# ============================================================================
# resolve_session_for_message Tests
# ============================================================================

class TestResolveSessionForMessage:
    """Tests for resolve_session_for_message function."""

    @pytest.fixture
    def session_manager(self):
        """Create a mock session manager."""
        return MockSessionManager()

    @pytest.fixture
    def mock_message(self):
        """Create a basic mock message without reference."""
        return MockMessage(
            id=1,
            content="Hello bot",
            author=MockAuthor(id=123, name="testuser"),
            channel=MockChannel(id=456),
        )

    @pytest.mark.asyncio
    async def test_resolves_session_for_basic_message(self, session_manager, mock_message):
        """resolve_session_for_message resolves session for basic message."""
        from persbot.bot.session_resolver import resolve_session_for_message

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello bot",
            session_manager=session_manager,
        )

        assert result is not None
        assert session_manager.resolve_session_called

    @pytest.mark.asyncio
    async def test_passes_correct_channel_id(self, session_manager, mock_message):
        """resolve_session_for_message passes correct channel_id."""
        from persbot.bot.session_resolver import resolve_session_for_message

        await resolve_session_for_message(
            message=mock_message,
            content="Hello bot",
            session_manager=session_manager,
        )

        assert session_manager.last_resolve_args['channel_id'] == 456

    @pytest.mark.asyncio
    async def test_passes_correct_author_id(self, session_manager, mock_message):
        """resolve_session_for_message passes correct author_id."""
        from persbot.bot.session_resolver import resolve_session_for_message

        await resolve_session_for_message(
            message=mock_message,
            content="Hello bot",
            session_manager=session_manager,
        )

        assert session_manager.last_resolve_args['author_id'] == 123

    @pytest.mark.asyncio
    async def test_passes_correct_username(self, session_manager, mock_message):
        """resolve_session_for_message passes correct username."""
        from persbot.bot.session_resolver import resolve_session_for_message

        await resolve_session_for_message(
            message=mock_message,
            content="Hello bot",
            session_manager=session_manager,
        )

        assert session_manager.last_resolve_args['username'] == "testuser"

    @pytest.mark.asyncio
    async def test_passes_correct_message_id(self, session_manager, mock_message):
        """resolve_session_for_message passes correct message_id as string."""
        from persbot.bot.session_resolver import resolve_session_for_message

        await resolve_session_for_message(
            message=mock_message,
            content="Hello bot",
            session_manager=session_manager,
        )

        assert session_manager.last_resolve_args['message_id'] == "1"

    @pytest.mark.asyncio
    async def test_passes_correct_created_at(self, session_manager):
        """resolve_session_for_message passes correct created_at."""
        from persbot.bot.session_resolver import resolve_session_for_message

        created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        message = MockMessage(created_at=created_at)

        await resolve_session_for_message(
            message=message,
            content="Hello bot",
            session_manager=session_manager,
        )

        assert session_manager.last_resolve_args['created_at'] == created_at

    @pytest.mark.asyncio
    async def test_returns_none_when_cleaned_message_is_empty(self, session_manager, mock_message):
        """resolve_session_for_message returns None when cleaned_message is empty."""
        from persbot.bot.session_resolver import resolve_session_for_message

        session_manager._return_value = MockResolvedSession(cleaned_message="")

        result = await resolve_session_for_message(
            message=mock_message,
            content="",
            session_manager=session_manager,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_resolution_is_none(self, session_manager, mock_message):
        """resolve_session_for_message returns None when session_manager returns None."""
        from persbot.bot.session_resolver import resolve_session_for_message

        session_manager._return_value = None

        result = await resolve_session_for_message(
            message=mock_message,
            content="Hello bot",
            session_manager=session_manager,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_passes_cancel_event(self, session_manager, mock_message):
        """resolve_session_for_message passes cancel_event."""
        from persbot.bot.session_resolver import resolve_session_for_message

        cancel_event = MagicMock()

        await resolve_session_for_message(
            message=mock_message,
            content="Hello bot",
            session_manager=session_manager,
            cancel_event=cancel_event,
        )

        assert session_manager.last_resolve_args['cancel_event'] is cancel_event


class TestResolveSessionForMessageWithReply:
    """Tests for resolve_session_for_message with reply handling."""

    @pytest.fixture
    def session_manager(self):
        """Create a mock session manager."""
        return MockSessionManager()

    @pytest.fixture
    def referenced_message(self):
        """Create a referenced (replied-to) message."""
        return MockMessage(
            id=999,
            content="Original message",
            author=MockAuthor(id=555, name="original_author"),
        )

    @pytest.fixture
    def message_with_reply(self, referenced_message):
        """Create a message that is a reply."""
        return MockMessage(
            id=1,
            content="Reply content",
            author=MockAuthor(id=123, name="replier"),
            channel=MockChannel(id=456),
            reference=MockMessageReference(
                message_id=999,
                resolved=referenced_message,
            ),
        )

    @pytest.mark.asyncio
    async def test_adds_reply_context_to_content(self, session_manager, message_with_reply):
        """resolve_session_for_message adds reply context to content."""
        from persbot.bot.session_resolver import resolve_session_for_message

        await resolve_session_for_message(
            message=message_with_reply,
            content="Reply content",
            session_manager=session_manager,
        )

        content = session_manager.last_resolve_args['message_content']
        assert "(답장 대상:" in content
        assert "555" in content  # author id
        assert "Reply content" in content

    @pytest.mark.asyncio
    async def test_includes_original_content_in_reply_context(self, session_manager, message_with_reply):
        """resolve_session_for_message includes original message content in context."""
        from persbot.bot.session_resolver import resolve_session_for_message

        await resolve_session_for_message(
            message=message_with_reply,
            content="Reply content",
            session_manager=session_manager,
        )

        content = session_manager.last_resolve_args['message_content']
        assert "Original message" in content

    @pytest.mark.asyncio
    async def test_detects_reply_to_summary_message(self, session_manager):
        """resolve_session_for_message detects reply to summary message."""
        from persbot.bot.session_resolver import resolve_session_for_message

        summary_msg = MockMessage(
            id=999,
            content="**요약:** This is a summary",
            author=MockAuthor(id=555, name="bot", bot=True),
        )
        reply_msg = MockMessage(
            id=1,
            content="Reply to summary",
            reference=MockMessageReference(message_id=999, resolved=summary_msg),
        )

        result = await resolve_session_for_message(
            message=reply_msg,
            content="Reply to summary",
            session_manager=session_manager,
        )

        assert result is not None
        assert result.is_reply_to_summary is True

    @pytest.mark.asyncio
    async def test_does_not_detect_reply_to_regular_bot_message(self, session_manager):
        """resolve_session_for_message does not detect reply to regular bot message as summary."""
        from persbot.bot.session_resolver import resolve_session_for_message

        bot_msg = MockMessage(
            id=999,
            content="Regular bot response",
            author=MockAuthor(id=555, name="bot", bot=True),
        )
        reply_msg = MockMessage(
            id=1,
            content="Reply to bot",
            reference=MockMessageReference(message_id=999, resolved=bot_msg),
        )

        result = await resolve_session_for_message(
            message=reply_msg,
            content="Reply to bot",
            session_manager=session_manager,
        )

        assert result is not None
        assert result.is_reply_to_summary is False

    @pytest.mark.asyncio
    async def test_does_not_detect_reply_to_user_message_as_summary(self, session_manager):
        """resolve_session_for_message does not mark reply to user message as summary."""
        from persbot.bot.session_resolver import resolve_session_for_message

        user_msg = MockMessage(
            id=999,
            content="**요약:** User message with summary keyword",
            author=MockAuthor(id=555, name="user", bot=False),
        )
        reply_msg = MockMessage(
            id=1,
            content="Reply",
            reference=MockMessageReference(message_id=999, resolved=user_msg),
        )

        result = await resolve_session_for_message(
            message=reply_msg,
            content="Reply",
            session_manager=session_manager,
        )

        assert result is not None
        assert result.is_reply_to_summary is False

    @pytest.mark.asyncio
    async def test_handles_fetch_returns_none_on_error(self, session_manager):
        """resolve_session_for_message handles when fetch returns None (simulating error)."""
        from persbot.bot.session_resolver import resolve_session_for_message

        # When resolved is None, the code tries to fetch
        reply_msg = MockMessage(
            id=1,
            content="Reply to missing",
            reference=MockMessageReference(message_id=999, resolved=None),
        )

        # Mock fetch to return None (simulating not found without exception)
        reply_msg.channel.fetch_message = AsyncMock(return_value=None)

        result = await resolve_session_for_message(
            message=reply_msg,
            content="Reply to missing",
            session_manager=session_manager,
        )

        # Should still resolve session without reply context
        assert result is not None
        # Content should NOT have reply context prefix
        content = session_manager.last_resolve_args['message_content']
        assert "(답장 대상:" not in content

    @pytest.mark.asyncio
    async def test_handles_none_reference_resolved(self, session_manager):
        """resolve_session_for_message handles None reference.resolved."""
        from persbot.bot.session_resolver import resolve_session_for_message

        reply_msg = MockMessage(
            id=1,
            content="Reply",
            reference=MockMessageReference(message_id=999, resolved=None),
        )

        # Mock fetch to return a message
        fetched_msg = MockMessage(id=999, content="Fetched message")
        reply_msg.channel.fetch_message = AsyncMock(return_value=fetched_msg)

        result = await resolve_session_for_message(
            message=reply_msg,
            content="Reply",
            session_manager=session_manager,
        )

        assert result is not None
        content = session_manager.last_resolve_args['message_content']
        assert "(답장 대상:" in content
        assert "Fetched message" in content

    @pytest.mark.asyncio
    async def test_handles_message_without_reference(self, session_manager):
        """resolve_session_for_message handles message without reference."""
        from persbot.bot.session_resolver import resolve_session_for_message

        message = MockMessage(
            id=1,
            content="Standalone message",
            reference=None,
        )

        result = await resolve_session_for_message(
            message=message,
            content="Standalone message",
            session_manager=session_manager,
        )

        assert result is not None
        content = session_manager.last_resolve_args['message_content']
        assert content == "Standalone message"

    @pytest.mark.asyncio
    async def test_handles_reference_without_message_id(self, session_manager):
        """resolve_session_for_message handles reference without message_id."""
        from persbot.bot.session_resolver import resolve_session_for_message

        message = MockMessage(
            id=1,
            content="Message",
            reference=MockMessageReference(message_id=None, resolved=None),
        )

        result = await resolve_session_for_message(
            message=message,
            content="Message",
            session_manager=session_manager,
        )

        assert result is not None
        content = session_manager.last_resolve_args['message_content']
        assert content == "Message"


# ============================================================================
# extract_session_context Tests
# ============================================================================

class TestExtractSessionContext:
    """Tests for extract_session_context function."""

    @pytest.fixture
    def single_message(self):
        """Create a single mock message."""
        return MockMessage(
            id=100,
            content="Test message",
            author=MockAuthor(id=123, name="testuser"),
            channel=MockChannel(id=456),
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def message_list(self):
        """Create a list of mock messages."""
        return [
            MockMessage(
                id=101,
                content="First message",
                author=MockAuthor(id=111, name="first_user"),
                channel=MockChannel(id=789),
                created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            ),
            MockMessage(
                id=102,
                content="Second message",
                author=MockAuthor(id=222, name="second_user"),
                channel=MockChannel(id=789),
                created_at=datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc),
            ),
        ]

    def test_extracts_context_from_single_message(self, single_message):
        """extract_session_context extracts context from single message."""
        from persbot.bot.session_resolver import extract_session_context

        result = extract_session_context(single_message)

        assert result.channel_id == 456
        assert result.user_id == 123
        assert result.username == "testuser"
        assert result.message_id == "100"

    def test_extracts_created_at_from_single_message(self, single_message):
        """extract_session_context extracts created_at from single message."""
        from persbot.bot.session_resolver import extract_session_context

        result = extract_session_context(single_message)

        assert result.created_at == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_extracts_context_from_list_uses_first_message(self, message_list):
        """extract_session_context uses first message from list."""
        from persbot.bot.session_resolver import extract_session_context

        result = extract_session_context(message_list)

        # Should use first message's data
        assert result.channel_id == 789
        assert result.user_id == 111
        assert result.username == "first_user"
        assert result.message_id == "101"

    def test_extracts_created_at_from_list_uses_first_message(self, message_list):
        """extract_session_context extracts created_at from first message in list."""
        from persbot.bot.session_resolver import extract_session_context

        result = extract_session_context(message_list)

        assert result.created_at == datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_ignores_subsequent_messages_in_list(self, message_list):
        """extract_session_context ignores subsequent messages in list."""
        from persbot.bot.session_resolver import extract_session_context

        result = extract_session_context(message_list)

        # Should NOT use second message's data
        assert result.user_id != 222
        assert result.username != "second_user"
        assert result.message_id != "102"

    def test_returns_session_context_type(self, single_message):
        """extract_session_context returns SessionContext instance."""
        from persbot.bot.session_resolver import extract_session_context
        from persbot.bot.chat_models import SessionContext

        result = extract_session_context(single_message)

        assert isinstance(result, SessionContext)

    def test_message_id_is_converted_to_string(self, single_message):
        """extract_session_context converts message_id to string."""
        from persbot.bot.session_resolver import extract_session_context

        result = extract_session_context(single_message)

        assert isinstance(result.message_id, str)
        assert result.message_id == "100"

    def test_handles_large_message_id(self):
        """extract_session_context handles large message IDs (snowflakes)."""
        from persbot.bot.session_resolver import extract_session_context

        large_id = 1234567890123456789
        message = MockMessage(
            id=large_id,
            author=MockAuthor(id=123),
            channel=MockChannel(id=456),
        )

        result = extract_session_context(message)

        assert result.message_id == str(large_id)

    def test_handles_large_channel_id(self):
        """extract_session_context handles large channel IDs (snowflakes)."""
        from persbot.bot.session_resolver import extract_session_context

        large_id = 987654321098765432
        message = MockMessage(
            id=1,
            author=MockAuthor(id=123),
            channel=MockChannel(id=large_id),
        )

        result = extract_session_context(message)

        assert result.channel_id == large_id

    def test_handles_large_user_id(self):
        """extract_session_context handles large user IDs (snowflakes)."""
        from persbot.bot.session_resolver import extract_session_context

        large_id = 555555555555555555
        message = MockMessage(
            id=1,
            author=MockAuthor(id=large_id),
            channel=MockChannel(id=456),
        )

        result = extract_session_context(message)

        assert result.user_id == large_id

    def test_empty_list_raises_index_error(self):
        """extract_session_context raises IndexError for empty list."""
        from persbot.bot.session_resolver import extract_session_context

        with pytest.raises(IndexError):
            extract_session_context([])

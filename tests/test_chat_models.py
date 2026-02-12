"""Feature tests for chat models.

Tests focus on behavior:
- ChatReply dataclass
- ToolProgress dataclass
- SessionContext dataclass
"""

import pytest

from persbot.bot.chat_models import ChatReply, ToolProgress, SessionContext


class TestChatReply:
    """Tests for ChatReply dataclass."""

    def test_creates_with_required_fields(self):
        """ChatReply can be created with required fields."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None
        )
        assert reply.text == "Hello"
        assert reply.session_key == "channel:123"
        assert reply.response is None

    def test_default_images_is_empty_list(self):
        """images defaults to empty list."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None
        )
        assert reply.images == []

    def test_default_notification_is_empty_string(self):
        """notification defaults to empty string."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None
        )
        assert reply.notification == ""

    def test_default_tool_rounds_is_zero(self):
        """tool_rounds defaults to 0."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None
        )
        assert reply.tool_rounds == 0

    def test_display_text_returns_text_without_notification(self):
        """display_text returns plain text when no notification."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None
        )
        assert reply.display_text == "Hello"

    def test_display_text_prepends_notification(self):
        """display_text prepends notification with emoji."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None,
            notification="Warning"
        )
        assert "📢" in reply.display_text
        assert "Warning" in reply.display_text
        assert "Hello" in reply.display_text

    def test_has_tools_returns_false_when_zero_rounds(self):
        """has_tools returns False when tool_rounds is 0."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None,
            tool_rounds=0
        )
        assert reply.has_tools is False

    def test_has_tools_returns_true_when_rounds_positive(self):
        """has_tools returns True when tool_rounds > 0."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None,
            tool_rounds=3
        )
        assert reply.has_tools is True

    def test_has_images_returns_false_for_empty_list(self):
        """has_images returns False when images is empty."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None,
            images=[]
        )
        assert reply.has_images is False

    def test_has_images_returns_true_when_images_present(self):
        """has_images returns True when images are present."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None,
            images=[b"fake_image_data"]
        )
        assert reply.has_images is True

    def test_can_store_multiple_images(self):
        """ChatReply can store multiple images."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None,
            images=[b"img1", b"img2", b"img3"]
        )
        assert len(reply.images) == 3

    def test_is_frozen(self):
        """ChatReply is frozen (immutable)."""
        reply = ChatReply(
            text="Hello",
            session_key="channel:123",
            response=None
        )
        with pytest.raises(AttributeError):
            reply.text = "Changed"


class TestToolProgress:
    """Tests for ToolProgress dataclass."""

    def test_creates_with_empty_tool_names(self):
        """ToolProgress can be created with empty tool names."""
        progress = ToolProgress()
        assert progress.tool_names == []

    def test_creates_with_tool_names(self):
        """ToolProgress can be created with tool names."""
        progress = ToolProgress(tool_names=["web_search", "get_weather"])
        assert progress.tool_names == ["web_search", "get_weather"]

    def test_korean_names_returns_list(self):
        """korean_names returns a list."""
        progress = ToolProgress(tool_names=["web_search"])
        assert isinstance(progress.korean_names, list)

    def test_notification_text_with_no_tools(self):
        """notification_text returns default when no tools."""
        progress = ToolProgress(tool_names=[])
        assert "도구 실행 중" in progress.notification_text

    def test_notification_text_with_tools(self):
        """notification_text includes tool emoji."""
        progress = ToolProgress(tool_names=["web_search"])
        assert "🔧" in progress.notification_text


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_creates_with_required_fields(self):
        """SessionContext can be created with required fields."""
        from datetime import datetime, timezone

        ctx = SessionContext(
            channel_id=123,
            user_id=456,
            username="testuser",
            message_id="789",
            created_at=datetime.now(timezone.utc)
        )
        assert ctx.channel_id == 123
        assert ctx.user_id == 456
        assert ctx.username == "testuser"
        assert ctx.message_id == "789"

    def test_default_is_reply_to_summary_is_false(self):
        """is_reply_to_summary defaults to False."""
        from datetime import datetime, timezone

        ctx = SessionContext(
            channel_id=123,
            user_id=456,
            username="testuser",
            message_id="789",
            created_at=datetime.now(timezone.utc)
        )
        assert ctx.is_reply_to_summary is False

    def test_default_reference_message_id_is_none(self):
        """reference_message_id defaults to None."""
        from datetime import datetime, timezone

        ctx = SessionContext(
            channel_id=123,
            user_id=456,
            username="testuser",
            message_id="789",
            created_at=datetime.now(timezone.utc)
        )
        assert ctx.reference_message_id is None

    def test_can_set_is_reply_to_summary(self):
        """is_reply_to_summary can be set."""
        from datetime import datetime, timezone

        ctx = SessionContext(
            channel_id=123,
            user_id=456,
            username="testuser",
            message_id="789",
            created_at=datetime.now(timezone.utc),
            is_reply_to_summary=True
        )
        assert ctx.is_reply_to_summary is True

    def test_can_set_reference_message_id(self):
        """reference_message_id can be set."""
        from datetime import datetime, timezone

        ctx = SessionContext(
            channel_id=123,
            user_id=456,
            username="testuser",
            message_id="789",
            created_at=datetime.now(timezone.utc),
            reference_message_id="111"
        )
        assert ctx.reference_message_id == "111"

    def test_is_not_frozen(self):
        """SessionContext is mutable (not frozen)."""
        from datetime import datetime, timezone

        ctx = SessionContext(
            channel_id=123,
            user_id=456,
            username="testuser",
            message_id="789",
            created_at=datetime.now(timezone.utc)
        )
        # Should not raise
        ctx.username = "newname"
        assert ctx.username == "newname"

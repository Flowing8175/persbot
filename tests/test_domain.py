"""Tests for all domain value objects.

This module provides comprehensive test coverage for:
- ChannelId, ThreadId
- MessageId
- Provider, ModelAlias
- SessionKey
- UserId, GuildId
"""

import pytest

from persbot.domain.channel import ChannelId, ThreadId
from persbot.domain.message import MessageId
from persbot.domain.model import Provider, ModelAlias, StandardModels
from persbot.domain.session import SessionKey
from persbot.domain.user import UserId, GuildId


# =============================================================================
# ChannelId Tests
# =============================================================================


class TestChannelId:
    """Tests for ChannelId value object."""

    def test_init_with_positive_integer(self):
        """Test ChannelId initialization with positive integer."""
        channel_id = ChannelId(123456)
        assert channel_id.value == 123456

    def test_init_with_zero_raises_error(self):
        """Test that zero raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ChannelId(0)

    def test_init_with_negative_raises_error(self):
        """Test that negative value raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ChannelId(-1)

    def test_init_with_non_integer_raises_error(self):
        """Test that non-integer raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ChannelId("not an int")

    def test_str_returns_value_as_string(self):
        """Test __str__ returns value as string."""
        channel_id = ChannelId(123456)
        assert str(channel_id) == "123456"

    def test_int_returns_value(self):
        """Test __int__ returns value."""
        channel_id = ChannelId(123456)
        assert int(channel_id) == 123456

    def test_from_raw_with_integer(self):
        """Test from_raw with integer."""
        channel_id = ChannelId.from_raw(123456)
        assert channel_id.value == 123456

    def test_from_raw_with_string(self):
        """Test from_raw with string."""
        channel_id = ChannelId.from_raw("123456")
        assert channel_id.value == 123456

    def test_from_raw_with_channel_id_returns_same(self):
        """Test from_raw with ChannelId returns same instance."""
        original = ChannelId(123456)
        channel_id = ChannelId.from_raw(original)
        assert channel_id is original

    def test_to_session_key(self):
        """Test to_session_key format."""
        channel_id = ChannelId(123456)
        assert channel_id.to_session_key() == "channel:123456"


class TestThreadId:
    """Tests for ThreadId value object."""

    def test_init_with_positive_integer(self):
        """Test ThreadId initialization."""
        thread_id = ThreadId(789012)
        assert thread_id.value == 789012

    def test_to_session_key_returns_thread_prefix(self):
        """Test to_session_key uses thread prefix."""
        thread_id = ThreadId(789012)
        assert thread_id.to_session_key() == "thread:789012"


# =============================================================================
# MessageId Tests
# =============================================================================


class TestMessageId:
    """Tests for MessageId value object."""

    def test_init_with_numeric_string(self):
        """Test MessageId with numeric string."""
        msg_id = MessageId("123456789")
        assert msg_id.value == "123456789"

    def test_init_with_integer(self):
        """Test MessageId with integer (converts to string)."""
        msg_id = MessageId(123456789)
        assert msg_id.value == "123456789"

    def test_init_with_non_numeric_string_raises_error(self):
        """Test non-numeric string raises ValueError."""
        with pytest.raises(ValueError, match="must be numeric"):
            MessageId("abc123")

    def test_init_with_empty_string_raises_error(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="must be numeric"):
            MessageId("")

    def test_str_returns_value(self):
        """Test __str__ returns value."""
        msg_id = MessageId("123456789")
        assert str(msg_id) == "123456789"

    def test_to_int_converts_to_integer(self):
        """Test to_int converts string value to int."""
        msg_id = MessageId("123456789")
        assert msg_id.to_int() == 123456789

    def test_from_raw_with_string(self):
        """Test from_raw with string."""
        msg_id = MessageId.from_raw("123456789")
        assert msg_id.value == "123456789"

    def test_from_raw_with_integer(self):
        """Test from_raw with integer."""
        msg_id = MessageId.from_raw(123456789)
        assert msg_id.value == "123456789"

    def test_from_raw_with_message_id_returns_same(self):
        """Test from_raw with MessageId returns same instance."""
        original = MessageId("123456789")
        msg_id = MessageId.from_raw(original)
        assert msg_id is original


# =============================================================================
# Provider Enum Tests
# =============================================================================


class TestProvider:
    """Tests for Provider enum."""

    def test_gemini_value(self):
        """Test GEMINI provider value."""
        assert Provider.GEMINI == "gemini"
        assert str(Provider.GEMINI) == "gemini"

    def test_openai_value(self):
        """Test OPENAI provider value."""
        assert Provider.OPENAI == "openai"
        assert str(Provider.OPENAI) == "openai"

    def test_zai_value(self):
        """Test ZAI provider value."""
        assert Provider.ZAI == "zai"
        assert str(Provider.ZAI) == "zai"

    def test_from_string_with_gemini(self):
        """Test from_string with gemini (case insensitive)."""
        assert Provider.from_string("gemini") == Provider.GEMINI
        assert Provider.from_string("GEMINI") == Provider.GEMINI
        assert Provider.from_string("GeMiNi") == Provider.GEMINI

    def test_from_string_with_openai(self):
        """Test from_string with openai variants."""
        assert Provider.from_string("openai") == Provider.OPENAI
        assert Provider.from_string("OPENAI") == Provider.OPENAI
        assert Provider.from_string("OpenAI") == Provider.OPENAI

    def test_from_string_with_zai(self):
        """Test from_string with zai (case insensitive)."""
        assert Provider.from_string("zai") == Provider.ZAI
        assert Provider.from_string("ZAI") == Provider.ZAI

    def test_from_string_with_unknown_raises_error(self):
        """Test from_string with unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Provider.from_string("unknown")


# =============================================================================
# ModelAlias Tests
# =============================================================================


class TestModelAlias:
    """Tests for ModelAlias value object."""

    def test_init_with_valid_string(self):
        """Test ModelAlias initialization."""
        model = ModelAlias("gemini-2.5-flash")
        assert model.value == "gemini-2.5-flash"

    def test_init_strips_whitespace(self):
        """Test that initialization strips whitespace."""
        model = ModelAlias("  gemini-2.5-flash  ")
        assert model.value == "gemini-2.5-flash"

    def test_init_with_empty_string_raises_error(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelAlias("")

        with pytest.raises(ValueError, match="cannot be empty"):
            ModelAlias("   ")

    def test_init_with_non_string_raises_error(self):
        """Test non-string raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            ModelAlias(123)

    def test_str_returns_value(self):
        """Test __str__ returns value."""
        model = ModelAlias("gemini-2.5-flash")
        assert str(model) == "gemini-2.5-flash"

    def test_provider_gemini(self):
        """Test provider property for Gemini models."""
        model = ModelAlias("gemini-2.5-flash")
        assert model.provider == Provider.GEMINI

        model = ModelAlias("Gemini 2.5 Flash")
        assert model.provider == Provider.GEMINI

    def test_provider_openai(self):
        """Test provider property for OpenAI models."""
        model = ModelAlias("gpt-4o")
        assert model.provider == Provider.OPENAI

        model = ModelAlias("GPT-4o Mini")
        assert model.provider == Provider.OPENAI

    def test_provider_zai(self):
        """Test provider property for ZAI models."""
        model = ModelAlias("glm-4.7")
        assert model.provider == Provider.ZAI

        model = ModelAlias("GLM 4 Flash")
        assert model.provider == Provider.ZAI

    def test_provider_defaults_to_gemini_for_unknown(self):
        """Test provider defaults to Gemini for unknown models."""
        model = ModelAlias("unknown-model")
        assert model.provider == Provider.GEMINI

    def test_from_raw_with_string(self):
        """Test from_raw with string."""
        model = ModelAlias.from_raw("gemini-2.5-flash")
        assert model.value == "gemini-2.5-flash"

    def test_from_raw_with_model_alias_returns_same(self):
        """Test from_raw with ModelAlias returns same instance."""
        original = ModelAlias("gemini-2.5-flash")
        model = ModelAlias.from_raw(original)
        assert model is original


class TestStandardModels:
    """Tests for StandardModels constants."""

    def test_gemini_models(self):
        """Test Gemini model constants."""
        assert StandardModels.GEMINI_FLASH.value == "Gemini 2.5 Flash"
        assert StandardModels.GEMINI_PRO.value == "Gemini 2.5 Pro"

    def test_openai_models(self):
        """Test OpenAI model constants."""
        assert StandardModels.GPT_4O.value == "GPT-4o"
        assert StandardModels.GPT_4O_MINI.value == "GPT-4o Mini"
        assert StandardModels.GPT_5_MINI.value == "GPT-5 Mini"

    def test_zai_models(self):
        """Test ZAI model constants."""
        assert StandardModels.GLM_4_7.value == "GLM 4.7"
        assert StandardModels.GLM_4_FLASH.value == "GLM 4 Flash"
        assert StandardModels.GLM_4_6V.value == "GLM 4.6V"

    def test_default_is_gemini_flash(self):
        """Test DEFAULT points to GEMINI_FLASH."""
        assert StandardModels.DEFAULT == StandardModels.GEMINI_FLASH


# =============================================================================
# SessionKey Tests
# =============================================================================


class TestSessionKey:
    """Tests for SessionKey value object."""

    def test_init_with_valid_key(self):
        """Test SessionKey initialization."""
        key = SessionKey("channel:123456")
        assert key.value == "channel:123456"

    def test_init_strips_whitespace(self):
        """Test that initialization strips whitespace."""
        key = SessionKey("  channel:123456  ")
        assert key.value == "channel:123456"

    def test_init_with_non_string_raises_error(self):
        """Test non-string raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            SessionKey(123456)

    def test_init_with_empty_string_raises_error(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SessionKey("")

        with pytest.raises(ValueError, match="cannot be empty"):
            SessionKey("   ")

    def test_init_without_colon_raises_error(self):
        """Test key without colon raises ValueError."""
        with pytest.raises(ValueError, match="must contain"):
            SessionKey("channel123456")

    def test_str_returns_value(self):
        """Test __str__ returns value."""
        key = SessionKey("channel:123456")
        assert str(key) == "channel:123456"

    def test_type_property_returns_prefix(self):
        """Test type property returns prefix before colon."""
        key = SessionKey("channel:123456")
        assert key.type == "channel"

        key = SessionKey("user:789")
        assert key.type == "user"

        key = SessionKey("thread:456")
        assert key.type == "thread"

    def test_id_property_returns_suffix(self):
        """Test id property returns suffix after colon."""
        key = SessionKey("channel:123456")
        assert key.id == "123456"

        key = SessionKey("user:789")
        assert key.id == "789"

    def test_from_channel(self):
        """Test from_channel class method."""
        key = SessionKey.from_channel(123456)
        assert key.value == "channel:123456"
        assert key.type == "channel"

    def test_from_user(self):
        """Test from_user class method."""
        key = SessionKey.from_user(789012)
        assert key.value == "user:789012"
        assert key.type == "user"

    def test_from_thread(self):
        """Test from_thread class method."""
        key = SessionKey.from_thread(456789)
        assert key.value == "thread:456789"
        assert key.type == "thread"

    def test_from_raw_with_string(self):
        """Test from_raw with string."""
        key = SessionKey.from_raw("channel:123456")
        assert key.value == "channel:123456"

    def test_from_raw_with_session_key_returns_same(self):
        """Test from_raw with SessionKey returns same instance."""
        original = SessionKey("channel:123456")
        key = SessionKey.from_raw(original)
        assert key is original

    def test_is_channel_session(self):
        """Test is_channel_session method."""
        key = SessionKey("channel:123456")
        assert key.is_channel_session() is True

        key = SessionKey("user:789")
        assert key.is_channel_session() is False

    def test_is_user_session(self):
        """Test is_user_session method."""
        key = SessionKey("user:789")
        assert key.is_user_session() is True

        key = SessionKey("channel:123456")
        assert key.is_user_session() is False

    def test_is_thread_session(self):
        """Test is_thread_session method."""
        key = SessionKey("thread:456789")
        assert key.is_thread_session() is True

        key = SessionKey("channel:123456")
        assert key.is_thread_session() is False


# =============================================================================
# UserId Tests
# =============================================================================


class TestUserId:
    """Tests for UserId value object."""

    def test_init_with_positive_integer(self):
        """Test UserId initialization."""
        user_id = UserId(123456)
        assert user_id.value == 123456

    def test_init_with_zero_raises_error(self):
        """Test that zero raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            UserId(0)

    def test_init_with_negative_raises_error(self):
        """Test that negative value raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            UserId(-1)

    def test_init_with_non_integer_raises_error(self):
        """Test non-integer raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            UserId("not an int")

    def test_str_returns_value_as_string(self):
        """Test __str__ returns value as string."""
        user_id = UserId(123456)
        assert str(user_id) == "123456"

    def test_int_returns_value(self):
        """Test __int__ returns value."""
        user_id = UserId(123456)
        assert int(user_id) == 123456

    def test_from_raw_with_integer(self):
        """Test from_raw with integer."""
        user_id = UserId.from_raw(123456)
        assert user_id.value == 123456

    def test_from_raw_with_string(self):
        """Test from raw with string that converts to int."""
        user_id = UserId.from_raw("123456")
        assert user_id.value == 123456

    def test_from_raw_with_user_id_returns_same(self):
        """Test from raw with UserId returns same instance."""
        original = UserId(123456)
        user_id = UserId.from_raw(original)
        assert user_id is original

    def test_to_session_key(self):
        """Test to_session_key format."""
        user_id = UserId(123456)
        assert user_id.to_session_key() == "user:123456"


# =============================================================================
# GuildId Tests
# =============================================================================


class TestGuildId:
    """Tests for GuildId value object."""

    def test_init_with_positive_integer(self):
        """Test GuildId initialization."""
        guild_id = GuildId(987654)
        assert guild_id.value == 987654

    def test_init_with_zero_raises_error(self):
        """Test that zero raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            GuildId(0)

    def test_init_with_negative_raises_error(self):
        """Test that negative value raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            GuildId(-1)

    def test_init_with_non_integer_raises_error(self):
        """Test non-integer raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            GuildId("not an int")

    def test_str_returns_value_as_string(self):
        """Test __str__ returns value as string."""
        guild_id = GuildId(987654)
        assert str(guild_id) == "987654"

    def test_int_returns_value(self):
        """Test __int__ returns value."""
        guild_id = GuildId(987654)
        assert int(guild_id) == 987654

    def test_from_raw_with_integer(self):
        """Test from raw with integer."""
        guild_id = GuildId.from_raw(987654)
        assert guild_id.value == 987654

    def test_from_raw_with_string(self):
        """Test from raw with string that converts to int."""
        guild_id = GuildId.from_raw("987654")
        assert guild_id.value == 987654

    def test_from_raw_with_guild_id_returns_same(self):
        """Test from raw with GuildId returns same instance."""
        original = GuildId(987654)
        guild_id = GuildId.from_raw(original)
        assert guild_id is original

    def test_from_user_id_creates_from_user_id(self):
        """Test from_user_id creates GuildId from UserId."""
        user_id = UserId(123456)
        guild_id = GuildId.from_user_id(user_id)
        assert guild_id.value == 123456

    def test_from_user_id_with_integer(self):
        """Test from_user_id with integer directly."""
        # from_user_id expects a UserId object, not raw int
        user_id = UserId(123456)
        guild_id = GuildId.from_user_id(user_id)
        assert guild_id.value == 123456


# =============================================================================
# Cross-Domain Tests
# =============================================================================


class TestDomainCrossFunctionality:
    """Tests for cross-domain functionality and integration."""

    def test_channel_and_user_ids_dont_conflict(self):
        """Test that ChannelId and UserId are distinct types."""
        channel = ChannelId(123456)
        user = UserId(123456)

        # Same numeric value but different types
        assert channel.value == user.value
        assert type(channel) != type(user)

    def test_session_key_from_different_sources(self):
        """Test SessionKey creation from different ID types."""
        channel_key = SessionKey.from_channel(123456)
        user_key = SessionKey.from_user(789012)
        thread_key = SessionKey.from_thread(456789)

        assert channel_key.value == "channel:123456"
        assert user_key.value == "user:789012"
        assert thread_key.value == "thread:456789"

    def test_model_alias_detection(self):
        """Test ModelAlias provider detection for all providers."""
        gemini_model = ModelAlias("gemini-2.5-flash")
        openai_model = ModelAlias("gpt-4o")
        zai_model = ModelAlias("glm-4.7")

        assert gemini_model.provider == Provider.GEMINI
        assert openai_model.provider == Provider.OPENAI
        assert zai_model.provider == Provider.ZAI

    def test_session_key_type_detection(self):
        """Test SessionKey type detection methods."""
        channel_key = SessionKey.from_channel(123)
        user_key = SessionKey.from_user(456)
        thread_key = SessionKey.from_thread(789)

        assert channel_key.is_channel_session() and not channel_key.is_user_session() and not channel_key.is_thread_session()
        assert user_key.is_user_session() and not user_key.is_channel_session() and not user_key.is_thread_session()
        assert thread_key.is_thread_session() and not thread_key.is_channel_session() and not thread_key.is_user_session()

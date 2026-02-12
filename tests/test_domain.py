"""Feature tests for domain layer value objects.

Tests focus on behavior rather than implementation details:
- SessionKey: session identifier value object
- Provider: LLM provider enum
- ModelAlias: model alias value object
- StandardModels: standard model constants
"""

import pytest

from persbot.domain.session import SessionKey
from persbot.domain.model import Provider, ModelAlias, StandardModels


# ==============================================================================
# SessionKey Feature Tests
# ==============================================================================

class TestSessionKeyCreation:
    """Tests for SessionKey instantiation and validation."""

    def test_creates_from_valid_string(self):
        """SessionKey can be created from a valid type:id string."""
        key = SessionKey("channel:123456")
        assert key.value == "channel:123456"

    def test_strips_whitespace_from_value(self):
        """SessionKey strips leading/trailing whitespace."""
        key = SessionKey("  channel:123456  ")
        assert key.value == "channel:123456"

    def test_rejects_empty_string(self):
        """SessionKey raises ValueError for empty string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SessionKey("")

    def test_rejects_whitespace_only_string(self):
        """SessionKey raises ValueError for whitespace-only string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SessionKey("   ")

    def test_rejects_string_without_colon(self):
        """SessionKey raises ValueError if no ':' separator present."""
        with pytest.raises(ValueError, match="must contain ':'"):
            SessionKey("invalidkey")

    def test_rejects_non_string_types(self):
        """SessionKey raises TypeError for non-string inputs."""
        with pytest.raises(TypeError, match="must be a string"):
            SessionKey(123)

        with pytest.raises(TypeError, match="must be a string"):
            SessionKey(None)

        with pytest.raises(TypeError, match="must be a string"):
            SessionKey(["channel:123"])

    def test_accepts_colon_at_start(self):
        """SessionKey accepts strings starting with colon (edge case)."""
        key = SessionKey(":123456")
        assert key.type == ""
        assert key.id == "123456"

    def test_accepts_colon_at_end(self):
        """SessionKey accepts strings ending with colon (edge case)."""
        key = SessionKey("channel:")
        assert key.type == "channel"
        assert key.id == ""

    def test_accepts_multiple_colons(self):
        """SessionKey uses first colon as separator, keeps others in id."""
        key = SessionKey("channel:123:456:789")
        assert key.type == "channel"
        assert key.id == "123:456:789"


class TestSessionKeyProperties:
    """Tests for SessionKey property accessors."""

    def test_type_property_returns_prefix(self):
        """type property returns the prefix before the colon."""
        key = SessionKey("channel:123456")
        assert key.type == "channel"

    def test_id_property_returns_suffix(self):
        """id property returns the suffix after the colon."""
        key = SessionKey("channel:123456")
        assert key.id == "123456"

    def test_properties_work_with_numeric_ids(self):
        """Properties correctly parse numeric IDs."""
        key = SessionKey("user:987654321")
        assert key.type == "user"
        assert key.id == "987654321"

    def test_properties_work_with_string_ids(self):
        """Properties correctly parse string IDs."""
        key = SessionKey("custom:some-identifier")
        assert key.type == "custom"
        assert key.id == "some-identifier"


class TestSessionKeyFactoryMethods:
    """Tests for SessionKey factory class methods."""

    def test_from_channel_creates_channel_key(self):
        """from_channel creates a channel-type session key."""
        key = SessionKey.from_channel(123456)
        assert key.value == "channel:123456"
        assert key.type == "channel"

    def test_from_user_creates_user_key(self):
        """from_user creates a user-type session key."""
        key = SessionKey.from_user(789012)
        assert key.value == "user:789012"
        assert key.type == "user"

    def test_from_thread_creates_thread_key(self):
        """from_thread creates a thread-type session key."""
        key = SessionKey.from_thread(345678)
        assert key.value == "thread:345678"
        assert key.type == "thread"

    def test_from_raw_with_string(self):
        """from_raw creates SessionKey from string input."""
        key = SessionKey.from_raw("channel:999")
        assert key.value == "channel:999"

    def test_from_raw_with_session_key(self):
        """from_raw returns the same SessionKey when passed a SessionKey."""
        original = SessionKey("channel:123")
        result = SessionKey.from_raw(original)
        assert result is original
        assert result.value == "channel:123"


class TestSessionKeyTypeChecks:
    """Tests for session type checking methods."""

    def test_is_channel_session_returns_true_for_channel(self):
        """is_channel_session returns True for channel-type keys."""
        key = SessionKey.from_channel(123)
        assert key.is_channel_session() is True

    def test_is_channel_session_returns_false_for_others(self):
        """is_channel_session returns False for non-channel keys."""
        key = SessionKey.from_user(123)
        assert key.is_channel_session() is False

    def test_is_user_session_returns_true_for_user(self):
        """is_user_session returns True for user-type keys."""
        key = SessionKey.from_user(456)
        assert key.is_user_session() is True

    def test_is_user_session_returns_false_for_others(self):
        """is_user_session returns False for non-user keys."""
        key = SessionKey.from_thread(456)
        assert key.is_user_session() is False

    def test_is_thread_session_returns_true_for_thread(self):
        """is_thread_session returns True for thread-type keys."""
        key = SessionKey.from_thread(789)
        assert key.is_thread_session() is True

    def test_is_thread_session_returns_false_for_others(self):
        """is_thread_session returns False for non-thread keys."""
        key = SessionKey.from_channel(789)
        assert key.is_thread_session() is False


class TestSessionKeyStringRepresentation:
    """Tests for SessionKey string representation."""

    def test_str_returns_value(self):
        """str() returns the underlying value."""
        key = SessionKey("channel:123456")
        assert str(key) == "channel:123456"

    def test_str_works_with_factory_methods(self):
        """str() works correctly with factory-created keys."""
        assert str(SessionKey.from_channel(1)) == "channel:1"
        assert str(SessionKey.from_user(2)) == "user:2"
        assert str(SessionKey.from_thread(3)) == "thread:3"


class TestSessionKeyImmutability:
    """Tests for SessionKey immutability (frozen dataclass)."""

    def test_value_cannot_be_changed(self):
        """SessionKey value cannot be modified after creation."""
        key = SessionKey("channel:123")
        with pytest.raises(AttributeError):
            key.value = "user:456"

    def test_cannot_add_new_attributes(self):
        """Cannot add new attributes to SessionKey (slots)."""
        key = SessionKey("channel:123")
        # Custom __init__ with frozen dataclass raises TypeError for unknown attrs
        with pytest.raises((AttributeError, TypeError)):
            key.new_attr = "value"


class TestSessionKeyEquality:
    """Tests for SessionKey equality and hashing."""

    def test_equal_keys_have_same_value(self):
        """Two SessionKeys with same value are equal."""
        key1 = SessionKey("channel:123")
        key2 = SessionKey("channel:123")
        assert key1 == key2

    def test_different_keys_have_different_values(self):
        """Two SessionKeys with different values are not equal."""
        key1 = SessionKey("channel:123")
        key2 = SessionKey("channel:456")
        assert key1 != key2

    def test_equal_keys_have_same_hash(self):
        """Two equal SessionKeys have the same hash."""
        key1 = SessionKey("channel:123")
        key2 = SessionKey("channel:123")
        assert hash(key1) == hash(key2)

    def test_can_be_used_in_set(self):
        """SessionKeys can be stored in a set."""
        key1 = SessionKey("channel:123")
        key2 = SessionKey("channel:123")
        key3 = SessionKey("user:456")
        key_set = {key1, key2, key3}
        assert len(key_set) == 2

    def test_can_be_used_as_dict_key(self):
        """SessionKeys can be used as dictionary keys."""
        key1 = SessionKey("channel:123")
        key2 = SessionKey("channel:123")
        d = {key1: "value1"}
        d[key2] = "value2"
        assert len(d) == 1
        assert d[SessionKey("channel:123")] == "value2"


# ==============================================================================
# Provider Enum Feature Tests
# ==============================================================================

class TestProviderEnum:
    """Tests for Provider enum values and methods."""

    def test_has_gemini_provider(self):
        """Provider has GEMINI member."""
        assert Provider.GEMINI.value == "gemini"

    def test_has_openai_provider(self):
        """Provider has OPENAI member."""
        assert Provider.OPENAI.value == "openai"

    def test_has_zai_provider(self):
        """Provider has ZAI member."""
        assert Provider.ZAI.value == "zai"


class TestProviderFromString:
    """Tests for Provider.from_string factory method."""

    def test_from_string_returns_gemini(self):
        """from_string returns GEMINI for 'gemini'."""
        assert Provider.from_string("gemini") == Provider.GEMINI

    def test_from_string_returns_openai(self):
        """from_string returns OPENAI for 'openai'."""
        assert Provider.from_string("openai") == Provider.OPENAI

    def test_from_string_returns_zai(self):
        """from_string returns ZAI for 'zai'."""
        assert Provider.from_string("zai") == Provider.ZAI

    def test_from_string_is_case_insensitive(self):
        """from_string handles mixed case input."""
        assert Provider.from_string("GEMINI") == Provider.GEMINI
        assert Provider.from_string("OpenAI") == Provider.OPENAI
        assert Provider.from_string("ZAI") == Provider.ZAI
        assert Provider.from_string("GeMiNi") == Provider.GEMINI

    def test_from_string_strips_whitespace(self):
        """from_string handles whitespace in input."""
        assert Provider.from_string("  gemini  ") == Provider.GEMINI
        assert Provider.from_string("\topenai\n") == Provider.OPENAI

    def test_from_string_raises_for_invalid_provider(self):
        """from_string raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Provider.from_string("unknown")

        with pytest.raises(ValueError, match="Unknown provider"):
            Provider.from_string("claude")

    def test_from_string_raises_for_empty_string(self):
        """from_string raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Provider.from_string("")

    def test_from_string_raises_for_whitespace_only(self):
        """from_string raises ValueError for whitespace-only string."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Provider.from_string("   ")


class TestProviderStringRepresentation:
    """Tests for Provider string representation."""

    def test_str_returns_value(self):
        """str() returns the provider value."""
        assert str(Provider.GEMINI) == "gemini"
        assert str(Provider.OPENAI) == "openai"
        assert str(Provider.ZAI) == "zai"

    def test_value_is_string(self):
        """Provider values are strings (str enum behavior)."""
        assert Provider.GEMINI.value.upper() == "GEMINI"


# ==============================================================================
# ModelAlias Feature Tests
# ==============================================================================

class TestModelAliasCreation:
    """Tests for ModelAlias instantiation and validation."""

    def test_creates_from_valid_string(self):
        """ModelAlias can be created from a valid string."""
        alias = ModelAlias("Gemini 2.5 Flash")
        assert alias.value == "Gemini 2.5 Flash"

    def test_strips_whitespace_from_value(self):
        """ModelAlias strips leading/trailing whitespace."""
        alias = ModelAlias("  GPT-4o  ")
        assert alias.value == "GPT-4o"

    def test_rejects_empty_string(self):
        """ModelAlias raises ValueError for empty string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelAlias("")

    def test_rejects_whitespace_only_string(self):
        """ModelAlias raises ValueError for whitespace-only string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelAlias("   ")

    def test_rejects_non_string_types(self):
        """ModelAlias raises TypeError for non-string inputs."""
        with pytest.raises(TypeError, match="must be a string"):
            ModelAlias(123)

        with pytest.raises(TypeError, match="must be a string"):
            ModelAlias(None)

        with pytest.raises(TypeError, match="must be a string"):
            ModelAlias(["GPT-4o"])


class TestModelAliasProvider:
    """Tests for ModelAlias provider detection."""

    def test_detects_gemini_provider(self):
        """Models with 'gemini' in name return GEMINI provider."""
        alias = ModelAlias("Gemini 2.5 Flash")
        assert alias.provider == Provider.GEMINI

    def test_detects_gemini_case_insensitive(self):
        """Provider detection is case-insensitive."""
        alias = ModelAlias("GEMINI PRO")
        assert alias.provider == Provider.GEMINI

    def test_detects_gpt_provider(self):
        """Models with 'gpt' in name return OPENAI provider."""
        alias = ModelAlias("GPT-4o")
        assert alias.provider == Provider.OPENAI

    def test_detects_gpt_case_insensitive(self):
        """GPT detection is case-insensitive."""
        alias = ModelAlias("gpt-4o-mini")
        assert alias.provider == Provider.OPENAI

    def test_detects_openai_provider(self):
        """Models with 'openai' in name return OPENAI provider."""
        alias = ModelAlias("OpenAI o1")
        assert alias.provider == Provider.OPENAI

    def test_detects_glm_provider(self):
        """Models with 'glm' in name return ZAI provider."""
        alias = ModelAlias("GLM 4.7")
        assert alias.provider == Provider.ZAI

    def test_detects_glm_case_insensitive(self):
        """GLM detection is case-insensitive."""
        alias = ModelAlias("glm-4-flash")
        assert alias.provider == Provider.ZAI

    def test_detects_zai_provider(self):
        """Models with 'zai' in name return ZAI provider."""
        alias = ModelAlias("ZAI Model")
        assert alias.provider == Provider.ZAI

    def test_defaults_to_gemini_for_unknown(self):
        """Unknown models default to GEMINI provider."""
        alias = ModelAlias("Claude 3.5")
        assert alias.provider == Provider.GEMINI

        alias = ModelAlias("Unknown Model XYZ")
        assert alias.provider == Provider.GEMINI

    def test_gpt_takes_precedence_over_gemini_word_boundary(self):
        """Provider detection checks patterns in order."""
        # Both patterns in same string - first match wins in our implementation
        alias = ModelAlias("gemini-gpt")
        assert alias.provider == Provider.GEMINI


class TestModelAliasFactoryMethods:
    """Tests for ModelAlias factory class methods."""

    def test_from_raw_with_string(self):
        """from_raw creates ModelAlias from string input."""
        alias = ModelAlias.from_raw("GPT-4o")
        assert alias.value == "GPT-4o"

    def test_from_raw_with_model_alias(self):
        """from_raw returns the same ModelAlias when passed a ModelAlias."""
        original = ModelAlias("Gemini Pro")
        result = ModelAlias.from_raw(original)
        assert result is original
        assert result.value == "Gemini Pro"


class TestModelAliasStringRepresentation:
    """Tests for ModelAlias string representation."""

    def test_str_returns_value(self):
        """str() returns the underlying value."""
        alias = ModelAlias("GPT-4o Mini")
        assert str(alias) == "GPT-4o Mini"


class TestModelAliasImmutability:
    """Tests for ModelAlias immutability (frozen dataclass)."""

    def test_value_cannot_be_changed(self):
        """ModelAlias value cannot be modified after creation."""
        alias = ModelAlias("GPT-4o")
        with pytest.raises(AttributeError):
            alias.value = "Gemini Pro"

    def test_cannot_add_new_attributes(self):
        """Cannot add new attributes to ModelAlias (slots)."""
        alias = ModelAlias("GPT-4o")
        # Custom __init__ with frozen dataclass raises TypeError for unknown attrs
        with pytest.raises((AttributeError, TypeError)):
            alias.new_attr = "value"


class TestModelAliasEquality:
    """Tests for ModelAlias equality and hashing."""

    def test_equal_aliases_have_same_value(self):
        """Two ModelAliases with same value are equal."""
        alias1 = ModelAlias("GPT-4o")
        alias2 = ModelAlias("GPT-4o")
        assert alias1 == alias2

    def test_different_aliases_have_different_values(self):
        """Two ModelAliases with different values are not equal."""
        alias1 = ModelAlias("GPT-4o")
        alias2 = ModelAlias("Gemini Pro")
        assert alias1 != alias2

    def test_equal_aliases_have_same_hash(self):
        """Two equal ModelAliases have the same hash."""
        alias1 = ModelAlias("GPT-4o")
        alias2 = ModelAlias("GPT-4o")
        assert hash(alias1) == hash(alias2)

    def test_can_be_used_in_set(self):
        """ModelAliases can be stored in a set."""
        alias1 = ModelAlias("GPT-4o")
        alias2 = ModelAlias("GPT-4o")
        alias3 = ModelAlias("Gemini Pro")
        alias_set = {alias1, alias2, alias3}
        assert len(alias_set) == 2


# ==============================================================================
# StandardModels Feature Tests
# ==============================================================================

class TestStandardModelsGemini:
    """Tests for Gemini standard model constants."""

    def test_gemini_flash_is_valid_alias(self):
        """GEMINI_FLASH is a valid ModelAlias."""
        assert isinstance(StandardModels.GEMINI_FLASH, ModelAlias)
        assert StandardModels.GEMINI_FLASH.value == "Gemini 2.5 Flash"
        assert StandardModels.GEMINI_FLASH.provider == Provider.GEMINI

    def test_gemini_pro_is_valid_alias(self):
        """GEMINI_PRO is a valid ModelAlias."""
        assert isinstance(StandardModels.GEMINI_PRO, ModelAlias)
        assert StandardModels.GEMINI_PRO.value == "Gemini 2.5 Pro"
        assert StandardModels.GEMINI_PRO.provider == Provider.GEMINI


class TestStandardModelsOpenAI:
    """Tests for OpenAI standard model constants."""

    def test_gpt_4o_is_valid_alias(self):
        """GPT_4O is a valid ModelAlias."""
        assert isinstance(StandardModels.GPT_4O, ModelAlias)
        assert StandardModels.GPT_4O.value == "GPT-4o"
        assert StandardModels.GPT_4O.provider == Provider.OPENAI

    def test_gpt_4o_mini_is_valid_alias(self):
        """GPT_4O_MINI is a valid ModelAlias."""
        assert isinstance(StandardModels.GPT_4O_MINI, ModelAlias)
        assert StandardModels.GPT_4O_MINI.value == "GPT-4o Mini"
        assert StandardModels.GPT_4O_MINI.provider == Provider.OPENAI

    def test_gpt_5_mini_is_valid_alias(self):
        """GPT_5_MINI is a valid ModelAlias."""
        assert isinstance(StandardModels.GPT_5_MINI, ModelAlias)
        assert StandardModels.GPT_5_MINI.value == "GPT-5 Mini"
        assert StandardModels.GPT_5_MINI.provider == Provider.OPENAI


class TestStandardModelsZAI:
    """Tests for ZAI standard model constants."""

    def test_glm_4_7_is_valid_alias(self):
        """GLM_4_7 is a valid ModelAlias."""
        assert isinstance(StandardModels.GLM_4_7, ModelAlias)
        assert StandardModels.GLM_4_7.value == "GLM 4.7"
        assert StandardModels.GLM_4_7.provider == Provider.ZAI

    def test_glm_4_flash_is_valid_alias(self):
        """GLM_4_FLASH is a valid ModelAlias."""
        assert isinstance(StandardModels.GLM_4_FLASH, ModelAlias)
        assert StandardModels.GLM_4_FLASH.value == "GLM 4 Flash"
        assert StandardModels.GLM_4_FLASH.provider == Provider.ZAI

    def test_glm_4_6v_is_valid_alias(self):
        """GLM_4_6V is a valid ModelAlias."""
        assert isinstance(StandardModels.GLM_4_6V, ModelAlias)
        assert StandardModels.GLM_4_6V.value == "GLM 4.6V"
        assert StandardModels.GLM_4_6V.provider == Provider.ZAI


class TestStandardModelsDefault:
    """Tests for default model constant."""

    def test_default_is_gemini_flash(self):
        """DEFAULT points to GEMINI_FLASH."""
        assert StandardModels.DEFAULT is StandardModels.GEMINI_FLASH

    def test_default_is_gemini_provider(self):
        """DEFAULT uses GEMINI provider."""
        assert StandardModels.DEFAULT.provider == Provider.GEMINI


# ==============================================================================
# Integration Feature Tests
# ==============================================================================

class TestDomainIntegration:
    """Integration tests combining domain value objects."""

    def test_session_key_with_model_alias(self):
        """SessionKey and ModelAlias can be used together in a context."""
        session = SessionKey.from_channel(123)
        model = StandardModels.GEMINI_PRO

        # Simulate a context dictionary
        context = {
            "session": session,
            "model": model,
        }

        assert context["session"].type == "channel"
        assert context["model"].provider == Provider.GEMINI

    def test_multiple_providers_distinct(self):
        """All provider values are distinct."""
        providers = [p.value for p in Provider]
        assert len(providers) == len(set(providers))

    def test_session_key_set_with_different_types(self):
        """SessionKeys with different types don't collide."""
        keys = {
            SessionKey.from_channel(123),
            SessionKey.from_user(123),
            SessionKey.from_thread(123),
        }
        assert len(keys) == 3

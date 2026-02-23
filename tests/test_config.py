"""Feature tests for config module.

Tests focus on behavior:
- AppConfig dataclass with default values
- _resolve_log_level() log level resolution
- _normalize_provider() provider string normalization
- _validate_provider() provider validation
- _first_nonempty_env() environment variable prioritization
- _parse_float_env() float parsing from env
- _parse_int_env() integer parsing from env
- _parse_bool_env() boolean parsing from env
- _parse_thinking_budget() thinking budget parsing with 'off' handling
- _parse_auto_channel_ids() channel ID parsing from comma-separated list
- _resolve_model_name() model name resolution by provider and role
- load_config() main configuration loading
"""

import logging
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from persbot.config import (
    AppConfig,
    _resolve_log_level,
    _normalize_provider,
    _validate_provider,
    _first_nonempty_env,
    _parse_float_env,
    _parse_int_env,
    _parse_bool_env,
    _parse_thinking_budget,
    _parse_auto_channel_ids,
    _resolve_model_name,
    load_config,
    DEFAULT_GEMINI_ASSISTANT_MODEL,
    DEFAULT_GEMINI_SUMMARY_MODEL,
    DEFAULT_OPENAI_ASSISTANT_MODEL,
    DEFAULT_OPENAI_SUMMARY_MODEL,
    DEFAULT_ZAI_ASSISTANT_MODEL,
    DEFAULT_ZAI_SUMMARY_MODEL,
)


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_creates_with_discord_token_only(self):
        """AppConfig creates with only required discord_token."""
        config = AppConfig(discord_token="test-token")
        assert config.discord_token == "test-token"

    def test_discord_token_is_required(self):
        """discord_token is a required positional argument."""
        with pytest.raises(TypeError):
            AppConfig()

    def test_default_assistant_llm_provider_is_gemini(self):
        """assistant_llm_provider defaults to 'gemini'."""
        config = AppConfig(discord_token="test-token")
        assert config.assistant_llm_provider == "gemini"

    def test_default_summarizer_llm_provider_is_gemini(self):
        """summarizer_llm_provider defaults to 'gemini'."""
        config = AppConfig(discord_token="test-token")
        assert config.summarizer_llm_provider == "gemini"

    def test_default_api_keys_are_none(self):
        """API key fields default to None."""
        config = AppConfig(discord_token="test-token")
        assert config.gemini_api_key is None
        assert config.openai_api_key is None
        assert config.zai_api_key is None
        assert config.openrouter_api_key is None
        assert config.weather_api_key is None
        assert config.search_api_key is None

    def test_default_openai_base_url_is_none(self):
        """openai_base_url defaults to None."""
        config = AppConfig(discord_token="test-token")
        assert config.openai_base_url is None

    def test_default_zai_base_url(self):
        """zai_base_url has default value."""
        config = AppConfig(discord_token="test-token")
        assert config.zai_base_url == "https://api.z.ai/api/paas/v4/"

    def test_default_zai_coding_plan_is_false(self):
        """zai_coding_plan defaults to False."""
        config = AppConfig(discord_token="test-token")
        assert config.zai_coding_plan is False

    def test_default_openrouter_image_model(self):
        """openrouter_image_model has default value."""
        config = AppConfig(discord_token="test-token")
        assert config.openrouter_image_model == "black-forest-labs/flux.2-klein-4b"

    def test_default_assistant_model_name(self):
        """assistant_model_name defaults to Gemini Flash."""
        config = AppConfig(discord_token="test-token")
        assert config.assistant_model_name == DEFAULT_GEMINI_ASSISTANT_MODEL

    def test_default_summarizer_model_name(self):
        """summarizer_model_name defaults to Gemini Pro."""
        config = AppConfig(discord_token="test-token")
        assert config.summarizer_model_name == DEFAULT_GEMINI_SUMMARY_MODEL

    def test_default_max_messages_per_fetch(self):
        """max_messages_per_fetch defaults to 300."""
        config = AppConfig(discord_token="test-token")
        assert config.max_messages_per_fetch == 300

    def test_default_api_max_retries(self):
        """api_max_retries defaults to 2."""
        config = AppConfig(discord_token="test-token")
        assert config.api_max_retries == 2

    def test_default_api_rate_limit_retry_after(self):
        """api_rate_limit_retry_after defaults to 5."""
        config = AppConfig(discord_token="test-token")
        assert config.api_rate_limit_retry_after == 5

    def test_default_api_request_timeout(self):
        """api_request_timeout defaults to 120.0 seconds."""
        config = AppConfig(discord_token="test-token")
        assert config.api_request_timeout == 120.0

    def test_default_zai_request_timeout(self):
        """zai_request_timeout defaults to 300.0 seconds (5 minutes)."""
        config = AppConfig(discord_token="test-token")
        assert config.zai_request_timeout == 300.0

    def test_default_api_retry_backoff_base(self):
        """api_retry_backoff_base defaults to 2.0."""
        config = AppConfig(discord_token="test-token")
        assert config.api_retry_backoff_base == 2.0

    def test_default_api_retry_backoff_max(self):
        """api_retry_backoff_max defaults to 32.0."""
        config = AppConfig(discord_token="test-token")
        assert config.api_retry_backoff_max == 32.0

    def test_default_progress_update_interval(self):
        """progress_update_interval defaults to 0.5."""
        config = AppConfig(discord_token="test-token")
        assert config.progress_update_interval == 0.5

    def test_default_countdown_update_interval(self):
        """countdown_update_interval defaults to 5."""
        config = AppConfig(discord_token="test-token")
        assert config.countdown_update_interval == 5

    def test_default_command_prefix(self):
        """command_prefix defaults to '!'."""
        config = AppConfig(discord_token="test-token")
        assert config.command_prefix == "!"

    def test_default_service_tier(self):
        """service_tier defaults to 'flex'."""
        config = AppConfig(discord_token="test-token")
        assert config.service_tier == "flex"

    def test_default_openai_finetuned_model_is_none(self):
        """openai_finetuned_model defaults to None."""
        config = AppConfig(discord_token="test-token")
        assert config.openai_finetuned_model is None

    def test_default_temperature(self):
        """temperature defaults to 1.0."""
        config = AppConfig(discord_token="test-token")
        assert config.temperature == 1.0

    def test_default_top_p(self):
        """top_p defaults to 1.0."""
        config = AppConfig(discord_token="test-token")
        assert config.top_p == 1.0

    def test_default_gemini_cache_min_tokens(self):
        """gemini_cache_min_tokens defaults to 1024."""
        config = AppConfig(discord_token="test-token")
        assert config.gemini_cache_min_tokens == 1024

    def test_default_gemini_cache_ttl_minutes(self):
        """gemini_cache_ttl_minutes defaults to 60."""
        config = AppConfig(discord_token="test-token")
        assert config.gemini_cache_ttl_minutes == 60

    def test_default_thinking_budget_is_none(self):
        """thinking_budget defaults to None."""
        config = AppConfig(discord_token="test-token")
        assert config.thinking_budget is None

    def test_default_max_history(self):
        """max_history defaults to 50."""
        config = AppConfig(discord_token="test-token")
        assert config.max_history == 50

    def test_default_summarization_threshold(self):
        """summarization_threshold defaults to 40."""
        config = AppConfig(discord_token="test-token")
        assert config.summarization_threshold == 40

    def test_default_summarization_keep_recent(self):
        """summarization_keep_recent defaults to 7."""
        config = AppConfig(discord_token="test-token")
        assert config.summarization_keep_recent == 7

    def test_default_summarization_model(self):
        """summarization_model defaults to gemini-2.5-flash."""
        config = AppConfig(discord_token="test-token")
        assert config.summarization_model == "gemini-2.5-flash"

    def test_default_summarization_max_tokens(self):
        """summarization_max_tokens defaults to 500."""
        config = AppConfig(discord_token="test-token")
        assert config.summarization_max_tokens == 500

    def test_default_auto_reply_channel_ids_is_empty(self):
        """auto_reply_channel_ids defaults to empty tuple."""
        config = AppConfig(discord_token="test-token")
        assert config.auto_reply_channel_ids == ()

    def test_default_log_level_is_info(self):
        """log_level defaults to logging.INFO."""
        config = AppConfig(discord_token="test-token")
        assert config.log_level == logging.INFO

    def test_default_session_cache_limit(self):
        """session_cache_limit defaults to 200."""
        config = AppConfig(discord_token="test-token")
        assert config.session_cache_limit == 200

    def test_default_session_inactive_minutes(self):
        """session_inactive_minutes defaults to 30."""
        config = AppConfig(discord_token="test-token")
        assert config.session_inactive_minutes == 30

    def test_default_message_buffer_delay(self):
        """message_buffer_delay defaults to 0.1."""
        config = AppConfig(discord_token="test-token")
        assert config.message_buffer_delay == 0.1

    def test_default_break_cut_mode_is_true(self):
        """break_cut_mode defaults to True."""
        config = AppConfig(discord_token="test-token")
        assert config.break_cut_mode is True

    def test_default_no_check_permission_is_false(self):
        """no_check_permission defaults to False."""
        config = AppConfig(discord_token="test-token")
        assert config.no_check_permission is False

    def test_default_enable_tools_is_true(self):
        """enable_tools defaults to True."""
        config = AppConfig(discord_token="test-token")
        assert config.enable_tools is True

    def test_default_enable_discord_tools_is_true(self):
        """enable_discord_tools defaults to True."""
        config = AppConfig(discord_token="test-token")
        assert config.enable_discord_tools is True

    def test_default_enable_api_tools_is_true(self):
        """enable_api_tools defaults to True."""
        config = AppConfig(discord_token="test-token")
        assert config.enable_api_tools is True

    def test_default_tool_timeout(self):
        """tool_timeout defaults to 10.0 seconds."""
        config = AppConfig(discord_token="test-token")
        assert config.tool_timeout == 10.0

    def test_default_image_rate_limit_per_minute(self):
        """image_rate_limit_per_minute defaults to 3."""
        config = AppConfig(discord_token="test-token")
        assert config.image_rate_limit_per_minute == 3

    def test_default_image_rate_limit_per_hour(self):
        """image_rate_limit_per_hour defaults to 15."""
        config = AppConfig(discord_token="test-token")
        assert config.image_rate_limit_per_hour == 15


class TestResolveLogLevel:
    """Tests for _resolve_log_level function."""

    def test_returns_info_for_none(self):
        """Returns INFO when level is None."""
        result = _resolve_log_level(None)
        assert result == logging.INFO

    def test_returns_info_for_empty_string(self):
        """Returns INFO when level is empty string."""
        result = _resolve_log_level("")
        assert result == logging.INFO

    def test_returns_debug_uppercase(self):
        """Returns DEBUG for 'DEBUG'."""
        result = _resolve_log_level("DEBUG")
        assert result == logging.DEBUG

    def test_returns_debug_lowercase(self):
        """Returns DEBUG for 'debug' (case insensitive)."""
        result = _resolve_log_level("debug")
        assert result == logging.DEBUG

    def test_returns_debug_mixed_case(self):
        """Returns DEBUG for 'DeBuG' (case insensitive)."""
        result = _resolve_log_level("DeBuG")
        assert result == logging.DEBUG

    def test_returns_info_uppercase(self):
        """Returns INFO for 'INFO'."""
        result = _resolve_log_level("INFO")
        assert result == logging.INFO

    def test_returns_warning_uppercase(self):
        """Returns WARNING for 'WARNING'."""
        result = _resolve_log_level("WARNING")
        assert result == logging.WARNING

    def test_returns_error_uppercase(self):
        """Returns ERROR for 'ERROR'."""
        result = _resolve_log_level("ERROR")
        assert result == logging.ERROR

    def test_returns_critical_uppercase(self):
        """Returns CRITICAL for 'CRITICAL'."""
        result = _resolve_log_level("CRITICAL")
        assert result == logging.CRITICAL

    def test_trims_whitespace(self):
        """Trims whitespace from level string."""
        result = _resolve_log_level("  DEBUG  ")
        assert result == logging.DEBUG

    def test_defaults_to_info_for_unknown_level(self):
        """Defaults to INFO for unknown level strings."""
        result = _resolve_log_level("INVALID")
        assert result == logging.INFO

    def test_defaults_to_info_for_numeric_string(self):
        """Defaults to INFO for numeric string."""
        result = _resolve_log_level("123")
        assert result == logging.INFO


class TestNormalizeProvider:
    """Tests for _normalize_provider function."""

    def test_returns_provider_when_valid(self):
        """Returns provider when valid and non-empty."""
        result = _normalize_provider("gemini", "openai")
        assert result == "gemini"

    def test_returns_default_when_none(self):
        """Returns default when provider is None."""
        result = _normalize_provider(None, "openai")
        assert result == "openai"

    def test_returns_default_when_empty(self):
        """Returns default when provider is empty string."""
        result = _normalize_provider("", "openai")
        assert result == "openai"

    def test_returns_default_when_whitespace_only(self):
        """Returns default when provider is whitespace only."""
        result = _normalize_provider("   ", "openai")
        assert result == "openai"

    def test_converts_to_lowercase(self):
        """Converts provider string to lowercase."""
        result = _normalize_provider("GEMINI", "openai")
        assert result == "gemini"

    def test_trims_whitespace(self):
        """Trims whitespace from provider string."""
        result = _normalize_provider("  gemini  ", "openai")
        assert result == "gemini"

    def test_trims_and_lowercases(self):
        """Trims and lowercases provider string."""
        result = _normalize_provider("  GEMINI  ", "openai")
        assert result == "gemini"


class TestValidateProvider:
    """Tests for _validate_provider function."""

    def test_returns_gemini(self):
        """Returns 'gemini' for valid gemini provider."""
        result = _validate_provider("gemini")
        assert result == "gemini"

    def test_returns_openai(self):
        """Returns 'openai' for valid openai provider."""
        result = _validate_provider("openai")
        assert result == "openai"

    def test_returns_zai(self):
        """Returns 'zai' for valid zai provider."""
        result = _validate_provider("zai")
        assert result == "zai"

    def test_exits_on_invalid_provider(self):
        """Calls sys.exit(1) for invalid provider."""
        with patch("sys.exit") as mock_exit:
            _validate_provider("invalid")
            mock_exit.assert_called_once_with(1)


class TestFirstNonemptyEnv:
    """Tests for _first_nonempty_env function."""

    def test_returns_value_of_first_env(self):
        """Returns value of first environment variable if set."""
        with patch.dict(os.environ, {"TEST_VAR1": "value1"}):
            result = _first_nonempty_env("TEST_VAR1", "TEST_VAR2")
            assert result == "value1"

    def test_returns_value_of_second_env_if_first_empty(self):
        """Returns value of second env if first is not set."""
        with patch.dict(os.environ, {"TEST_VAR2": "value2"}, clear=False):
            result = _first_nonempty_env("TEST_VAR1", "TEST_VAR2")
            assert result == "value2"

    def test_skips_empty_string_values(self):
        """Skips environment variables with empty string values."""
        with patch.dict(os.environ, {"TEST_VAR1": "", "TEST_VAR2": "value2"}):
            result = _first_nonempty_env("TEST_VAR1", "TEST_VAR2")
            assert result == "value2"

    def test_skips_whitespace_only_values(self):
        """Skips environment variables with whitespace only."""
        with patch.dict(os.environ, {"TEST_VAR1": "   ", "TEST_VAR2": "value2"}):
            result = _first_nonempty_env("TEST_VAR1", "TEST_VAR2")
            assert result == "value2"

    def test_returns_none_if_all_unset(self):
        """Returns None if all environment variables are unset."""
        result = _first_nonempty_env("NONEXISTENT1", "NONEXISTENT2")
        assert result is None

    def test_trims_whitespace_from_values(self):
        """Trims whitespace from returned value."""
        with patch.dict(os.environ, {"TEST_VAR1": "  value1  "}):
            result = _first_nonempty_env("TEST_VAR1")
            assert result == "value1"


class TestParseFloatEnv:
    """Tests for _parse_float_env function."""

    def test_returns_default_when_env_not_set(self):
        """Returns default when environment variable is not set."""
        result = _parse_float_env("NONEXISTENT_VAR", 5.0)
        assert result == 5.0

    def test_returns_float_value(self):
        """Returns float value from environment variable."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = _parse_float_env("TEST_FLOAT", 0.0)
            assert result == 3.14

    def test_returns_integer_as_float(self):
        """Returns integer value as float."""
        with patch.dict(os.environ, {"TEST_FLOAT": "42"}):
            result = _parse_float_env("TEST_FLOAT", 0.0)
            assert result == 42.0

    def test_returns_default_on_invalid_value(self):
        """Returns default when environment value is invalid."""
        with patch.dict(os.environ, {"TEST_FLOAT": "invalid"}):
            result = _parse_float_env("TEST_FLOAT", 5.0)
            assert result == 5.0

    def test_returns_zero(self):
        """Returns zero when env is '0'."""
        with patch.dict(os.environ, {"TEST_FLOAT": "0"}):
            result = _parse_float_env("TEST_FLOAT", 1.0)
            assert result == 0.0

    def test_returns_negative_float(self):
        """Returns negative float value."""
        with patch.dict(os.environ, {"TEST_FLOAT": "-2.5"}):
            result = _parse_float_env("TEST_FLOAT", 0.0)
            assert result == -2.5


class TestParseIntEnv:
    """Tests for _parse_int_env function."""

    def test_returns_default_when_env_not_set(self):
        """Returns default when environment variable is not set."""
        result = _parse_int_env("NONEXISTENT_VAR", 10)
        assert result == 10

    def test_returns_int_value(self):
        """Returns int value from environment variable."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = _parse_int_env("TEST_INT", 0)
            assert result == 42

    def test_returns_negative_int(self):
        """Returns negative int value."""
        with patch.dict(os.environ, {"TEST_INT": "-10"}):
            result = _parse_int_env("TEST_INT", 0)
            assert result == -10

    def test_returns_default_on_invalid_value(self):
        """Returns default when environment value is invalid."""
        with patch.dict(os.environ, {"TEST_INT": "invalid"}):
            result = _parse_int_env("TEST_INT", 10)
            assert result == 10

    def test_returns_default_on_float_value(self):
        """Returns default when env value is a float string."""
        with patch.dict(os.environ, {"TEST_INT": "3.99"}):
            result = _parse_int_env("TEST_INT", 0)
            assert result == 0

    def test_returns_zero(self):
        """Returns zero when env is '0'."""
        with patch.dict(os.environ, {"TEST_INT": "0"}):
            result = _parse_int_env("TEST_INT", 10)
            assert result == 0


class TestParseBoolEnv:
    """Tests for _parse_bool_env function."""

    def test_returns_default_when_env_not_set(self):
        """Returns default when environment variable is not set."""
        result = _parse_bool_env("NONEXISTENT_VAR", True)
        assert result is True

    def test_returns_true_for_true_lower(self):
        """Returns True for 'true'."""
        with patch.dict(os.environ, {"TEST_BOOL": "true"}):
            result = _parse_bool_env("TEST_BOOL", False)
            assert result is True

    def test_returns_true_for_true_upper(self):
        """Returns True for 'TRUE'."""
        with patch.dict(os.environ, {"TEST_BOOL": "TRUE"}):
            result = _parse_bool_env("TEST_BOOL", False)
            assert result is True

    def test_returns_true_for_true_mixed(self):
        """Returns True for 'True'."""
        with patch.dict(os.environ, {"TEST_BOOL": "True"}):
            result = _parse_bool_env("TEST_BOOL", False)
            assert result is True

    def test_returns_true_for_1(self):
        """Returns True for '1'."""
        with patch.dict(os.environ, {"TEST_BOOL": "1"}):
            result = _parse_bool_env("TEST_BOOL", False)
            assert result is True

    def test_returns_true_for_yes(self):
        """Returns True for 'yes'."""
        with patch.dict(os.environ, {"TEST_BOOL": "yes"}):
            result = _parse_bool_env("TEST_BOOL", False)
            assert result is True

    def test_returns_true_for_yes_upper(self):
        """Returns True for 'YES'."""
        with patch.dict(os.environ, {"TEST_BOOL": "YES"}):
            result = _parse_bool_env("TEST_BOOL", False)
            assert result is True

    def test_returns_false_for_false(self):
        """Returns False for 'false'."""
        with patch.dict(os.environ, {"TEST_BOOL": "false"}):
            result = _parse_bool_env("TEST_BOOL", True)
            assert result is False

    def test_returns_false_for_0(self):
        """Returns False for '0'."""
        with patch.dict(os.environ, {"TEST_BOOL": "0"}):
            result = _parse_bool_env("TEST_BOOL", True)
            assert result is False

    def test_returns_false_for_no(self):
        """Returns False for 'no'."""
        with patch.dict(os.environ, {"TEST_BOOL": "no"}):
            result = _parse_bool_env("TEST_BOOL", True)
            assert result is False

    def test_returns_false_for_arbitrary_string(self):
        """Returns False for arbitrary string."""
        with patch.dict(os.environ, {"TEST_BOOL": "random"}):
            result = _parse_bool_env("TEST_BOOL", True)
            assert result is False

    def test_trims_whitespace(self):
        """Trims whitespace from value."""
        with patch.dict(os.environ, {"TEST_BOOL": "  true  "}):
            result = _parse_bool_env("TEST_BOOL", False)
            assert result is True


class TestParseThinkingBudget:
    """Tests for _parse_thinking_budget function."""

    def test_returns_none_for_off(self):
        """Returns None for 'off' string."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "off"}):
            result = _parse_thinking_budget()
            assert result is None

    def test_returns_none_for_off_uppercase(self):
        """Returns None for 'OFF' string."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "OFF"}):
            result = _parse_thinking_budget()
            assert result is None

    def test_returns_none_for_off_mixed_case(self):
        """Returns None for 'Off' string."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "Off"}):
            result = _parse_thinking_budget()
            assert result is None

    def test_returns_none_for_off_with_whitespace(self):
        """Returns None for '  off  '."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "  off  "}):
            result = _parse_thinking_budget()
            assert result is None

    def test_returns_none_when_not_set(self):
        """Returns None when environment variable is not set (default 'off')."""
        result = _parse_thinking_budget()
        assert result is None

    def test_returns_int_for_valid_number(self):
        """Returns int value for valid number string."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "1000"}):
            result = _parse_thinking_budget()
            assert result == 1000

    def test_returns_negative_int(self):
        """Returns negative int value."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "-500"}):
            result = _parse_thinking_budget()
            assert result == -500

    def test_returns_none_for_invalid_value(self):
        """Returns None for invalid value (non-numeric)."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "invalid"}):
            result = _parse_thinking_budget()
            assert result is None


class TestParseAutoChannelIds:
    """Tests for _parse_auto_channel_ids function."""

    def test_returns_empty_tuple_when_not_set(self):
        """Returns empty tuple when env var is not set."""
        with patch.dict(os.environ, {}, clear=False):
            if "AUTO_REPLY_CHANNEL_IDS" in os.environ:
                del os.environ["AUTO_REPLY_CHANNEL_IDS"]
            result = _parse_auto_channel_ids()
            assert result == ()

    def test_returns_empty_tuple_for_empty_string(self):
        """Returns empty tuple for empty string."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": ""}):
            result = _parse_auto_channel_ids()
            assert result == ()

    def test_returns_single_channel_id(self):
        """Returns tuple with single channel ID."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": "123456789"}):
            result = _parse_auto_channel_ids()
            assert result == (123456789,)

    def test_returns_multiple_channel_ids(self):
        """Returns tuple with multiple channel IDs."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": "123,456,789"}):
            result = _parse_auto_channel_ids()
            assert result == (123, 456, 789)

    def test_ignores_whitespace(self):
        """Ignores whitespace around IDs."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": " 123 , 456 , 789 "}):
            result = _parse_auto_channel_ids()
            assert result == (123, 456, 789)

    def test_ignores_empty_entries(self):
        """Ignores empty entries between commas."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": "123,,456"}):
            result = _parse_auto_channel_ids()
            assert result == (123, 456)

    def test_skips_invalid_ids(self):
        """Skips invalid (non-numeric) IDs."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": "123,invalid,456"}):
            result = _parse_auto_channel_ids()
            assert result == (123, 456)

    def test_returns_negative_ids(self):
        """Returns negative channel IDs (though unusual)."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": "-123,-456"}):
            result = _parse_auto_channel_ids()
            assert result == (-123, -456)


class TestResolveModelName:
    """Tests for _resolve_model_name function."""

    def test_gemini_assistant_default(self):
        """Returns default Gemini assistant model for gemini provider."""
        result = _resolve_model_name("gemini", role="assistant")
        assert result == DEFAULT_GEMINI_ASSISTANT_MODEL

    def test_gemini_summary_default(self):
        """Returns default Gemini summary model for gemini provider."""
        result = _resolve_model_name("gemini", role="summary")
        assert result == DEFAULT_GEMINI_SUMMARY_MODEL

    def test_openai_assistant_default(self):
        """Returns default OpenAI assistant model for openai provider."""
        result = _resolve_model_name("openai", role="assistant")
        assert result == DEFAULT_OPENAI_ASSISTANT_MODEL

    def test_openai_summary_default(self):
        """Returns default OpenAI summary model for openai provider."""
        result = _resolve_model_name("openai", role="summary")
        assert result == DEFAULT_OPENAI_SUMMARY_MODEL

    def test_zai_assistant_default(self):
        """Returns default Z.AI assistant model for zai provider."""
        result = _resolve_model_name("zai", role="assistant")
        assert result == DEFAULT_ZAI_ASSISTANT_MODEL

    def test_zai_summary_default(self):
        """Returns default Z.AI summary model for zai provider."""
        result = _resolve_model_name("zai", role="summary")
        assert result == DEFAULT_ZAI_SUMMARY_MODEL

    def test_gemini_assistant_env_override(self):
        """Respects GEMINI_ASSISTANT_MODEL_NAME env override."""
        with patch.dict(os.environ, {"GEMINI_ASSISTANT_MODEL_NAME": "custom-gemini-model"}):
            result = _resolve_model_name("gemini", role="assistant")
            assert result == "custom-gemini-model"

    def test_gemini_summary_env_override(self):
        """Respects GEMINI_SUMMARY_MODEL_NAME env override."""
        with patch.dict(os.environ, {"GEMINI_SUMMARY_MODEL_NAME": "custom-gemini-summary"}):
            result = _resolve_model_name("gemini", role="summary")
            assert result == "custom-gemini-summary"

    def test_openai_assistant_env_override(self):
        """Respects OPENAI_ASSISTANT_MODEL_NAME env override."""
        with patch.dict(os.environ, {"OPENAI_ASSISTANT_MODEL_NAME": "custom-openai-model"}):
            result = _resolve_model_name("openai", role="assistant")
            assert result == "custom-openai-model"

    def test_openai_summary_env_override(self):
        """Respects OPENAI_SUMMARY_MODEL_NAME env override."""
        with patch.dict(os.environ, {"OPENAI_SUMMARY_MODEL_NAME": "custom-openai-summary"}):
            result = _resolve_model_name("openai", role="summary")
            assert result == "custom-openai-summary"

    def test_zai_assistant_env_override(self):
        """Respects ZAI_ASSISTANT_MODEL_NAME env override."""
        with patch.dict(os.environ, {"ZAI_ASSISTANT_MODEL_NAME": "custom-zai-model"}):
            result = _resolve_model_name("zai", role="assistant")
            assert result == "custom-zai-model"

    def test_zai_summary_env_override(self):
        """Respects ZAI_SUMMARY_MODEL_NAME env override."""
        with patch.dict(os.environ, {"ZAI_SUMMARY_MODEL_NAME": "custom-zai-summary"}):
            result = _resolve_model_name("zai", role="summary")
            assert result == "custom-zai-summary"

    def test_unknown_provider_defaults_to_gemini(self):
        """Defaults to Gemini models for unknown provider."""
        result = _resolve_model_name("unknown", role="assistant")
        assert result == DEFAULT_GEMINI_ASSISTANT_MODEL


class TestLoadConfig:
    """Tests for load_config function."""

    @pytest.fixture
    def minimal_env(self):
        """Provide minimal environment for load_config."""
        return {
            "DISCORD_TOKEN": "test-discord-token",
            "GEMINI_API_KEY": "test-gemini-key",
        }

    def test_loads_minimal_config(self, minimal_env):
        """Loads config with minimal required environment."""
        with patch.dict(os.environ, minimal_env, clear=False):
            config = load_config()
            assert config.discord_token == "test-discord-token"
            assert config.gemini_api_key == "test-gemini-key"
            assert config.assistant_llm_provider == "gemini"
            assert config.summarizer_llm_provider == "gemini"

    def test_exits_without_discord_token(self):
        """Exits when DISCORD_TOKEN is not set."""
        with patch.dict(os.environ, {}, clear=False):
            if "DISCORD_TOKEN" in os.environ:
                del os.environ["DISCORD_TOKEN"]
        with patch("sys.exit") as mock_exit:
            load_config()
            mock_exit.assert_called_with(1)

    def test_exits_without_gemini_key_when_using_gemini(self):
        """Exits when GEMINI_API_KEY is not set but gemini is provider."""
        with patch.dict(os.environ, {"DISCORD_TOKEN": "test"}, clear=False):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
        with patch("sys.exit") as mock_exit:
            load_config()
            mock_exit.assert_called_with(1)

    def test_exits_without_openai_key_when_using_openai(self):
        """Exits when OPENAI_API_KEY is not set but openai is provider."""
        env = {
            "DISCORD_TOKEN": "test",
            "OPENAI_API_KEY": "",
            "ASSISTANT_LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("sys.exit") as mock_exit:
                load_config()
                mock_exit.assert_called_once_with(1)

    def test_exits_without_zai_key_when_using_zai(self):
        """Exits when ZAI_API_KEY is not set but zai is provider."""
        env = {
            "DISCORD_TOKEN": "test",
            "ZAI_API_KEY": "",
            "ASSISTANT_LLM_PROVIDER": "zai",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("sys.exit") as mock_exit:
                load_config()
                mock_exit.assert_called_once_with(1)

    def test_loads_openai_base_url(self, minimal_env):
        """Loads OPENAI_BASE_URL from environment."""
        env = {**minimal_env, "OPENAI_BASE_URL": "https://custom.openai.com"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.openai_base_url == "https://custom.openai.com"

    def test_loads_zai_base_url(self, minimal_env):
        """Loads ZAI_BASE_URL from environment."""
        env = {**minimal_env, "ZAI_BASE_URL": "https://custom.z.ai"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.zai_base_url == "https://custom.z.ai"

    def test_uses_coding_plan_base_url_when_enabled(self, minimal_env):
        """Uses coding plan API endpoint when ZAI_CODING_PLAN is enabled."""
        env = {**minimal_env, "ZAI_CODING_PLAN": "true", "ZAI_API_KEY": "test-zai"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.zai_coding_plan is True
            assert config.zai_base_url == "https://api.z.ai/api/coding/paas/v4/"

    def test_respects_base_url_env_over_coding_plan(self, minimal_env):
        """ZAI_BASE_URL env overrides coding plan default."""
        env = {
            **minimal_env,
            "ZAI_CODING_PLAN": "true",
            "ZAI_BASE_URL": "https://custom.z.ai",
            "ZAI_API_KEY": "test-zai",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.zai_base_url == "https://custom.z.ai"

    def test_loads_openrouter_api_key(self, minimal_env):
        """Loads OPENROUTER_API_KEY from environment."""
        env = {**minimal_env, "OPENROUTER_API_KEY": "test-openrouter-key"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.openrouter_api_key == "test-openrouter-key"

    def test_loads_custom_openrouter_image_model(self, minimal_env):
        """Loads OPENROUTER_IMAGE_MODEL from environment."""
        env = {**minimal_env, "OPENROUTER_IMAGE_MODEL": "custom/image-model"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.openrouter_image_model == "custom/image-model"

    def test_loads_service_tier(self, minimal_env):
        """Loads SERVICE_TIER from environment."""
        env = {**minimal_env, "SERVICE_TIER": "premium"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.service_tier == "premium"

    def test_loads_openai_finetuned_model(self, minimal_env):
        """Loads OPENAI_FINETUNED_MODEL from environment."""
        env = {
            **minimal_env,
            "OPENAI_FINETUNED_MODEL": "ft:gpt-5:custom:abc123",
            "OPENAI_API_KEY": "test-openai",
            "ASSISTANT_LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.openai_finetuned_model == "ft:gpt-5:custom:abc123"

    def test_openai_finetuned_overrides_assistant_model(self, minimal_env):
        """OpenAI finetuned model overrides assistant model name."""
        env = {
            **minimal_env,
            "OPENAI_FINETUNED_MODEL": "ft:gpt-5:custom:abc123",
            "OPENAI_API_KEY": "test-openai",
            "ASSISTANT_LLM_PROVIDER": "openai",
            "OPENAI_ASSISTANT_MODEL_NAME": "gpt-5-mini",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.assistant_model_name == "ft:gpt-5:custom:abc123"

    def test_loads_assistant_llm_provider(self, minimal_env):
        """Loads ASSISTANT_LLM_PROVIDER from environment."""
        env = {
            **minimal_env,
            "ASSISTANT_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.assistant_llm_provider == "openai"

    def test_loads_summarizer_llm_provider(self, minimal_env):
        """Loads SUMMARIZER_LLM_PROVIDER from environment."""
        env = {
            **minimal_env,
            "SUMMARIZER_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.summarizer_llm_provider == "openai"

    def test_summarizer_defaults_to_assistant_provider(self, minimal_env):
        """Summarizer provider defaults to assistant provider."""
        env = {
            **minimal_env,
            "ASSISTANT_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.summarizer_llm_provider == "openai"

    def test_loads_message_buffer_delay(self, minimal_env):
        """Loads MESSAGE_BUFFER_DELAY from environment."""
        env = {**minimal_env, "MESSAGE_BUFFER_DELAY": "5.0"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.message_buffer_delay == 5.0

    def test_loads_temperature(self, minimal_env):
        """Loads TEMPERATURE from environment."""
        env = {**minimal_env, "TEMPERATURE": "0.7"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.temperature == 0.7

    def test_loads_top_p(self, minimal_env):
        """Loads TOP_P from environment."""
        env = {**minimal_env, "TOP_P": "0.9"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.top_p == 0.9

    def test_loads_thinking_budget(self, minimal_env):
        """Loads THINKING_BUDGET from environment."""
        env = {**minimal_env, "THINKING_BUDGET": "5000"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.thinking_budget == 5000

    def test_loads_thinking_budget_off(self, minimal_env):
        """Loads THINKING_BUDGET=off as None."""
        env = {**minimal_env, "THINKING_BUDGET": "off"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.thinking_budget is None

    def test_loads_max_history(self, minimal_env):
        """Loads MAX_HISTORY from environment."""
        env = {**minimal_env, "MAX_HISTORY": "100"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.max_history == 100

    def test_loads_gemini_cache_min_tokens(self, minimal_env):
        """Loads GEMINI_CACHE_MIN_TOKENS from environment."""
        env = {**minimal_env, "GEMINI_CACHE_MIN_TOKENS": "2048"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.gemini_cache_min_tokens == 2048

    def test_loads_gemini_cache_ttl_minutes(self, minimal_env):
        """Loads GEMINI_CACHE_TTL_MINUTES from environment."""
        env = {**minimal_env, "GEMINI_CACHE_TTL_MINUTES": "120"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.gemini_cache_ttl_minutes == 120

    def test_loads_no_check_permission(self, minimal_env):
        """Loads NO_CHECK_PERMISSION from environment."""
        env = {**minimal_env, "NO_CHECK_PERMISSION": "true"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.no_check_permission is True

    def test_loads_enable_tools(self, minimal_env):
        """Loads ENABLE_TOOLS from environment."""
        env = {**minimal_env, "ENABLE_TOOLS": "false"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.enable_tools is False

    def test_loads_enable_discord_tools(self, minimal_env):
        """Loads ENABLE_DISCORD_TOOLS from environment."""
        env = {**minimal_env, "ENABLE_DISCORD_TOOLS": "false"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.enable_discord_tools is False

    def test_loads_enable_api_tools(self, minimal_env):
        """Loads ENABLE_API_TOOLS from environment."""
        env = {**minimal_env, "ENABLE_API_TOOLS": "false"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.enable_api_tools is False

    def test_loads_tool_timeout(self, minimal_env):
        """Loads TOOL_TIMEOUT from environment."""
        env = {**minimal_env, "TOOL_TIMEOUT": "30.0"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.tool_timeout == 30.0

    def test_loads_weather_api_key(self, minimal_env):
        """Loads WEATHER_API_KEY from environment."""
        env = {**minimal_env, "WEATHER_API_KEY": "test-weather-key"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.weather_api_key == "test-weather-key"

    def test_loads_search_api_key(self, minimal_env):
        """Loads SEARCH_API_KEY from environment."""
        env = {**minimal_env, "SEARCH_API_KEY": "test-search-key"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.search_api_key == "test-search-key"

    def test_loads_api_request_timeout(self, minimal_env):
        """Loads API_REQUEST_TIMEOUT from environment."""
        env = {**minimal_env, "API_REQUEST_TIMEOUT": "180.0"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.api_request_timeout == 180.0

    def test_loads_zai_request_timeout(self, minimal_env):
        """Loads ZAI_REQUEST_TIMEOUT from environment."""
        env = {**minimal_env, "ZAI_REQUEST_TIMEOUT": "600.0"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.zai_request_timeout == 600.0

    def test_loads_auto_reply_channel_ids(self, minimal_env):
        """Loads AUTO_REPLY_CHANNEL_IDS from environment."""
        env = {**minimal_env, "AUTO_REPLY_CHANNEL_IDS": "123,456,789"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.auto_reply_channel_ids == (123, 456, 789)

    def test_loads_log_level(self, minimal_env):
        """Loads LOG_LEVEL from environment."""
        env = {**minimal_env, "LOG_LEVEL": "DEBUG"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.log_level == logging.DEBUG

    def test_loads_image_rate_limit_per_minute(self, minimal_env):
        """Loads IMAGE_RATE_LIMIT_PER_MINUTE from environment."""
        env = {**minimal_env, "IMAGE_RATE_LIMIT_PER_MINUTE": "5"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.image_rate_limit_per_minute == 5

    def test_loads_image_rate_limit_per_hour(self, minimal_env):
        """Loads IMAGE_RATE_LIMIT_PER_HOUR from environment."""
        env = {**minimal_env, "IMAGE_RATE_LIMIT_PER_HOUR": "30"}
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.image_rate_limit_per_hour == 30

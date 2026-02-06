"""Comprehensive tests for AppConfig configuration loading and validation."""

import os
import pytest
import logging
from unittest.mock import patch
import sys

from soyebot.config import (
    AppConfig,
    load_config,
    _normalize_provider,
    _validate_provider,
    _first_nonempty_env,
    _parse_float_env,
    _parse_int_env,
    _parse_thinking_budget,
    _parse_auto_channel_ids,
    _resolve_model_name,
    _resolve_log_level,
    DEFAULT_GEMINI_ASSISTANT_MODEL,
    DEFAULT_GEMINI_SUMMARY_MODEL,
    DEFAULT_OPENAI_ASSISTANT_MODEL,
    DEFAULT_OPENAI_SUMMARY_MODEL,
    DEFAULT_ZAI_ASSISTANT_MODEL,
    DEFAULT_ZAI_SUMMARY_MODEL,
)


class TestNormalizeProvider:
    """Tests for _normalize_provider helper function."""

    def test_normalize_provider_with_none(self):
        """Test normalize_provider with None returns default."""
        assert _normalize_provider(None, "gemini") == "gemini"

    def test_normalize_provider_with_empty_string(self):
        """Test normalize_provider with empty string returns default."""
        assert _normalize_provider("", "gemini") == "gemini"
        assert _normalize_provider("   ", "gemini") == "gemini"

    def test_normalize_provider_with_whitespace(self):
        """Test normalize_provider strips whitespace."""
        assert _normalize_provider("  gemini  ", "openai") == "gemini"
        assert _normalize_provider("\topenai\n", "gemini") == "openai"

    def test_normalize_provider_lowercases(self):
        """Test normalize_provider converts to lowercase."""
        assert _normalize_provider("GEMINI", "openai") == "gemini"
        assert _normalize_provider("OpenAI", "gemini") == "openai"
        assert _normalize_provider("ZAI", "gemini") == "zai"


class TestValidateProvider:
    """Tests for _validate_provider helper function."""

    def test_validate_provider_gemini(self):
        """Test validate_provider accepts 'gemini'."""
        assert _validate_provider("gemini") == "gemini"

    def test_validate_provider_openai(self):
        """Test validate_provider accepts 'openai'."""
        assert _validate_provider("openai") == "openai"

    def test_validate_provider_zai(self):
        """Test validate_provider accepts 'zai'."""
        assert _validate_provider("zai") == "zai"

    def test_validate_provider_invalid(self, caplog):
        """Test validate_provider exits on invalid provider."""
        with patch.object(sys, "exit") as mock_exit:
            _validate_provider("invalid_provider")
            mock_exit.assert_called_once_with(1)


class TestFirstNonemptyEnv:
    """Tests for _first_nonempty_env helper function."""

    def test_first_nonempty_env_returns_first_value(self):
        """Test returns first non-empty environment variable."""
        with patch.dict(os.environ, {"VAR1": "value1", "VAR2": "value2"}):
            assert _first_nonempty_env("VAR1", "VAR2") == "value1"

    def test_first_nonempty_env_skips_empty(self):
        """Test skips empty environment variables."""
        with patch.dict(os.environ, {"VAR1": "", "VAR2": "value2", "VAR3": "value3"}):
            assert _first_nonempty_env("VAR1", "VAR2", "VAR3") == "value2"

    def test_first_nonempty_env_skips_whitespace_only(self):
        """Test skips whitespace-only environment variables."""
        with patch.dict(os.environ, {"VAR1": "   ", "VAR2": "value2"}):
            assert _first_nonempty_env("VAR1", "VAR2") == "value2"

    def test_first_nonempty_env_returns_none(self):
        """Test returns None when all variables are empty/missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert _first_nonempty_env("VAR1", "VAR2") is None


class TestParseFloatEnv:
    """Tests for _parse_float_env helper function."""

    def test_parse_float_env_with_valid_value(self):
        """Test parsing valid float value."""
        with patch.dict(os.environ, {"FLOAT_VAR": "3.14"}):
            assert _parse_float_env("FLOAT_VAR", 1.0) == 3.14

    def test_parse_float_env_with_integer_value(self):
        """Test parsing integer value as float."""
        with patch.dict(os.environ, {"FLOAT_VAR": "42"}):
            assert _parse_float_env("FLOAT_VAR", 1.0) == 42.0

    def test_parse_float_env_with_missing_var(self):
        """Test returns default when variable missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert _parse_float_env("FLOAT_VAR", 1.5) == 1.5

    def test_parse_float_env_with_invalid_value(self, caplog):
        """Test returns default on invalid value and logs warning."""
        with patch.dict(os.environ, {"FLOAT_VAR": "invalid"}):
            assert _parse_float_env("FLOAT_VAR", 1.5) == 1.5
            assert any("FLOAT_VAR" in record.message for record in caplog.records)


class TestParseIntEnv:
    """Tests for _parse_int_env helper function."""

    def test_parse_int_env_with_valid_value(self):
        """Test parsing valid integer value."""
        with patch.dict(os.environ, {"INT_VAR": "42"}):
            assert _parse_int_env("INT_VAR", 10) == 42

    def test_parse_int_env_with_negative_value(self):
        """Test parsing negative integer value."""
        with patch.dict(os.environ, {"INT_VAR": "-10"}):
            assert _parse_int_env("INT_VAR", 0) == -10

    def test_parse_int_env_with_missing_var(self):
        """Test returns default when variable missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert _parse_int_env("INT_VAR", 10) == 10

    def test_parse_int_env_with_invalid_value(self, caplog):
        """Test returns default on invalid value and logs warning."""
        with patch.dict(os.environ, {"INT_VAR": "not_a_number"}):
            assert _parse_int_env("INT_VAR", 10) == 10
            assert any("INT_VAR" in record.message for record in caplog.records)


class TestParseThinkingBudget:
    """Tests for _parse_thinking_budget helper function."""

    def test_parse_thinking_budget_with_number(self):
        """Test parsing thinking budget with number."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "65000"}):
            assert _parse_thinking_budget() == 65000

    def test_parse_thinking_budget_with_off(self):
        """Test parsing thinking budget with 'off' returns None."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "off"}):
            assert _parse_thinking_budget() is None

    def test_parse_thinking_budget_with_off_case_insensitive(self):
        """Test parsing thinking budget with 'OFF' returns None."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "OFF"}):
            assert _parse_thinking_budget() is None

    def test_parse_thinking_budget_with_missing_var(self):
        """Test default to 'off' when variable missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert _parse_thinking_budget() is None

    def test_parse_thinking_budget_with_invalid_value(self, caplog):
        """Test returns None on invalid value and logs warning."""
        with patch.dict(os.environ, {"THINKING_BUDGET": "invalid"}):
            assert _parse_thinking_budget() is None
            assert any("THINKING_BUDGET" in record.message for record in caplog.records)


class TestParseAutoChannelIds:
    """Tests for _parse_auto_channel_ids helper function."""

    def test_parse_auto_channel_ids_empty(self):
        """Test parsing empty string returns empty tuple."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": ""}):
            assert _parse_auto_channel_ids() == ()

    def test_parse_auto_channel_ids_single(self):
        """Test parsing single channel ID."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": "123456789"}):
            assert _parse_auto_channel_ids() == (123456789,)

    def test_parse_auto_channel_ids_multiple(self):
        """Test parsing multiple channel IDs."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": "123,456,789"}):
            assert _parse_auto_channel_ids() == (123, 456, 789)

    def test_parse_auto_channel_ids_with_whitespace(self):
        """Test parsing channel IDs with whitespace."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": " 123 , 456 , 789 "}):
            assert _parse_auto_channel_ids() == (123, 456, 789)

    def test_parse_auto_channel_ids_with_invalid(self, caplog):
        """Test parsing with invalid IDs filters them out."""
        with patch.dict(os.environ, {"AUTO_REPLY_CHANNEL_IDS": "123,invalid,456"}):
            assert _parse_auto_channel_ids() == (123, 456)
            assert any(
                "AUTO_REPLY_CHANNEL_IDS" in record.message for record in caplog.records
            )

    def test_parse_auto_channel_ids_missing(self):
        """Test returns empty tuple when variable missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert _parse_auto_channel_ids() == ()


class TestResolveModelName:
    """Tests for _resolve_model_name helper function."""

    def test_resolve_model_name_gemini_assistant(self):
        """Test resolving Gemini assistant model name."""
        with patch.dict(os.environ, {}, clear=True):
            assert (
                _resolve_model_name("gemini", role="assistant")
                == DEFAULT_GEMINI_ASSISTANT_MODEL
            )

    def test_resolve_model_name_gemini_summary(self):
        """Test resolving Gemini summary model name."""
        with patch.dict(os.environ, {}, clear=True):
            assert (
                _resolve_model_name("gemini", role="summary")
                == DEFAULT_GEMINI_SUMMARY_MODEL
            )

    def test_resolve_model_name_openai_assistant(self):
        """Test resolving OpenAI assistant model name."""
        with patch.dict(os.environ, {}, clear=True):
            assert (
                _resolve_model_name("openai", role="assistant")
                == DEFAULT_OPENAI_ASSISTANT_MODEL
            )

    def test_resolve_model_name_openai_summary(self):
        """Test resolving OpenAI summary model name."""
        with patch.dict(os.environ, {}, clear=True):
            assert (
                _resolve_model_name("openai", role="summary")
                == DEFAULT_OPENAI_SUMMARY_MODEL
            )

    def test_resolve_model_name_zai_assistant(self):
        """Test resolving ZAI assistant model name."""
        with patch.dict(os.environ, {}, clear=True):
            assert (
                _resolve_model_name("zai", role="assistant")
                == DEFAULT_ZAI_ASSISTANT_MODEL
            )

    def test_resolve_model_name_zai_summary(self):
        """Test resolving ZAI summary model name."""
        with patch.dict(os.environ, {}, clear=True):
            assert (
                _resolve_model_name("zai", role="summary") == DEFAULT_ZAI_SUMMARY_MODEL
            )

    def test_resolve_model_name_gemini_override(self):
        """Test overriding Gemini model name with env var."""
        with patch.dict(
            os.environ, {"GEMINI_ASSISTANT_MODEL_NAME": "custom-gemini-model"}
        ):
            assert (
                _resolve_model_name("gemini", role="assistant") == "custom-gemini-model"
            )

    def test_resolve_model_name_openai_override(self):
        """Test overriding OpenAI model name with env var."""
        with patch.dict(
            os.environ, {"OPENAI_SUMMARY_MODEL_NAME": "custom-openai-model"}
        ):
            assert (
                _resolve_model_name("openai", role="summary") == "custom-openai-model"
            )

    def test_resolve_model_name_zai_override(self):
        """Test overriding ZAI model name with env var."""
        with patch.dict(os.environ, {"ZAI_ASSISTANT_MODEL_NAME": "custom-zai-model"}):
            assert _resolve_model_name("zai", role="assistant") == "custom-zai-model"


class TestResolveLogLevel:
    """Tests for _resolve_log_level helper function."""

    def test_resolve_log_level_none(self):
        """Test resolve_log_level with None returns INFO."""
        assert _resolve_log_level(None) == logging.INFO  # type: ignore

    def test_resolve_log_level_empty(self):
        """Test resolve_log_level with empty string returns INFO."""
        assert _resolve_log_level("") == logging.INFO

    def test_resolve_log_level_debug(self):
        """Test resolve_log_level with DEBUG."""
        assert _resolve_log_level("DEBUG") == logging.DEBUG

    def test_resolve_log_level_info(self):
        """Test resolve_log_level with INFO."""
        assert _resolve_log_level("INFO") == logging.INFO

    def test_resolve_log_level_warning(self):
        """Test resolve_log_level with WARNING."""
        assert _resolve_log_level("WARNING") == logging.WARNING

    def test_resolve_log_level_error(self):
        """Test resolve_log_level with ERROR."""
        assert _resolve_log_level("ERROR") == logging.ERROR

    def test_resolve_log_level_critical(self):
        """Test resolve_log_level with CRITICAL."""
        assert _resolve_log_level("CRITICAL") == logging.CRITICAL

    def test_resolve_log_level_case_insensitive(self):
        """Test resolve_log_level is case insensitive."""
        assert _resolve_log_level("debug") == logging.DEBUG
        assert _resolve_log_level("info") == logging.INFO

    def test_resolve_log_level_whitespace(self):
        """Test resolve_log_level strips whitespace."""
        assert _resolve_log_level(" DEBUG ") == logging.DEBUG

    def test_resolve_log_level_invalid(self, caplog):
        """Test resolve_log_level returns INFO on invalid level."""
        assert _resolve_log_level("INVALID") == logging.INFO
        assert any("LOG_LEVEL" in record.message for record in caplog.records)


class TestLoadConfigMinimal:
    """Tests for load_config with minimal environment setup."""

    def test_load_config_minimal_gemini(self):
        """Test loading config with only required Gemini variables."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()

            assert config.discord_token == "test_discord_token"
            assert config.gemini_api_key == "test_gemini_key"
            assert config.openai_api_key is None
            assert config.zai_api_key is None
            assert config.assistant_llm_provider == "gemini"
            assert config.summarizer_llm_provider == "gemini"

    def test_load_config_minimal_openai(self):
        """Test loading config with only required OpenAI variables."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "OPENAI_API_KEY": "test_openai_key",
            "ASSISTANT_LLM_PROVIDER": "openai",
            "SUMMARIZER_LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()

            assert config.discord_token == "test_discord_token"
            assert config.openai_api_key == "test_openai_key"
            assert config.gemini_api_key is None
            assert config.zai_api_key is None
            assert config.assistant_llm_provider == "openai"
            assert config.summarizer_llm_provider == "openai"

    def test_load_config_minimal_zai(self):
        """Test loading config with only required ZAI variables."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "ZAI_API_KEY": "test_zai_key",
            "ASSISTANT_LLM_PROVIDER": "zai",
            "SUMMARIZER_LLM_PROVIDER": "zai",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()

            assert config.discord_token == "test_discord_token"
            assert config.zai_api_key == "test_zai_key"
            assert config.gemini_api_key is None
            assert config.openai_api_key is None
            assert config.assistant_llm_provider == "zai"
            assert config.summarizer_llm_provider == "zai"


class TestLoadConfigMissingRequired:
    """Tests for load_config with missing required variables."""

    def test_load_config_missing_discord_token(self, caplog):
        """Test loading config without DISCORD_TOKEN exits."""
        env = {
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys, "exit") as mock_exit:
                load_config()
                mock_exit.assert_called_once_with(1)
                assert any(
                    "DISCORD_TOKEN" in record.message for record in caplog.records
                )

    def test_load_config_missing_gemini_key(self, caplog):
        """Test loading config without GEMINI_API_KEY when using Gemini exits."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys, "exit") as mock_exit:
                load_config()
                mock_exit.assert_called_once_with(1)
                assert any(
                    "GEMINI_API_KEY" in record.message for record in caplog.records
                )

    def test_load_config_missing_openai_key(self, caplog):
        """Test loading config without OPENAI_API_KEY when using OpenAI exits."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "ASSISTANT_LLM_PROVIDER": "openai",
            "SUMMARIZER_LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys, "exit") as mock_exit:
                load_config()
                mock_exit.assert_called_once_with(1)
                assert any(
                    "OPENAI_API_KEY" in record.message for record in caplog.records
                )

    def test_load_config_missing_zai_key(self, caplog):
        """Test loading config without ZAI_API_KEY when using ZAI exits."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "ASSISTANT_LLM_PROVIDER": "zai",
            "SUMMARIZER_LLM_PROVIDER": "zai",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys, "exit") as mock_exit:
                load_config()
                mock_exit.assert_called_once_with(1)
                assert any("ZAI_API_KEY" in record.message for record in caplog.records)


class TestLoadConfigAllProviders:
    """Tests for load_config with all providers configured."""

    def test_load_config_all_providers(self):
        """Test loading config with all API keys set."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "OPENAI_API_KEY": "test_openai_key",
            "ZAI_API_KEY": "test_zai_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()

            assert config.discord_token == "test_discord_token"
            assert config.gemini_api_key == "test_gemini_key"
            assert config.openai_api_key == "test_openai_key"
            assert config.zai_api_key == "test_zai_key"


class TestLoadConfigProviders:
    """Tests for LLM provider configuration."""

    def test_load_config_default_provider(self):
        """Test default provider is Gemini."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_llm_provider == "gemini"
            assert config.summarizer_llm_provider == "gemini"

    def test_load_config_assistant_provider_only(self):
        """Test setting only assistant provider."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "ASSISTANT_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test_openai_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_llm_provider == "openai"
            assert config.summarizer_llm_provider == "openai"

    def test_load_config_separate_providers(self):
        """Test setting different providers for assistant and summarizer."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "OPENAI_API_KEY": "test_openai_key",
            "ASSISTANT_LLM_PROVIDER": "gemini",
            "SUMMARIZER_LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_llm_provider == "gemini"
            assert config.summarizer_llm_provider == "openai"

    def test_load_config_provider_case_insensitive(self):
        """Test provider names are case insensitive."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "ASSISTANT_LLM_PROVIDER": "GEMINI",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_llm_provider == "gemini"


class TestLoadConfigModelNames:
    """Tests for model name configuration."""

    def test_load_config_default_models(self):
        """Test default model names for Gemini."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_model_name == DEFAULT_GEMINI_ASSISTANT_MODEL
            assert config.summarizer_model_name == DEFAULT_GEMINI_SUMMARY_MODEL

    def test_load_config_custom_gemini_models(self):
        """Test custom Gemini model names."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "GEMINI_ASSISTANT_MODEL_NAME": "custom-gemini-assistant",
            "GEMINI_SUMMARY_MODEL_NAME": "custom-gemini-summary",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_model_name == "custom-gemini-assistant"
            assert config.summarizer_model_name == "custom-gemini-summary"

    def test_load_config_custom_openai_models(self):
        """Test custom OpenAI model names."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "OPENAI_API_KEY": "test_openai_key",
            "ASSISTANT_LLM_PROVIDER": "openai",
            "SUMMARIZER_LLM_PROVIDER": "openai",
            "OPENAI_ASSISTANT_MODEL_NAME": "gpt-4-custom",
            "OPENAI_SUMMARY_MODEL_NAME": "gpt-3.5-custom",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_model_name == "gpt-4-custom"
            assert config.summarizer_model_name == "gpt-3.5-custom"

    def test_load_config_custom_zai_models(self):
        """Test custom ZAI model names."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "ZAI_API_KEY": "test_zai_key",
            "ASSISTANT_LLM_PROVIDER": "zai",
            "SUMMARIZER_LLM_PROVIDER": "zai",
            "ZAI_ASSISTANT_MODEL_NAME": "custom-zai-assistant",
            "ZAI_SUMMARY_MODEL_NAME": "custom-zai-summary",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_model_name == "custom-zai-assistant"
            assert config.summarizer_model_name == "custom-zai-summary"

    def test_load_config_openai_finetuned_override(self):
        """Test OpenAI fine-tuned model overrides assistant model."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "OPENAI_API_KEY": "test_openai_key",
            "ASSISTANT_LLM_PROVIDER": "openai",
            "OPENAI_FINETUNED_MODEL": "ft:gpt-4-custom",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.assistant_model_name == "ft:gpt-4-custom"
            assert config.openai_finetuned_model == "ft:gpt-4-custom"


class TestLoadConfigNumericalSettings:
    """Tests for numerical configuration settings."""

    def test_load_config_api_request_timeout_default(self):
        """Test API request timeout default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.api_request_timeout == 30

    def test_load_config_api_max_retries_default(self):
        """Test API max retries default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.api_max_retries == 2

    def test_load_config_api_rate_limit_retry_after_default(self):
        """Test API rate limit retry after default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.api_rate_limit_retry_after == 5

    def test_load_config_api_retry_backoff_base_default(self):
        """Test API retry backoff base default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.api_retry_backoff_base == 2.0

    def test_load_config_api_retry_backoff_max_default(self):
        """Test API retry backoff max default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.api_retry_backoff_max == 32.0

    def test_load_config_session_cache_limit_default(self):
        """Test session cache limit default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.session_cache_limit == 200

    def test_load_config_session_ttl_minutes_default(self):
        """Test session TTL minutes default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.session_inactive_minutes == 30

    def test_load_config_max_messages_per_fetch(self):
        """Test max messages per fetch default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.max_messages_per_fetch == 300

    def test_load_config_max_history(self):
        """Test max history configuration."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "MAX_HISTORY": "100",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.max_history == 100


class TestLoadConfigLLMParameters:
    """Tests for LLM generation parameters."""

    def test_load_config_temperature(self):
        """Test temperature configuration."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "TEMPERATURE": "0.7",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.temperature == 0.7

    def test_load_config_top_p(self):
        """Test top_p configuration."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "TOP_P": "0.9",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.top_p == 0.9

    def test_load_config_thinking_budget_number(self):
        """Test thinking budget with number."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "THINKING_BUDGET": "100000",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.thinking_budget == 100000

    def test_load_config_thinking_budget_off(self):
        """Test thinking budget with 'off' returns None."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "THINKING_BUDGET": "off",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.thinking_budget is None

    def test_load_config_gemini_cache_settings(self):
        """Test Gemini cache settings."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "GEMINI_CACHE_MIN_TOKENS": "65536",
            "GEMINI_CACHE_TTL_MINUTES": "120",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.gemini_cache_min_tokens == 65536
            assert config.gemini_cache_ttl_minutes == 120


class TestLoadConfigBotSettings:
    """Tests for bot-specific settings."""

    def test_load_config_command_prefix_default(self):
        """Test command prefix default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.command_prefix == "!"

    def test_load_config_auto_reply_channel_ids(self):
        """Test auto-reply channel IDs configuration."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "AUTO_REPLY_CHANNEL_IDS": "123456789,987654321",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.auto_reply_channel_ids == (123456789, 987654321)

    def test_load_config_auto_reply_channel_ids_empty(self):
        """Test auto-reply channel IDs empty."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "AUTO_REPLY_CHANNEL_IDS": "",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.auto_reply_channel_ids == ()

    def test_load_config_no_check_permission_true(self):
        """Test NO_CHECK_PERMISSION true."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "NO_CHECK_PERMISSION": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.no_check_permission is True

    def test_load_config_no_check_permission_false(self):
        """Test NO_CHECK_PERMISSION false."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "NO_CHECK_PERMISSION": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.no_check_permission is False

    def test_load_config_no_check_permission_1(self):
        """Test NO_CHECK_PERMISSION with '1'."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "NO_CHECK_PERMISSION": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.no_check_permission is True

    def test_load_config_no_check_permission_yes(self):
        """Test NO_CHECK_PERMISSION with 'yes'."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "NO_CHECK_PERMISSION": "yes",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.no_check_permission is True

    def test_load_config_no_check_permission_missing(self):
        """Test NO_CHECK_PERMISSION missing defaults to False."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.no_check_permission is False

    def test_load_config_break_cut_mode_default(self):
        """Test BREAK_CUT_MODE default value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.break_cut_mode is True


class TestLoadConfigOtherSettings:
    """Tests for other configuration settings."""

    def test_load_config_zai_base_url(self):
        """Test ZAI base URL configuration."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "ZAI_BASE_URL": "https://custom.z.ai/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.zai_base_url == "https://custom.z.ai/v1"

    def test_load_config_zai_coding_plan_default(self):
        """Test ZAI_CODING_PLAN defaults to False."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.zai_coding_plan is False
            assert config.zai_base_url == "https://api.z.ai/api/paas/v4/"

    def test_load_config_zai_coding_plan_true(self):
        """Test ZAI_CODING_PLAN true uses Coding Plan API endpoint."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "ZAI_CODING_PLAN": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.zai_coding_plan is True
            assert config.zai_base_url == "https://api.z.ai/api/coding/paas/v4/"

    def test_load_config_zai_coding_plan_variations(self):
        """Test ZAI_CODING_PLAN accepts various true values."""
        for true_value in ["true", "TRUE", "1", "yes", "YES"]:
            env = {
                "DISCORD_TOKEN": "test_discord_token",
                "GEMINI_API_KEY": "test_gemini_key",
                "ZAI_CODING_PLAN": true_value,
            }
            with patch.dict(os.environ, env, clear=True):
                config = load_config()
                assert config.zai_coding_plan is True

    def test_load_config_zai_coding_plan_false(self):
        """Test ZAI_CODING_PLAN false variations."""
        for false_value in ["false", "FALSE", "0", "no", "NO", "random"]:
            env = {
                "DISCORD_TOKEN": "test_discord_token",
                "GEMINI_API_KEY": "test_gemini_key",
                "ZAI_CODING_PLAN": false_value,
            }
            with patch.dict(os.environ, env, clear=True):
                config = load_config()
                assert config.zai_coding_plan is False

    def test_load_config_zai_coding_plan_with_custom_url(self):
        """Test ZAI_BASE_URL overrides ZAI_CODING_PLAN default endpoint."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "ZAI_CODING_PLAN": "true",
            "ZAI_BASE_URL": "https://custom.z.ai/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.zai_coding_plan is True
            assert config.zai_base_url == "https://custom.z.ai/v1"

    def test_load_config_service_tier(self):
        """Test service tier configuration."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "SERVICE_TIER": "premium",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.service_tier == "premium"

    def test_load_config_log_level(self):
        """Test log level configuration."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.log_level == logging.DEBUG

    def test_load_config_message_buffer_delay(self):
        """Test message buffer delay configuration."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "MESSAGE_BUFFER_DELAY": "1.5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.message_buffer_delay == 1.5


class TestLoadConfigTypeConversions:
    """Tests for type conversion edge cases."""

    def test_load_config_invalid_number_uses_default(self, caplog):
        """Test invalid numeric values use defaults."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "TEMPERATURE": "not_a_number",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.temperature == 1.0  # Default value
            assert any("TEMPERATURE" in record.message for record in caplog.records)

    def test_load_config_float_for_int_field(self, caplog):
        """Test float value for int field."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "MAX_HISTORY": "50.5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.max_history == 50  # Parsed as int

    def test_load_config_negative_temperature(self):
        """Test negative temperature value."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "TEMPERATURE": "-0.5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.temperature == -0.5

    def test_load_config_zero_values(self):
        """Test zero values are preserved."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
            "TEMPERATURE": "0",
            "MAX_HISTORY": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.temperature == 0.0
            assert config.max_history == 0


class TestLoadConfigDefaults:
    """Tests for default configuration values."""

    def test_load_config_all_defaults(self):
        """Test all default values when no optional env vars set."""
        env = {
            "DISCORD_TOKEN": "test_discord_token",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()

            # Check defaults
            assert config.assistant_llm_provider == "gemini"
            assert config.summarizer_llm_provider == "gemini"
            assert config.assistant_model_name == DEFAULT_GEMINI_ASSISTANT_MODEL
            assert config.summarizer_model_name == DEFAULT_GEMINI_SUMMARY_MODEL
            assert config.max_messages_per_fetch == 300
            assert config.api_max_retries == 2
            assert config.api_rate_limit_retry_after == 5
            assert config.api_request_timeout == 30
            assert config.api_retry_backoff_base == 2.0
            assert config.api_retry_backoff_max == 32.0
            assert config.progress_update_interval == 0.5
            assert config.countdown_update_interval == 5
            assert config.command_prefix == "!"
            assert config.service_tier == "flex"
            assert config.temperature == 1.0
            assert config.top_p == 1.0
            assert config.gemini_cache_min_tokens == 32768
            assert config.gemini_cache_ttl_minutes == 60
            assert config.thinking_budget is None
            assert config.max_history == 50
            assert config.auto_reply_channel_ids == ()
            assert config.log_level == logging.INFO
            assert config.session_cache_limit == 200
            assert config.session_inactive_minutes == 30
            assert config.message_buffer_delay == 2.5
            assert config.break_cut_mode is True
            assert config.no_check_permission is False
            assert config.openai_finetuned_model is None


class TestAppConfigDataclass:
    """Tests for AppConfig dataclass functionality."""

    def test_app_config_attributes(self):
        """Test AppConfig has all expected attributes."""
        config = AppConfig(discord_token="test_token")

        # Check all expected attributes exist
        assert hasattr(config, "discord_token")
        assert hasattr(config, "assistant_llm_provider")
        assert hasattr(config, "summarizer_llm_provider")
        assert hasattr(config, "gemini_api_key")
        assert hasattr(config, "openai_api_key")
        assert hasattr(config, "zai_api_key")
        assert hasattr(config, "zai_base_url")
        assert hasattr(config, "assistant_model_name")
        assert hasattr(config, "summarizer_model_name")
        assert hasattr(config, "max_messages_per_fetch")
        assert hasattr(config, "api_max_retries")
        assert hasattr(config, "api_rate_limit_retry_after")
        assert hasattr(config, "api_request_timeout")
        assert hasattr(config, "api_retry_backoff_base")
        assert hasattr(config, "api_retry_backoff_max")
        assert hasattr(config, "progress_update_interval")
        assert hasattr(config, "countdown_update_interval")
        assert hasattr(config, "command_prefix")
        assert hasattr(config, "service_tier")
        assert hasattr(config, "openai_finetuned_model")
        assert hasattr(config, "temperature")
        assert hasattr(config, "top_p")
        assert hasattr(config, "gemini_cache_min_tokens")
        assert hasattr(config, "gemini_cache_ttl_minutes")
        assert hasattr(config, "thinking_budget")
        assert hasattr(config, "max_history")
        assert hasattr(config, "auto_reply_channel_ids")
        assert hasattr(config, "log_level")
        assert hasattr(config, "session_cache_limit")
        assert hasattr(config, "session_inactive_minutes")
        assert hasattr(config, "message_buffer_delay")
        assert hasattr(config, "break_cut_mode")
        assert hasattr(config, "no_check_permission")

    def test_app_config_mutable(self):
        """Test AppConfig fields can be modified."""
        config = AppConfig(discord_token="test_token")
        config.temperature = 0.5
        config.command_prefix = "?"
        assert config.temperature == 0.5
        assert config.command_prefix == "?"

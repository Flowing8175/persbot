"""Configuration module for SoyeBot."""

from persbot.config.base import (
    AppConfig,
    CacheConfig,
    DEFAULT_GEMINI_ASSISTANT_MODEL,
    DEFAULT_GEMINI_SUMMARY_MODEL,
    DEFAULT_OPENAI_ASSISTANT_MODEL,
    DEFAULT_OPENAI_SUMMARY_MODEL,
    DEFAULT_ZAI_ASSISTANT_MODEL,
    DEFAULT_ZAI_SUMMARY_MODEL,
    LLMModelConfig,
    ProviderConfig,
    SessionConfig,
    ToolConfig,
)
from persbot.config.parsers import (
    _first_nonempty_env,
    _parse_auto_channel_ids,
    _parse_bool_env,
    _parse_float_env,
    _parse_int_env,
    _parse_thinking_budget,
    _normalize_provider,
    _resolve_log_level,
    _validate_provider,
)
from persbot.config.validators import (
    get_default_model,
    validate_buffer_delay,
    validate_model_name,
    validate_required_keys,
    validate_temperature,
    validate_thinking_budget,
    validate_top_p,
)

__all__ = [
    # Base
    "AppConfig",
    "LLMModelConfig",
    "ProviderConfig",
    "CacheConfig",
    "ToolConfig",
    "SessionConfig",
    # Defaults
    "DEFAULT_GEMINI_ASSISTANT_MODEL",
    "DEFAULT_GEMINI_SUMMARY_MODEL",
    "DEFAULT_OPENAI_ASSISTANT_MODEL",
    "DEFAULT_OPENAI_SUMMARY_MODEL",
    "DEFAULT_ZAI_ASSISTANT_MODEL",
    "DEFAULT_ZAI_SUMMARY_MODEL",
    # Parsers
    "_resolve_log_level",
    "_parse_float_env",
    "_parse_int_env",
    "_parse_bool_env",
    "_parse_thinking_budget",
    "_parse_auto_channel_ids",
    "_first_nonempty_env",
    "_normalize_provider",
    "_validate_provider",
    # Validators
    "validate_required_keys",
    "validate_model_name",
    "validate_temperature",
    "validate_top_p",
    "validate_thinking_budget",
    "validate_buffer_delay",
    "get_default_model",
]

"""Base configuration classes for SoyeBot."""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Default Model Constants
DEFAULT_GEMINI_ASSISTANT_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-2.5-pro"
DEFAULT_OPENAI_ASSISTANT_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_SUMMARY_MODEL = "gpt-5-mini"
DEFAULT_ZAI_ASSISTANT_MODEL = "glm-4.7"
DEFAULT_ZAI_SUMMARY_MODEL = "glm-4-flash"


@dataclass
class LLMModelConfig:
    """Configuration for a specific LLM model."""

    api_model_name: str
    """The actual model name used for API calls."""

    provider: str
    """The provider (gemini, openai, zai)."""

    alias: Optional[str] = None
    """Optional alias for referencing this model."""

    display_name: Optional[str] = None
    """Display name for UI purposes."""


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 2
    rate_limit_retry_after: int = 5
    request_timeout: float = 120.0
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 32.0


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""

    min_tokens: int = 32768
    ttl_minutes: int = 60
    refresh_buffer_min: int = 1
    refresh_buffer_max: int = 5


@dataclass
class ToolConfig:
    """Configuration for tool behavior."""

    enabled: bool = True
    enable_discord_tools: bool = True
    enable_api_tools: bool = True
    enable_persona_tools: bool = True
    rate_limit: int = 0
    timeout: float = 10.0
    weather_api_key: Optional[str] = None
    search_api_key: Optional[str] = None


@dataclass
class SessionConfig:
    """Configuration for session management."""

    cache_limit: int = 200
    inactive_minutes: int = 30
    message_buffer_delay: float = 0.1
    break_cut_mode: bool = True


@dataclass
class AppConfig:
    """
    Application configuration.

    This is the main configuration class that holds all settings
    for the SoyeBot application.
    """

    # Discord
    discord_token: str

    # Provider Selection
    assistant_llm_provider: str = "gemini"
    summarizer_llm_provider: str = "gemini"

    # API Keys
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    zai_api_key: Optional[str] = None
    zai_base_url: str = "https://api.z.ai/api/paas/v4/"
    zai_coding_plan: bool = False
    openrouter_api_key: Optional[str] = None

    # Model Configuration
    assistant_model_name: str = DEFAULT_GEMINI_ASSISTANT_MODEL
    summarizer_model_name: str = DEFAULT_GEMINI_SUMMARY_MODEL
    openai_finetuned_model: Optional[str] = None
    openrouter_image_model: str = "black-forest-labs/flux.2-klein-4b"

    # API Behavior
    max_messages_per_fetch: int = 300
    api_max_retries: int = 2
    api_rate_limit_retry_after: int = 5
    api_request_timeout: float = 120.0
    api_retry_backoff_base: float = 2.0
    api_retry_backoff_max: float = 32.0
    progress_update_interval: float = 0.5
    countdown_update_interval: int = 5

    # Bot Configuration
    command_prefix: str = "!"
    service_tier: str = "flex"
    no_check_permission: bool = False

    # LLM Parameters
    temperature: float = 1.0
    top_p: float = 1.0
    thinking_budget: Optional[int] = None
    max_history: int = 50

    # Gemini-specific
    gemini_cache_min_tokens: int = 32768
    gemini_cache_ttl_minutes: int = 60

    # Channel Configuration
    auto_reply_channel_ids: Tuple[int, ...] = ()

    # Logging
    log_level: int = 20  # logging.INFO

    # Session Configuration
    session_cache_limit: int = 200
    session_inactive_minutes: int = 30
    message_buffer_delay: float = 0.1
    break_cut_mode: bool = True

    # Tool Configuration
    enable_tools: bool = True
    enable_discord_tools: bool = True
    enable_api_tools: bool = True
    tool_rate_limit: int = 0
    tool_timeout: float = 10.0
    weather_api_key: Optional[str] = None
    search_api_key: Optional[str] = None

    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get configuration for a specific provider."""
        if provider == "gemini":
            return ProviderConfig(
                api_key=self.gemini_api_key,
                max_retries=self.api_max_retries,
                rate_limit_retry_after=self.api_rate_limit_retry_after,
                request_timeout=self.api_request_timeout,
                retry_backoff_base=self.api_retry_backoff_base,
                retry_backoff_max=self.api_retry_backoff_max,
            )
        elif provider == "openai":
            return ProviderConfig(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                max_retries=self.api_max_retries,
                rate_limit_retry_after=self.api_rate_limit_retry_after,
                request_timeout=self.api_request_timeout,
                retry_backoff_base=self.api_retry_backoff_base,
                retry_backoff_max=self.api_retry_backoff_max,
            )
        elif provider == "zai":
            return ProviderConfig(
                api_key=self.zai_api_key,
                base_url=self.zai_base_url,
                max_retries=self.api_max_retries,
                rate_limit_retry_after=self.api_rate_limit_retry_after,
                request_timeout=self.api_request_timeout,
                retry_backoff_base=self.api_retry_backoff_base,
                retry_backoff_max=self.api_retry_backoff_max,
            )
        else:
            return ProviderConfig()

    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        return CacheConfig(
            min_tokens=self.gemini_cache_min_tokens,
            ttl_minutes=self.gemini_cache_ttl_minutes,
        )

    def get_tool_config(self) -> ToolConfig:
        """Get tool configuration."""
        return ToolConfig(
            enabled=self.enable_tools,
            enable_discord_tools=self.enable_discord_tools,
            enable_api_tools=self.enable_api_tools,
            rate_limit=self.tool_rate_limit,
            timeout=self.tool_timeout,
            weather_api_key=self.weather_api_key,
            search_api_key=self.search_api_key,
        )

    def get_session_config(self) -> SessionConfig:
        """Get session configuration."""
        return SessionConfig(
            cache_limit=self.session_cache_limit,
            inactive_minutes=self.session_inactive_minutes,
            message_buffer_delay=self.message_buffer_delay,
            break_cut_mode=self.break_cut_mode,
        )


# Keep the old config file working by importing from here
def _migrate_from_dataclass():
    """Migrate from the old dataclass-based config to the new class."""
    import sys
    import os

    # Check if the old config.py still uses dataclass
    old_config_path = os.path.join(
        os.path.dirname(__file__), "..", "config.py"
    )
    if not os.path.exists(old_config_path):
        return

    # The old config.py will be refactored in stages
    pass

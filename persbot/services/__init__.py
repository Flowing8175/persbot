"""Services module for SoyeBot LLM integrations."""

from persbot.services.base import BaseLLMService, ChatMessage
from persbot.services.cache_manager import CacheManager, HashBasedCacheStrategy
from persbot.services.retry_handler import (
    BackoffStrategy,
    FatalError,
    GeminiRetryHandler,
    OpenAIRetryHandler,
    RetryConfig,
    RetryHandler,
    RetryableError,
    ZAIRetryHandler,
)

__all__ = [
    # Base
    "BaseLLMService",
    "ChatMessage",
    # Cache
    "CacheManager",
    "HashBasedCacheStrategy",
    # Retry
    "RetryHandler",
    "RetryConfig",
    "BackoffStrategy",
    "RetryableError",
    "FatalError",
    "GeminiRetryHandler",
    "OpenAIRetryHandler",
    "ZAIRetryHandler",
]

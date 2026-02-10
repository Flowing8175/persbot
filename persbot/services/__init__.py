"""Services module for SoyeBot LLM integrations."""

from persbot.exceptions import FatalError, RetryableError
from persbot.services.base import BaseLLMService, ChatMessage
from persbot.services.cache_manager import CacheManager, HashBasedCacheStrategy
from persbot.services.retry_handler import (
    BackoffStrategy,
    GeminiRetryHandler,
    OpenAIRetryHandler,
    RetryConfig,
    RetryHandler,
    ZAIRetryHandler,
)

__all__ = [
    # Base
    "BaseLLMService",
    "ChatMessage",
    # Cache
    "CacheManager",
    "HashBasedCacheStrategy",
    # Exceptions
    "FatalError",
    "RetryableError",
    # Retry
    "RetryHandler",
    "RetryConfig",
    "BackoffStrategy",
    "GeminiRetryHandler",
    "OpenAIRetryHandler",
    "ZAIRetryHandler",
]

"""Custom exceptions for SoyeBot."""

from typing import Any, Optional


class SoyeBotException(Exception):
    """Base exception for all SoyeBot errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class APIException(SoyeBotException):
    """Exception raised when an API call fails."""

    pass


class RateLimitException(APIException):
    """Exception raised when rate limit is exceeded."""

    pass


class AuthenticationException(APIException):
    """Exception raised when API authentication fails."""

    pass


class QuotaException(APIException):
    """Exception raised when API quota is exceeded."""

    pass


class CacheException(SoyeBotException):
    """Exception raised when cache operations fail."""

    pass


class ToolException(SoyeBotException):
    """Exception raised when tool execution fails."""

    pass


class ToolTimeoutException(ToolException):
    """Exception raised when tool execution times out."""

    pass


class ToolRateLimitException(ToolException):
    """Exception raised when tool rate limit is exceeded."""

    pass


class SessionException(SoyeBotException):
    """Exception raised when session operations fail."""

    pass


class ConfigurationException(SoyeBotException):
    """Exception raised when configuration is invalid."""

    pass


class ValidationException(ConfigurationException):
    """Exception raised when configuration validation fails."""

    pass


class FatalError(SoyeBotException):
    """Exception raised for fatal errors that should stop retry attempts immediately."""

    pass


class RetryableError(SoyeBotException):
    """Base exception for errors that can be retried."""

    pass


# =============================================================================
# Provider-Specific Exceptions
# =============================================================================

class ProviderException(SoyeBotException):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, provider: str, details: Optional[dict[str, Any]] = None):
        self.provider = provider
        super().__init__(message, details)


class ProviderUnavailableException(ProviderException):
    """Exception raised when a provider is unavailable or misconfigured."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider)


class ModelNotFoundException(ProviderException):
    """Exception raised when a requested model is not found."""

    def __init__(self, model_name: str, provider: str):
        super().__init__(
            f"Model '{model_name}' not found in provider '{provider}'",
            provider,
            {"model_name": model_name}
        )
        self.model_name = model_name


class ContextCacheException(ProviderException):
    """Exception raised when context cache operations fail."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider)


# =============================================================================
# Session-Specific Exceptions
# =============================================================================

class SessionNotFoundException(SessionException):
    """Exception raised when a requested session is not found."""

    def __init__(self, session_key: str):
        super().__init__(f"Session '{session_key}' not found", {"session_key": session_key})
        self.session_key = session_key


class SessionExpiredException(SessionException):
    """Exception raised when a session has expired."""

    pass


class SessionConflictException(SessionException):
    """Exception raised when there's a conflict in session state."""

    pass


# =============================================================================
# Image-Specific Exceptions
# =============================================================================

class ImageException(SoyeBotException):
    """Base exception for image-related errors."""

    pass


class ImageSizeException(ImageException):
    """Exception raised when an image is too large."""

    pass


class ImageFormatException(ImageException):
    """Exception raised when an image format is not supported."""

    pass


class ImageRateLimitException(ImageException):
    """Exception raised when image generation rate limit is exceeded."""

    def __init__(self, limit_type: str, current: int, limit: int):
        super().__init__(
            f"Image {limit_type} limit exceeded: {current}/{limit}",
            {"limit_type": limit_type, "current": current, "limit": limit}
        )
        self.limit_type = limit_type
        self.current = current
        self.limit = limit


# =============================================================================
# Message Processing Exceptions
# =============================================================================

class MessageProcessingException(SoyeBotException):
    """Base exception for message processing errors."""

    pass


class MessageTooLongException(MessageProcessingException):
    """Exception raised when a message exceeds maximum length."""

    pass


class MessageContentNotFoundException(MessageProcessingException):
    """Exception raised when message content cannot be extracted."""

    pass


# =============================================================================
# Cancellation Exceptions
# =============================================================================

class CancellationException(SoyeBotException):
    """Exception raised when an operation is cancelled by user request."""

    pass


class AbortSignalException(CancellationException):
    """Exception raised when an abort signal is received."""

    pass


# =============================================================================
# Validation Exceptions
# =============================================================================

class DomainValidationException(SoyeBotException):
    """Exception raised when domain object validation fails."""

    pass


class InvalidValueObjectException(DomainValidationException):
    """Exception raised when a value object is invalid."""

    pass


# =============================================================================
# HTTP/API Client Exceptions
# =============================================================================

class HTTPClientException(APIException):
    """Base exception for HTTP client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict[str, Any]] = None):
        self.status_code = status_code
        full_details = {**(details or {}), "status_code": status_code} if status_code else details
        super().__init__(message, full_details)


class ServerException(HTTPClientException):
    """Exception raised for 5xx server errors."""

    pass


class ClientException(HTTPClientException):
    """Exception raised for 4xx client errors."""

    pass


# =============================================================================
# Prompt Generation Exceptions
# =============================================================================

class PromptGenerationException(SoyeBotException):
    """Exception raised when prompt generation fails."""

    pass


class PersonaCreationException(PromptGenerationException):
    """Exception raised when persona creation fails."""

    pass

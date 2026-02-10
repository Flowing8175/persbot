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

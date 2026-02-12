"""Feature tests for custom exception hierarchy.

Tests focus on behavior:
- Exception inheritance chain
- Message formatting with details
- Specialized exception fields
"""

import pytest

from persbot.exceptions import (
    SoyeBotException,
    APIException,
    RateLimitException,
    AuthenticationException,
    QuotaException,
    CacheException,
    ToolException,
    ToolTimeoutException,
    ToolRateLimitException,
    SessionException,
    ConfigurationException,
    ValidationException,
    FatalError,
    RetryableError,
    ProviderException,
    ProviderUnavailableException,
    ModelNotFoundException,
    ContextCacheException,
    SessionNotFoundException,
    SessionExpiredException,
    SessionConflictException,
    ImageException,
    ImageSizeException,
    ImageFormatException,
    ImageRateLimitException,
    MessageProcessingException,
    MessageTooLongException,
    MessageContentNotFoundException,
    CancellationException,
    AbortSignalException,
    DomainValidationException,
    InvalidValueObjectException,
    HTTPClientException,
    ServerException,
    ClientException,
    PromptGenerationException,
    PersonaCreationException,
)


class TestSoyeBotException:
    """Tests for the base SoyeBotException."""

    def test_creates_with_message_only(self):
        """SoyeBotException can be created with just a message."""
        exc = SoyeBotException("Test error")
        assert exc.message == "Test error"
        assert exc.details == {}

    def test_creates_with_message_and_details(self):
        """SoyeBotException can be created with message and details."""
        exc = SoyeBotException("Test error", details={"key": "value", "count": 42})
        assert exc.message == "Test error"
        assert exc.details == {"key": "value", "count": 42}

    def test_str_returns_message_only_without_details(self):
        """str() returns just the message when no details."""
        exc = SoyeBotException("Test error")
        assert str(exc) == "Test error"

    def test_str_includes_details_when_present(self):
        """str() includes details in formatted output."""
        exc = SoyeBotException("Test error", details={"key": "value"})
        assert "Test error" in str(exc)
        assert "key=value" in str(exc)

    def test_str_formats_multiple_details(self):
        """str() formats multiple details correctly."""
        exc = SoyeBotException("Error", details={"a": 1, "b": 2})
        result = str(exc)
        assert "a=1" in result
        assert "b=2" in result


class TestAPIExceptions:
    """Tests for API-related exceptions."""

    def test_api_exception_inherits_from_base(self):
        """APIException inherits from SoyeBotException."""
        exc = APIException("API error")
        assert isinstance(exc, SoyeBotException)

    def test_rate_limit_exception_inherits_from_api(self):
        """RateLimitException inherits from APIException."""
        exc = RateLimitException("Rate limited")
        assert isinstance(exc, APIException)
        assert isinstance(exc, SoyeBotException)

    def test_authentication_exception_inherits_from_api(self):
        """AuthenticationException inherits from APIException."""
        exc = AuthenticationException("Auth failed")
        assert isinstance(exc, APIException)

    def test_quota_exception_inherits_from_api(self):
        """QuotaException inherits from APIException."""
        exc = QuotaException("Quota exceeded")
        assert isinstance(exc, APIException)


class TestCacheException:
    """Tests for CacheException."""

    def test_inherits_from_base(self):
        """CacheException inherits from SoyeBotException."""
        exc = CacheException("Cache error")
        assert isinstance(exc, SoyeBotException)

    def test_accepts_details(self):
        """CacheException accepts details."""
        exc = CacheException("Cache miss", details={"key": "test_key"})
        assert exc.details == {"key": "test_key"}


class TestToolExceptions:
    """Tests for tool-related exceptions."""

    def test_tool_exception_inherits_from_base(self):
        """ToolException inherits from SoyeBotException."""
        exc = ToolException("Tool error")
        assert isinstance(exc, SoyeBotException)

    def test_tool_timeout_exception_inherits_from_tool(self):
        """ToolTimeoutException inherits from ToolException."""
        exc = ToolTimeoutException("Timed out")
        assert isinstance(exc, ToolException)
        assert isinstance(exc, SoyeBotException)

    def test_tool_rate_limit_exception_inherits_from_tool(self):
        """ToolRateLimitException inherits from ToolException."""
        exc = ToolRateLimitException("Rate limited")
        assert isinstance(exc, ToolException)


class TestProviderExceptions:
    """Tests for provider-specific exceptions."""

    def test_provider_exception_has_provider_field(self):
        """ProviderException stores provider name."""
        exc = ProviderException("Error", provider="gemini")
        assert exc.provider == "gemini"

    def test_provider_exception_inherits_from_base(self):
        """ProviderException inherits from SoyeBotException."""
        exc = ProviderException("Error", provider="openai")
        assert isinstance(exc, SoyeBotException)

    def test_provider_unavailable_exception_defaults_provider(self):
        """ProviderUnavailableException has default provider."""
        exc = ProviderUnavailableException("Unavailable")
        assert exc.provider == "unknown"

    def test_provider_unavailable_exception_accepts_provider(self):
        """ProviderUnavailableException accepts provider argument."""
        exc = ProviderUnavailableException("Unavailable", provider="zai")
        assert exc.provider == "zai"

    def test_model_not_found_exception_stores_model_name(self):
        """ModelNotFoundException stores model name."""
        exc = ModelNotFoundException("gpt-5", provider="openai")
        assert exc.model_name == "gpt-5"
        assert exc.provider == "openai"

    def test_model_not_found_exception_message_includes_model(self):
        """ModelNotFoundException message includes model name."""
        exc = ModelNotFoundException("gpt-5", provider="openai")
        assert "gpt-5" in exc.message
        assert "openai" in exc.message


class TestSessionExceptions:
    """Tests for session-related exceptions."""

    def test_session_exception_inherits_from_base(self):
        """SessionException inherits from SoyeBotException."""
        exc = SessionException("Session error")
        assert isinstance(exc, SoyeBotException)

    def test_session_not_found_exception_stores_key(self):
        """SessionNotFoundException stores session key."""
        exc = SessionNotFoundException("channel:123")
        assert exc.session_key == "channel:123"

    def test_session_not_found_exception_message_includes_key(self):
        """SessionNotFoundException message includes key."""
        exc = SessionNotFoundException("channel:123")
        assert "channel:123" in exc.message

    def test_session_expired_exception_inherits_from_session(self):
        """SessionExpiredException inherits from SessionException."""
        exc = SessionExpiredException("Expired")
        assert isinstance(exc, SessionException)

    def test_session_conflict_exception_inherits_from_session(self):
        """SessionConflictException inherits from SessionException."""
        exc = SessionConflictException("Conflict")
        assert isinstance(exc, SessionException)


class TestImageExceptions:
    """Tests for image-related exceptions."""

    def test_image_exception_inherits_from_base(self):
        """ImageException inherits from SoyeBotException."""
        exc = ImageException("Image error")
        assert isinstance(exc, SoyeBotException)

    def test_image_size_exception_inherits_from_image(self):
        """ImageSizeException inherits from ImageException."""
        exc = ImageSizeException("Too large")
        assert isinstance(exc, ImageException)

    def test_image_format_exception_inherits_from_image(self):
        """ImageFormatException inherits from ImageException."""
        exc = ImageFormatException("Invalid format")
        assert isinstance(exc, ImageException)

    def test_image_rate_limit_exception_has_limit_fields(self):
        """ImageRateLimitException stores limit info."""
        exc = ImageRateLimitException("per_minute", current=5, limit=3)
        assert exc.limit_type == "per_minute"
        assert exc.current == 5
        assert exc.limit == 3

    def test_image_rate_limit_exception_message_includes_limits(self):
        """ImageRateLimitException message includes limit info."""
        exc = ImageRateLimitException("per_minute", current=5, limit=3)
        assert "5/3" in exc.message


class TestMessageProcessingExceptions:
    """Tests for message processing exceptions."""

    def test_message_processing_exception_inherits_from_base(self):
        """MessageProcessingException inherits from SoyeBotException."""
        exc = MessageProcessingException("Processing error")
        assert isinstance(exc, SoyeBotException)

    def test_message_too_long_exception_inherits(self):
        """MessageTooLongException inherits from MessageProcessingException."""
        exc = MessageTooLongException("Too long")
        assert isinstance(exc, MessageProcessingException)

    def test_message_content_not_found_exception_inherits(self):
        """MessageContentNotFoundException inherits from MessageProcessingException."""
        exc = MessageContentNotFoundException("Not found")
        assert isinstance(exc, MessageProcessingException)


class TestHTTPClientExceptions:
    """Tests for HTTP client exceptions."""

    def test_http_client_exception_inherits_from_api(self):
        """HTTPClientException inherits from APIException."""
        exc = HTTPClientException("HTTP error")
        assert isinstance(exc, APIException)

    def test_http_client_exception_stores_status_code(self):
        """HTTPClientException stores status code."""
        exc = HTTPClientException("Error", status_code=404)
        assert exc.status_code == 404

    def test_http_client_exception_includes_status_in_details(self):
        """HTTPClientException includes status code in details."""
        exc = HTTPClientException("Error", status_code=500, details={"reason": "crash"})
        assert exc.details["status_code"] == 500
        assert exc.details["reason"] == "crash"

    def test_server_exception_inherits_from_http_client(self):
        """ServerException inherits from HTTPClientException."""
        exc = ServerException("Server error", status_code=500)
        assert isinstance(exc, HTTPClientException)

    def test_client_exception_inherits_from_http_client(self):
        """ClientException inherits from HTTPClientException."""
        exc = ClientException("Client error", status_code=400)
        assert isinstance(exc, HTTPClientException)


class TestCancellationExceptions:
    """Tests for cancellation-related exceptions."""

    def test_cancellation_exception_inherits_from_base(self):
        """CancellationException inherits from SoyeBotException."""
        exc = CancellationException("Cancelled")
        assert isinstance(exc, SoyeBotException)

    def test_abort_signal_exception_inherits_from_cancellation(self):
        """AbortSignalException inherits from CancellationException."""
        exc = AbortSignalException("Aborted")
        assert isinstance(exc, CancellationException)


class TestValidationExceptions:
    """Tests for validation exceptions."""

    def test_domain_validation_exception_inherits_from_base(self):
        """DomainValidationException inherits from SoyeBotException."""
        exc = DomainValidationException("Invalid")
        assert isinstance(exc, SoyeBotException)

    def test_invalid_value_object_exception_inherits(self):
        """InvalidValueObjectException inherits from DomainValidationException."""
        exc = InvalidValueObjectException("Invalid value")
        assert isinstance(exc, DomainValidationException)

    def test_validation_exception_inherits_from_configuration(self):
        """ValidationException inherits from ConfigurationException."""
        exc = ValidationException("Invalid config")
        assert isinstance(exc, ConfigurationException)


class TestPromptExceptions:
    """Tests for prompt-related exceptions."""

    def test_prompt_generation_exception_inherits_from_base(self):
        """PromptGenerationException inherits from SoyeBotException."""
        exc = PromptGenerationException("Generation failed")
        assert isinstance(exc, SoyeBotException)

    def test_persona_creation_exception_inherits_from_prompt(self):
        """PersonaCreationException inherits from PromptGenerationException."""
        exc = PersonaCreationException("Creation failed")
        assert isinstance(exc, PromptGenerationException)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_fatal_error_is_soyebot_exception(self):
        """FatalError is a SoyeBotException."""
        exc = FatalError("Fatal")
        assert isinstance(exc, SoyeBotException)

    def test_retryable_error_is_soyebot_exception(self):
        """RetryableError is a SoyeBotException."""
        exc = RetryableError("Retry")
        assert isinstance(exc, SoyeBotException)

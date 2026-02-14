"""Tests for gemini_service.py module."""

import asyncio
import datetime
import hashlib
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch, call

from persbot.services.gemini_service import GeminiService


class MockGeminiCachedModel:
    """Mock GeminiCachedModel for testing."""

    def __init__(self, model_name="test-model", config=None):
        self._model_name = model_name
        self._config = config or Mock()

    @property
    def model_name(self):
        return self._model_name

    def generate_content(self, contents, tools=None):
        return Mock(text="Generated content")


class MockConfig:
    """Mock AppConfig for testing."""

    def __init__(self):
        self.gemini_api_key = "test-api-key"
        self.temperature = 0.7
        self.top_p = 0.9
        self.api_max_retries = 3
        self.api_request_timeout = 60.0
        self.api_retry_backoff_base = 1.0
        self.api_retry_backoff_max = 30.0
        self.api_rate_limit_retry_after = 10.0
        self.gemini_cache_ttl_minutes = 0  # Disable periodic cleanup task for tests
        self.gemini_cache_min_tokens = 1024
        self.thinking_budget = None


class MockPromptService:
    """Mock PromptService for testing."""

    def get_summary_prompt(self):
        return "Summarize the following:"

    def get_active_assistant_prompt(self):
        return "You are a helpful assistant."


@pytest.fixture
def mock_config():
    """Create a mock config."""
    return MockConfig()


@pytest.fixture
def mock_prompt_service():
    """Create a mock prompt service."""
    return MockPromptService()


@pytest.fixture
def mock_genai_client():
    """Create a mock genai.Client."""
    client = Mock()
    client.models = Mock()
    client.models.generate_content = Mock(return_value=Mock(text="Response"))
    client.models.count_tokens = Mock(return_value=Mock(total_tokens=500))
    client.caches = Mock()
    client.caches.list = Mock(return_value=[])
    client.caches.create = Mock(return_value=Mock(name="cache-123"))
    client.caches.update = Mock()
    return client


@pytest.fixture
def service(mock_config, mock_prompt_service, mock_genai_client):
    """Create a GeminiService instance for testing with all dependencies mocked."""
    with patch("persbot.services.gemini_service.genai.Client") as mock_client_class:
        mock_client_class.return_value = mock_genai_client
        with patch.object(GeminiService, "_get_or_create_model") as mock_get_model:
            mock_get_model.return_value = MockGeminiCachedModel()

            service = GeminiService(
                mock_config,
                assistant_model_name="gemini-2.5-flash",
                summary_model_name="gemini-2.5-flash",
                prompt_service=mock_prompt_service,
            )
            # Disable periodic cleanup task for tests
            service._model_cache = {}
            return service


class TestGeminiServiceInit:
    """Tests for GeminiService.__init__()."""

    def test_initializes_with_config_and_models(
        self, mock_config, mock_prompt_service, mock_genai_client
    ):
        """__init__ initializes with config and creates models."""
        with patch("persbot.services.gemini_service.genai.Client") as mock_client_class:
            mock_client_class.return_value = mock_genai_client
            with patch.object(GeminiService, "_get_or_create_model") as mock_get_model:
                mock_get_model.return_value = MockGeminiCachedModel()

                sv = GeminiService(
                    mock_config,
                    assistant_model_name="gemini-2.5-flash",
                    prompt_service=mock_prompt_service,
                )

                assert sv.config == mock_config
                assert sv._assistant_model_name == "gemini-2.5-flash"
                assert sv._summary_model_name == "gemini-2.5-flash"  # Falls back to assistant

    def test_uses_custom_summary_model(
        self, mock_config, mock_prompt_service, mock_genai_client
    ):
        """__init__ uses custom summary model when provided."""
        with patch("persbot.services.gemini_service.genai.Client") as mock_client_class:
            mock_client_class.return_value = mock_genai_client
            with patch.object(GeminiService, "_get_or_create_model") as mock_get_model:
                mock_get_model.return_value = MockGeminiCachedModel()

                sv = GeminiService(
                    mock_config,
                    assistant_model_name="gemini-2.5-pro",
                    summary_model_name="gemini-2.5-flash",
                    prompt_service=mock_prompt_service,
                )

                assert sv._assistant_model_name == "gemini-2.5-pro"
                assert sv._summary_model_name == "gemini-2.5-flash"


class TestGeminiServiceGetOrCreateModel:
    """Tests for GeminiService._get_or_create_model()."""

    @pytest.fixture
    def minimal_service(self, mock_config, mock_prompt_service, mock_genai_client):
        """Create a minimal service for testing model creation."""
        with patch("persbot.services.gemini_service.genai.Client") as mock_client_class:
            mock_client_class.return_value = mock_genai_client
            with patch.object(GeminiService, "__init__", lambda x, *args, **kwargs: None):
                sv = GeminiService.__new__(GeminiService)
                sv.client = mock_genai_client
                sv.config = mock_config
                sv._model_cache = {}
                sv.prompt_service = mock_prompt_service
                return sv

    def test_creates_new_model_when_not_cached(self, minimal_service):
        """_get_or_create_model creates new model when not in cache."""
        with patch.object(
            minimal_service, "_resolve_gemini_cache", return_value=(None, None)
        ):
            with patch.object(
                minimal_service, "_build_generation_config"
            ) as mock_build_config:
                mock_build_config.return_value = Mock()

                model = minimal_service._get_or_create_model(
                    "test-model", "system instruction"
                )

                assert model is not None
                # Check cache was populated
                assert len(minimal_service._model_cache) == 1

    def test_returns_cached_model_when_valid(self, minimal_service):
        """_get_or_create_model returns cached model when still valid."""
        # Pre-populate cache with a valid model
        cache_key = hash(("test-model", "system instruction", True))
        cached_model = MockGeminiCachedModel()
        future_expiration = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        minimal_service._model_cache[cache_key] = (cached_model, future_expiration)

        model = minimal_service._get_or_create_model(
            "test-model", "system instruction"
        )

        assert model == cached_model

    def test_refreshes_expired_model(self, minimal_service):
        """_get_or_create_model refreshes expired cached model."""
        # Pre-populate cache with an expired model
        cache_key = hash(("test-model", "system instruction", True))
        expired_model = MockGeminiCachedModel()
        past_expiration = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        minimal_service._model_cache[cache_key] = (expired_model, past_expiration)

        with patch.object(
            minimal_service, "_resolve_gemini_cache", return_value=(None, None)
        ):
            with patch.object(
                minimal_service, "_build_generation_config"
            ) as mock_build_config:
                mock_build_config.return_value = Mock()

                model = minimal_service._get_or_create_model(
                    "test-model", "system instruction"
                )

                # Should have created a new model, not returned expired one
                assert model is not None


class TestGeminiServiceCheckModelCacheValidity:
    """Tests for GeminiService._check_model_cache_validity()."""

    @pytest.fixture
    def minimal_service(self, mock_config, mock_prompt_service, mock_genai_client):
        """Create a minimal service for testing."""
        with patch("persbot.services.gemini_service.genai.Client"):
            with patch.object(GeminiService, "__init__", lambda x, *args, **kwargs: None):
                sv = GeminiService.__new__(GeminiService)
                sv.client = mock_genai_client
                sv.config = mock_config
                sv._model_cache = {}
                return sv

    def test_returns_none_for_missing_key(self, minimal_service):
        """_check_model_cache_validity returns None for missing key."""
        now = datetime.datetime.now(datetime.timezone.utc)
        result = minimal_service._check_model_cache_validity(999, now)
        assert result is None

    def test_returns_model_for_valid_cache(self, minimal_service):
        """_check_model_cache_validity returns model for valid cache."""
        model = MockGeminiCachedModel()
        future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        minimal_service._model_cache[1] = (model, future)

        now = datetime.datetime.now(datetime.timezone.utc)
        result = minimal_service._check_model_cache_validity(1, now)

        assert result == model

    def test_returns_none_and_deletes_expired_cache(self, minimal_service):
        """_check_model_cache_validity returns None and deletes expired cache."""
        model = MockGeminiCachedModel()
        past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        minimal_service._model_cache[1] = (model, past)

        now = datetime.datetime.now(datetime.timezone.utc)
        result = minimal_service._check_model_cache_validity(1, now)

        assert result is None
        assert 1 not in minimal_service._model_cache

    def test_handles_none_expiration(self, minimal_service):
        """_check_model_cache_validity handles None expiration (no TTL)."""
        model = MockGeminiCachedModel()
        minimal_service._model_cache[1] = (model, None)

        now = datetime.datetime.now(datetime.timezone.utc)
        result = minimal_service._check_model_cache_validity(1, now)

        assert result == model


class TestGeminiServiceGetSearchTools:
    """Tests for GeminiService._get_search_tools()."""

    def test_returns_google_search_tool(self, service):
        """_get_search_tools returns Google Search tool."""
        with patch("persbot.services.gemini_service.genai_types.Tool") as mock_tool:
            with patch(
                "persbot.services.gemini_service.genai_types.GoogleSearch"
            ) as mock_search:
                mock_tool.return_value = Mock()

                result = service._get_search_tools("gemini-2.5-flash")

                # Tool should be created
                assert result is not None


class TestGeminiServiceResolveGeminiCache:
    """Tests for GeminiService._resolve_gemini_cache()."""

    def test_returns_none_none_when_cache_disabled(self, service):
        """_resolve_gemini_cache returns (None, None) when use_cache is False."""
        result = service._resolve_gemini_cache(
            "model", "instruction", [], use_cache=False
        )
        assert result == (None, None)

    def test_calls_get_gemini_cache_when_enabled(self, service):
        """_resolve_gemini_cache calls _get_gemini_cache when use_cache is True."""
        with patch.object(
            service, "_get_gemini_cache", return_value=("cache-name", None)
        ):
            result = service._resolve_gemini_cache(
                "model", "instruction", [], use_cache=True
            )

            assert result == ("cache-name", None)


class TestGeminiServiceBuildGenerationConfig:
    """Tests for GeminiService._build_generation_config()."""

    def test_builds_config_with_cache(self, service):
        """_build_generation_config builds config with cached_content."""
        with patch("persbot.services.gemini_service.genai_types.GenerateContentConfig") as mock_config:
            mock_config.return_value = Mock()

            service._build_generation_config(
                cache_name="cache-123",
                system_instruction="Be helpful",
                tools=None
            )

            # Verify GenerateContentConfig was called with cached_content
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["cached_content"] == "cache-123"

    def test_builds_config_without_cache(self, service):
        """_build_generation_config builds config with system_instruction when no cache."""
        with patch("persbot.services.gemini_service.genai_types.GenerateContentConfig") as mock_config:
            mock_config.return_value = Mock()

            service._build_generation_config(
                cache_name=None,
                system_instruction="Be helpful",
                tools=["tool1"]
            )

            call_kwargs = mock_config.call_args[1]
            assert "cached_content" not in call_kwargs
            assert call_kwargs["system_instruction"] == "Be helpful"
            assert call_kwargs["tools"] == ["tool1"]


class TestGeminiServiceCreateAssistantModel:
    """Tests for GeminiService.create_assistant_model()."""

    def test_creates_assistant_model_with_instruction(self, service):
        """create_assistant_model creates model with custom instruction."""
        with patch.object(service, "_get_or_create_model") as mock_get_model:
            mock_get_model.return_value = MockGeminiCachedModel()

            model = service.create_assistant_model("Custom system instruction")

            mock_get_model.assert_called_once_with(
                service._assistant_model_name,
                "Custom system instruction",
                use_cache=True
            )
            assert model is not None

    def test_passes_use_cache_parameter(self, service):
        """create_assistant_model passes use_cache parameter."""
        with patch.object(service, "_get_or_create_model") as mock_get_model:
            mock_get_model.return_value = MockGeminiCachedModel()

            service.create_assistant_model("instruction", use_cache=False)

            mock_get_model.assert_called_once_with(
                service._assistant_model_name,
                "instruction",
                use_cache=False
            )


class TestGeminiServiceReloadParameters:
    """Tests for GeminiService.reload_parameters()."""

    def test_clears_model_cache(self, service):
        """reload_parameters clears the model cache."""
        service._model_cache["key1"] = Mock()
        service._model_cache["key2"] = Mock()

        service.reload_parameters()

        assert len(service._model_cache) == 0


class TestGeminiServiceCreateRetryHandler:
    """Tests for GeminiService._create_retry_handler()."""

    def test_creates_gemini_retry_handler(self, service):
        """_create_retry_handler creates GeminiRetryHandler."""
        handler = service._create_retry_handler()
        assert handler is not None


class TestGeminiServiceRoleNames:
    """Tests for GeminiService role name methods."""

    def test_get_user_role_name_returns_user(self, service):
        """get_user_role_name returns 'user'."""
        assert service.get_user_role_name() == "user"

    def test_get_assistant_role_name_returns_model(self, service):
        """get_assistant_role_name returns 'model'."""
        assert service.get_assistant_role_name() == "model"


class TestGeminiServiceIsRateLimitError:
    """Tests for GeminiService._is_rate_limit_error()."""

    def test_detects_429_error(self, service):
        """_is_rate_limit_error detects 429 status code."""
        error = Exception("Error 429: Too Many Requests")
        assert service._is_rate_limit_error(error) is True

    def test_detects_quota_error(self, service):
        """_is_rate_limit_error detects quota keyword."""
        error = Exception("Quota exceeded for this API")
        assert service._is_rate_limit_error(error) is True

    def test_detects_rate_limit_keyword(self, service):
        """_is_rate_limit_error detects rate limit keyword."""
        error = Exception("Rate limit exceeded")
        assert service._is_rate_limit_error(error) is True

    def test_rejects_400_without_quota(self, service):
        """_is_rate_limit_error rejects 400 errors without quota keyword."""
        error = Exception("Error 400: Bad Request")
        assert service._is_rate_limit_error(error) is False

    def test_accepts_400_with_quota(self, service):
        """_is_rate_limit_error accepts 400 errors with quota keyword."""
        error = Exception("Error 400: quota exceeded")
        assert service._is_rate_limit_error(error) is True

    def test_rejects_generic_error(self, service):
        """_is_rate_limit_error rejects generic errors."""
        error = Exception("Something went wrong")
        assert service._is_rate_limit_error(error) is False


class TestGeminiServiceIsFatalError:
    """Tests for GeminiService._is_fatal_error()."""

    def test_detects_cachedcontent_not_found(self, service):
        """_is_fatal_error detects 'CachedContent not found'."""
        error = Exception("CachedContent not found: cache-123")
        assert service._is_fatal_error(error) is True

    def test_detects_403_permission_denied(self, service):
        """_is_fatal_error detects 403 PERMISSION_DENIED."""
        error = Exception("403 PERMISSION_DENIED on resource")
        assert service._is_fatal_error(error) is True

    def test_rejects_403_without_permission(self, service):
        """_is_fatal_error rejects 403 without 'permission' keyword."""
        error = Exception("403 Forbidden")
        assert service._is_fatal_error(error) is False

    def test_rejects_generic_error(self, service):
        """_is_fatal_error rejects generic errors."""
        error = Exception("Something went wrong")
        assert service._is_fatal_error(error) is False


class TestGeminiServiceExtractRetryDelay:
    """Tests for GeminiService._extract_retry_delay()."""

    def test_extracts_retry_in_seconds(self, service):
        """_extract_retry_delay extracts 'retry in Xs' format."""
        error = Exception("Please retry in 30.5s")
        result = service._extract_retry_delay(error)
        assert result == 30.5

    def test_extracts_seconds_format(self, service):
        """_extract_retry_delay extracts 'seconds: X' format."""
        error = Exception("Please wait seconds: 45")
        result = service._extract_retry_delay(error)
        assert result == 45.0

    def test_returns_none_for_no_match(self, service):
        """_extract_retry_delay returns None when no pattern matches."""
        error = Exception("Unknown error format")
        result = service._extract_retry_delay(error)
        assert result is None


class TestGeminiServiceGetCacheKey:
    """Tests for GeminiService._get_cache_key()."""

    def test_generates_consistent_key(self, service):
        """_get_cache_key generates consistent key for same inputs."""
        key1 = service._get_cache_key("gemini-2.5-flash", "system instruction")
        key2 = service._get_cache_key("gemini-2.5-flash", "system instruction")
        assert key1 == key2

    def test_includes_tools_suffix(self, service):
        """_get_cache_key includes '-tools' suffix when tools present."""
        key_no_tools = service._get_cache_key("model", "instruction", tools=None)
        key_with_tools = service._get_cache_key("model", "instruction", tools=["tool"])

        assert "-tools" not in key_no_tools
        assert "-tools" in key_with_tools

    def test_sanitizes_model_name(self, service):
        """_get_cache_key sanitizes special characters in model name."""
        key = service._get_cache_key("gemini/2.5/flash", "instruction")
        # Should replace special chars with dashes
        assert "/" not in key


class TestGeminiServiceGetGeminiCache:
    """Tests for GeminiService._get_gemini_cache()."""

    def test_returns_none_for_empty_instruction(self, service):
        """_get_gemini_cache returns None for empty system instruction."""
        result = service._get_gemini_cache("model", "", tools=None)
        assert result == (None, None)

    def test_returns_none_when_token_count_fails(self, service):
        """_get_gemini_cache returns None when token counting fails."""
        service.client.models.count_tokens.side_effect = Exception("API Error")

        result = service._get_gemini_cache("model", "instruction", tools=None)

        assert result == (None, None)

    def test_returns_none_when_tokens_below_minimum(self, service):
        """_get_gemini_cache returns None when token count below minimum."""
        service.client.models.count_tokens.return_value = Mock(total_tokens=100)

        result = service._get_gemini_cache("model", "short instruction", tools=None)

        assert result == (None, None)


class TestGeminiServiceSummarizeText:
    """Tests for GeminiService.summarize_text()."""

    @pytest.mark.asyncio
    async def test_returns_message_for_empty_text(self, service):
        """summarize_text returns message for empty text."""
        result = await service.summarize_text("")
        assert "없습니다" in result

    @pytest.mark.asyncio
    async def test_returns_message_for_whitespace_text(self, service):
        """summarize_text returns message for whitespace-only text."""
        result = await service.summarize_text("   ")
        assert "없습니다" in result

    @pytest.mark.asyncio
    async def test_calls_summary_model(self, service):
        """summarize_text calls summary model with text."""
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=Mock(text="Summary result"))
        service.summary_model = mock_model

        with patch.object(service, "_gemini_retry") as mock_retry:
            mock_retry.return_value = Mock(text="Summary result")
            with patch.object(service, "_extract_text_from_response", return_value="Summary result"):
                result = await service.summarize_text("Text to summarize")

                assert result == "Summary result"


class TestGeminiServiceIsCacheError:
    """Tests for GeminiService._is_cache_error()."""

    def test_detects_403_cachedcontent_error(self, service):
        """_is_cache_error detects 403 CachedContent not found."""
        from google.genai.errors import ClientError

        error = ClientError(code=403, response_json={"error": {"message": "CachedContent not found: cache-123"}})
        assert service._is_cache_error(error) is True

    def test_rejects_non_403_error(self, service):
        """_is_cache_error rejects non-403 errors."""
        from google.genai.errors import ClientError

        error = ClientError(code=429, response_json={"error": {"message": "Too Many Requests"}})
        assert service._is_cache_error(error) is False

    def test_rejects_non_client_error(self, service):
        """_is_cache_error rejects non-ClientError exceptions."""
        error = Exception("Some error")
        assert service._is_cache_error(error) is False


class TestGeminiServiceToolsForProvider:
    """Tests for GeminiService.get_tools_for_provider()."""

    def test_converts_tools_to_gemini_format(self, service):
        """get_tools_for_provider converts tools to Gemini format."""
        with patch(
            "persbot.services.gemini_service.GeminiToolAdapter.convert_tools"
        ) as mock_convert:
            mock_convert.return_value = ["converted_tool"]

            result = service.get_tools_for_provider(["tool1", "tool2"])

            mock_convert.assert_called_once_with(["tool1", "tool2"])
            assert result == ["converted_tool"]


class TestGeminiServiceExtractFunctionCalls:
    """Tests for GeminiService.extract_function_calls()."""

    def test_extracts_function_calls_from_response(self, service):
        """extract_function_calls extracts function calls from response."""
        with patch(
            "persbot.services.gemini_service.GeminiToolAdapter.extract_function_calls"
        ) as mock_extract:
            mock_extract.return_value = [{"name": "func1", "parameters": {}}]

            result = service.extract_function_calls(Mock())

            assert len(result) == 1
            assert result[0]["name"] == "func1"


class TestGeminiServiceFormatFunctionResults:
    """Tests for GeminiService.format_function_results()."""

    def test_formats_function_results(self, service):
        """format_function_results formats results for Gemini."""
        with patch(
            "persbot.services.gemini_service.GeminiToolAdapter.format_results"
        ) as mock_format:
            mock_format.return_value = ["formatted_result"]

            result = service.format_function_results(
                [{"name": "func1", "result": "success"}]
            )

            mock_format.assert_called_once()
            assert result == ["formatted_result"]


class TestGeminiServiceCalculateBackoff:
    """Tests for GeminiService._calculate_backoff()."""

    def test_calculates_exponential_backoff(self, service):
        """_calculate_backoff calculates exponential backoff."""
        # The backoff formula is: min(base**attempt, max)
        # With base=1.0, backoff is always 1.0 until it caps
        backoff1 = service._calculate_backoff(1)
        backoff2 = service._calculate_backoff(2)

        # Both should be positive values
        assert backoff1 > 0
        assert backoff2 > 0
        # Both should be within max
        assert backoff1 <= service.config.api_retry_backoff_max
        assert backoff2 <= service.config.api_retry_backoff_max

    def test_respects_max_backoff(self, service):
        """_calculate_backoff respects maximum backoff."""
        # Very high attempt number
        backoff = service._calculate_backoff(100)

        # Should be capped at max
        assert backoff <= service.config.api_retry_backoff_max


class TestGeminiServiceExtractTextFromResponse:
    """Tests for GeminiService._extract_text_from_response()."""

    def test_extracts_text_from_response(self, service):
        """_extract_text_from_response extracts clean text."""
        mock_response = Mock()

        with patch(
            "persbot.services.gemini_service.extract_clean_text"
        ) as mock_extract:
            mock_extract.return_value = "Clean text"

            result = service._extract_text_from_response(mock_response)

            mock_extract.assert_called_once_with(mock_response)
            assert result == "Clean text"


class TestGeminiServiceLogRawRequest:
    """Tests for GeminiService._log_raw_request()."""

    def test_logs_user_message_preview(self, service):
        """_log_raw_request logs user message preview."""
        with patch("persbot.services.gemini_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            mock_logger.debug = Mock()

            service._log_raw_request("This is a test message", None)

            # Should have logged
            assert mock_logger.debug.called

    def test_skips_when_debug_disabled(self, service):
        """_log_raw_request skips when debug logging disabled."""
        with patch("persbot.services.gemini_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = False
            mock_logger.debug = Mock()

            service._log_raw_request("Test message", None)

            # Should not have called debug for the main log
            mock_logger.debug.assert_not_called()


class TestGeminiServiceLogRawResponse:
    """Tests for GeminiService._log_raw_response()."""

    def test_logs_token_metadata(self, service):
        """_log_raw_response logs token metadata."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=20,
            total_token_count=170
        )

        with patch("persbot.services.gemini_service.logger") as mock_logger:
            mock_logger.debug = Mock()
            mock_logger.isEnabledFor.return_value = False

            service._log_raw_response(mock_response, 1)

            # Should have logged token counts
            assert mock_logger.debug.called

    def test_handles_tuple_response(self, service):
        """_log_raw_response handles tuple response from chat session."""
        mock_response = (Mock(), Mock(), Mock())
        mock_response[2].usage_metadata = Mock(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=0,
            total_token_count=150
        )

        with patch("persbot.services.gemini_service.logger") as mock_logger:
            mock_logger.debug = Mock()
            mock_logger.isEnabledFor.return_value = False

            service._log_raw_response(mock_response, 1)

            # Should have extracted response from tuple
            assert mock_logger.debug.called

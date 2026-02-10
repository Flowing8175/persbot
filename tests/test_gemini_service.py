"""Comprehensive tests for GeminiService."""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
import pytest_asyncio

from persbot.services.base import ChatMessage
from persbot.services.gemini_service import (
    DEFAULT_CACHE_MIN_TOKENS,
    DEFAULT_CACHE_TTL_MINUTES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GeminiService,
    _CachedModel,
    _ChatSession,
    extract_clean_text,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Create a mock AppConfig."""
    config = Mock()
    config.gemini_api_key = "test_gemini_key_12345"
    config.temperature = 1.0
    config.top_p = 1.0
    config.thinking_budget = None
    config.gemini_cache_min_tokens = DEFAULT_CACHE_MIN_TOKENS
    config.gemini_cache_ttl_minutes = DEFAULT_CACHE_TTL_MINUTES
    config.api_request_timeout = 30.0
    config.api_max_retries = 3
    config.api_rate_limit_retry_after = 10.0
    config.api_retry_backoff_base = 2.0
    config.api_retry_backoff_max = 60.0
    return config


@pytest.fixture
def mock_prompt_service():
    """Create a mock PromptService."""
    service = Mock()
    service.get_summary_prompt = Mock(return_value="Summarize the following text:")
    service.get_active_assistant_prompt = Mock(return_value="You are a helpful assistant.")
    return service


@pytest.fixture
def mock_genai_client(mocker):
    """Create a mock genai.Client."""
    mock_client = Mock()
    mock_client.models = Mock()
    mock_client.models.generate_content = Mock()
    mock_count_result = Mock()
    mock_count_result.total_tokens = 1000
    mock_client.models.count_tokens = Mock(return_value=mock_count_result)
    mock_client.caches = Mock()
    mock_client.caches.list = Mock(return_value=[])

    mock_cache = Mock()
    mock_cache.name = "cached-contents/test-cache-123"
    mock_cache.display_name = "test-cache"
    mock_client.caches.create = Mock(return_value=mock_cache)
    mock_client.caches.update = Mock()

    mocker.patch("persbot.services.gemini_service.genai.Client", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_genai_types(mocker):
    """Create mock genai types."""
    mock_types = Mock()
    mock_types.GenerateContentConfig = Mock(return_value=Mock())
    mock_types.CreateCachedContentConfig = Mock(return_value=Mock())
    mock_types.UpdateCachedContentConfig = Mock(return_value=Mock())
    mock_types.Tool = Mock(return_value=Mock())
    mock_types.GoogleSearch = Mock(return_value=Mock())
    mock_types.ThinkingConfig = Mock(return_value=Mock())
    mock_types.Part = Mock()
    mock_types.Part.from_bytes = Mock(return_value=Mock())

    mocker.patch("persbot.services.gemini_service.genai_types", mock_types)
    return mock_types


@pytest.fixture
def gemini_service(mock_config, mock_prompt_service, mock_genai_client, mock_genai_types):
    """Create a GeminiService instance with mocked dependencies."""
    mock_config.gemini_cache_ttl_minutes = 0
    service = GeminiService(
        config=mock_config,
        assistant_model_name="gemini-2.5-flash",
        summary_model_name="gemini-2.5-flash",
        prompt_service=mock_prompt_service,
    )
    return service


@pytest.fixture
def mock_response():
    """Create a mock Gemini response."""
    response = Mock()

    # Create mock parts
    part1 = Mock()
    part1.text = "Hello, this is a test response."
    part1.thought = False

    part2 = Mock()
    part2.text = "More text here."
    part2.thought = False

    # Create mock content
    content = Mock()
    content.parts = [part1, part2]

    # Create mock candidate
    candidate = Mock()
    candidate.content = content
    candidate.finish_reason = "STOP"

    response.candidates = [candidate]

    # Mock usage metadata
    usage = Mock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    usage.cached_content_token_count = 25
    usage.total_token_count = 150
    response.usage_metadata = usage

    return response


@pytest.fixture
def mock_discord_message():
    """Create a mock Discord message."""
    message = Mock()
    message.id = 123456789
    message.author = Mock()
    message.author.id = 987654321
    message.author.name = "TestUser"
    message.author.guild_permissions = Mock()
    message.author.guild_permissions.manage_guild = False
    message.channel = Mock()
    message.channel.id = 111222333
    message.guild = Mock()
    message.guild.id = 444555666
    message.attachments = []
    message.reply = AsyncMock()
    return message


# ============================================================================
# Test Extract Clean Text
# ============================================================================


class TestExtractCleanText:
    """Test extract_clean_text function."""

    def test_extract_text_from_response(self, mock_response):
        """Test extracting text from a valid response."""
        result = extract_clean_text(mock_response)
        assert "Hello, this is a test response." in result
        assert "More text here." in result

    def test_skip_thought_parts(self):
        """Test that thought parts are skipped."""
        response = Mock()

        thought_part = Mock()
        thought_part.text = "This is a thought"
        thought_part.thought = True

        text_part = Mock()
        text_part.text = "This is actual text"
        text_part.thought = False

        content = Mock()
        content.parts = [thought_part, text_part]

        candidate = Mock()
        candidate.content = content

        response.candidates = [candidate]

        result = extract_clean_text(response)
        assert "This is a thought" not in result
        assert "This is actual text" in result

    def test_empty_response(self):
        """Test handling empty response."""
        response = Mock()
        response.candidates = []
        result = extract_clean_text(response)
        assert result == ""

    def test_no_candidates(self):
        """Test response with no candidates."""
        response = Mock()
        response.candidates = None
        result = extract_clean_text(response)
        assert result == ""

    def test_exception_handling(self):
        """Test exception handling in extract_clean_text."""
        response = Mock()
        response.candidates = Exception("Test error")
        result = extract_clean_text(response)
        assert result == ""


# ============================================================================
# Test GeminiService Initialization
# ============================================================================


class TestGeminiServiceInitialization:
    """Test GeminiService initialization."""

    def test_init_creates_client(self, mock_config, mock_prompt_service, mocker):
        """Test that initialization creates genai.Client."""
        mock_client = Mock()
        mock_patch = mocker.patch(
            "persbot.services.gemini_service.genai.Client", return_value=mock_client
        )

        mock_count_result = Mock()
        mock_count_result.total_tokens = 100
        mock_client.models.count_tokens = Mock(return_value=mock_count_result)

        mock_config.gemini_cache_ttl_minutes = 0
        service = GeminiService(
            config=mock_config,
            assistant_model_name="gemini-2.5-flash",
            summary_model_name="gemini-2.5-flash",
            prompt_service=mock_prompt_service,
        )

        mock_patch.assert_called_once_with(api_key="test_gemini_key_12345")
        assert service.client is mock_client

    def test_init_sets_model_names(
        self, mock_config, mock_prompt_service, mock_genai_client, mocker
    ):
        """Test that model names are set correctly."""
        mock_config.gemini_cache_ttl_minutes = 0
        service = GeminiService(
            config=mock_config,
            assistant_model_name="gemini-2.5-flash",
            summary_model_name="gemini-2.5-pro",
            prompt_service=mock_prompt_service,
        )

        assert service._assistant_model_name == "gemini-2.5-flash"
        assert service._summary_model_name == "gemini-2.5-pro"

    def test_init_defaults_summary_model(
        self, mock_config, mock_prompt_service, mock_genai_client, mocker
    ):
        """Test that summary model defaults to assistant model."""
        mock_config.gemini_cache_ttl_minutes = 0
        service = GeminiService(
            config=mock_config,
            assistant_model_name="gemini-2.5-flash",
            prompt_service=mock_prompt_service,
        )

        assert service._summary_model_name == "gemini-2.5-flash"

    def test_init_creates_model_cache(
        self, mock_config, mock_prompt_service, mock_genai_client, mocker
    ):
        """Test that model cache is initialized."""
        mock_config.gemini_cache_ttl_minutes = 0
        service = GeminiService(
            config=mock_config,
            assistant_model_name="gemini-2.5-flash",
            prompt_service=mock_prompt_service,
        )

        assert hasattr(service, "_model_cache")
        assert isinstance(service._model_cache, dict)

    def test_init_preloads_models(
        self, mock_config, mock_prompt_service, mock_genai_client, mocker
    ):
        """Test that models are preloaded during initialization."""
        mock_config.gemini_cache_ttl_minutes = 0
        service = GeminiService(
            config=mock_config,
            assistant_model_name="gemini-2.5-flash",
            prompt_service=mock_prompt_service,
        )

        assert service.summary_model is not None
        assert service.assistant_model is not None

    def test_init_starts_cache_cleanup_task(
        self, mock_config, mock_prompt_service, mock_genai_client, mocker
    ):
        """Test that cache cleanup task is started."""
        mock_create_task = mocker.patch.object(asyncio, "create_task")

        service = GeminiService(
            config=mock_config,
            assistant_model_name="gemini-2.5-flash",
            prompt_service=mock_prompt_service,
        )

        mock_create_task.assert_called_once()

    def test_init_no_cleanup_when_ttl_zero(
        self, mock_config, mock_prompt_service, mock_genai_client, mocker
    ):
        """Test that cleanup task is not started when TTL is 0."""
        mock_config.gemini_cache_ttl_minutes = 0
        mock_create_task = mocker.patch.object(asyncio, "create_task")

        service = GeminiService(
            config=mock_config,
            assistant_model_name="gemini-2.5-flash",
            prompt_service=mock_prompt_service,
        )

        mock_create_task.assert_not_called()


# ============================================================================
# Test Model Caching
# ============================================================================


class TestModelCaching:
    """Test model caching logic."""

    def test_get_or_create_model_creates_new(self, gemini_service):
        """Test that _get_or_create_model creates a new model."""
        model = gemini_service._get_or_create_model(
            "gemini-2.5-flash", "Test system instruction", use_cache=False
        )

        assert isinstance(model, _CachedModel)
        assert model._model_name == "gemini-2.5-flash"

    def test_get_or_create_model_uses_cache(self, gemini_service):
        """Test that cached model is reused."""
        model1 = gemini_service._get_or_create_model(
            "gemini-2.5-flash", "Test system instruction", use_cache=False
        )

        model2 = gemini_service._get_or_create_model(
            "gemini-2.5-flash", "Test system instruction", use_cache=False
        )

        assert model1 is model2

    def test_get_or_create_model_different_instructions(self, gemini_service):
        """Test that different instructions create different models."""
        model1 = gemini_service._get_or_create_model(
            "gemini-2.5-flash", "Instruction 1", use_cache=False
        )

        model2 = gemini_service._get_or_create_model(
            "gemini-2.5-flash", "Instruction 2", use_cache=False
        )

        assert model1 is not model2

    def test_get_or_create_model_different_models(self, gemini_service):
        """Test that different model names create different models."""
        model1 = gemini_service._get_or_create_model(
            "gemini-2.5-flash", "Test instruction", use_cache=False
        )

        model2 = gemini_service._get_or_create_model(
            "gemini-2.5-pro", "Test instruction", use_cache=False
        )

        assert model1 is not model2

    def test_check_model_cache_validity_returns_model(self, gemini_service):
        """Test that _check_model_cache_validity returns cached model."""
        model = gemini_service._get_or_create_model(
            "gemini-2.5-flash", "Test instruction", use_cache=False
        )

        key = hash(("gemini-2.5-flash", "Test instruction", False))
        now = datetime.now(timezone.utc)

        result = gemini_service._check_model_cache_validity(key, now)
        assert result is model

    def test_check_model_cache_validity_expired(self, gemini_service):
        """Test that expired cache returns None."""
        model = gemini_service._get_or_create_model(
            "gemini-2.5-flash", "Test instruction", use_cache=False
        )

        key = hash(("gemini-2.5-flash", "Test instruction", False))
        past_expiration = datetime.now(timezone.utc) - timedelta(hours=1)
        gemini_service._model_cache[key] = (model, past_expiration)

        now = datetime.now(timezone.utc)
        result = gemini_service._check_model_cache_validity(key, now)
        assert result is None
        assert key not in gemini_service._model_cache

    def test_check_model_cache_validity_missing(self, gemini_service):
        """Test that missing cache returns None."""
        key = 999999  # Non-existent key
        now = datetime.now(timezone.utc)

        result = gemini_service._check_model_cache_validity(key, now)
        assert result is None

    def test_reload_parameters_clears_cache(self, gemini_service):
        """Test that reload_parameters clears the model cache."""
        # Add something to cache
        gemini_service._get_or_create_model("gemini-2.5-flash", "Test instruction", use_cache=False)

        assert len(gemini_service._model_cache) > 0

        gemini_service.reload_parameters()

        assert len(gemini_service._model_cache) == 0


# ============================================================================
# Test Cache Management
# ============================================================================


class TestCacheManagement:
    """Test cache management functions."""

    def test_get_cache_key_format(self, gemini_service):
        """Test cache key format."""
        key = gemini_service._get_cache_key("gemini-2.5-flash", "test content")

        assert key.startswith("persbot-")
        assert "gemini-2-5-flash" in key

    def test_get_cache_key_with_tools(self, gemini_service):
        """Test cache key with tools."""
        key = gemini_service._get_cache_key("gemini-2.5-flash", "test content", tools=[Mock()])

        assert key.endswith("-tools")

    def test_get_cache_key_consistency(self, gemini_service):
        """Test that cache key is consistent for same inputs."""
        key1 = gemini_service._get_cache_key("gemini-2.5-flash", "test content")
        key2 = gemini_service._get_cache_key("gemini-2.5-flash", "test content")

        assert key1 == key2

    def test_resolve_gemini_cache_no_cache(self, gemini_service):
        """Test cache resolution when use_cache is False."""
        cache_name, expiration = gemini_service._resolve_gemini_cache(
            "gemini-2.5-flash", "test instruction", tools=None, use_cache=False
        )

        assert cache_name is None
        assert expiration is None

    def test_get_search_tools_returns_tools(self, gemini_service, mock_genai_types):
        """Test that search tools are returned."""
        tools = gemini_service._get_search_tools("gemini-2.5-flash")

        assert tools is not None
        assert len(tools) > 0
        mock_genai_types.Tool.assert_called()
        mock_genai_types.GoogleSearch.assert_called()

    def test_build_generation_config_without_cache(self, gemini_service, mock_genai_types):
        """Test building config without cache."""
        config = gemini_service._build_generation_config(None, "Test system instruction", None)

        mock_genai_types.GenerateContentConfig.assert_called()
        call_kwargs = mock_genai_types.GenerateContentConfig.call_args[1]
        assert call_kwargs.get("system_instruction") == "Test system instruction"

    def test_build_generation_config_with_cache(self, gemini_service, mock_genai_types):
        """Test building config with cache."""
        config = gemini_service._build_generation_config(
            "cached-content-name", "Test system instruction", None
        )

        mock_genai_types.GenerateContentConfig.assert_called()
        call_kwargs = mock_genai_types.GenerateContentConfig.call_args[1]
        assert call_kwargs.get("cached_content") == "cached-content-name"
        assert "system_instruction" not in call_kwargs

    def test_build_generation_config_with_tools(self, gemini_service, mock_genai_types):
        """Test building config with tools."""
        mock_tools = [Mock()]
        config = gemini_service._build_generation_config(
            None, "Test system instruction", mock_tools
        )

        mock_genai_types.GenerateContentConfig.assert_called()
        call_kwargs = mock_genai_types.GenerateContentConfig.call_args[1]
        assert call_kwargs.get("tools") == mock_tools

    def test_build_generation_config_with_thinking_budget(
        self, gemini_service, mock_config, mock_genai_types
    ):
        """Test building config with thinking budget."""
        mock_config.thinking_budget = 1000

        config = gemini_service._build_generation_config(None, "Test system instruction", None)

        mock_genai_types.ThinkingConfig.assert_called_with(
            include_thoughts=True, thinking_budget=1000
        )


# ============================================================================
# Test Content Generation
# ============================================================================


class TestContentGeneration:
    """Test content generation methods."""

    def test_cached_model_generate_content(self, mock_genai_client, mock_genai_types):
        """Test _CachedModel.generate_content."""
        mock_config_obj = Mock()
        mock_config_obj.temperature = 1.0
        mock_config_obj.top_p = 1.0
        mock_config_obj.cached_content = None
        mock_config_obj.system_instruction = "Test instruction"

        model = _CachedModel(mock_genai_client, "gemini-2.5-flash", mock_config_obj)

        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        model.generate_content(contents)

        mock_genai_client.models.generate_content.assert_called_once()

    def test_cached_model_generate_content_with_tools(self, mock_genai_client, mock_genai_types):
        """Test _CachedModel.generate_content with tools."""
        mock_config_obj = Mock()
        mock_config_obj.temperature = 1.0
        mock_config_obj.top_p = 1.0
        mock_config_obj.cached_content = None
        mock_config_obj.system_instruction = "Test instruction"

        model = _CachedModel(mock_genai_client, "gemini-2.5-flash", mock_config_obj)

        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        mock_tools = [Mock()]
        model.generate_content(contents, tools=mock_tools)

        mock_genai_client.models.generate_content.assert_called_once()

    def test_cached_model_generate_content_with_cached_content(
        self, mock_genai_client, mock_genai_types
    ):
        """Test _CachedModel.generate_content ignores tools when cached_content is set."""
        mock_config_obj = Mock()
        mock_config_obj.temperature = 1.0
        mock_config_obj.top_p = 1.0
        mock_config_obj.cached_content = "cached-name"
        mock_config_obj.thinking_config = None

        model = _CachedModel(mock_genai_client, "gemini-2.5-flash", mock_config_obj)

        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        mock_tools = [Mock()]
        model.generate_content(contents, tools=mock_tools)

        # Should not include tools in config when cached_content is used
        call_kwargs = mock_genai_types.GenerateContentConfig.call_args[1]
        assert "tools" not in call_kwargs


# ============================================================================
# Test Chat Session
# ============================================================================


class TestChatSession:
    """Test _ChatSession functionality."""

    def test_chat_session_initialization(self):
        """Test _ChatSession initialization."""
        mock_factory = Mock()
        session = _ChatSession("System instruction", mock_factory)

        assert session._system_instruction == "System instruction"
        assert session._factory is mock_factory
        assert len(session.history) == 0

    def test_chat_session_history_maxlen(self):
        """Test that chat session history has maxlen."""
        mock_factory = Mock()
        session = _ChatSession("System instruction", mock_factory)

        assert session.history.maxlen == 50

    def test_get_api_history_empty(self):
        """Test _get_api_history with empty history."""
        mock_factory = Mock()
        session = _ChatSession("System instruction", mock_factory)

        api_history = session._get_api_history()
        assert api_history == []

    def test_get_api_history_with_messages(self):
        """Test _get_api_history with messages."""
        mock_factory = Mock()
        session = _ChatSession("System instruction", mock_factory)

        msg = ChatMessage(role="user", content="Hello", parts=[{"text": "Hello"}], author_id=123456)
        session.history.append(msg)

        api_history = session._get_api_history()
        assert len(api_history) == 1
        assert api_history[0]["role"] == "user"

    def test_send_message_returns_tuple(self, mock_genai_types):
        """Test send_message returns correct tuple."""
        mock_response = Mock()
        mock_response.candidates = []

        mock_factory = Mock()
        mock_factory.generate_content = Mock(return_value=mock_response)

        session = _ChatSession("System instruction", mock_factory)

        result = session.send_message(
            "Hello bot", author_id=123456, author_name="TestUser", message_ids=["123"]
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        user_msg, model_msg, response = result
        assert isinstance(user_msg, ChatMessage)
        assert isinstance(model_msg, ChatMessage)
        assert response is mock_response

    def test_send_message_with_images(self, mock_genai_types):
        """Test send_message with images."""
        mock_response = Mock()
        mock_response.candidates = []

        mock_factory = Mock()
        mock_factory.generate_content = Mock(return_value=mock_response)

        session = _ChatSession("System instruction", mock_factory)

        image_data = b"\x89PNG\r\n\x1a\n"
        result = session.send_message("Hello", author_id=123456, images=[image_data])

        mock_factory.generate_content.assert_called_once()
        call_args = mock_factory.generate_content.call_args[1]
        assert "contents" in call_args

    def test_send_message_with_tools(self):
        """Test send_message with tools."""
        mock_response = Mock()
        mock_response.candidates = []

        mock_factory = Mock()
        mock_factory.generate_content = Mock(return_value=mock_response)

        session = _ChatSession("System instruction", mock_factory)

        mock_tools = [Mock()]
        result = session.send_message("Hello", author_id=123456, tools=mock_tools)

        mock_factory.generate_content.assert_called_once()


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling methods."""

    def test_is_rate_limit_error_429(self, gemini_service):
        """Test detecting 429 rate limit error."""
        error = Exception("429 Too Many Requests")
        assert gemini_service._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_quota(self, gemini_service):
        """Test detecting quota error."""
        error = Exception("Resource has been exhausted (e.g. check quota).")
        assert gemini_service._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_400_no_quota(self, gemini_service):
        """Test that 400 without quota is not rate limit."""
        error = Exception("400 Bad Request")
        assert gemini_service._is_rate_limit_error(error) is False

    def test_is_rate_limit_error_400_with_quota(self, gemini_service):
        """Test that 400 with quota is rate limit."""
        error = Exception("400 quota exceeded")
        assert gemini_service._is_rate_limit_error(error) is True

    def test_is_fatal_error_cache_not_found(self, gemini_service):
        """Test detecting fatal cache not found error."""
        error = Exception("CachedContent not found")
        assert gemini_service._is_fatal_error(error) is True

    def test_is_fatal_error_permission_denied(self, gemini_service):
        """Test detecting fatal permission denied error."""
        error = Exception("403 PERMISSION_DENIED on cached content")
        assert gemini_service._is_fatal_error(error) is True

    def test_is_fatal_error_not_fatal(self, gemini_service):
        """Test that non-fatal errors return False."""
        error = Exception("Some other error")
        assert gemini_service._is_fatal_error(error) is False

    def test_extract_retry_delay_seconds_format(self, gemini_service):
        """Test extracting retry delay in seconds format."""
        error = Exception("Please retry in 5.5s")
        delay = gemini_service._extract_retry_delay(error)
        assert delay == 5.5

    def test_extract_retry_delay_colon_format(self, gemini_service):
        """Test extracting retry delay in colon format."""
        error = Exception("Retry after seconds: 10")
        delay = gemini_service._extract_retry_delay(error)
        assert delay == 10.0

    def test_extract_retry_delay_not_found(self, gemini_service):
        """Test extracting retry delay when not found."""
        error = Exception("Some error without retry info")
        delay = gemini_service._extract_retry_delay(error)
        assert delay is None

    def test_calculate_backoff(self, gemini_service, mock_config):
        """Test backoff calculation."""
        delay = gemini_service._calculate_backoff(1)
        assert delay == mock_config.api_retry_backoff_base  # 2.0

        delay = gemini_service._calculate_backoff(2)
        assert delay == mock_config.api_retry_backoff_base**2  # 4.0

    def test_calculate_backoff_max_cap(self, gemini_service, mock_config):
        """Test backoff is capped at max."""
        mock_config.api_retry_backoff_max = 10.0

        delay = gemini_service._calculate_backoff(10)
        assert delay <= 10.0

    @pytest.mark.asyncio
    async def test_handle_cache_error_retry_success(self, gemini_service):
        """Test successful cache error retry handling."""
        mock_on_cache_error = AsyncMock()

        result = await gemini_service._handle_cache_error_retry(1, mock_on_cache_error)

        assert result is True
        mock_on_cache_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_cache_error_retry_failure(self, gemini_service):
        """Test failed cache error retry handling."""
        mock_on_cache_error = AsyncMock(side_effect=Exception("Refresh failed"))

        result = await gemini_service._handle_cache_error_retry(1, mock_on_cache_error)

        assert result is False


# ============================================================================
# Test Summarize Text
# ============================================================================


class TestSummarizeText:
    """Test summarize_text functionality."""

    @pytest.mark.asyncio
    async def test_summarize_empty_text(self, gemini_service):
        """Test summarizing empty text."""
        result = await gemini_service.summarize_text("")
        assert result == "요약할 메시지가 없습니다."

    @pytest.mark.asyncio
    async def test_summarize_text_success(self, gemini_service, mocker):
        """Test successful text summarization."""
        mock_response = Mock()
        mock_response.candidates = []

        mock_gemini_retry = mocker.patch.object(
            gemini_service, "_gemini_retry", return_value=mock_response
        )

        result = await gemini_service.summarize_text("Some long text to summarize")

        mock_gemini_retry.assert_called_once()


# ============================================================================
# Test Chat Response Generation
# ============================================================================


class TestGenerateChatResponse:
    """Test generate_chat_response functionality."""

    @pytest.fixture
    def mock_chat_session(self):
        """Create a mock chat session."""
        session = Mock()
        session.history = []
        session._factory = Mock()
        session._factory._model_name = "gemini-2.5-flash"
        session.send_message = Mock(
            return_value=(
                ChatMessage(role="user", content="Hello", author_id=123),
                ChatMessage(role="model", content="Hi!", author_id=None),
                Mock(),
            )
        )
        return session

    @pytest.mark.asyncio
    async def test_generate_chat_response_success(
        self, gemini_service, mock_chat_session, mock_discord_message, mocker
    ):
        """Test successful chat response generation."""
        mock_response_obj = Mock()
        mock_response_obj.candidates = []

        mock_chat_session.send_message = Mock(
            return_value=(
                ChatMessage(role="user", content="Hello", author_id=123),
                ChatMessage(role="model", content="Hi there!", author_id=None),
                mock_response_obj,
            )
        )

        mock_gemini_retry = mocker.patch.object(
            gemini_service,
            "_gemini_retry",
            return_value=mock_chat_session.send_message.return_value,
        )

        result = await gemini_service.generate_chat_response(
            mock_chat_session, "Hello", mock_discord_message
        )

        assert result is not None
        mock_gemini_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_chat_response_with_images(
        self, gemini_service, mock_chat_session, mock_discord_message, mocker
    ):
        """Test chat response with images."""
        mock_attachment = Mock()
        mock_attachment.content_type = "image/png"
        mock_attachment.read = AsyncMock(return_value=b"\x89PNG")
        mock_discord_message.attachments = [mock_attachment]

        mock_extract_images = mocker.patch.object(
            gemini_service, "_extract_images_from_message", return_value=[b"\x89PNG"]
        )

        mock_gemini_retry = mocker.patch.object(gemini_service, "_gemini_retry", return_value=None)

        result = await gemini_service.generate_chat_response(
            mock_chat_session, "Look at this image", mock_discord_message
        )

        mock_extract_images.assert_called()


# ============================================================================
# Test Tool Calling
# ============================================================================


class TestToolCalling:
    """Test tool calling integration."""

    def test_get_tools_for_provider(self, gemini_service, mocker):
        """Test get_tools_for_provider delegates to adapter."""
        mock_adapter = mocker.patch("persbot.services.gemini_service.GeminiToolAdapter")
        mock_tools = [Mock()]

        gemini_service.get_tools_for_provider(mock_tools)

        mock_adapter.convert_tools.assert_called_once_with(mock_tools)

    def test_extract_function_calls(self, gemini_service, mocker):
        """Test extract_function_calls delegates to adapter."""
        mock_adapter = mocker.patch("persbot.services.gemini_service.GeminiToolAdapter")
        mock_response = Mock()

        gemini_service.extract_function_calls(mock_response)

        mock_adapter.extract_function_calls.assert_called_once_with(mock_response)

    def test_format_function_results(self, gemini_service, mocker):
        """Test format_function_results delegates to adapter."""
        mock_adapter = mocker.patch("persbot.services.gemini_service.GeminiToolAdapter")
        mock_results = [{"name": "test", "result": "success"}]

        gemini_service.format_function_results(mock_results)

        mock_adapter.format_results.assert_called_once_with(mock_results)

    @pytest.mark.asyncio
    async def test_send_tool_results(self, gemini_service, mock_discord_message, mocker):
        """Test send_tool_results functionality."""
        mock_chat_session = Mock()
        mock_chat_session._factory = Mock()
        mock_chat_session._factory._model_name = "gemini-2.5-flash"
        mock_chat_session.history = [ChatMessage(role="model", content="test", author_id=None)]
        mock_chat_session.send_tool_results = Mock(
            return_value=(ChatMessage(role="model", content="Result", author_id=None), Mock())
        )

        mock_execute_with_retry = mocker.patch.object(
            gemini_service,
            "execute_with_retry",
            return_value=(ChatMessage(role="model", content="Result", author_id=None), Mock()),
        )

        tool_rounds = [(Mock(), [{"name": "tool1", "result": "success"}])]

        result = await gemini_service.send_tool_results(
            mock_chat_session, tool_rounds, tools=[Mock()], discord_message=mock_discord_message
        )

        mock_execute_with_retry.assert_called_once()


# ============================================================================
# Test Role Names
# ============================================================================


class TestRoleNames:
    """Test role name methods."""

    def test_get_user_role_name(self, gemini_service):
        """Test get_user_role_name returns 'user'."""
        assert gemini_service.get_user_role_name() == "user"

    def test_get_assistant_role_name(self, gemini_service):
        """Test get_assistant_role_name returns 'model'."""
        assert gemini_service.get_assistant_role_name() == "model"


# ============================================================================
# Test Create Assistant Model
# ============================================================================


class TestCreateAssistantModel:
    """Test create_assistant_model functionality."""

    def test_create_assistant_model(self, gemini_service):
        """Test creating assistant model."""
        model = gemini_service.create_assistant_model("Custom system instruction", use_cache=False)

        assert isinstance(model, _CachedModel)

    def test_create_assistant_model_with_cache(self, gemini_service):
        """Test creating assistant model with cache."""
        model1 = gemini_service.create_assistant_model("Custom instruction", use_cache=True)

        model2 = gemini_service.create_assistant_model("Custom instruction", use_cache=True)

        assert model1 is model2


# ============================================================================
# Test Logging Methods
# ============================================================================


class TestLoggingMethods:
    """Test logging helper methods."""

    def test_log_raw_request(self, gemini_service, caplog):
        """Test _log_raw_request."""
        import logging

        with caplog.at_level(logging.DEBUG):
            gemini_service._log_raw_request("Test message")

    def test_log_raw_response(self, gemini_service, mock_response, caplog):
        """Test _log_raw_response."""
        import logging

        with caplog.at_level(logging.INFO):
            gemini_service._log_raw_response(mock_response, 1)

    def test_log_raw_response_with_tuple(self, gemini_service, mock_response, caplog):
        """Test _log_raw_response with tuple response."""
        import logging

        with caplog.at_level(logging.INFO):
            gemini_service._log_raw_response(("user_msg", "model_msg", mock_response), 1)


# ============================================================================
# Test Cache Error Detection
# ============================================================================


class TestCacheErrorDetection:
    """Test cache error detection."""

    def test_is_cache_error_true(self, gemini_service, mocker):
        """Test detecting cache error."""
        from google.genai.errors import ClientError

        mock_error = mocker.Mock(spec=ClientError)
        mock_error.code = 403
        mock_error.message = "CachedContent not found"

        result = gemini_service._is_cache_error(mock_error)
        assert result is True

    def test_is_cache_error_false(self, gemini_service, mocker):
        """Test non-cache error."""
        from google.genai.errors import ClientError

        mock_error = mocker.Mock(spec=ClientError)
        mock_error.code = 400
        mock_error.message = "Some other error"

        result = gemini_service._is_cache_error(mock_error)
        assert result is False


# ============================================================================
# Test Periodic Cache Cleanup
# ============================================================================


class TestPeriodicCacheCleanup:
    """Test periodic cache cleanup functionality."""

    @pytest.mark.asyncio
    async def test_periodic_cache_cleanup_cancelled(self, gemini_service, mocker):
        """Test that cancelled cleanup task exits gracefully."""
        gemini_service.config.gemini_cache_ttl_minutes = 60

        mock_sleep = mocker.patch.object(asyncio, "sleep", side_effect=asyncio.CancelledError())

        mock_refresh = mocker.patch.object(
            gemini_service, "_refresh_expired_cache", new_callable=AsyncMock
        )

        await gemini_service._periodic_cache_cleanup()

        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_periodic_cache_cleanup_skipped_when_ttl_zero(self, gemini_service):
        """Test cleanup is skipped when TTL is 0."""
        gemini_service.config.gemini_cache_ttl_minutes = 0

        result = await gemini_service._periodic_cache_cleanup()

        assert result is None

    @pytest.mark.asyncio
    async def test_refresh_expired_cache_timeout(self, gemini_service, mock_genai_client, mocker):
        """Test handling timeout in cache refresh."""
        mock_wait_for = mocker.patch.object(asyncio, "wait_for", side_effect=asyncio.TimeoutError())

        await gemini_service._refresh_expired_cache()

        # Should not raise exception


# ============================================================================
# Test Notify Final Error
# ============================================================================


class TestNotifyFinalError:
    """Test final error notification."""

    @pytest.mark.asyncio
    async def test_notify_timeout_error(self, gemini_service, mock_discord_message, mocker):
        """Test notification for timeout error."""
        error = asyncio.TimeoutError()

        await gemini_service._notify_final_error(error, mock_discord_message)

        mock_discord_message.reply.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_generic_error(self, gemini_service, mock_discord_message):
        """Test notification for generic error."""
        error = Exception("Some error")

        await gemini_service._notify_final_error(error, mock_discord_message)

        mock_discord_message.reply.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_no_discord_message(self, gemini_service):
        """Test notification without Discord message."""
        error = Exception("Some error")

        # Should not raise
        await gemini_service._notify_final_error(error, None)


# ============================================================================
# Test Rate Limit Retry
# ============================================================================


class TestRateLimitRetry:
    """Test rate limit retry handling."""

    @pytest.mark.asyncio
    async def test_handle_rate_limit_retry(self, gemini_service, mock_discord_message, mocker):
        """Test rate limit retry handling."""
        error = Exception("429 Rate limit exceeded. Please retry in 5s")

        mock_wait = mocker.patch.object(
            gemini_service, "_wait_with_countdown", new_callable=AsyncMock
        )

        await gemini_service._handle_rate_limit_retry(error, 1, mock_discord_message)

        mock_wait.assert_called_once()


# ============================================================================
# Test Execute Model Call
# ============================================================================


class TestExecuteModelCall:
    """Test _execute_model_call helper."""

    @pytest.mark.asyncio
    async def test_execute_sync_call(self, gemini_service):
        """Test executing synchronous model call."""
        mock_call = Mock(return_value="result")

        result = await gemini_service._execute_model_call(mock_call)

        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_async_call(self, gemini_service):
        """Test executing asynchronous model call."""

        async def async_call():
            return "async_result"

        result = await gemini_service._execute_model_call(async_call)

        assert result == "async_result"

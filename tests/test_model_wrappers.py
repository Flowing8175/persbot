"""Tests for model wrappers.

Tests cover:
- GeminiCachedModel: Wrapper for Gemini models with cached content support
- OpenAIChatCompletionModel: Wrapper for OpenAI chat completion models
"""

from typing import Any, List
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest

from persbot.services.model_wrappers.gemini_model import GeminiCachedModel
from persbot.services.model_wrappers.openai_model import OpenAIChatCompletionModel
from persbot.services.session_wrappers.openai_session import (
    ChatCompletionSession,
    ResponseSession,
)
from persbot.services.session_wrappers.gemini_session import GeminiChatSession


# Helper class to simulate genai_types config
class MockGenerateContentConfig:
    """Mock GenerateContentConfig for testing."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        system_instruction: str = None,
        cached_content=None,
        thinking_config=None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.system_instruction = system_instruction
        self.cached_content = cached_content
        self.thinking_config = thinking_config

    def __eq__(self, other):
        if not isinstance(other, MockGenerateContentConfig):
            return False
        return (
            self.temperature == other.temperature
            and self.top_p == other.top_p
            and self.system_instruction == other.system_instruction
        )


class MockThinkingConfig:
    """Mock ThinkingConfig for testing."""

    def __init__(self, thinking_budget: int = 0):
        self.thinking_budget = thinking_budget


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_genai_client():
    """Create a mock genai.Client."""
    client = Mock()
    client.models = Mock()
    client.models.generate_content = Mock(return_value=Mock(text="Response"))
    client.models.count_tokens = Mock(return_value=Mock(total_tokens=500))
    client.aio = Mock()
    client.aio.models = Mock()
    return client


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    ))
    client.responses = Mock()
    client.responses.create = Mock(return_value=Mock(
        output_text="Test response"
    ))
    return client


@pytest.fixture
def gemini_config():
    """Create a basic Gemini config."""
    return MockGenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        system_instruction="You are a helpful assistant.",
    )


@pytest.fixture
def gemini_config_with_cache():
    """Create a Gemini config with cached content."""
    mock_cache = Mock()
    mock_cache.name = "cache-123"
    return MockGenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        cached_content=mock_cache,
    )


@pytest.fixture
def gemini_config_with_thinking():
    """Create a Gemini config with thinking config."""
    thinking_config = MockThinkingConfig(thinking_budget=5000)
    return MockGenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        system_instruction="You are a helpful assistant.",
        thinking_config=thinking_config,
    )


# ============================================================================
# GeminiCachedModel Tests
# ============================================================================


class TestGeminiCachedModelInit:
    """Tests for GeminiCachedModel.__init__()."""

    def test_initializes_with_client_and_config(self, mock_genai_client, gemini_config):
        """__init__ stores client, model_name, and config."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        assert model._client == mock_genai_client
        assert model._model_name == "gemini-2.5-flash"
        assert model._config == gemini_config

    def test_initializes_with_cached_content_config(
        self, mock_genai_client, gemini_config_with_cache
    ):
        """__init__ works with cached content config."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config_with_cache,
        )

        assert model._config == gemini_config_with_cache
        assert model.has_cached_content is True


class TestGeminiCachedModelProperties:
    """Tests for GeminiCachedModel properties."""

    def test_model_name_property(self, mock_genai_client, gemini_config):
        """model_name returns the configured model name."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-pro",
            config=gemini_config,
        )

        assert model.model_name == "gemini-2.5-pro"

    def test_has_cached_content_true(self, mock_genai_client, gemini_config_with_cache):
        """has_cached_content returns True when config has cached_content."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config_with_cache,
        )

        assert model.has_cached_content is True

    def test_has_cached_content_false(self, mock_genai_client, gemini_config):
        """has_cached_content returns False when config has no cached_content."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        assert model.has_cached_content is False


class TestGeminiCachedModelGenerateContent:
    """Tests for GeminiCachedModel.generate_content()."""

    def test_generate_content_without_tools(
        self, mock_genai_client, gemini_config
    ):
        """generate_content calls client with contents and base config."""
        mock_response = Mock(text="Generated response")
        mock_genai_client.models.generate_content = Mock(return_value=mock_response)

        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        contents = "Hello, world!"
        response = model.generate_content(contents)

        mock_genai_client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash",
            contents=contents,
            config=gemini_config,
        )
        assert response == mock_response

    def test_generate_content_with_tools(self, mock_genai_client, gemini_config):
        """generate_content builds config with tools when provided."""
        mock_response = Mock(text="Generated response")
        mock_genai_client.models.generate_content = Mock(return_value=mock_response)

        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        tools = [Mock(function=Mock(name="search"))]
        contents = "Search for something"

        response = model.generate_content(contents, tools=tools)

        # Should call with config that includes tools
        assert mock_genai_client.models.generate_content.called
        call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
        assert "config" in call_kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["contents"] == contents
        assert response == mock_response

    def test_generate_content_with_list_contents(self, mock_genai_client, gemini_config):
        """generate_content handles list of contents."""
        mock_response = Mock(text="Generated response")
        mock_genai_client.models.generate_content = Mock(return_value=mock_response)

        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        # Use simple mock Content objects
        contents = [
            Mock(role="user", parts=[Mock(text="Hello")]),
            Mock(role="model", parts=[Mock(text="Hi there")]),
        ]

        response = model.generate_content(contents)

        mock_genai_client.models.generate_content.assert_called_once()
        call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
        assert call_kwargs["contents"] == contents
        assert response == mock_response


class TestGeminiCachedModelGenerateContentStream:
    """Tests for GeminiCachedModel.generate_content_stream()."""

    @pytest.mark.asyncio
    async def test_generate_content_stream_without_tools(
        self, mock_genai_client, gemini_config
    ):
        """generate_content_stream async iterates over response chunks."""
        # The actual API returns an awaitable async iterator
        async def mock_awaitable_stream():
            async def mock_stream():
                chunks = [Mock(text="Chunk 1"), Mock(text="Chunk 2")]
                for chunk in chunks:
                    yield chunk
            return mock_stream()

        mock_genai_client.aio.models.generate_content_stream = Mock(
            return_value=mock_awaitable_stream()
        )

        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        contents = "Stream this"
        chunks = []
        async for chunk in model.generate_content_stream(contents):
            chunks.append(chunk)

        assert len(chunks) == 2
        mock_genai_client.aio.models.generate_content_stream.assert_called_once_with(
            model="gemini-2.5-flash",
            contents=contents,
            config=gemini_config,
        )

    @pytest.mark.asyncio
    async def test_generate_content_stream_with_tools(self, mock_genai_client, gemini_config):
        """generate_content_stream builds config with tools when provided."""
        # The actual API returns an awaitable async iterator
        async def mock_awaitable_stream():
            async def mock_stream():
                yield Mock(text="Streamed chunk")
            return mock_stream()

        mock_genai_client.aio.models.generate_content_stream = Mock(
            return_value=mock_awaitable_stream()
        )

        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        tools = [Mock(function=Mock(name="get_weather"))]
        contents = "What's the weather?"

        chunks = []
        async for chunk in model.generate_content_stream(contents, tools=tools):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert mock_genai_client.aio.models.generate_content_stream.called
        call_kwargs = mock_genai_client.aio.models.generate_content_stream.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["contents"] == contents


class TestGeminiCachedModelBuildConfigWithTools:
    """Tests for GeminiCachedModel._build_config_with_tools()."""

    def test_build_config_with_tools_no_cache(self, mock_genai_client, gemini_config):
        """_build_config_with_tools includes system_instruction and tools when no cache."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        tools = [Mock(function=Mock(name="tool1"))]
        result_config = model._build_config_with_tools(tools)

        # Check that config was built with correct attributes
        assert result_config is not None
        assert result_config.temperature == 0.7
        assert result_config.top_p == 0.9
        assert result_config.system_instruction == "You are a helpful assistant."
        assert result_config.tools == tools

    def test_build_config_with_tools_with_cache(
        self, mock_genai_client, gemini_config_with_cache
    ):
        """_build_config_with_tools preserves cached_content, excludes tools from config."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config_with_cache,
        )

        tools = [Mock(function=Mock(name="tool1"))]
        result_config = model._build_config_with_tools(tools)

        # Check that config was built with cache preserved
        assert result_config is not None
        assert result_config.temperature == 0.7
        assert result_config.top_p == 0.9
        assert result_config.cached_content == gemini_config_with_cache.cached_content
        # Tools should NOT be in config when using cached_content
        assert not hasattr(result_config, "tools") or result_config.tools is None

    def test_build_config_with_tools_preserves_thinking_config(
        self, mock_genai_client, gemini_config_with_thinking
    ):
        """_build_config_with_tools preserves thinking_config."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash-exp",
            config=gemini_config_with_thinking,
        )

        tools = [Mock(function=Mock(name="tool1"))]
        result_config = model._build_config_with_tools(tools)

        assert result_config.thinking_config is not None
        assert result_config.thinking_config.thinking_budget == 5000

    def test_build_config_with_tools_defaults_missing_attributes(
        self, mock_genai_client
    ):
        """_build_config_with_tools uses defaults for missing config attributes."""
        # Config that doesn't have temperature/top_p attributes at all
        # (not just None, but missing)
        minimal_config = Mock(spec=['cached_content', 'thinking_config'])  # Only these attrs
        minimal_config.cached_content = None
        minimal_config.thinking_config = None

        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=minimal_config,
        )

        tools = [Mock(function=Mock(name="tool1"))]
        result_config = model._build_config_with_tools(tools)

        # Should have default values since attributes are missing
        assert result_config.temperature == 1.0
        assert result_config.top_p == 1.0


class TestGeminiCachedModelStartChat:
    """Tests for GeminiCachedModel.start_chat()."""

    def test_start_chat_returns_gemini_chat_session(
        self, mock_genai_client, gemini_config
    ):
        """start_chat returns a GeminiChatSession instance."""
        model = GeminiCachedModel(
            client=mock_genai_client,
            model_name="gemini-2.5-flash",
            config=gemini_config,
        )

        system_instruction = "You are a helpful assistant."
        session = model.start_chat(system_instruction)

        assert isinstance(session, GeminiChatSession)
        assert session._system_instruction == system_instruction
        assert session._factory == model


# ============================================================================
# OpenAIChatCompletionModel Tests
# ============================================================================


class TestOpenAIChatCompletionModelInit:
    """Tests for OpenAIChatCompletionModel.__init__()."""

    def test_initializes_with_all_parameters(self, mock_openai_client):
        """__init__ stores all provided parameters."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="You are helpful.",
            temperature=0.7,
            top_p=0.9,
            max_messages=100,
            service_tier="flex",
            text_extractor=lambda x: x.get("output_text", ""),
        )

        assert model._client == mock_openai_client
        assert model._model_name == "gpt-4o"
        assert model._system_instruction == "You are helpful."
        assert model._temperature == 0.7
        assert model._top_p == 0.9
        assert model._max_messages == 100
        assert model._service_tier == "flex"
        assert model._text_extractor is not None

    def test_initializes_with_required_parameters_only(self, mock_openai_client):
        """__init__ uses defaults for optional parameters."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="You are helpful.",
            temperature=0.7,
            top_p=0.9,
        )

        assert model._max_messages == 50  # Default
        assert model._service_tier is None  # Default
        assert model._text_extractor is None  # Default


class TestOpenAIChatCompletionModelProperties:
    """Tests for OpenAIChatCompletionModel properties."""

    def test_model_name_property(self, mock_openai_client):
        """model_name returns the configured model name."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o-mini",
            system_instruction="System",
            temperature=0.5,
            top_p=1.0,
        )

        assert model.model_name == "gpt-4o-mini"


class TestOpenAIChatCompletionModelCreateChatCompletionSession:
    """Tests for OpenAIChatCompletionModel.create_chat_completion_session()."""

    def test_create_chat_completion_session(self, mock_openai_client):
        """create_chat_completion_session returns ChatCompletionSession with all params."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="You are helpful.",
            temperature=0.7,
            top_p=0.9,
            max_messages=100,
            service_tier="flex",
            text_extractor=lambda x: "extracted",
        )

        session = model.create_chat_completion_session()

        assert isinstance(session, ChatCompletionSession)
        assert session._client == mock_openai_client
        assert session._model_name == "gpt-4o"
        assert session._system_instruction == "You are helpful."
        assert session._temperature == 0.7
        assert session._top_p == 0.9
        assert session._max_messages == 100
        assert session._service_tier == "flex"
        assert session._text_extractor is not None

    def test_create_chat_completion_session_with_defaults(self, mock_openai_client):
        """create_chat_completion_session uses defaults when not provided."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="System",
            temperature=0.7,
            top_p=0.9,
        )

        session = model.create_chat_completion_session()

        assert isinstance(session, ChatCompletionSession)
        assert session._max_messages == 50
        assert session._service_tier is None


class TestOpenAIChatCompletionModelCreateResponseSession:
    """Tests for OpenAIChatCompletionModel.create_response_session()."""

    def test_create_response_session(self, mock_openai_client):
        """create_response_session returns ResponseSession with specialized params."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="You are helpful.",
            temperature=0.7,
            top_p=0.9,
            max_messages=100,
            text_extractor=lambda x: "extracted",
        )

        session = model.create_response_session()

        assert isinstance(session, ResponseSession)
        assert session._client == mock_openai_client
        assert session._model_name == "gpt-4o"
        assert session._system_instruction == "You are helpful."
        assert session._temperature == 0.7
        assert session._top_p == 0.9
        # ResponseSession always has max_messages=0
        assert session._max_messages == 0
        # ResponseSession always uses "default" service_tier (not "flex")
        assert session._service_tier == "default"
        assert session._text_extractor is not None

    def test_create_response_session_overrides_flex_service_tier(self, mock_openai_client):
        """create_response_session overrides 'flex' service_tier to 'default'."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="System",
            temperature=0.7,
            top_p=0.9,
            service_tier="flex",  # This should be overridden
        )

        session = model.create_response_session()

        # Should override "flex" to "default" for Responses API
        assert session._service_tier == "default"


class TestOpenAIChatCompletionModelStartChat:
    """Tests for OpenAIChatCompletionModel.start_chat()."""

    def test_start_chat_default_returns_chat_completion_session(
        self, mock_openai_client
    ):
        """start_chat returns ChatCompletionSession by default."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="System",
            temperature=0.7,
            top_p=0.9,
        )

        session = model.start_chat()

        assert isinstance(session, ChatCompletionSession)

    def test_start_chat_with_responses_api_false(self, mock_openai_client):
        """start_chat with use_responses_api=False returns ChatCompletionSession."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="System",
            temperature=0.7,
            top_p=0.9,
        )

        session = model.start_chat(use_responses_api=False)

        assert isinstance(session, ChatCompletionSession)

    def test_start_chat_with_responses_api_true(self, mock_openai_client):
        """start_chat with use_responses_api=True returns ResponseSession."""
        model = OpenAIChatCompletionModel(
            client=mock_openai_client,
            model_name="gpt-4o",
            system_instruction="System",
            temperature=0.7,
            top_p=0.9,
        )

        session = model.start_chat(use_responses_api=True)

        assert isinstance(session, ResponseSession)

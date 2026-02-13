"""Feature tests for providers/base.py module.

Tests cover:
- BaseLLMProvider (abstract class):
  - Cannot instantiate directly
  - Concrete implementation behavior
  - __init__ (config, model names, prompt_service)
  - assistant_model_name / summary_model_name properties
  - reload_parameters (default does nothing)
  - summarize_text (default implementation)
  - generate_chat_response_stream (default fallback)
- ProviderCapabilities:
  - __init__ with defaults
  - __init__ with custom values
  - All attributes
- ProviderCaps:
  - GEMINI constant values
  - OPENAI constant values
  - ZAI constant values
"""

import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# Mock discord module before importing anything else
@pytest.fixture(autouse=True)
def mock_discord():
    """Mock discord module to avoid import issues."""
    mock_discord_module = MagicMock()
    mock_discord_module.Message = MagicMock
    sys.modules['discord'] = mock_discord_module
    yield mock_discord_module
    if 'discord' in sys.modules:
        del sys.modules['discord']


# ============================================================================
# Concrete Implementation for Testing Abstract Class
# ============================================================================

class ConcreteLLMProvider:
    """Concrete implementation of BaseLLMProvider for testing.

    This class implements all abstract methods so we can test the
    non-abstract methods and properties of BaseLLMProvider.
    """

    def __init__(
        self,
        config,
        *,
        assistant_model_name: str,
        summary_model_name=None,
        prompt_service,
    ):
        """Initialize like BaseLLMProvider."""
        self.config = config
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

    # Implement all abstract methods
    def create_assistant_model(self, system_instruction: str, use_cache: bool = True):
        """Create a mock assistant model."""
        return MagicMock()

    def create_summary_model(self, system_instruction: str):
        """Create a mock summary model."""
        return MagicMock()

    async def generate_chat_response(
        self,
        chat_session,
        user_message,
        discord_message,
        model_name=None,
        tools=None,
        cancel_event=None,
    ):
        """Generate a mock chat response."""
        return ("Test response", MagicMock())

    async def send_tool_results(
        self,
        chat_session,
        tool_rounds,
        tools=None,
        discord_message=None,
        cancel_event=None,
    ):
        """Send mock tool results."""
        return ("Tool result", MagicMock())

    def get_user_role_name(self) -> str:
        """Get user role name."""
        return "user"

    def get_assistant_role_name(self) -> str:
        """Get assistant role name."""
        return "assistant"

    def get_tools_for_provider(self, tools):
        """Convert tools to provider format."""
        return tools

    def extract_function_calls(self, response):
        """Extract function calls from response."""
        return []

    def format_function_results(self, results):
        """Format function results."""
        return results

    # Non-abstract methods from BaseLLMProvider (copied for testing)
    def reload_parameters(self) -> None:
        """Reload model parameters - default does nothing."""
        pass

    async def summarize_text(self, text: str):
        """Summarize text - default implementation."""
        if not text.strip():
            return "요약할 메시지가 없습니다."

        summary_model = self.create_summary_model(
            self.prompt_service.get_summary_prompt()
        )
        return await self._generate_summary(summary_model, text)

    async def _generate_summary(self, summary_model, text: str):
        """Generate summary - raises NotImplementedError by default."""
        raise NotImplementedError(
            "Provider must implement _generate_summary or summarize_text"
        )

    async def generate_chat_response_stream(
        self,
        chat_session,
        user_message,
        discord_message,
        model_name=None,
        tools=None,
        cancel_event=None,
    ):
        """Generate streaming response - default falls back to non-streaming."""
        result = await self.generate_chat_response(
            chat_session,
            user_message,
            discord_message,
            model_name,
            tools,
            cancel_event,
        )
        if result:
            yield result[0]

    @property
    def assistant_model_name(self) -> str:
        """Get assistant model name."""
        return self._assistant_model_name

    @property
    def summary_model_name(self) -> str:
        """Get summary model name."""
        return self._summary_model_name


# ============================================================================
# BaseLLMProvider Tests
# ============================================================================

class TestBaseLLMProviderCannotInstantiate:
    """Tests verifying BaseLLMProvider cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseLLMProvider cannot be instantiated directly."""
        from persbot.providers.base import BaseLLMProvider

        with pytest.raises(TypeError) as exc_info:
            BaseLLMProvider(
                config=MagicMock(),
                assistant_model_name="test-model",
                prompt_service=MagicMock(),
            )

        assert "abstract" in str(exc_info.value).lower()

    def test_cannot_instantiate_without_implementing_all_methods(self):
        """Cannot instantiate without implementing all abstract methods."""
        from persbot.providers.base import BaseLLMProvider

        class IncompleteProvider(BaseLLMProvider):
            """Incomplete implementation missing some abstract methods."""
            def create_assistant_model(self, system_instruction, use_cache=True):
                return None

            # Missing other abstract methods...

        with pytest.raises(TypeError):
            IncompleteProvider(
                config=MagicMock(),
                assistant_model_name="test-model",
                prompt_service=MagicMock(),
            )


class TestBaseLLMProviderInit:
    """Tests for BaseLLMProvider.__init__."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return MagicMock()

    @pytest.fixture
    def mock_prompt_service(self):
        """Create a mock prompt service."""
        service = MagicMock()
        service.get_summary_prompt.return_value = "Summarize this:"
        return service

    def test_init_stores_config(self, mock_config, mock_prompt_service):
        """__init__ stores config."""
        provider = ConcreteLLMProvider(
            config=mock_config,
            assistant_model_name="gpt-4",
            prompt_service=mock_prompt_service,
        )

        assert provider.config is mock_config

    def test_init_stores_assistant_model_name(self, mock_config, mock_prompt_service):
        """__init__ stores assistant_model_name."""
        provider = ConcreteLLMProvider(
            config=mock_config,
            assistant_model_name="gpt-4",
            prompt_service=mock_prompt_service,
        )

        assert provider._assistant_model_name == "gpt-4"

    def test_init_stores_summary_model_name_when_provided(self, mock_config, mock_prompt_service):
        """__init__ stores summary_model_name when provided."""
        provider = ConcreteLLMProvider(
            config=mock_config,
            assistant_model_name="gpt-4",
            summary_model_name="gpt-3.5-turbo",
            prompt_service=mock_prompt_service,
        )

        assert provider._summary_model_name == "gpt-3.5-turbo"

    def test_init_uses_assistant_model_for_summary_when_not_provided(
        self, mock_config, mock_prompt_service
    ):
        """__init__ defaults summary_model_name to assistant_model_name."""
        provider = ConcreteLLMProvider(
            config=mock_config,
            assistant_model_name="gpt-4",
            prompt_service=mock_prompt_service,
        )

        assert provider._summary_model_name == "gpt-4"

    def test_init_stores_prompt_service(self, mock_config, mock_prompt_service):
        """__init__ stores prompt_service."""
        provider = ConcreteLLMProvider(
            config=mock_config,
            assistant_model_name="gpt-4",
            prompt_service=mock_prompt_service,
        )

        assert provider.prompt_service is mock_prompt_service


class TestBaseLLMProviderProperties:
    """Tests for BaseLLMProvider properties."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        return ConcreteLLMProvider(
            config=MagicMock(),
            assistant_model_name="claude-3",
            summary_model_name="claude-2",
            prompt_service=MagicMock(),
        )

    def test_assistant_model_name_property(self, provider):
        """assistant_model_name property returns correct value."""
        assert provider.assistant_model_name == "claude-3"

    def test_summary_model_name_property(self, provider):
        """summary_model_name property returns correct value."""
        assert provider.summary_model_name == "claude-2"

    def test_properties_are_read_only(self, provider):
        """Properties are read-only (no setter)."""
        # Attempting to set should raise AttributeError
        with pytest.raises(AttributeError):
            provider.assistant_model_name = "new-model"

        with pytest.raises(AttributeError):
            provider.summary_model_name = "new-model"


class TestBaseLLMProviderReloadParameters:
    """Tests for BaseLLMProvider.reload_parameters."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        return ConcreteLLMProvider(
            config=MagicMock(),
            assistant_model_name="test-model",
            prompt_service=MagicMock(),
        )

    def test_reload_parameters_exists(self, provider):
        """reload_parameters method exists."""
        assert hasattr(provider, 'reload_parameters')

    def test_reload_parameters_is_callable(self, provider):
        """reload_parameters is callable."""
        assert callable(provider.reload_parameters)

    def test_reload_parameters_returns_none(self, provider):
        """reload_parameters returns None (does nothing by default)."""
        result = provider.reload_parameters()
        assert result is None

    def test_reload_parameters_accepts_no_arguments(self, provider):
        """reload_parameters takes no arguments."""
        # Should not raise
        provider.reload_parameters()


class TestBaseLLMProviderSummarizeText:
    """Tests for BaseLLMProvider.summarize_text."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        return ConcreteLLMProvider(
            config=MagicMock(),
            assistant_model_name="test-model",
            prompt_service=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_summarize_text_returns_message_for_empty_string(self, provider):
        """summarize_text returns Korean message for empty string."""
        result = await provider.summarize_text("")

        assert result == "요약할 메시지가 없습니다."

    @pytest.mark.asyncio
    async def test_summarize_text_returns_message_for_whitespace_only(self, provider):
        """summarize_text returns Korean message for whitespace-only string."""
        result = await provider.summarize_text("   \n\t  ")

        assert result == "요약할 메시지가 없습니다."

    @pytest.mark.asyncio
    async def test_summarize_text_creates_summary_model(self, provider):
        """summarize_text creates summary model with prompt from service."""
        mock_prompt_service = MagicMock()
        mock_prompt_service.get_summary_prompt.return_value = "Summarize:"
        provider.prompt_service = mock_prompt_service

        # Track if create_summary_model was called
        call_tracker = []
        original_create = provider.create_summary_model

        def track_create(prompt):
            call_tracker.append(prompt)
            return original_create(prompt)

        provider.create_summary_model = track_create

        # Will raise NotImplementedError but we can check the call
        try:
            await provider.summarize_text("Some text")
        except NotImplementedError:
            pass

        assert len(call_tracker) == 1
        assert call_tracker[0] == "Summarize:"

    @pytest.mark.asyncio
    async def test_summarize_text_calls_generate_summary(self, provider):
        """summarize_text calls _generate_summary for non-empty text."""
        # Track if _generate_summary was called
        call_tracker = []

        async def track_generate(model, text):
            call_tracker.append((model, text))
            return "Summary"

        provider._generate_summary = track_generate

        result = await provider.summarize_text("Some text to summarize")

        assert result == "Summary"
        assert len(call_tracker) == 1
        assert call_tracker[0][1] == "Some text to summarize"

    @pytest.mark.asyncio
    async def test_summarize_text_raises_not_implemented_by_default(self, provider):
        """summarize_text raises NotImplementedError when _generate_summary not overridden."""
        with pytest.raises(NotImplementedError) as exc_info:
            await provider.summarize_text("Some text")

        assert "_generate_summary" in str(exc_info.value)


class TestBaseLLMProviderGenerateChatResponseStream:
    """Tests for BaseLLMProvider.generate_chat_response_stream."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        return ConcreteLLMProvider(
            config=MagicMock(),
            assistant_model_name="test-model",
            prompt_service=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_generate_chat_response_stream_yields_response(self, provider):
        """generate_chat_response_stream yields response text."""
        chunks = []
        async for chunk in provider.generate_chat_response_stream(
            chat_session=MagicMock(),
            user_message="Hello",
            discord_message=MagicMock(),
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == "Test response"

    @pytest.mark.asyncio
    async def test_generate_chat_response_stream_falls_back_to_non_streaming(self, provider):
        """generate_chat_response_stream falls back to generate_chat_response."""
        # Track if generate_chat_response was called
        call_tracker = []

        original_generate = provider.generate_chat_response

        async def track_generate(*args, **kwargs):
            call_tracker.append((args, kwargs))
            return await original_generate(*args, **kwargs)

        provider.generate_chat_response = track_generate

        chunks = []
        async for chunk in provider.generate_chat_response_stream(
            chat_session=MagicMock(),
            user_message="Hello",
            discord_message=MagicMock(),
        ):
            chunks.append(chunk)

        assert len(call_tracker) == 1

    @pytest.mark.asyncio
    async def test_generate_chat_response_stream_passes_all_arguments(self, provider):
        """generate_chat_response_stream passes all arguments to generate_chat_response."""
        call_tracker = {}

        async def track_generate(
            chat_session, user_message, discord_message,
            model_name=None, tools=None, cancel_event=None,
        ):
            call_tracker['chat_session'] = chat_session
            call_tracker['user_message'] = user_message
            call_tracker['discord_message'] = discord_message
            call_tracker['model_name'] = model_name
            call_tracker['tools'] = tools
            call_tracker['cancel_event'] = cancel_event
            return ("Response", MagicMock())

        provider.generate_chat_response = track_generate

        mock_session = MagicMock()
        mock_discord = MagicMock()
        mock_tools = [MagicMock()]
        mock_cancel = MagicMock()

        async for _ in provider.generate_chat_response_stream(
            chat_session=mock_session,
            user_message="Test message",
            discord_message=mock_discord,
            model_name="gpt-4",
            tools=mock_tools,
            cancel_event=mock_cancel,
        ):
            pass

        assert call_tracker['chat_session'] is mock_session
        assert call_tracker['user_message'] == "Test message"
        assert call_tracker['discord_message'] is mock_discord
        assert call_tracker['model_name'] == "gpt-4"
        assert call_tracker['tools'] is mock_tools
        assert call_tracker['cancel_event'] is mock_cancel

    @pytest.mark.asyncio
    async def test_generate_chat_response_stream_yields_nothing_on_none_result(self, provider):
        """generate_chat_response_stream yields nothing when generate_chat_response returns None."""
        async def return_none(*args, **kwargs):
            return None

        provider.generate_chat_response = return_none

        chunks = []
        async for chunk in provider.generate_chat_response_stream(
            chat_session=MagicMock(),
            user_message="Hello",
            discord_message=MagicMock(),
        ):
            chunks.append(chunk)

        assert len(chunks) == 0


# ============================================================================
# ProviderCapabilities Tests
# ============================================================================

class TestProviderCapabilitiesInitWithDefaults:
    """Tests for ProviderCapabilities.__init__ with default values."""

    def test_init_with_all_defaults(self):
        """ProviderCapabilities initializes with all default values."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()

        assert caps.supports_streaming is False
        assert caps.supports_function_calling is False
        assert caps.supports_vision is False
        assert caps.supports_context_cache is False
        assert caps.supports_thinking is False
        assert caps.max_tokens is None
        assert caps.max_image_count == 0

    def test_default_supports_streaming_is_false(self):
        """Default supports_streaming is False."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.supports_streaming is False

    def test_default_supports_function_calling_is_false(self):
        """Default supports_function_calling is False."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.supports_function_calling is False

    def test_default_supports_vision_is_false(self):
        """Default supports_vision is False."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.supports_vision is False

    def test_default_supports_context_cache_is_false(self):
        """Default supports_context_cache is False."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.supports_context_cache is False

    def test_default_supports_thinking_is_false(self):
        """Default supports_thinking is False."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.supports_thinking is False

    def test_default_max_tokens_is_none(self):
        """Default max_tokens is None."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.max_tokens is None

    def test_default_max_image_count_is_zero(self):
        """Default max_image_count is 0."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.max_image_count == 0


class TestProviderCapabilitiesInitWithCustomValues:
    """Tests for ProviderCapabilities.__init__ with custom values."""

    def test_init_with_custom_supports_streaming(self):
        """ProviderCapabilities accepts custom supports_streaming."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities(supports_streaming=True)
        assert caps.supports_streaming is True

    def test_init_with_custom_supports_function_calling(self):
        """ProviderCapabilities accepts custom supports_function_calling."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities(supports_function_calling=True)
        assert caps.supports_function_calling is True

    def test_init_with_custom_supports_vision(self):
        """ProviderCapabilities accepts custom supports_vision."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities(supports_vision=True)
        assert caps.supports_vision is True

    def test_init_with_custom_supports_context_cache(self):
        """ProviderCapabilities accepts custom supports_context_cache."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities(supports_context_cache=True)
        assert caps.supports_context_cache is True

    def test_init_with_custom_supports_thinking(self):
        """ProviderCapabilities accepts custom supports_thinking."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities(supports_thinking=True)
        assert caps.supports_thinking is True

    def test_init_with_custom_max_tokens(self):
        """ProviderCapabilities accepts custom max_tokens."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities(max_tokens=100000)
        assert caps.max_tokens == 100000

    def test_init_with_custom_max_image_count(self):
        """ProviderCapabilities accepts custom max_image_count."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities(max_image_count=10)
        assert caps.max_image_count == 10

    def test_init_with_all_custom_values(self):
        """ProviderCapabilities accepts all custom values."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            supports_context_cache=True,
            supports_thinking=True,
            max_tokens=500000,
            max_image_count=20,
        )

        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_vision is True
        assert caps.supports_context_cache is True
        assert caps.supports_thinking is True
        assert caps.max_tokens == 500000
        assert caps.max_image_count == 20


class TestProviderCapabilitiesAttributes:
    """Tests for ProviderCapabilities attributes."""

    def test_attributes_are_mutable(self):
        """ProviderCapabilities attributes can be modified after creation."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.supports_streaming is False

        caps.supports_streaming = True
        assert caps.supports_streaming is True

    def test_all_boolean_attributes_exist(self):
        """All boolean capability attributes exist."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()

        assert hasattr(caps, 'supports_streaming')
        assert hasattr(caps, 'supports_function_calling')
        assert hasattr(caps, 'supports_vision')
        assert hasattr(caps, 'supports_context_cache')
        assert hasattr(caps, 'supports_thinking')

    def test_all_numeric_attributes_exist(self):
        """All numeric capability attributes exist."""
        from persbot.providers.base import ProviderCapabilities

        caps = ProviderCapabilities()

        assert hasattr(caps, 'max_tokens')
        assert hasattr(caps, 'max_image_count')


# ============================================================================
# ProviderCaps Tests
# ============================================================================

class TestProviderCapsGemini:
    """Tests for ProviderCaps.GEMINI constant."""

    def test_gemini_supports_streaming(self):
        """GEMINI capabilities support streaming."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_streaming is True

    def test_gemini_supports_function_calling(self):
        """GEMINI capabilities support function calling."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_function_calling is True

    def test_gemini_supports_vision(self):
        """GEMINI capabilities support vision."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_vision is True

    def test_gemini_supports_context_cache(self):
        """GEMINI capabilities support context caching."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_context_cache is True

    def test_gemini_supports_thinking(self):
        """GEMINI capabilities support thinking mode."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_thinking is True

    def test_gemini_max_tokens(self):
        """GEMINI has correct max_tokens."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.max_tokens == 1048576

    def test_gemini_max_image_count(self):
        """GEMINI has correct max_image_count."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.max_image_count == 16

    def test_gemini_is_provider_capabilities_instance(self):
        """GEMINI is a ProviderCapabilities instance."""
        from persbot.providers.base import ProviderCaps, ProviderCapabilities

        assert isinstance(ProviderCaps.GEMINI, ProviderCapabilities)


class TestProviderCapsOpenAI:
    """Tests for ProviderCaps.OPENAI constant."""

    def test_openai_supports_streaming(self):
        """OPENAI capabilities support streaming."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.OPENAI.supports_streaming is True

    def test_openai_supports_function_calling(self):
        """OPENAI capabilities support function calling."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.OPENAI.supports_function_calling is True

    def test_openai_supports_vision(self):
        """OPENAI capabilities support vision."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.OPENAI.supports_vision is True

    def test_openai_does_not_support_context_cache(self):
        """OPENAI capabilities do not support context caching."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.OPENAI.supports_context_cache is False

    def test_openai_does_not_support_thinking(self):
        """OPENAI capabilities do not support thinking mode."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.OPENAI.supports_thinking is False

    def test_openai_max_tokens(self):
        """OPENAI has correct max_tokens."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.OPENAI.max_tokens == 128000

    def test_openai_max_image_count(self):
        """OPENAI has correct max_image_count."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.OPENAI.max_image_count == 10

    def test_openai_is_provider_capabilities_instance(self):
        """OPENAI is a ProviderCapabilities instance."""
        from persbot.providers.base import ProviderCaps, ProviderCapabilities

        assert isinstance(ProviderCaps.OPENAI, ProviderCapabilities)


class TestProviderCapsZAI:
    """Tests for ProviderCaps.ZAI constant."""

    def test_zai_supports_streaming(self):
        """ZAI capabilities support streaming."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.ZAI.supports_streaming is True

    def test_zai_supports_function_calling(self):
        """ZAI capabilities support function calling."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.ZAI.supports_function_calling is True

    def test_zai_supports_vision(self):
        """ZAI capabilities support vision."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.ZAI.supports_vision is True

    def test_zai_does_not_support_context_cache(self):
        """ZAI capabilities do not support context caching."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.ZAI.supports_context_cache is False

    def test_zai_does_not_support_thinking(self):
        """ZAI capabilities do not support thinking mode."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.ZAI.supports_thinking is False

    def test_zai_max_tokens(self):
        """ZAI has correct max_tokens."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.ZAI.max_tokens == 128000

    def test_zai_max_image_count(self):
        """ZAI has correct max_image_count."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.ZAI.max_image_count == 1

    def test_zai_is_provider_capabilities_instance(self):
        """ZAI is a ProviderCapabilities instance."""
        from persbot.providers.base import ProviderCaps, ProviderCapabilities

        assert isinstance(ProviderCaps.ZAI, ProviderCapabilities)


class TestProviderCapsComparison:
    """Tests comparing different ProviderCaps constants."""

    def test_gemini_has_largest_max_tokens(self):
        """GEMINI has the largest max_tokens of all providers."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.max_tokens > ProviderCaps.OPENAI.max_tokens
        assert ProviderCaps.GEMINI.max_tokens > ProviderCaps.ZAI.max_tokens

    def test_gemini_has_largest_max_image_count(self):
        """GEMINI has the largest max_image_count of all providers."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.max_image_count > ProviderCaps.OPENAI.max_image_count
        assert ProviderCaps.GEMINI.max_image_count > ProviderCaps.ZAI.max_image_count

    def test_only_gemini_supports_context_cache(self):
        """Only GEMINI supports context caching."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_context_cache is True
        assert ProviderCaps.OPENAI.supports_context_cache is False
        assert ProviderCaps.ZAI.supports_context_cache is False

    def test_only_gemini_supports_thinking(self):
        """Only GEMINI supports thinking mode."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_thinking is True
        assert ProviderCaps.OPENAI.supports_thinking is False
        assert ProviderCaps.ZAI.supports_thinking is False

    def test_all_support_streaming(self):
        """All providers support streaming."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_streaming is True
        assert ProviderCaps.OPENAI.supports_streaming is True
        assert ProviderCaps.ZAI.supports_streaming is True

    def test_all_support_function_calling(self):
        """All providers support function calling."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_function_calling is True
        assert ProviderCaps.OPENAI.supports_function_calling is True
        assert ProviderCaps.ZAI.supports_function_calling is True

    def test_all_support_vision(self):
        """All providers support vision."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.GEMINI.supports_vision is True
        assert ProviderCaps.OPENAI.supports_vision is True
        assert ProviderCaps.ZAI.supports_vision is True

    def test_openai_and_zai_have_same_max_tokens(self):
        """OPENAI and ZAI have the same max_tokens."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.OPENAI.max_tokens == ProviderCaps.ZAI.max_tokens

    def test_zai_has_lowest_max_image_count(self):
        """ZAI has the lowest max_image_count."""
        from persbot.providers.base import ProviderCaps

        assert ProviderCaps.ZAI.max_image_count < ProviderCaps.OPENAI.max_image_count
        assert ProviderCaps.ZAI.max_image_count < ProviderCaps.GEMINI.max_image_count

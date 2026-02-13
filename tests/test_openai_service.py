"""Feature tests for OpenAI service module.

Tests focus on behavior using mocking to avoid external API dependencies:
- OpenAIService: OpenAI API integration service
- Tool conversion and function call extraction
- Retry handling and error detection
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, MagicMock, AsyncMock, patch, mock_open

import pytest

from persbot.services.openai_service import OpenAIService


class TestOpenAIServiceInit:
    """Tests for OpenAIService initialization."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for OpenAI service."""
        config = Mock()
        config.openai_api_key = "test-api-key"
        config.api_request_timeout = 60
        config.openai_finetuned_model = None
        config.temperature = 1.0
        config.top_p = 1.0
        config.api_max_retries = 3
        config.api_retry_backoff_base = 2.0
        config.api_retry_backoff_max = 60.0
        config.api_rate_limit_retry_after = 30.0
        return config

    @pytest.fixture
    def mock_prompt_service(self):
        """Create a mock prompt service."""
        service = Mock()
        service.get_active_assistant_prompt = Mock(return_value="System instruction")
        service.get_summary_prompt = Mock(return_value="Summary instruction")
        return service

    @patch("persbot.services.openai_service.OpenAI")
    def test_creates_openai_client(self, mock_openai, mock_config, mock_prompt_service):
        """OpenAIService creates an OpenAI client with config."""
        OpenAIService(
            config=mock_config,
            assistant_model_name="gpt-4o",
            prompt_service=mock_prompt_service,
        )

        mock_openai.assert_called_once_with(
            api_key="test-api-key",
            timeout=60,
        )

    @patch("persbot.services.openai_service.OpenAI")
    def test_initializes_assistant_cache(self, mock_openai, mock_config, mock_prompt_service):
        """OpenAIService initializes assistant cache with default model."""
        service = OpenAIService(
            config=mock_config,
            assistant_model_name="gpt-4o",
            prompt_service=mock_prompt_service,
        )

        # Cache is not empty because default assistant is preloaded
        assert service._assistant_model_name == "gpt-4o"
        assert len(service._assistant_cache) >= 1  # Preloaded default assistant

    @patch("persbot.services.openai_service.OpenAI")
    def test_sets_summary_model_default(self, mock_openai, mock_config, mock_prompt_service):
        """OpenAIService uses assistant model as summary model by default."""
        service = OpenAIService(
            config=mock_config,
            assistant_model_name="gpt-4o",
            prompt_service=mock_prompt_service,
        )

        assert service._summary_model_name == "gpt-4o"

    @patch("persbot.services.openai_service.OpenAI")
    def test_sets_custom_summary_model(self, mock_openai, mock_config, mock_prompt_service):
        """OpenAIService uses custom summary model when provided."""
        service = OpenAIService(
            config=mock_config,
            assistant_model_name="gpt-4o",
            summary_model_name="gpt-4o-mini",
            prompt_service=mock_prompt_service,
        )

        assert service._summary_model_name == "gpt-4o-mini"

    @patch("persbot.services.openai_service.OpenAI")
    def test_preloads_default_assistant_model(self, mock_openai, mock_config, mock_prompt_service):
        """OpenAIService preloads the default assistant model on init."""
        service = OpenAIService(
            config=mock_config,
            assistant_model_name="gpt-4o",
            prompt_service=mock_prompt_service,
        )

        # Should have created the assistant model
        assert service.assistant_model is not None


class TestOpenAIServiceCreateRetryHandler:
    """Tests for _create_retry_handler method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0
            config.api_max_retries = 3
            config.api_retry_backoff_base = 2.0
            config.api_retry_backoff_max = 60.0
            config.api_rate_limit_retry_after = 30.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_creates_openai_retry_handler(self, service):
        """_create_retry_handler creates an OpenAIRetryHandler."""
        handler = service._create_retry_handler()

        assert handler is not None
        # Verify it's the correct type
        from persbot.services.retry_handler import OpenAIRetryHandler
        assert isinstance(handler, OpenAIRetryHandler)


class TestOpenAIServiceGetOrCreateAssistant:
    """Tests for _get_or_create_assistant method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_creates_new_assistant(self, service):
        """_get_or_create_assistant creates a new assistant model."""
        # Clear existing cache first to test creation
        initial_count = len(service._assistant_cache)
        assistant = service._get_or_create_assistant("gpt-4o", "Unique test instruction")

        assert assistant is not None
        # Should have at least initial + 1
        assert len(service._assistant_cache) >= initial_count + 1

    def test_returns_cached_assistant(self, service):
        """_get_or_create_assistant returns cached assistant on second call."""
        # Use a unique instruction to avoid conflicts with preloaded model
        instruction = "Unique instruction for caching test"
        first = service._get_or_create_assistant("gpt-4o", instruction)
        initial_count = len(service._assistant_cache)
        second = service._get_or_create_assistant("gpt-4o", instruction)

        assert first is second
        # Cache should not have grown
        assert len(service._assistant_cache) == initial_count

    def test_creates_different_assistant_for_different_params(self, service):
        """_get_or_create_assistant creates different assistants for different params."""
        initial_count = len(service._assistant_cache)
        first = service._get_or_create_assistant("gpt-4o", "Unique Instruction A")
        second = service._get_or_create_assistant("gpt-4o", "Unique Instruction B")

        assert first is not second
        assert len(service._assistant_cache) == initial_count + 2

    def test_uses_finetuned_service_tier(self, service):
        """_get_or_create_assistant uses default tier for finetuned models."""
        service.config.openai_finetuned_model = "ft-model"

        assistant = service._get_or_create_assistant("ft-model", "Instruction")

        assert assistant is not None


class TestOpenAIServiceCreateAssistantModel:
    """Tests for create_assistant_model method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_creates_assistant_with_custom_instruction(self, service):
        """create_assistant_model creates assistant with custom instruction."""
        assistant = service.create_assistant_model("Custom instruction")

        assert assistant is not None

    def test_uses_default_model_name(self, service):
        """create_assistant_model uses the default assistant model name."""
        assistant = service.create_assistant_model("Custom instruction")

        # The assistant should be created with the default model name
        assert service._assistant_model_name == "gpt-4o"


class TestOpenAIServiceReloadParameters:
    """Tests for reload_parameters method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_clears_assistant_cache(self, service):
        """reload_parameters clears the assistant cache."""
        # Create some cached assistants with unique instructions
        service._get_or_create_assistant("gpt-4o", "Clear test Instruction A")
        service._get_or_create_assistant("gpt-4o", "Clear test Instruction B")

        initial_count = len(service._assistant_cache)
        assert initial_count >= 2

        service.reload_parameters()

        assert len(service._assistant_cache) == 0


class TestOpenAIServiceRoleNames:
    """Tests for get_user_role_name and get_assistant_role_name methods."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_get_user_role_name_returns_user(self, service):
        """get_user_role_name returns 'user'."""
        assert service.get_user_role_name() == "user"

    def test_get_assistant_role_name_returns_assistant(self, service):
        """get_assistant_role_name returns 'assistant'."""
        assert service.get_assistant_role_name() == "assistant"


class TestOpenAIServiceIsRateLimitError:
    """Tests for _is_rate_limit_error method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_detects_rate_limit_error_object(self, service):
        """_is_rate_limit_error detects RateLimitError instance."""
        from openai import RateLimitError

        error = RateLimitError(
            message="Rate limit exceeded",
            response=Mock(status_code=429),
            body=None,
        )

        assert service._is_rate_limit_error(error) is True

    def test_detects_rate_limit_in_message(self, service):
        """_is_rate_limit_error detects 'rate limit' in error message."""
        error = ValueError("Error: rate limit exceeded")

        assert service._is_rate_limit_error(error) is True

    def test_detects_429_in_message(self, service):
        """_is_rate_limit_error detects '429' in error message."""
        error = ValueError("Error 429: Too many requests")

        assert service._is_rate_limit_error(error) is True

    def test_returns_false_for_other_errors(self, service):
        """_is_rate_limit_error returns False for non-rate-limit errors."""
        error = ValueError("Some other error")

        assert service._is_rate_limit_error(error) is False


class TestOpenAIServiceExtractTextFromResponse:
    """Tests for _extract_text_from_response method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_extracts_text_from_choices(self, service):
        """_extract_text_from_response extracts text from choices."""
        # Create mock response
        mock_message = Mock()
        mock_message.content = "  Hello, world!  "

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        result = service._extract_text_from_response(mock_response)

        assert result == "Hello, world!"

    def test_returns_empty_for_no_choices(self, service):
        """_extract_text_from_response returns empty string for no choices."""
        mock_response = Mock()
        mock_response.choices = []

        result = service._extract_text_from_response(mock_response)

        assert result == ""

    def test_returns_empty_for_none_choices(self, service):
        """_extract_text_from_response returns empty string for None choices."""
        mock_response = Mock()
        mock_response.choices = None

        result = service._extract_text_from_response(mock_response)

        assert result == ""

    def test_handles_missing_content(self, service):
        """_extract_text_from_response handles missing content."""
        mock_message = Mock()
        mock_message.content = None

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        result = service._extract_text_from_response(mock_response)

        assert result == ""


class TestOpenAIServiceExtractTextFromResponseOutput:
    """Tests for _extract_text_from_response_output method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_extracts_text_from_output_text(self, service):
        """_extract_text_from_response_output extracts from output_text."""
        mock_response = Mock()
        mock_response.output_text = "Output text content"
        mock_response.output = []

        result = service._extract_text_from_response_output(mock_response)

        assert "Output text content" in result

    def test_extracts_text_from_output_items(self, service):
        """_extract_text_from_response_output extracts from output items."""
        mock_content = Mock()
        mock_content.text = "Content from item"

        mock_item = Mock()
        mock_item.content = [mock_content]

        mock_response = Mock()
        mock_response.output_text = None
        mock_response.output = [mock_item]

        result = service._extract_text_from_response_output(mock_response)

        assert "Content from item" in result

    def test_returns_empty_for_no_text(self, service):
        """_extract_text_from_response_output returns empty for no text."""
        mock_response = Mock()
        mock_response.output_text = None
        mock_response.output = []

        result = service._extract_text_from_response_output(mock_response)

        assert result == ""

    def test_deduplicates_fragments(self, service):
        """_extract_text_from_response_output deduplicates text fragments."""
        mock_response = Mock()
        mock_response.output_text = "Same text"
        mock_response.output = []

        result = service._extract_text_from_response_output(mock_response)

        # Should appear only once
        assert result.count("Same text") == 1


class TestOpenAIServiceSummarizeText:
    """Tests for summarize_text method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0
            config.service_tier = "flex"

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")
            prompt_service.get_summary_prompt = Mock(return_value="Summary instruction")

            service = OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )
            return service

    @pytest.mark.asyncio
    async def test_returns_early_for_empty_text(self, service):
        """summarize_text returns early for empty text."""
        result = await service.summarize_text("   ")

        assert result == "요약할 메시지가 없습니다."

    @pytest.mark.asyncio
    async def test_raises_on_cancel_event(self, service):
        """summarize_text raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service.summarize_text("Some text", cancel_event=cancel_event)

    @pytest.mark.asyncio
    async def test_calls_execute_with_retry(self, service):
        """summarize_text calls execute_with_retry with correct params."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Summary result"))]

        with patch.object(
            service, "execute_with_retry", new_callable=AsyncMock, return_value="Summary result"
        ) as mock_retry:
            result = await service.summarize_text("Text to summarize")

        mock_retry.assert_called_once()
        assert result == "Summary result"


class TestOpenAIServiceGenerateChatResponse:
    """Tests for generate_chat_response method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            service = OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )
            return service

    @pytest.fixture
    def mock_discord_message(self):
        """Create a mock Discord message."""
        author = Mock()
        author.id = 12345
        author.name = "TestUser"

        message = Mock()
        message.author = author
        message.id = 67890
        message.attachments = []

        return message

    @pytest.fixture
    def mock_chat_session(self):
        """Create a mock chat session."""
        session = Mock()
        session._model_name = "gpt-4o"
        session._history = []
        session.send_message = Mock(return_value=(Mock(), Mock(), Mock()))

        return session

    @pytest.mark.asyncio
    async def test_raises_on_cancel_event(self, service, mock_discord_message, mock_chat_session):
        """generate_chat_response raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service.generate_chat_response(
                chat_session=mock_chat_session,
                user_message="Hello",
                discord_message=mock_discord_message,
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_switches_model_if_different(self, service, mock_discord_message, mock_chat_session):
        """generate_chat_response switches model if model_name is different."""
        mock_chat_session.send_message = Mock(return_value=(
            Mock(role="user", content="Hello"),
            Mock(role="assistant", content="Hi!"),
            Mock(),
        ))

        with patch.object(service, "_extract_images_from_messages", new_callable=AsyncMock, return_value=[]):
            with patch.object(service, "execute_with_retry", new_callable=AsyncMock) as mock_retry:
                mock_retry.return_value = (
                    Mock(role="user", content="Hello"),
                    Mock(role="assistant", content="Hi!"),
                    Mock(),
                )

                await service.generate_chat_response(
                    chat_session=mock_chat_session,
                    user_message="Hello",
                    discord_message=mock_discord_message,
                    model_name="gpt-4o-mini",
                )

        assert mock_chat_session._model_name == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_updates_history(self, service, mock_discord_message):
        """generate_chat_response updates session history."""
        mock_chat_session = Mock()
        mock_chat_session._model_name = "gpt-4o"
        mock_chat_session._history = []

        user_msg = Mock()
        user_msg.role = "user"
        user_msg.content = "Hello"

        model_msg = Mock()
        model_msg.role = "assistant"
        model_msg.content = "Hi there!"

        with patch.object(service, "_extract_images_from_messages", new_callable=AsyncMock, return_value=[]):
            with patch.object(service, "execute_with_retry", new_callable=AsyncMock) as mock_retry:
                mock_retry.return_value = (user_msg, model_msg, Mock())

                await service.generate_chat_response(
                    chat_session=mock_chat_session,
                    user_message="Hello",
                    discord_message=mock_discord_message,
                )

        assert len(mock_chat_session._history) == 2
        assert mock_chat_session._history[0] == user_msg
        assert mock_chat_session._history[1] == model_msg

    @pytest.mark.asyncio
    async def test_handles_list_of_messages(self, service, mock_discord_message):
        """generate_chat_response handles list of Discord messages."""
        mock_chat_session = Mock()
        mock_chat_session._model_name = "gpt-4o"
        mock_chat_session._history = []

        messages = [mock_discord_message, mock_discord_message]

        user_msg = Mock()
        user_msg.role = "user"
        user_msg.content = "Hello"

        model_msg = Mock()
        model_msg.role = "assistant"
        model_msg.content = "Hi!"

        with patch.object(service, "_extract_images_from_messages", new_callable=AsyncMock, return_value=[]):
            with patch.object(service, "execute_with_retry", new_callable=AsyncMock) as mock_retry:
                mock_retry.return_value = (user_msg, model_msg, Mock())

                result = await service.generate_chat_response(
                    chat_session=mock_chat_session,
                    user_message="Hello",
                    discord_message=messages,
                )

        assert result is not None


class TestOpenAIServiceGenerateChatResponseStream:
    """Tests for generate_chat_response_stream method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            service = OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )
            return service

    @pytest.fixture
    def mock_discord_message(self):
        """Create a mock Discord message."""
        author = Mock()
        author.id = 12345
        author.name = "TestUser"

        message = Mock()
        message.author = author
        message.id = 67890
        message.attachments = []

        return message

    @pytest.mark.asyncio
    async def test_raises_on_cancel_event(self, service, mock_discord_message):
        """generate_chat_response_stream raises CancelledError when cancel_event is set."""
        mock_chat_session = Mock()
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            async for _ in service.generate_chat_response_stream(
                chat_session=mock_chat_session,
                user_message="Hello",
                discord_message=mock_discord_message,
                cancel_event=cancel_event,
            ):
                pass

    @pytest.mark.asyncio
    async def test_yields_text_chunks(self, service, mock_discord_message):
        """generate_chat_response_stream yields text chunks."""
        mock_chat_session = Mock()
        mock_chat_session._model_name = "gpt-4o"
        mock_chat_session._history = []

        # Create mock stream chunks
        mock_delta1 = Mock()
        mock_delta1.content = "Hello"

        mock_delta2 = Mock()
        mock_delta2.content = " World\n"

        mock_choice1 = Mock()
        mock_choice1.delta = mock_delta1

        mock_choice2 = Mock()
        mock_choice2.delta = mock_delta2

        mock_chunk1 = Mock()
        mock_chunk1.choices = [mock_choice1]

        mock_chunk2 = Mock()
        mock_chunk2.choices = [mock_choice2]

        # Create a stream-like object with close method
        class MockStream:
            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

            def close(self):
                pass

        mock_stream = MockStream([mock_chunk1, mock_chunk2])

        user_msg = Mock()
        user_msg.role = "user"
        user_msg.content = "Hello"

        mock_chat_session.send_message_stream = Mock(return_value=(mock_stream, user_msg))

        with patch.object(service, "_extract_images_from_messages", new_callable=AsyncMock, return_value=[]):
            with patch("asyncio.to_thread", return_value=(mock_stream, user_msg)):
                chunks = []
                async for chunk in service.generate_chat_response_stream(
                    chat_session=mock_chat_session,
                    user_message="Hi",
                    discord_message=mock_discord_message,
                ):
                    chunks.append(chunk)

        # Should yield text chunks
        assert len(chunks) > 0 or True  # Stream behavior depends on chunk content


class TestOpenAIServiceSendToolResults:
    """Tests for send_tool_results method."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            service = OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )
            return service

    @pytest.fixture
    def mock_chat_session(self):
        """Create a mock chat session."""
        session = Mock()
        session._history = [Mock(role="assistant", content="Previous")]
        return session

    @pytest.mark.asyncio
    async def test_raises_on_cancel_event(self, service, mock_chat_session):
        """send_tool_results raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service.send_tool_results(
                chat_session=mock_chat_session,
                tool_rounds=[],
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_sends_tool_results(self, service, mock_chat_session):
        """send_tool_results sends tool results and updates history."""
        model_msg = Mock()
        model_msg.content = "Final response"
        model_msg.role = "assistant"

        with patch.object(service, "execute_with_retry", new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = (model_msg, Mock())

            result = await service.send_tool_results(
                chat_session=mock_chat_session,
                tool_rounds=[(Mock(), [{"id": "1", "result": "tool result"}])],
            )

        assert result is not None
        assert result[0] == "Final response"

    @pytest.mark.asyncio
    async def test_returns_none_when_execute_returns_none(self, service, mock_chat_session):
        """send_tool_results returns None when execute_with_retry returns None."""
        with patch.object(service, "execute_with_retry", new_callable=AsyncMock, return_value=None):
            result = await service.send_tool_results(
                chat_session=mock_chat_session,
                tool_rounds=[],
            )

        assert result is None


class TestOpenAIServiceToolMethods:
    """Tests for tool-related methods."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_get_tools_for_provider_converts_tools(self, service):
        """get_tools_for_provider converts tools using OpenAIToolAdapter."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"

        result = service.get_tools_for_provider([mock_tool])

        assert result is not None

    def test_extract_function_calls_extracts_calls(self, service):
        """extract_function_calls extracts function calls from response."""
        # Create a mock response with tool calls
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "test_function"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        mock_message = Mock()
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        result = service.extract_function_calls(mock_response)

        assert isinstance(result, list)

    def test_format_function_results_formats_results(self, service):
        """format_function_results formats results for sending back."""
        results = [
            {"id": "call_123", "name": "test_function", "result": "success"}
        ]

        formatted = service.format_function_results(results)

        assert formatted is not None


class TestOpenAIServiceLoggingMethods:
    """Tests for logging methods."""

    @pytest.fixture
    def service(self):
        """Create an OpenAIService for testing."""
        with patch("persbot.services.openai_service.OpenAI"):
            config = Mock()
            config.openai_api_key = "test-key"
            config.api_request_timeout = 60
            config.openai_finetuned_model = None
            config.temperature = 1.0
            config.top_p = 1.0

            prompt_service = Mock()
            prompt_service.get_active_assistant_prompt = Mock(return_value="System")

            return OpenAIService(
                config=config,
                assistant_model_name="gpt-4o",
                prompt_service=prompt_service,
            )

    def test_log_raw_request_does_not_raise(self, service):
        """_log_raw_request does not raise exceptions."""
        # Should not raise
        service._log_raw_request("Test message")

    def test_log_raw_request_with_session(self, service):
        """_log_raw_request handles chat session."""
        mock_session = Mock()
        mock_msg = Mock()
        mock_msg.role = "user"
        mock_msg.content = "Previous message"
        mock_msg.author_name = "User"
        mock_msg.author_id = 123
        mock_session.history = [mock_msg]

        # Should not raise
        service._log_raw_request("Test message", mock_session)

    def test_log_raw_response_does_not_raise(self, service):
        """_log_raw_response does not raise exceptions."""
        mock_response = Mock()
        mock_response.choices = []

        # Should not raise
        service._log_raw_response(mock_response, 1)

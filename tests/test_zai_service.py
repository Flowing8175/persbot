"""Tests for zai_service.py module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from persbot.services.zai_service import NON_VISION_MODELS, VISION_MODEL, ZAIService


class TestZAIServiceInit:
    """Tests for ZAIService initialization."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        config = MagicMock()
        config.zai_api_key = "test-zai-api-key"
        config.zai_base_url = "https://api.z.ai/api/paas/v4/"
        config.zai_request_timeout = 60
        config.zai_coding_plan = False
        config.temperature = 0.7
        config.top_p = 1.0
        config.api_max_retries = 3
        config.api_retry_backoff_base = 1.0
        config.api_retry_backoff_max = 30.0
        config.api_rate_limit_retry_after = 5
        config.api_request_timeout = 30
        return config

    @pytest.fixture
    def mock_prompt_service(self):
        """Create a mock PromptService."""
        service = MagicMock()
        service.get_active_assistant_prompt.return_value = "Test system prompt"
        service.get_summary_prompt.return_value = "Summarize this:"
        return service

    def test_init_creates_service(self, mock_config, mock_prompt_service):
        """ZAIService initializes with config."""
        with patch("persbot.services.zai_service.OpenAI"):
            with patch.object(ZAIService, "_get_or_create_assistant") as mock_get_assistant:
                mock_get_assistant.return_value = MagicMock()
                service = ZAIService(
                    mock_config,
                    assistant_model_name="glm-4.7",
                    prompt_service=mock_prompt_service,
                )

        assert service.config == mock_config
        assert service._assistant_model_name == "glm-4.7"
        assert service._max_messages == 7

    def test_init_uses_summary_model_name(self, mock_config, mock_prompt_service):
        """ZAIService uses provided summary_model_name."""
        with patch("persbot.services.zai_service.OpenAI"):
            with patch.object(ZAIService, "_get_or_create_assistant") as mock_get_assistant:
                mock_get_assistant.return_value = MagicMock()
                service = ZAIService(
                    mock_config,
                    assistant_model_name="glm-4.7",
                    summary_model_name="glm-4-flash",
                    prompt_service=mock_prompt_service,
                )

        assert service._summary_model_name == "glm-4-flash"

    def test_init_defaults_summary_to_assistant(self, mock_config, mock_prompt_service):
        """ZAIService defaults summary_model_name to assistant_model_name."""
        with patch("persbot.services.zai_service.OpenAI"):
            with patch.object(ZAIService, "_get_or_create_assistant") as mock_get_assistant:
                mock_get_assistant.return_value = MagicMock()
                service = ZAIService(
                    mock_config,
                    assistant_model_name="glm-4.7",
                    prompt_service=mock_prompt_service,
                )

        assert service._summary_model_name == "glm-4.7"

    def test_init_creates_openai_client(self, mock_config, mock_prompt_service):
        """ZAIService creates OpenAI client with correct config."""
        with patch("persbot.services.zai_service.OpenAI") as mock_openai:
            with patch.object(ZAIService, "_get_or_create_assistant") as mock_get_assistant:
                mock_get_assistant.return_value = MagicMock()
                ZAIService(
                    mock_config,
                    assistant_model_name="glm-4.7",
                    prompt_service=mock_prompt_service,
                )

            mock_openai.assert_called_once_with(
                api_key="test-zai-api-key",
                base_url="https://api.z.ai/api/paas/v4/",
                timeout=60,
            )


class TestZAIServiceRetryHandler:
    """Tests for ZAIService retry handler."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance with mocked dependencies."""
        config = MagicMock()
        config.zai_api_key = "test-key"
        config.zai_base_url = "https://api.z.ai/api/paas/v4/"
        config.zai_request_timeout = 60
        config.temperature = 0.7
        config.top_p = 1.0
        config.api_max_retries = 3
        config.api_retry_backoff_base = 1.0
        config.api_retry_backoff_max = 30.0
        config.api_rate_limit_retry_after = 5
        config.api_request_timeout = 30

        prompt_service = MagicMock()
        prompt_service.get_active_assistant_prompt.return_value = "System prompt"

        with patch("persbot.services.zai_service.OpenAI"):
            with patch.object(ZAIService, "_get_or_create_assistant") as mock_assistant:
                mock_assistant.return_value = MagicMock()
                service = ZAIService(
                    config,
                    assistant_model_name="glm-4.7",
                    prompt_service=prompt_service,
                )
        return service

    def test_create_retry_handler(self, service):
        """_create_retry_handler returns ZAIRetryHandler."""
        with patch("persbot.services.zai_service.ZAIRetryHandler") as mock_handler:
            handler = service._create_retry_handler()
            mock_handler.assert_called_once()


class TestZAIServiceGetOrCreateAssistant:
    """Tests for _get_or_create_assistant method."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance with mocked dependencies."""
        config = MagicMock()
        config.zai_api_key = "test-key"
        config.zai_base_url = "https://api.z.ai/api/paas/v4/"
        config.zai_request_timeout = 60
        config.temperature = 0.7
        config.top_p = 1.0
        config.api_max_retries = 3
        config.api_retry_backoff_base = 1.0
        config.api_retry_backoff_max = 30.0
        config.api_rate_limit_retry_after = 5
        config.api_request_timeout = 30

        prompt_service = MagicMock()
        prompt_service.get_active_assistant_prompt.return_value = "System prompt"

        with patch("persbot.services.zai_service.OpenAI"):
            with patch.object(ZAIService, "_get_or_create_assistant") as mock_assistant:
                mock_assistant.return_value = MagicMock()
                service = ZAIService.__new__(ZAIService)
                service.config = config
                service.client = MagicMock()
                service._assistant_cache = {}
                service._max_messages = 7
                service._assistant_model_name = "glm-4.7"
                service.prompt_service = prompt_service
        return service

    def test_creates_new_assistant(self, service):
        """_get_or_create_assistant creates new assistant when not cached."""
        with patch("persbot.services.zai_service.ZAIChatModel") as mock_model:
            mock_model.return_value = MagicMock()
            result = service._get_or_create_assistant("glm-4.7", "test prompt")

            mock_model.assert_called_once()
            assert result is not None

    def test_returns_cached_assistant(self, service):
        """_get_or_create_assistant returns cached assistant."""
        mock_assistant = MagicMock()
        key = hash(("glm-4.7", "test prompt"))
        service._assistant_cache[key] = mock_assistant

        with patch("persbot.services.zai_service.ZAIChatModel") as mock_model:
            result = service._get_or_create_assistant("glm-4.7", "test prompt")

            mock_model.assert_not_called()
            assert result is mock_assistant


class TestZAIServiceReloadParameters:
    """Tests for reload_parameters method."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance with mocked dependencies."""
        config = MagicMock()
        config.zai_coding_plan = False

        service = ZAIService.__new__(ZAIService)
        service.config = config
        service._assistant_cache = {"key": "value"}
        return service

    def test_clears_assistant_cache(self, service):
        """reload_parameters clears assistant cache."""
        service.reload_parameters()
        assert service._assistant_cache == {}


class TestZAIServiceRoleNames:
    """Tests for role name methods."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance."""
        service = ZAIService.__new__(ZAIService)
        return service

    def test_get_user_role_name(self, service):
        """get_user_role_name returns 'user'."""
        assert service.get_user_role_name() == "user"

    def test_get_assistant_role_name(self, service):
        """get_assistant_role_name returns 'assistant'."""
        assert service.get_assistant_role_name() == "assistant"


class TestZAIServiceGetModelForImages:
    """Tests for _get_model_for_images method."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance."""
        service = ZAIService.__new__(ZAIService)
        return service

    def test_switches_to_vision_model_when_has_images(self, service):
        """_get_model_for_images switches to vision model when images present."""
        for model in NON_VISION_MODELS:
            result = service._get_model_for_images(model, has_images=True)
            assert result == VISION_MODEL

    def test_keeps_model_when_no_images(self, service):
        """_get_model_for_images keeps model when no images."""
        result = service._get_model_for_images("glm-4.7", has_images=False)
        assert result == "glm-4.7"

    def test_keeps_vision_model_when_has_images(self, service):
        """_get_model_for_images keeps vision-capable model even with images."""
        result = service._get_model_for_images("glm-4.6v", has_images=True)
        assert result == "glm-4.6v"

    def test_non_vision_models_constant(self):
        """NON_VISION_MODELS contains expected models."""
        assert "glm-4.7" in NON_VISION_MODELS
        assert "glm-4-flash" in NON_VISION_MODELS

    def test_vision_model_constant(self):
        """VISION_MODEL is set correctly."""
        assert VISION_MODEL == "glm-4.6v"


class TestZAIServiceRateLimitError:
    """Tests for _is_rate_limit_error method."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance."""
        service = ZAIService.__new__(ZAIService)
        return service

    def test_detects_rate_limit_in_message(self, service):
        """_is_rate_limit_error detects 'rate limit' in message."""
        error = Exception("Rate limit exceeded")
        assert service._is_rate_limit_error(error) is True

    def test_detects_429_in_message(self, service):
        """_is_rate_limit_error detects '429' in message."""
        error = Exception("Error 429: Too many requests")
        assert service._is_rate_limit_error(error) is True

    def test_returns_false_for_other_errors(self, service):
        """_is_rate_limit_error returns False for other errors."""
        error = Exception("Some other error")
        assert service._is_rate_limit_error(error) is False

    def test_case_insensitive(self, service):
        """_is_rate_limit_error is case insensitive."""
        error = Exception("RATE LIMIT exceeded")
        assert service._is_rate_limit_error(error) is True


class TestZAIServiceLogging:
    """Tests for logging methods."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance."""
        service = ZAIService.__new__(ZAIService)
        return service

    def test_log_raw_request_logs_when_debug_enabled(self, service):
        """_log_raw_request no longer logs (debug logging removed)."""
        with patch("persbot.services.zai_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            service._log_raw_request("test message")
            # Debug logging has been removed, so debug should not be called
            mock_logger.debug.assert_not_called()

    def test_log_raw_request_skips_when_debug_disabled(self, service):
        """_log_raw_request skips logging when DEBUG level is disabled."""
        with patch("persbot.services.zai_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = False
            service._log_raw_request("test message")
            mock_logger.debug.assert_not_called()

    def test_log_raw_response_logs_when_debug_enabled(self, service):
        """_log_raw_response no longer logs (debug logging removed)."""
        with patch("persbot.services.zai_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            service._log_raw_response({"response": "data"}, attempt=1)
            # Debug logging has been removed, so debug should not be called
            mock_logger.debug.assert_not_called()

    def test_log_raw_response_skips_when_debug_disabled(self, service):
        """_log_raw_response skips logging when DEBUG level is disabled."""
        with patch("persbot.services.zai_service.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = False
            service._log_raw_response({"response": "data"}, attempt=1)
            mock_logger.debug.assert_not_called()


class TestZAIServiceExtractTextFromResponse:
    """Tests for _extract_text_from_response method."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance."""
        service = ZAIService.__new__(ZAIService)
        return service

    def test_extracts_string_content(self, service):
        """_extract_text_from_response extracts string content."""
        mock_message = MagicMock()
        mock_message.content = "Hello, world!"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = service._extract_text_from_response(mock_response)
        assert result == "Hello, world!"

    def test_extracts_list_content(self, service):
        """_extract_text_from_response extracts text from list content."""
        mock_message = MagicMock()
        mock_message.content = [
            {"text": "Hello"},
            {"text": "world!"},
        ]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = service._extract_text_from_response(mock_response)
        assert result == "Hello world!"

    def test_returns_empty_for_no_choices(self, service):
        """_extract_text_from_response returns empty string for no choices."""
        mock_response = MagicMock()
        mock_response.choices = []

        result = service._extract_text_from_response(mock_response)
        assert result == ""

    def test_returns_empty_for_no_message(self, service):
        """_extract_text_from_response returns empty string for no message."""
        mock_choice = MagicMock()
        mock_choice.message = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = service._extract_text_from_response(mock_response)
        assert result == ""

    def test_returns_empty_for_no_content(self, service):
        """_extract_text_from_response returns empty string for no content."""
        mock_message = MagicMock()
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = service._extract_text_from_response(mock_response)
        assert result == ""

    def test_handles_object_content_blocks(self, service):
        """_extract_text_from_response handles object content blocks."""
        mock_block = MagicMock()
        mock_block.text = "Block text"

        mock_message = MagicMock()
        mock_message.content = [mock_block]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = service._extract_text_from_response(mock_response)
        assert result == "Block text"

    def test_returns_empty_on_exception(self, service):
        """_extract_text_from_response returns empty string on exception."""
        result = service._extract_text_from_response(None)
        assert result == ""


class TestZAIServiceSummarizeText:
    """Tests for summarize_text method."""

    @pytest.fixture
    def service(self):
        """Create ZAIService with mocked dependencies."""
        config = MagicMock()
        config.temperature = 0.7
        config.top_p = 1.0

        prompt_service = MagicMock()
        prompt_service.get_summary_prompt.return_value = "Summarize:"

        service = ZAIService.__new__(ZAIService)
        service.config = config
        service.prompt_service = prompt_service
        service._summary_model_name = "glm-4-flash"
        service.client = MagicMock()

        return service

    @pytest.mark.asyncio
    async def test_returns_default_for_empty_text(self, service):
        """summarize_text returns default message for empty text."""
        result = await service.summarize_text("")
        assert result == "요약할 메시지가 없습니다."

    @pytest.mark.asyncio
    async def test_returns_default_for_whitespace_text(self, service):
        """summarize_text returns default message for whitespace text."""
        result = await service.summarize_text("   ")
        assert result == "요약할 메시지가 없습니다."

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_when_cancel_event_set(self, service):
        """summarize_text raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service.summarize_text("test text", cancel_event=cancel_event)

    @pytest.mark.asyncio
    async def test_calls_execute_with_retry(self, service):
        """summarize_text calls execute_with_retry with correct parameters."""
        with patch.object(service, "execute_with_retry", return_value="Summary result"):
            result = await service.summarize_text("Text to summarize")

            assert result == "Summary result"
            service.execute_with_retry.assert_called_once()


class TestZAIServiceGenerateChatResponse:
    """Tests for generate_chat_response method."""

    @pytest.fixture
    def mock_discord_message(self):
        """Create a mock Discord message."""
        msg = MagicMock()
        msg.author.id = 12345
        msg.author.name = "testuser"
        msg.id = 67890
        msg.attachments = []
        return msg

    @pytest.fixture
    def mock_chat_session(self):
        """Create a mock chat session."""
        session = MagicMock()
        session._model_name = "glm-4.7"
        session._history = []
        session.send_message = MagicMock(return_value=(
            MagicMock(content="user msg"),
            MagicMock(content="assistant msg"),
            MagicMock(),
        ))
        return session

    @pytest.fixture
    def service(self):
        """Create ZAIService with mocked dependencies."""
        config = MagicMock()
        config.temperature = 0.7
        config.top_p = 1.0

        prompt_service = MagicMock()

        service = ZAIService.__new__(ZAIService)
        service.config = config
        service.prompt_service = prompt_service
        service._assistant_model_name = "glm-4.7"
        service.client = MagicMock()

        return service

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_when_cancel_event_set(
        self, service, mock_chat_session, mock_discord_message
    ):
        """generate_chat_response raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service.generate_chat_response(
                mock_chat_session,
                "test message",
                mock_discord_message,
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_calls_execute_with_retry(
        self, service, mock_chat_session, mock_discord_message
    ):
        """generate_chat_response calls execute_with_retry."""
        user_msg = MagicMock()
        user_msg.content = "user message"

        model_msg = MagicMock()
        model_msg.content = "assistant response"

        mock_response = MagicMock()

        with patch.object(service, "_extract_images_from_messages", return_value=[]):
            with patch.object(service, "execute_with_retry", return_value=(user_msg, model_msg, mock_response)):
                result = await service.generate_chat_response(
                    mock_chat_session,
                    "test message",
                    mock_discord_message,
                )

        assert result == ("assistant response", mock_response)

    @pytest.mark.asyncio
    async def test_returns_none_when_result_is_none(
        self, service, mock_chat_session, mock_discord_message
    ):
        """generate_chat_response returns None when execute_with_retry returns None."""
        with patch.object(service, "_extract_images_from_messages", return_value=[]):
            with patch.object(service, "execute_with_retry", return_value=None):
                result = await service.generate_chat_response(
                    mock_chat_session,
                    "test message",
                    mock_discord_message,
                )

        assert result is None

    @pytest.mark.asyncio
    async def test_switches_to_vision_model_with_images(
        self, service, mock_chat_session, mock_discord_message
    ):
        """generate_chat_response switches to vision model when images present."""
        user_msg = MagicMock()
        user_msg.content = "user message"

        model_msg = MagicMock()
        model_msg.content = "response"

        with patch.object(service, "_extract_images_from_messages", return_value=[b"image data"]):
            with patch.object(service, "execute_with_retry", return_value=(user_msg, model_msg, MagicMock())):
                await service.generate_chat_response(
                    mock_chat_session,
                    "test message",
                    mock_discord_message,
                )

        # Should have switched to vision model
        assert mock_chat_session._model_name == VISION_MODEL


class TestZAIServiceSendToolResults:
    """Tests for send_tool_results method."""

    @pytest.fixture
    def mock_chat_session(self):
        """Create a mock chat session."""
        session = MagicMock()
        session._history = [MagicMock(role="assistant")]
        return session

    @pytest.fixture
    def service(self):
        """Create ZAIService with mocked dependencies."""
        service = ZAIService.__new__(ZAIService)
        service._assistant_model_name = "glm-4.7"
        service.client = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_when_cancel_event_set(self, service, mock_chat_session):
        """send_tool_results raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service.send_tool_results(
                mock_chat_session,
                [],
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_calls_execute_with_retry(self, service, mock_chat_session):
        """send_tool_results calls execute_with_retry."""
        model_msg = MagicMock()
        model_msg.content = "Tool response"

        mock_response = MagicMock()

        with patch.object(service, "execute_with_retry", return_value=(model_msg, mock_response)):
            result = await service.send_tool_results(
                mock_chat_session,
                [("tool_call", {"result": "data"})],
            )

        assert result == ("Tool response", mock_response)

    @pytest.mark.asyncio
    async def test_returns_none_when_result_is_none(self, service, mock_chat_session):
        """send_tool_results returns None when execute_with_retry returns None."""
        with patch.object(service, "execute_with_retry", return_value=None):
            result = await service.send_tool_results(
                mock_chat_session,
                [],
            )

        assert result is None


class TestZAIServiceToolSupport:
    """Tests for tool support methods."""

    @pytest.fixture
    def service(self):
        """Create ZAIService instance."""
        service = ZAIService.__new__(ZAIService)
        return service

    def test_get_tools_for_provider(self, service):
        """get_tools_for_provider delegates to ZAIToolAdapter."""
        with patch("persbot.services.zai_service.ZAIToolAdapter") as mock_adapter:
            mock_adapter.convert_tools.return_value = [{"type": "function"}]
            result = service.get_tools_for_provider([{"tool": "def"}])

            mock_adapter.convert_tools.assert_called_once_with([{"tool": "def"}])
            assert result == [{"type": "function"}]

    def test_extract_function_calls(self, service):
        """extract_function_calls delegates to ZAIToolAdapter."""
        with patch("persbot.services.zai_service.ZAIToolAdapter") as mock_adapter:
            mock_adapter.extract_function_calls.return_value = [{"name": "test_func"}]
            result = service.extract_function_calls({"response": "data"})

            mock_adapter.extract_function_calls.assert_called_once()
            assert result == [{"name": "test_func"}]

    def test_format_function_results(self, service):
        """format_function_results delegates to ZAIToolAdapter."""
        with patch("persbot.services.zai_service.ZAIToolAdapter") as mock_adapter:
            mock_adapter.format_results.return_value = [{"role": "tool", "content": "result"}]
            result = service.format_function_results([{"name": "func", "result": "data"}])

            mock_adapter.format_results.assert_called_once()
            assert result == [{"role": "tool", "content": "result"}]


class TestZAIServiceGenerateChatResponseStream:
    """Tests for generate_chat_response_stream method."""

    @pytest.fixture
    def mock_discord_message(self):
        """Create a mock Discord message."""
        msg = MagicMock()
        msg.author.id = 12345
        msg.author.name = "testuser"
        msg.id = 67890
        msg.attachments = []
        return msg

    @pytest.fixture
    def mock_chat_session(self):
        """Create a mock chat session."""
        session = MagicMock()
        session._model_name = "glm-4.7"
        session._history = []
        return session

    @pytest.fixture
    def service(self):
        """Create ZAIService with mocked dependencies."""
        service = ZAIService.__new__(ZAIService)
        service._assistant_model_name = "glm-4.7"
        service.client = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_when_cancel_event_set(
        self, service, mock_chat_session, mock_discord_message
    ):
        """generate_chat_response_stream raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            async for _ in service.generate_chat_response_stream(
                mock_chat_session,
                "test message",
                mock_discord_message,
                cancel_event=cancel_event,
            ):
                pass

    @pytest.mark.asyncio
    async def test_yields_text_chunks(self, service, mock_chat_session, mock_discord_message):
        """generate_chat_response_stream yields text chunks."""
        # Create mock stream chunks
        mock_delta = MagicMock()
        mock_delta.content = "Hello\nWorld"

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]

        # Create mock stream
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([mock_chunk]))
        mock_stream.close = MagicMock()

        mock_user_msg = MagicMock()
        mock_user_msg.content = "user message"

        with patch.object(service, "_extract_images_from_messages", return_value=[]):
            with patch.object(service, "_log_raw_request"):
                with patch("asyncio.to_thread", return_value=(mock_stream, mock_user_msg)):
                    chunks = []
                    async for chunk in service.generate_chat_response_stream(
                        mock_chat_session,
                        "test message",
                        mock_discord_message,
                    ):
                        chunks.append(chunk)

        # Should yield lines
        assert len(chunks) > 0


class TestZAIServiceCreateAssistantModel:
    """Tests for create_assistant_model method."""

    @pytest.fixture
    def service(self):
        """Create ZAIService with mocked dependencies."""
        config = MagicMock()
        config.temperature = 0.7
        config.top_p = 1.0

        prompt_service = MagicMock()

        service = ZAIService.__new__(ZAIService)
        service.config = config
        service.prompt_service = prompt_service
        service._assistant_model_name = "glm-4.7"
        service._assistant_cache = {}
        service.client = MagicMock()
        service._max_messages = 7

        return service

    def test_creates_assistant_model(self, service):
        """create_assistant_model creates and returns assistant model."""
        with patch.object(service, "_get_or_create_assistant") as mock_get:
            mock_get.return_value = MagicMock()
            result = service.create_assistant_model("Test prompt")

            mock_get.assert_called_once_with("glm-4.7", "Test prompt")
            assert result is not None

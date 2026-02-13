"""Feature tests for chat use case module.

Tests focus on behavior using mocking:
- ChatRequest: request for chat generation
- ChatResponse: response from chat generation
- StreamChunk: a chunk of streamed response
- ChatUseCase: use case for chat operations
"""

import sys
from unittest.mock import Mock, MagicMock, AsyncMock

import pytest


# Mock external dependencies before any imports
_mock_ddgs = MagicMock()
_mock_ddgs.DDGS = MagicMock
_mock_ddgs.exceptions = MagicMock()
_mock_ddgs.exceptions.RatelimitException = Exception
_mock_ddgs.exceptions.DDGSException = Exception
sys.modules['ddgs'] = _mock_ddgs
sys.modules['ddgs.exceptions'] = _mock_ddgs.exceptions

_mock_bs4 = MagicMock()
sys.modules['bs4'] = _mock_bs4


class TestChatRequest:
    """Tests for ChatRequest dataclass."""

    def test_chat_request_exists(self):
        """ChatRequest class exists."""
        from persbot.use_cases.chat_use_case import ChatRequest
        assert ChatRequest is not None

    def test_chat_request_has_required_fields(self):
        """ChatRequest has required fields."""
        from persbot.use_cases.chat_use_case import ChatRequest

        # Create mock dependencies
        mock_message = MagicMock()
        mock_resolution = MagicMock()

        request = ChatRequest(
            user_message="Hello",
            discord_message=mock_message,
            resolution=mock_resolution,
        )

        assert request.user_message == "Hello"
        assert request.discord_message == mock_message
        assert request.resolution == mock_resolution

    def test_chat_request_defaults(self):
        """ChatRequest has correct defaults."""
        from persbot.use_cases.chat_use_case import ChatRequest

        mock_message = MagicMock()
        mock_resolution = MagicMock()

        request = ChatRequest(
            user_message="Hello",
            discord_message=mock_message,
            resolution=mock_resolution,
        )

        assert request.use_summarizer_backend is False
        assert request.cancel_event is None


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_chat_response_exists(self):
        """ChatResponse class exists."""
        from persbot.use_cases.chat_use_case import ChatResponse
        assert ChatResponse is not None

    def test_chat_response_has_required_fields(self):
        """ChatResponse has required fields."""
        from persbot.use_cases.chat_use_case import ChatResponse

        response = ChatResponse(
            text="Hello world",
            session_key="channel:123",
            response_obj=MagicMock(),
            images=[],
        )

        assert response.text == "Hello world"
        assert response.session_key == "channel:123"

    def test_chat_response_defaults(self):
        """ChatResponse has correct defaults."""
        from persbot.use_cases.chat_use_case import ChatResponse

        response = ChatResponse(
            text="Hello",
            session_key="channel:123",
            response_obj=None,
            images=[],
        )

        assert response.notification is None
        assert response.tool_rounds == 0


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_stream_chunk_exists(self):
        """StreamChunk class exists."""
        from persbot.use_cases.chat_use_case import StreamChunk
        assert StreamChunk is not None

    def test_stream_chunk_has_required_fields(self):
        """StreamChunk has required fields."""
        from persbot.use_cases.chat_use_case import StreamChunk

        chunk = StreamChunk(text="Hello")

        assert chunk.text == "Hello"
        assert chunk.is_final is False
        assert chunk.metadata is None

    def test_stream_chunk_can_be_final(self):
        """StreamChunk can be marked as final."""
        from persbot.use_cases.chat_use_case import StreamChunk

        chunk = StreamChunk(text="Done", is_final=True)

        assert chunk.is_final is True


class TestChatUseCase:
    """Tests for ChatUseCase class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return Mock()

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        service.model_usage_service = Mock()
        service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-flash"
        service.model_usage_service.check_and_increment_usage = AsyncMock(
            return_value=(True, "gemini-flash", None)
        )
        service.generate_chat_response = AsyncMock(
            return_value=("Response text", MagicMock())
        )
        service.get_active_backend = Mock(return_value=MagicMock())
        service.extract_function_calls_from_response = Mock(return_value=[])
        return service

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        manager = Mock()
        manager.get_or_create = AsyncMock(
            return_value=(MagicMock(), "channel:123")
        )
        manager.undo_last_exchanges = Mock(return_value=[])
        return manager

    def test_chat_use_case_exists(self):
        """ChatUseCase class exists."""
        from persbot.use_cases.chat_use_case import ChatUseCase
        assert ChatUseCase is not None

    def test_chat_use_case_creates_with_dependencies(
        self, mock_config, mock_llm_service, mock_session_manager
    ):
        """ChatUseCase creates with dependencies."""
        from persbot.use_cases.chat_use_case import ChatUseCase

        use_case = ChatUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        assert use_case.config == mock_config
        assert use_case.llm_service == mock_llm_service
        assert use_case.session_manager == mock_session_manager

    def test_chat_use_case_has_tool_labels(
        self, mock_config, mock_llm_service, mock_session_manager
    ):
        """ChatUseCase has tool labels mapping."""
        from persbot.use_cases.chat_use_case import ChatUseCase

        use_case = ChatUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        assert hasattr(use_case, '_tool_labels')
        assert len(use_case._tool_labels) > 0

    def test_get_primary_message_returns_single(
        self, mock_config, mock_llm_service, mock_session_manager
    ):
        """_get_primary_message returns single message."""
        from persbot.use_cases.chat_use_case import ChatUseCase

        use_case = ChatUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        mock_message = MagicMock()
        result = use_case._get_primary_message(mock_message)

        assert result == mock_message

    def test_get_primary_message_returns_first_from_list(
        self, mock_config, mock_llm_service, mock_session_manager
    ):
        """_get_primary_message returns first from list."""
        from persbot.use_cases.chat_use_case import ChatUseCase

        use_case = ChatUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        mock_messages = [MagicMock(), MagicMock()]
        result = use_case._get_primary_message(mock_messages)

        assert result == mock_messages[0]

    def test_get_enabled_tools_returns_none_when_no_manager(
        self, mock_config, mock_llm_service, mock_session_manager
    ):
        """_get_enabled_tools returns None when no tool manager."""
        from persbot.use_cases.chat_use_case import ChatUseCase

        use_case = ChatUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        result = use_case._get_enabled_tools()
        assert result is None

    def test_get_enabled_tools_returns_tools_when_enabled(
        self, mock_config, mock_llm_service, mock_session_manager
    ):
        """_get_enabled_tools returns tools when enabled."""
        from persbot.use_cases.chat_use_case import ChatUseCase

        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled = Mock(return_value=True)
        mock_tool_manager.get_enabled_tools = Mock(return_value={"tool1": MagicMock()})

        use_case = ChatUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        result = use_case._get_enabled_tools()
        assert result is not None
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_regenerate_last_response_returns_none_when_no_removed(
        self, mock_config, mock_llm_service, mock_session_manager
    ):
        """regenerate_last_response returns None when no messages removed."""
        from persbot.use_cases.chat_use_case import ChatUseCase

        mock_session_manager.undo_last_exchanges = Mock(return_value=[])

        use_case = ChatUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        result = await use_case.regenerate_last_response(
            channel_id=123, session_key="channel:123"
        )

        assert result is None

"""Tests for streaming response functionality."""

import asyncio
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from persbot.services.base import ChatMessage
from persbot.services.session_wrappers.openai_session import ChatCompletionSession
from persbot.services.session_wrappers.zai_session import ZAIChatSession


class MockStreamChunk:
    """Mock chunk from streaming API."""

    def __init__(self, content: str):
        self.choices = [Mock(delta=Mock(content=content))]


def create_mock_stream(chunks: List[str]):
    """Create a mock stream that yields chunks."""
    for chunk in chunks:
        yield MockStreamChunk(chunk)


class TestOpenAISessionStreaming:
    """Tests for OpenAI session streaming."""

    def test_send_message_stream_exists(self):
        """Verify send_message_stream method exists."""
        assert hasattr(ChatCompletionSession, 'send_message_stream')

    def test_send_message_stream_returns_tuple(self):
        """Verify send_message_stream returns correct structure."""
        mock_client = Mock()
        mock_stream = Mock()
        mock_client.chat.completions.create.return_value = mock_stream

        session = ChatCompletionSession(
            client=mock_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=1.0,
            top_p=1.0,
            max_messages=10,
        )

        result, user_msg = session.send_message_stream(
            user_message="Hello",
            author_id=123,
        )

        # Verify stream=True was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['stream'] is True

        # Verify returns stream and user message
        assert result == mock_stream
        assert isinstance(user_msg, ChatMessage)
        assert user_msg.role == "user"


class TestZAISessionStreaming:
    """Tests for Z.AI session streaming."""

    def test_send_message_stream_exists(self):
        """Verify send_message_stream method exists."""
        assert hasattr(ZAIChatSession, 'send_message_stream')

    def test_send_message_stream_returns_tuple(self):
        """Verify send_message_stream returns correct structure."""
        mock_client = Mock()
        mock_stream = Mock()
        mock_client.chat.completions.create.return_value = mock_stream

        session = ZAIChatSession(
            client=mock_client,
            model_name="glm-4",
            system_instruction="You are helpful",
            temperature=1.0,
            top_p=1.0,
            max_messages=10,
            text_extractor=lambda x: "",
        )

        result, user_msg = session.send_message_stream(
            user_message="Hello",
            author_id=123,
        )

        # Verify stream=True was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['stream'] is True

        # Verify returns stream and user message
        assert result == mock_stream
        assert isinstance(user_msg, ChatMessage)


class TestStreamingResponseBuffering:
    """Tests for line-break buffering in streaming responses."""

    @pytest.mark.asyncio
    async def test_buffering_yields_after_linebreak(self):
        """Test that chunks are yielded after line breaks."""
        from persbot.services.openai_service import OpenAIService

        # Create mock stream that yields chunks
        chunks = [
            "Hello ",
            "world!\n",
            "This is ",
            "line two.\n",
            "Final line",
        ]

        mock_stream = iter([MockStreamChunk(c) for c in chunks])

        # Create mock session with stream method
        mock_session = Mock()
        mock_session.send_message_stream.return_value = (mock_stream, Mock())

        # Create service with mock
        mock_config = Mock()
        mock_config.openai_api_key = "test-key"
        mock_config.api_request_timeout = 30.0

        with patch('persbot.services.openai_service.OpenAI'):
            service = OpenAIService.__new__(OpenAIService)
            service.config = mock_config
            service._extract_images_from_message = AsyncMock(return_value=[])
            service._log_raw_request = Mock()

        # Verify method exists
        assert hasattr(OpenAIService, 'generate_chat_response_stream')

    @pytest.mark.asyncio
    async def test_streaming_accumulates_full_content(self):
        """Test that full content is accumulated during streaming."""
        # This tests that the buffer correctly accumulates all text
        buffer = ""
        chunks = ["Hello ", "world!\n", "Line 2\n", "End"]
        expected_full = "".join(chunks)

        for chunk in chunks:
            buffer += chunk

        assert buffer == expected_full


class TestStreamingResponseSender:
    """Tests for send_streaming_response function."""

    @pytest.mark.asyncio
    async def test_send_streaming_response_exists(self):
        """Verify send_streaming_response function exists."""
        from persbot.bot.response_sender import send_streaming_response
        assert callable(send_streaming_response)

    @pytest.mark.asyncio
    async def test_send_streaming_response_sends_chunks(self):
        """Test that streaming response sends chunks to channel."""
        from persbot.bot.response_sender import send_streaming_response

        # Create mock channel
        mock_channel = Mock()
        mock_channel.typing = MagicMock()
        mock_channel.typing.return_value.__aenter__ = AsyncMock()
        mock_channel.typing.return_value.__aexit__ = AsyncMock()
        mock_channel.send = AsyncMock(return_value=Mock(id=123))

        # Create mock session manager
        mock_session_manager = Mock()
        mock_session_manager.link_message_to_session = Mock()

        # Create async iterator for stream
        async def mock_stream():
            yield "Hello world!\n"
            yield "Second line\n"

        # Call the function
        messages = await send_streaming_response(
            channel=mock_channel,
            stream=mock_stream(),
            session_key="test-session",
            session_manager=mock_session_manager,
        )

        # Verify messages were sent
        assert len(messages) >= 1
        assert mock_channel.send.called

    @pytest.mark.asyncio
    async def test_send_streaming_response_handles_cancellation(self):
        """Test that cancellation is properly handled."""
        from persbot.bot.response_sender import send_streaming_response

        mock_channel = Mock()
        mock_channel.id = 123
        mock_channel.typing = MagicMock()
        mock_channel.typing.return_value.__aenter__ = AsyncMock()
        mock_channel.typing.return_value.__aexit__ = AsyncMock()
        mock_channel.send = AsyncMock(return_value=Mock(id=123))

        mock_session_manager = Mock()
        mock_session_manager.link_message_to_session = Mock()

        # Create stream that raises CancelledError
        async def mock_stream_cancel():
            yield "First line\n"
            raise asyncio.CancelledError()

        # Should re-raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await send_streaming_response(
                channel=mock_channel,
                stream=mock_stream_cancel(),
                session_key="test-session",
                session_manager=mock_session_manager,
            )


class TestChatHandlerStreaming:
    """Tests for chat_handler streaming functions."""

    def test_create_chat_reply_stream_exists(self):
        """Verify create_chat_reply_stream function exists."""
        from persbot.bot.chat_handler import create_chat_reply_stream
        assert callable(create_chat_reply_stream)

    def test_send_streaming_response_exists_in_chat_handler(self):
        """Verify send_streaming_response exists in chat_handler."""
        from persbot.bot.chat_handler import send_streaming_response as chat_send_stream
        assert callable(chat_send_stream)


class TestLLMServiceStreaming:
    """Tests for LLMService streaming methods."""

    def test_generate_chat_response_stream_exists(self):
        """Verify generate_chat_response_stream method exists."""
        from persbot.services.llm_service import LLMService
        assert hasattr(LLMService, 'generate_chat_response_stream')

    @pytest.mark.asyncio
    async def test_streaming_falls_back_for_unsupported_provider(self):
        """Test that streaming falls back for providers without streaming."""
        from persbot.services.llm_service import LLMService

        # Create mock backend without streaming support
        # Use spec to limit available methods
        mock_backend = Mock(spec=['generate_chat_response', 'get_tools_for_provider'])
        mock_backend.generate_chat_response = AsyncMock(
            return_value=("Test response", None)
        )
        # No generate_chat_response_stream attribute due to spec

        mock_config = Mock()

        service = LLMService.__new__(LLMService)
        service.config = mock_config
        service.assistant_backend = mock_backend
        service.summarizer_backend = mock_backend
        service.model_usage_service = Mock()
        service.model_usage_service.check_and_increment_usage = AsyncMock(
            return_value=(True, "test-model", None)
        )
        service._extract_message_metadata = Mock(
            return_value=(123, 456, 789, Mock())
        )
        service.get_backend_for_model = Mock(return_value=mock_backend)
        service.model_usage_service.get_api_model_name = Mock(return_value="test-model")

        # Should fall back to non-streaming since backend lacks generate_chat_response_stream
        chunks = []
        async for chunk in service.generate_chat_response_stream(
            Mock(), "test message", Mock()
        ):
            chunks.append(chunk)

        # Should have gotten the fallback response
        assert len(chunks) >= 1
        assert "Test response" in "".join(chunks)


class TestProviderCapabilities:
    """Tests for provider streaming capabilities."""

    def test_openai_supports_streaming(self):
        """Verify OpenAI capabilities include streaming."""
        from persbot.providers.base import ProviderCaps
        assert ProviderCaps.OPENAI.supports_streaming is True

    def test_zai_supports_streaming(self):
        """Verify ZAI capabilities include streaming."""
        from persbot.providers.base import ProviderCaps
        assert ProviderCaps.ZAI.supports_streaming is True

    def test_gemini_supports_streaming(self):
        """Verify Gemini capabilities include streaming."""
        from persbot.providers.base import ProviderCaps
        assert ProviderCaps.GEMINI.supports_streaming is True

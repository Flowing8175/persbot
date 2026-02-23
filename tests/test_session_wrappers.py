"""Comprehensive tests for session wrapper modules.

Tests for:
- persbot.services.session_wrappers.gemini_session: Helper functions and GeminiChatSession
- persbot.services.session_wrappers.openai_session: Dataclasses and session classes
"""

import base64
import sys
from collections import deque
from dataclasses import asdict
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from typing import Any
import pytest

# Mock ddgs module before any imports that depend on it
_mock_ddgs = MagicMock()
_mock_ddgs.DDGS = MagicMock
_mock_ddgs.exceptions = MagicMock()
_mock_ddgs.exceptions.RatelimitException = Exception
_mock_ddgs.exceptions.DDGSException = Exception
sys.modules['ddgs'] = _mock_ddgs
sys.modules['ddgs.exceptions'] = _mock_ddgs.exceptions

# Mock bs4 module before any imports that depend on it
_mock_bs4 = MagicMock()
sys.modules['bs4'] = _mock_bs4

from persbot.services.session_wrappers.gemini_session import (
    _build_content,
    _build_text_part,
    _build_function_call_part,
    _build_function_response_part,
    _build_inline_data_part,
    _deep_serialize,
    _sanitize_for_json,
    _content_to_safe_dict,
    extract_clean_text,
    GeminiChatSession,
)

from persbot.services.session_wrappers.openai_session import (
    FakeDelta,
    FakeChoice,
    FakeChunk,
    ResponsesAPIStreamAdapter,
    encode_image_to_url,
    OpenAIMessage,
    BaseOpenAISession,
    ChatCompletionSession,
    ResponseSession,
)

from persbot.services.base import ChatMessage


# =============================================================================
# Mock Helpers and Fixtures
# =============================================================================

class MockPart:
    """Mock genai_types.Part for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockContent:
    """Mock genai_types.Content for testing."""
    def __init__(self, role="user", parts=None):
        self.role = role
        self.parts = parts or []


class MockCandidate:
    """Mock genai_types.Candidate for testing."""
    def __init__(self, content=None):
        self.content = content


class MockResponse:
    """Mock Gemini response object."""
    def __init__(self, candidates=None):
        self.candidates = candidates or []


class MockCachedModel:
    """Mock _CachedModel factory for testing GeminiChatSession."""
    def __init__(self):
        self.generate_content_calls = []
        self.generate_content_stream_calls = []
        self.last_contents = None
        self.last_tools = None

    def generate_content(self, contents=None, tools=None):
        self.generate_content_calls.append((contents, tools))
        self.last_contents = contents
        self.last_tools = tools

        # Create mock response with text content
        mock_part = MockPart(text="Test response")
        mock_content = MockContent(role="model", parts=[mock_part])
        mock_candidate = MockCandidate(content=mock_content)
        return MockResponse(candidates=[mock_candidate])

    def generate_content_stream(self, contents=None, tools=None):
        self.generate_content_stream_calls.append((contents, tools))
        self.last_contents = contents
        self.last_tools = tools

        # Return an async iterator that yields chunks
        async def stream_gen():
            mock_part = MockPart(text="Streaming")
            mock_content = MockContent(role="model", parts=[mock_part])
            mock_candidate = MockCandidate(content=mock_content)
            yield MockResponse(candidates=[mock_candidate])

        return stream_gen()


@pytest.fixture
def mock_gemini_factory():
    """Create a mock Gemini cached model factory."""
    return MockCachedModel()


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.responses = Mock()

    # Mock chat completions
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message = Mock()
    mock_completion.choices[0].message.content = "Test response"
    mock_completion.choices[0].message.tool_calls = None
    client.chat.completions.create = Mock(return_value=mock_completion)

    # Mock responses
    mock_response = Mock()
    mock_response.output_text = "Test response"
    client.responses.create = Mock(return_value=mock_response)

    return client


# =============================================================================
# Tests for gemini_session.py Helper Functions
# =============================================================================

class TestBuildContent:
    """Tests for _build_content() function."""

    def test_builds_content_with_role_and_parts(self):
        """_build_content creates Content with role and parts."""
        parts = [_build_text_part("Hello")]
        content = _build_content("user", parts)

        assert content.role == "user"
        assert len(content.parts) == 1
        assert content.parts[0].text == "Hello"

    def test_builds_content_with_model_role(self):
        """_build_content works with model role."""
        parts = [_build_text_part("Response")]
        content = _build_content("model", parts)

        assert content.role == "model"
        assert content.parts[0].text == "Response"


class TestBuildTextPart:
    """Tests for _build_text_part() function."""

    def test_builds_text_part(self):
        """_build_text_part creates Part with text."""
        part = _build_text_part("Hello world")

        assert hasattr(part, 'text')
        assert part.text == "Hello world"

    def test_builds_empty_text_part(self):
        """_build_text_part works with empty string."""
        part = _build_text_part("")

        assert hasattr(part, 'text')
        assert part.text == ""


class TestBuildFunctionCallPart:
    """Tests for _build_function_call_part() function."""

    def test_builds_function_call_part(self):
        """_build_function_call_part creates Part with function_call."""
        part = _build_function_call_part("test_func", {"arg1": "value1"})

        assert hasattr(part, 'function_call')
        assert part.function_call.name == "test_func"
        assert part.function_call.args == {"arg1": "value1"}

    def test_builds_function_call_with_empty_args(self):
        """_build_function_call_part works with empty args."""
        part = _build_function_call_part("test_func", {})

        assert hasattr(part, 'function_call')
        assert part.function_call.name == "test_func"
        assert part.function_call.args == {}


class TestBuildFunctionResponsePart:
    """Tests for _build_function_response_part() function."""

    def test_builds_function_response_part_with_result(self):
        """_build_function_response_part creates Part with function_response result."""
        part = _build_function_response_part("test_func", {"result": "success"})

        assert hasattr(part, 'function_response')
        assert part.function_response.name == "test_func"
        assert part.function_response.response == {"result": "success"}

    def test_builds_function_response_part_with_error(self):
        """_build_function_response_part creates Part with error."""
        part = _build_function_response_part("test_func", {"error": "Failed"})

        assert hasattr(part, 'function_response')
        assert part.function_response.name == "test_func"
        assert part.function_response.response == {"error": "Failed"}

    def test_builds_function_response_part_with_empty_response(self):
        """_build_function_response_part works with empty response."""
        part = _build_function_response_part("test_func", {})

        assert hasattr(part, 'function_response')
        assert part.function_response.name == "test_func"
        assert part.function_response.response == {}


class TestBuildInlineDataPart:
    """Tests for _build_inline_data_part() function."""

    def test_builds_inline_data_part_for_jpeg(self):
        """_build_inline_data_part creates Part with JPEG image data."""
        # JPEG magic bytes
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        part = _build_inline_data_part(jpeg_data, "image/jpeg")

        assert hasattr(part, 'inline_data')
        assert part.inline_data.mime_type == "image/jpeg"
        # Should be base64 encoded
        assert isinstance(part.inline_data.data, str)

    def test_builds_inline_data_part_for_png(self):
        """_build_inline_data_part creates Part with PNG image data."""
        # PNG magic bytes
        png_data = b"\x89PNG\r\n\x1a\n"
        part = _build_inline_data_part(png_data, "image/png")

        assert hasattr(part, 'inline_data')
        assert part.inline_data.mime_type == "image/png"
        assert isinstance(part.inline_data.data, str)

    def test_base64_encoding_is_correct(self):
        """_build_inline_data_part correctly base64 encodes data."""
        test_data = b"Hello"
        part = _build_inline_data_part(test_data, "text/plain")

        expected_b64 = base64.b64encode(test_data).decode("utf-8")
        assert part.inline_data.data == expected_b64


class TestDeepSerialize:
    """Tests for _deep_serialize() function."""

    def test_serializes_primitives(self):
        """_deep_serialize returns primitives unchanged."""
        assert _deep_serialize(None) == ""
        assert _deep_serialize("text") == "text"
        assert _deep_serialize(42) == 42
        assert _deep_serialize(3.14) == 3.14
        assert _deep_serialize(True) is True

    def test_serializes_bytes(self):
        """_deep_serialize converts bytes to len string."""
        result = _deep_serialize(b"hello")
        assert result == "<bytes len=5>"

    def test_serializes_dict(self):
        """_deep_serialize recursively serializes dict values."""
        result = _deep_serialize({"key": "value", "num": 42})
        assert result == {"key": "value", "num": 42}

    def test_serializes_nested_dict(self):
        """_deep_serialize handles nested dicts."""
        result = _deep_serialize({"outer": {"inner": "value"}})
        assert result == {"outer": {"inner": "value"}}

    def test_serializes_list(self):
        """_deep_serialize recursively serializes list items."""
        result = _deep_serialize([1, 2, "three"])
        assert result == [1, 2, "three"]

    def test_serializes_tuple(self):
        """_deep_serialize converts tuple to list."""
        result = _deep_serialize((1, 2, 3))
        assert result == [1, 2, 3]

    def test_serializes_set(self):
        """_deep_serialize converts set to list."""
        result = _deep_serialize({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_serializes_custom_object_to_string(self):
        """_deep_serialize converts custom objects to string."""
        class CustomObj:
            def __str__(self):
                return "CustomObject"

        result = _deep_serialize(CustomObj())
        assert result == "CustomObject"

    def test_serializes_nested_complex_structure(self):
        """_deep_serialize handles deeply nested structures."""
        result = _deep_serialize({
            "list": [1, 2, {"nested": "value"}],
            "dict": {"key": [1, 2, 3]}
        })
        assert result == {
            "list": [1, 2, {"nested": "value"}],
            "dict": {"key": [1, 2, 3]}
        }


class TestSanitizeForJson:
    """Tests for _sanitize_for_json() function."""

    def test_preserves_primitives(self):
        """_sanitize_for_json preserves primitive types."""
        assert _sanitize_for_json(None) is None
        assert _sanitize_for_json("text") == "text"
        assert _sanitize_for_json(42) == 42
        assert _sanitize_for_json(3.14) == 3.14
        assert _sanitize_for_json(True) is True

    def test_converts_bytes(self):
        """_sanitize_for_json converts bytes to len string."""
        result = _sanitize_for_json(b"hello")
        assert result == "<bytes len=5>"

    def test_preserves_dict_structure(self):
        """_sanitize_for_json preserves dict structure."""
        result = _sanitize_for_json({"key": "value", "num": 42})
        assert result == {"key": "value", "num": 42}

    def test_preserves_nested_dict(self):
        """_sanitize_for_json preserves nested dicts."""
        result = _sanitize_for_json({"outer": {"inner": "value"}})
        assert result == {"outer": {"inner": "value"}}

    def test_preserves_list(self):
        """_sanitize_for_json preserves lists."""
        result = _sanitize_for_json([1, 2, "three"])
        assert result == [1, 2, "three"]

    def test_preserves_tuple(self):
        """_sanitize_for_json converts tuple to list."""
        result = _sanitize_for_json((1, 2, 3))
        assert result == [1, 2, 3]

    def test_preserves_set(self):
        """_sanitize_for_json converts set to list."""
        result = _sanitize_for_json({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_converts_custom_object_to_string(self):
        """_sanitize_for_json converts custom objects to string."""
        class CustomObj:
            def __str__(self):
                return "CustomObject"

        result = _sanitize_for_json(CustomObj())
        assert result == "CustomObject"

    def test_does_not_modify_json_safe_objects(self):
        """_sanitize_for_json doesn't modify JSON-safe objects."""
        result = _sanitize_for_json({
            "str": "text",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        })
        assert result == {
            "str": "text",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }


class TestContentToSafeDict:
    """Tests for _content_to_safe_dict() function."""

    def test_converts_content_with_text_part(self):
        """_content_to_safe_dict converts content with text part."""
        part = MockPart(text="Hello world")
        content = MockContent(role="user", parts=[part])

        result = _content_to_safe_dict(content)

        assert result["role"] == "user"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["text"] == "Hello world"

    def test_converts_content_with_function_call(self):
        """_content_to_safe_dict converts function_call part."""
        mock_fc = Mock()
        mock_fc.name = "test_func"
        mock_fc.args = {"arg1": "value1"}
        part = MockPart(function_call=mock_fc)
        content = MockContent(role="model", parts=[part])

        result = _content_to_safe_dict(content)

        assert result["role"] == "model"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["function_call"]["name"] == "test_func"
        assert result["parts"][0]["function_call"]["args"] == {"arg1": "value1"}

    def test_converts_content_with_function_response(self):
        """_content_to_safe_dict converts function_response part."""
        mock_fr = Mock()
        mock_fr.name = "test_func"
        mock_fr.response = {"result": "success"}
        part = MockPart(function_response=mock_fr)
        content = MockContent(role="tool", parts=[part])

        result = _content_to_safe_dict(content)

        assert result["role"] == "tool"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["function_response"]["name"] == "test_func"
        assert result["parts"][0]["function_response"]["response"] == {"result": "success"}

    def test_converts_content_with_inline_data(self):
        """_content_to_safe_dict converts inline_data part."""
        mock_blob = Mock()
        mock_blob.mime_type = "image/png"
        mock_blob.data = "base64data"
        part = MockPart(inline_data=mock_blob)
        content = MockContent(role="user", parts=[part])

        result = _content_to_safe_dict(content)

        assert result["role"] == "user"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["inline_data"]["mime_type"] == "image/png"
        assert result["parts"][0]["inline_data"]["data"] == "base64data"

    def test_preserves_thought_signature(self):
        """_content_to_safe_dict preserves thought_signature."""
        part = MockPart(text="Hello", thought_signature="sig123")
        content = MockContent(role="model", parts=[part])

        result = _content_to_safe_dict(content)

        assert result["parts"][0]["thought_signature"] == "sig123"

    def test_skips_empty_parts(self):
        """_content_to_safe_dict skips empty parts to avoid validation errors."""
        # Create a part with no actual content
        part = MockPart()  # No text, function_call, function_response, or inline_data
        content = MockContent(role="user", parts=[part])

        result = _content_to_safe_dict(content)

        assert result["role"] == "user"
        assert len(result["parts"]) == 0

    def test_handles_multiple_parts(self):
        """_content_to_safe_dict handles content with multiple parts."""
        part1 = MockPart(text="Hello")
        mock_fc = Mock()
        mock_fc.name = "test_func"
        mock_fc.args = {}
        part2 = MockPart(function_call=mock_fc)
        content = MockContent(role="user", parts=[part1, part2])

        result = _content_to_safe_dict(content)

        assert result["role"] == "user"
        assert len(result["parts"]) == 2
        assert result["parts"][0]["text"] == "Hello"
        assert result["parts"][1]["function_call"]["name"] == "test_func"


class TestExtractCleanText:
    """Tests for extract_clean_text() function."""

    def test_extracts_text_from_single_candidate(self):
        """extract_clean_text extracts text from single candidate."""
        part = MockPart(text="Hello world")
        content = MockContent(role="model", parts=[part])
        candidate = MockCandidate(content=content)
        response = MockResponse(candidates=[candidate])

        result = extract_clean_text(response)

        assert result == "Hello world"

    def test_extracts_text_from_multiple_candidates(self):
        """extract_clean_text combines text from multiple candidates."""
        part1 = MockPart(text="Hello")
        part2 = MockPart(text="world")
        content1 = MockContent(role="model", parts=[part1])
        content2 = MockContent(role="model", parts=[part2])
        candidate1 = MockCandidate(content=content1)
        candidate2 = MockCandidate(content=content2)
        response = MockResponse(candidates=[candidate1, candidate2])

        result = extract_clean_text(response)

        assert result == "Hello world"

    def test_filters_out_thought_parts(self):
        """extract_clean_text skips parts marked as thoughts."""
        thought_part = MockPart(text="Thinking...", thought=True)
        text_part = MockPart(text="Actual response", thought=False)
        content = MockContent(role="model", parts=[thought_part, text_part])
        candidate = MockCandidate(content=content)
        response = MockResponse(candidates=[candidate])

        result = extract_clean_text(response)

        assert result == "Actual response"
        assert "Thinking" not in result

    def test_returns_empty_string_for_no_candidates(self):
        """extract_clean_text returns empty string when no candidates."""
        response = MockResponse(candidates=[])

        result = extract_clean_text(response)

        assert result == ""

    def test_returns_empty_string_for_no_parts(self):
        """extract_clean_text returns empty string when content has no parts."""
        content = MockContent(role="model", parts=[])
        candidate = MockCandidate(content=content)
        response = MockResponse(candidates=[candidate])

        result = extract_clean_text(response)

        assert result == ""

    def test_returns_empty_string_on_exception(self):
        """extract_clean_text returns empty string on exception."""
        # Create a malformed response that will cause an exception
        response = Mock(spec=[])

        result = extract_clean_text(response)

        assert result == ""


class TestGeminiChatSessionInit:
    """Tests for GeminiChatSession.__init__()."""

    def test_initializes_with_required_params(self, mock_gemini_factory):
        """__init__ initializes with system instruction and factory."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        assert session._system_instruction == "You are helpful"
        assert session._factory == mock_gemini_factory

    def test_initializes_with_empty_history(self, mock_gemini_factory):
        """__init__ creates empty history deque."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        assert len(session.history) == 0
        assert isinstance(session.history, deque)

    def test_uses_default_max_history(self, mock_gemini_factory):
        """__init__ uses default max_history of 50."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        assert session.history.maxlen == 50

    def test_uses_custom_max_history(self, mock_gemini_factory):
        """__init__ respects custom max_history."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory,
            max_history=100
        )

        assert session.history.maxlen == 100


class TestGeminiChatSessionGetApiHistory:
    """Tests for GeminiChatSession._get_api_history()."""

    def test_returns_empty_list_for_empty_history(self, mock_gemini_factory):
        """_get_api_history returns empty list when history is empty."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        result = session._get_api_history()

        assert result == []

    def test_converts_chat_message_with_parts(self, mock_gemini_factory):
        """_get_api_history converts ChatMessage with parts."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        msg = ChatMessage(
            role="user",
            content="Hello",
            parts=[{"text": "Hello"}]
        )
        session.history.append(msg)

        result = session._get_api_history()

        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].parts) == 1
        assert result[0].parts[0].text == "Hello"

    def test_converts_simple_message_with_content(self, mock_gemini_factory):
        """_get_api_history converts SimpleMessage with content."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        # Create a message with content but no parts
        msg = ChatMessage(role="user", content="Hello")
        session.history.append(msg)

        result = session._get_api_history()

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].parts[0].text == "Hello"

    def test_includes_image_parts(self, mock_gemini_factory):
        """_get_api_history includes inline data for images."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        # JPEG magic bytes
        jpeg_data = b"\xff\xd8\xff\xe0"
        msg = ChatMessage(
            role="user",
            content="Look at this",
            images=[jpeg_data]
        )
        session.history.append(msg)

        result = session._get_api_history()

        assert len(result) == 1
        assert result[0].role == "user"
        # Should have text part + inline data part
        assert len(result[0].parts) >= 1
        # Check text part exists
        text_parts = [p for p in result[0].parts if hasattr(p, 'text') and p.text]
        assert len(text_parts) >= 1
        # Check inline data part exists (has actual inline_data with mime_type)
        inline_parts = [p for p in result[0].parts if hasattr(p, 'inline_data') and p.inline_data is not None]
        assert len(inline_parts) == 1


class TestGeminiChatSessionSendMessage:
    """Tests for GeminiChatSession.send_message()."""

    def test_sends_text_message(self, mock_gemini_factory):
        """send_message sends text message to model."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        user_msg, model_msg, response = session.send_message(
            user_message="Hello",
            author_id=123
        )

        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert user_msg.author_id == 123
        assert model_msg.role == "model"
        assert model_msg.content == "Test response"
        assert response is not None

    def test_sends_with_optional_params(self, mock_gemini_factory):
        """send_message includes optional parameters."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        user_msg, model_msg, response = session.send_message(
            user_message="Hello",
            author_id=123,
            author_name="TestUser",
            message_ids=["msg1", "msg2"]
        )

        assert user_msg.author_name == "TestUser"
        assert user_msg.message_ids == ["msg1", "msg2"]

    def test_sends_with_images(self, mock_gemini_factory):
        """send_message includes images in current parts."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        jpeg_data = b"\xff\xd8\xff\xe0"
        user_msg, model_msg, response = session.send_message(
            user_message="Look at this",
            author_id=123,
            images=[jpeg_data]
        )

        assert user_msg.images == [jpeg_data]
        # Verify factory was called
        assert len(mock_gemini_factory.generate_content_calls) == 1

    def test_sends_with_tools(self, mock_gemini_factory):
        """send_message passes tools to factory."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        tools = [{"name": "test_tool"}]
        user_msg, model_msg, response = session.send_message(
            user_message="Hello",
            author_id=123,
            tools=tools
        )

        # Verify tools were passed
        call = mock_gemini_factory.generate_content_calls[0]
        assert call[1] == tools

    def test_includes_history_in_contents(self, mock_gemini_factory):
        """send_message includes history in API contents."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        # Add to history
        msg = ChatMessage(role="user", content="Previous", parts=[{"text": "Previous"}])
        session.history.append(msg)

        user_msg, model_msg, response = session.send_message(
            user_message="Hello",
            author_id=123
        )

        # Verify contents include history + current
        call = mock_gemini_factory.generate_content_calls[0]
        contents = call[0]
        # Should have previous message + current user message
        assert len(contents) >= 2


class TestGeminiChatSessionSendMessageStream:
    """Tests for GeminiChatSession.send_message_stream()."""

    @pytest.mark.asyncio
    async def test_returns_user_message_and_stream(self, mock_gemini_factory):
        """send_message_stream returns user message and async iterator."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        user_msg, stream = await session.send_message_stream(
            user_message="Hello",
            author_id=123
        )

        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert user_msg.author_id == 123
        assert stream is not None

    @pytest.mark.asyncio
    async def test_stream_is_iterable(self, mock_gemini_factory):
        """send_message_stream returns iterable async stream."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        user_msg, stream = await session.send_message_stream(
            user_message="Hello",
            author_id=123
        )

        # Should be able to iterate
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_includes_images(self, mock_gemini_factory):
        """send_message_stream includes images in current parts."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        jpeg_data = b"\xff\xd8\xff\xe0"
        user_msg, stream = await session.send_message_stream(
            user_message="Look at this",
            author_id=123,
            images=[jpeg_data]
        )

        assert user_msg.images == [jpeg_data]


class TestGeminiChatSessionSendToolResults:
    """Tests for GeminiChatSession.send_tool_results()."""

    def test_sends_tool_results(self, mock_gemini_factory):
        """send_tool_results sends tool execution results."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        # Create mock response with function_call
        mock_fc = Mock()
        mock_fc.name = "test_func"
        mock_fc.args = {}
        fc_part = MockPart(function_call=mock_fc)

        mock_content = MockContent(role="model", parts=[fc_part])
        mock_candidate = MockCandidate(content=mock_content)
        mock_response = MockResponse(candidates=[mock_candidate])

        # Tool results
        results = [{"name": "test_func", "result": "success"}]

        model_msg, response = session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        assert model_msg.role == "model"
        assert model_msg.content == "Test response"
        assert response is not None

    def test_handles_multiple_tool_rounds(self, mock_gemini_factory):
        """send_tool_results handles multiple rounds of tool calls."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        # Create mock responses
        mock_fc1 = Mock()
        mock_fc1.name = "func1"
        mock_fc1.args = {}
        fc_part1 = MockPart(function_call=mock_fc1)
        mock_content1 = MockContent(role="model", parts=[fc_part1])
        mock_candidate1 = MockCandidate(content=mock_content1)
        mock_response1 = MockResponse(candidates=[mock_candidate1])

        mock_fc2 = Mock()
        mock_fc2.name = "func2"
        mock_fc2.args = {}
        fc_part2 = MockPart(function_call=mock_fc2)
        mock_content2 = MockContent(role="model", parts=[fc_part2])
        mock_candidate2 = MockCandidate(content=mock_content2)
        mock_response2 = MockResponse(candidates=[mock_candidate2])

        results1 = [{"name": "func1", "result": "success1"}]
        results2 = [{"name": "func2", "result": "success2"}]

        model_msg, response = session.send_tool_results(
            tool_rounds=[
                (mock_response1, results1),
                (mock_response2, results2)
            ]
        )

        assert model_msg.role == "model"
        # Should have called generate_content
        assert len(mock_gemini_factory.generate_content_calls) == 1

    def test_handles_tool_errors(self, mock_gemini_factory):
        """send_tool_results handles tool results with errors."""
        session = GeminiChatSession(
            system_instruction="You are helpful",
            factory=mock_gemini_factory
        )

        # Create mock response
        mock_fc = Mock()
        mock_fc.name = "test_func"
        mock_fc.args = {}
        fc_part = MockPart(function_call=mock_fc)
        mock_content = MockContent(role="model", parts=[fc_part])
        mock_candidate = MockCandidate(content=mock_content)
        mock_response = MockResponse(candidates=[mock_candidate])

        # Tool results with error
        results = [{"name": "test_func", "error": "Failed"}]

        model_msg, response = session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        assert model_msg.role == "model"
        # Verify the call was made
        assert len(mock_gemini_factory.generate_content_calls) == 1


# =============================================================================
# Tests for openai_session.py Dataclasses
# =============================================================================

class TestFakeDelta:
    """Tests for FakeDelta dataclass."""

    def test_creates_with_content(self):
        """FakeDelta creates with content."""
        delta = FakeDelta(content="Hello")
        assert delta.content == "Hello"

    def test_default_content_is_none(self):
        """FakeDelta content defaults to None."""
        delta = FakeDelta()
        assert delta.content is None


class TestFakeChoice:
    """Tests for FakeChoice dataclass."""

    def test_creates_with_delta(self):
        """FakeChoice creates with delta."""
        delta = FakeDelta(content="Hello")
        choice = FakeChoice(delta=delta)
        assert choice.delta == delta

    def test_default_delta_is_empty(self):
        """FakeChoice delta defaults to FakeDelta()."""
        choice = FakeChoice()
        assert isinstance(choice.delta, FakeDelta)
        assert choice.delta.content is None


class TestFakeChunk:
    """Tests for FakeChunk dataclass."""

    def test_creates_with_choices(self):
        """FakeChunk creates with choices."""
        delta = FakeDelta(content="Hello")
        choice = FakeChoice(delta=delta)
        chunk = FakeChunk(choices=[choice])
        assert len(chunk.choices) == 1
        assert chunk.choices[0].delta.content == "Hello"

    def test_default_choices_is_empty_list(self):
        """FakeChunk choices defaults to empty list."""
        chunk = FakeChunk()
        assert chunk.choices == []


class TestOpenAIMessage:
    """Tests for OpenAIMessage dataclass."""

    def test_creates_with_required_fields(self):
        """OpenAIMessage creates with role and content."""
        msg = OpenAIMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_allows_optional_name(self):
        """OpenAIMessage allows optional name."""
        msg = OpenAIMessage(role="user", content="Hello", name="TestUser")
        assert msg.name == "TestUser"

    def test_allows_tool_call_id(self):
        """OpenAIMessage allows tool_call_id."""
        msg = OpenAIMessage(
            role="tool",
            content="Result",
            tool_call_id="call_123"
        )
        assert msg.tool_call_id == "call_123"

    def test_allows_tool_calls(self):
        """OpenAIMessage allows tool_calls."""
        tool_calls = [{"id": "call_123", "function": {"name": "test"}}]
        msg = OpenAIMessage(
            role="assistant",
            content=None,
            tool_calls=tool_calls
        )
        assert msg.tool_calls == tool_calls


# =============================================================================
# Tests for ResponsesAPIStreamAdapter
# =============================================================================

class TestResponsesAPIStreamAdapter:
    """Tests for ResponsesAPIStreamAdapter class."""

    def test_creates_with_raw_stream(self):
        """ResponsesAPIStreamAdapter creates with raw stream."""
        raw_stream = iter([])
        adapter = ResponsesAPIStreamAdapter(raw_stream)
        assert adapter._raw_stream == raw_stream

    def test_aiter_returns_self(self):
        """__aiter__ returns self."""
        raw_stream = iter([])
        adapter = ResponsesAPIStreamAdapter(raw_stream)
        assert adapter.__aiter__() is adapter

    def test_iter_returns_self(self):
        """__iter__ returns self as iterator."""
        raw_stream = iter([])
        adapter = ResponsesAPIStreamAdapter(raw_stream)
        result = adapter.__iter__()
        assert result == adapter

    def test_process_event_with_output_text_delta(self):
        """_process_event extracts text from response.output_text.delta events."""
        adapter = ResponsesAPIStreamAdapter(iter([]))

        event = Mock()
        event.type = "response.output_text.delta"
        event.delta = "Hello"

        result = adapter._process_event(event)

        assert isinstance(result, FakeChunk)
        assert result.choices[0].delta.content == "Hello"

    def test_process_event_with_response_in_progress(self):
        """_process_event handles response.in_progress with output_text."""
        adapter = ResponsesAPIStreamAdapter(iter([]))

        event = Mock()
        event.type = "response.in_progress"
        mock_response = Mock()
        mock_response.output_text = "Partial text"
        event.response = mock_response

        result = adapter._process_event(event)

        assert isinstance(result, FakeChunk)
        assert result.choices[0].delta.content == "Partial text"

    def test_process_event_with_response_created(self):
        """_process_event handles response.created with output_text."""
        adapter = ResponsesAPIStreamAdapter(iter([]))

        event = Mock()
        event.type = "response.created"
        mock_response = Mock()
        mock_response.output_text = "Initial text"
        event.response = mock_response

        result = adapter._process_event(event)

        assert isinstance(result, FakeChunk)
        assert result.choices[0].delta.content == "Initial text"

    def test_process_event_returns_none_for_non_text_events(self):
        """_process_event returns None for events without text."""
        adapter = ResponsesAPIStreamAdapter(iter([]))

        event = Mock()
        event.type = "response.done"
        event.response = Mock()
        event.response.output_text = None

        result = adapter._process_event(event)

        assert result is None

    @pytest.mark.asyncio
    async def test_anext_yields_chunks(self):
        """__anext__ yields chunks from stream events."""
        events = [
            Mock(type="response.output_text.delta", delta="Hello "),
            Mock(type="response.output_text.delta", delta="world"),
        ]
        adapter = ResponsesAPIStreamAdapter(iter(events))

        chunks = []
        async for chunk in adapter:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello "
        assert chunks[1].choices[0].delta.content == "world"

    @pytest.mark.asyncio
    async def test_anext_raises_stop_async_iteration(self):
        """__anext__ raises StopAsyncIteration when exhausted."""
        adapter = ResponsesAPIStreamAdapter(iter([]))

        with pytest.raises(StopAsyncIteration):
            await adapter.__anext__()

    def test_next_yields_chunks(self):
        """__next__ yields chunks from stream events (sync)."""
        events = [
            Mock(type="response.output_text.delta", delta="Hello"),
        ]
        adapter = ResponsesAPIStreamAdapter(iter(events))
        # Initialize the iterator by calling __iter__
        it = iter(adapter)

        result = next(it)
        assert isinstance(result, FakeChunk)
        assert result.choices[0].delta.content == "Hello"

    def test_next_raises_stop_iteration(self):
        """__next__ raises StopIteration when exhausted."""
        adapter = ResponsesAPIStreamAdapter(iter([]))
        it = iter(adapter)

        with pytest.raises(StopIteration):
            next(it)

    def test_close_calls_raw_stream_close(self):
        """close calls close on raw stream if available."""
        raw_stream = Mock()
        adapter = ResponsesAPIStreamAdapter(raw_stream)
        adapter.close()
        raw_stream.close.assert_called_once()

    def test_close_does_not_raise_without_close_method(self):
        """close does not raise when raw stream has no close method."""
        raw_stream = iter([])
        adapter = ResponsesAPIStreamAdapter(raw_stream)
        adapter.close()  # Should not raise


# =============================================================================
# Tests for encode_image_to_url
# =============================================================================

class TestEncodeImageToUrl:
    """Tests for encode_image_to_url() function."""

    def test_encodes_jpeg_to_base64_url(self):
        """encode_image_to_url converts JPEG to base64 data URL."""
        # JPEG magic bytes
        jpeg_data = b"\xff\xd8\xff\xe0"
        result = encode_image_to_url(jpeg_data)

        assert result["type"] == "image_url"
        assert "url" in result["image_url"]
        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_encodes_png_to_base64_url(self):
        """encode_image_to_url converts PNG to base64 data URL."""
        # PNG magic bytes
        png_data = b"\x89PNG\r\n\x1a\n"
        result = encode_image_to_url(png_data)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_base64_encoding_is_correct(self):
        """encode_image_to_url produces correct base64 encoding."""
        test_data = b"test"
        result = encode_image_to_url(test_data)

        expected_b64 = base64.b64encode(test_data).decode("utf-8")
        assert result["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


# =============================================================================
# Tests for BaseOpenAISession
# =============================================================================

class TestBaseOpenAISessionInit:
    """Tests for BaseOpenAISession.__init__()."""

    def test_initializes_with_required_params(self, mock_openai_client):
        """__init__ initializes with required parameters."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        assert session._client == mock_openai_client
        assert session._model_name == "gpt-4"
        assert session._system_instruction == "You are helpful"
        assert session._temperature == 0.7
        assert session._top_p == 0.9
        assert session._max_messages == 50

    def test_initializes_with_optional_service_tier(self, mock_openai_client):
        """__init__ accepts service_tier parameter."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
            service_tier="auto"
        )

        assert session._service_tier == "auto"

    def test_initializes_with_custom_text_extractor(self, mock_openai_client):
        """__init__ accepts custom text extractor."""
        extractor = lambda x: "custom"
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
            text_extractor=extractor
        )

        assert session._text_extractor == extractor


class TestBaseOpenAISessionSupportsTopP:
    """Tests for BaseOpenAISession._supports_top_p()."""

    def test_returns_true_for_standard_models(self, mock_openai_client):
        """_supports_top_p returns True for standard models."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        assert session._supports_top_p() is True

    def test_returns_false_for_gpt5(self, mock_openai_client):
        """_supports_top_p returns False for gpt-5."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-5",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        assert session._supports_top_p() is False

    def test_returns_false_for_o1(self, mock_openai_client):
        """_supports_top_p returns False for o1."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="o1",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        assert session._supports_top_p() is False

    def test_case_insensitive_model_check(self, mock_openai_client):
        """_supports_top_p is case insensitive."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="GPT-5-PREVIEW",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        assert session._supports_top_p() is False


class TestBaseOpenAISessionHistory:
    """Tests for BaseOpenAISession.history property."""

    def test_history_returns_list(self, mock_openai_client):
        """history property returns list of messages."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        result = session.history
        assert isinstance(result, list)
        assert result == []

    def test_history_setter_replaces_history(self, mock_openai_client):
        """history setter replaces existing history."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        new_history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi")
        ]
        session.history = new_history

        assert len(session.history) == 2
        assert session.history[0].content == "Hello"


class TestBaseOpenAISessionAppendHistory:
    """Tests for BaseOpenAISession._append_history()."""

    def test_appends_message_with_content(self, mock_openai_client):
        """_append_history adds message to history."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        session._append_history("user", "Hello")

        assert len(session.history) == 1
        assert session.history[0].role == "user"
        assert session.history[0].content == "Hello"

    def test_skips_empty_content(self, mock_openai_client):
        """_append_history does not add empty content."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        session._append_history("user", "")

        assert len(session.history) == 0

    def test_includes_author_metadata(self, mock_openai_client):
        """_append_history includes author_id and author_name."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        session._append_history(
            "user",
            "Hello",
            author_id=123,
            author_name="TestUser"
        )

        msg = session.history[0]
        assert msg.author_id == 123
        assert msg.author_name == "TestUser"


class TestBaseOpenAISessionCreateUserMessage:
    """Tests for BaseOpenAISession._create_user_message()."""

    def test_creates_user_message(self, mock_openai_client):
        """_create_user_message creates ChatMessage."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        msg = session._create_user_message("Hello", 123)

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.author_id == 123

    def test_includes_optional_params(self, mock_openai_client):
        """_create_user_message includes optional parameters."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        msg = session._create_user_message(
            "Hello",
            123,
            author_name="TestUser",
            message_ids=["msg1"]
        )

        assert msg.author_name == "TestUser"
        assert msg.message_ids == ["msg1"]

    def test_includes_images(self, mock_openai_client):
        """_create_user_message includes images."""
        session = BaseOpenAISession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        jpeg_data = b"\xff\xd8\xff\xe0"
        msg = session._create_user_message(
            "Hello",
            123,
            images=[jpeg_data]
        )

        assert msg.images == [jpeg_data]


# =============================================================================
# Tests for ChatCompletionSession
# =============================================================================

class TestChatCompletionSessionSendMessage:
    """Tests for ChatCompletionSession.send_message()."""

    def test_sends_message_and_returns_responses(self, mock_openai_client):
        """send_message sends message and returns user/model messages."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        user_msg, model_msg, response = session.send_message(
            user_message="Hello",
            author_id=123
        )

        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert model_msg.role == "assistant"
        assert model_msg.content == "Test response"
        assert response is not None

    def test_includes_system_message(self, mock_openai_client):
        """send_message includes system instruction."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        session.send_message(user_message="Hello", author_id=123)

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    def test_includes_tools(self, mock_openai_client):
        """send_message passes tools to API."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        tools = [{"type": "function", "function": {"name": "test"}}]
        session.send_message(
            user_message="Hello",
            author_id=123,
            tools=tools
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["tools"] == tools


class TestChatCompletionSessionSendMessageStream:
    """Tests for ChatCompletionSession.send_message_stream()."""

    def test_returns_stream_and_user_message(self, mock_openai_client):
        """send_message_stream returns stream and user message."""
        mock_stream = Mock()
        mock_openai_client.chat.completions.create.return_value = mock_stream

        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        stream, user_msg = session.send_message_stream(
            user_message="Hello",
            author_id=123
        )

        assert stream == mock_stream
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"


class TestChatCompletionSessionBuildSystemMessage:
    """Tests for ChatCompletionSession._build_system_message()."""

    def test_returns_system_message_when_set(self, mock_openai_client):
        """_build_system_message returns system message when instruction set."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        result = session._build_system_message()

        assert result == [{"role": "system", "content": "You are helpful"}]

    def test_returns_empty_list_when_no_instruction(self, mock_openai_client):
        """_build_system_message returns empty list when no instruction."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        result = session._build_system_message()

        assert result == []


class TestChatCompletionSessionConvertHistoryToApiFormat:
    """Tests for ChatCompletionSession._convert_history_to_api_format()."""

    def test_converts_text_only_messages(self, mock_openai_client):
        """_convert_history_to_api_format converts text messages."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        session._history.append(ChatMessage(role="user", content="Hello"))
        session._history.append(ChatMessage(role="assistant", content="Hi"))

        result = session._convert_history_to_api_format()

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi"

    def test_converts_messages_with_images(self, mock_openai_client):
        """_convert_history_to_api_format handles images in content blocks."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        jpeg_data = b"\xff\xd8\xff\xe0"
        session._history.append(
            ChatMessage(role="user", content="Look", images=[jpeg_data])
        )

        result = session._convert_history_to_api_format()

        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Should be content blocks with text + image_url
        assert "content" in result[0]
        assert isinstance(result[0]["content"], list)


class TestChatCompletionSessionBuildUserContent:
    """Tests for ChatCompletionSession._build_user_content()."""

    def test_builds_text_only_content(self, mock_openai_client):
        """_build_user_content builds text-only content."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        result = session._build_user_content("Hello", None)

        assert result == {"role": "user", "content": "Hello"}

    def test_builds_content_with_images(self, mock_openai_client):
        """_build_user_content builds content blocks with images."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        jpeg_data = b"\xff\xd8\xff\xe0"
        result = session._build_user_content("Look", [jpeg_data])

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        # Should have text + image_url blocks
        assert any(block.get("type") == "text" for block in result["content"])
        assert any(block.get("type") == "image_url" for block in result["content"])

    def test_builds_image_only_content(self, mock_openai_client):
        """_build_user_content builds image-only content."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        jpeg_data = b"\xff\xd8\xff\xe0"
        result = session._build_user_content("", [jpeg_data])

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "image_url"


class TestChatCompletionSessionExtractResponseContent:
    """Tests for ChatCompletionSession._extract_response_content()."""

    def test_extracts_content_from_response(self, mock_openai_client):
        """_extract_response_content extracts text from response."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"

        result = session._extract_response_content(mock_response)

        assert result == "Test response"

    def test_falls_back_to_text_extractor(self, mock_openai_client):
        """_extract_response_content uses text extractor when no content."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
            text_extractor=lambda x: "extracted"
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None

        result = session._extract_response_content(mock_response)

        assert result == "extracted"


class TestChatCompletionSessionSendToolResults:
    """Tests for ChatCompletionSession.send_tool_results()."""

    def test_sends_tool_results_to_api(self, mock_openai_client):
        """send_tool_results sends results to API."""
        session = ChatCompletionSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        # Mock response with tool_calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Let me check"
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_func"
        mock_tc.function.arguments = '{"arg": "value"}'
        mock_response.choices[0].message.tool_calls = [mock_tc]

        results = [{"name": "test_func", "result": "success"}]

        model_msg, response = session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        assert model_msg.role == "assistant"
        assert model_msg.content == "Test response"
        assert response is not None


# =============================================================================
# Tests for ResponseSession
# =============================================================================

class TestResponseSessionBuildInputPayload:
    """Tests for ResponseSession._build_input_payload()."""

    def test_builds_payload_with_system_instruction(self, mock_openai_client):
        """_build_input_payload includes system instruction."""
        session = ResponseSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        result = session._build_input_payload()

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"][0]["type"] == "input_text"

    def test_builds_payload_with_history(self, mock_openai_client):
        """_build_input_payload includes history messages."""
        session = ResponseSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        session._history.append(ChatMessage(role="user", content="Hello"))
        session._history.append(ChatMessage(role="assistant", content="Hi"))

        result = session._build_input_payload()

        # System + 2 history messages
        assert len(result) == 3
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["type"] == "input_text"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"][0]["type"] == "output_text"

    def test_user_messages_use_input_text(self, mock_openai_client):
        """_build_input_payload uses input_text for user messages."""
        session = ResponseSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        session._history.append(ChatMessage(role="user", content="Hello"))

        result = session._build_input_payload()

        assert result[1]["content"][0]["type"] == "input_text"

    def test_assistant_messages_use_output_text(self, mock_openai_client):
        """_build_input_payload uses output_text for assistant messages."""
        session = ResponseSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        session._history.append(ChatMessage(role="assistant", content="Hi"))

        result = session._build_input_payload()

        assert result[1]["content"][0]["type"] == "output_text"


class TestResponseSessionSendMessage:
    """Tests for ResponseSession.send_message()."""

    def test_sends_message_and_returns_responses(self, mock_openai_client):
        """send_message sends message and returns user/model messages."""
        # ResponseSession uses text_extractor to get content from Responses API
        session = ResponseSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
            text_extractor=lambda x: getattr(x, 'output_text', '')
        )

        user_msg, model_msg, response = session.send_message(
            user_message="Hello",
            author_id=123
        )

        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert model_msg.role == "assistant"
        assert model_msg.content == "Test response"
        assert response is not None


class TestResponseSessionSendMessageStream:
    """Tests for ResponseSession.send_message_stream()."""

    def test_returns_adapted_stream_and_user_message(self, mock_openai_client):
        """send_message_stream returns adapted stream and user message."""
        mock_stream = iter([])
        mock_openai_client.responses.create.return_value = mock_stream

        session = ResponseSession(
            client=mock_openai_client,
            model_name="gpt-4",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50
        )

        stream, user_msg = session.send_message_stream(
            user_message="Hello",
            author_id=123
        )

        assert isinstance(stream, ResponsesAPIStreamAdapter)
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"

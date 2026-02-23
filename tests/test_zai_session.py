"""Comprehensive tests for persbot.services.session_wrappers.zai_session.

Tests for:
- encode_image_to_zai_format() helper function
- ZAIChatSession class initialization
- ZAIChatSession.send_message()
- ZAIChatSession.send_message_stream()
- ZAIChatSession._build_messages()
- ZAIChatSession.send_tool_results()
"""

import base64
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import pytest

from persbot.services.session_wrappers.zai_session import (
    encode_image_to_zai_format,
    ZAIChatSession,
)
from persbot.services.base import ChatMessage


# =============================================================================
# Mock Helpers and Fixtures
# =============================================================================

def default_text_extractor(response):
    """Default text extractor that extracts content from response."""
    if response and response.choices and len(response.choices) > 0:
        msg = response.choices[0].message
        if hasattr(msg, 'content') and msg.content:
            return msg.content
    return ""


@pytest.fixture
def mock_zai_client():
    """Create a mock Z.AI OpenAI-compatible client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()

    # Mock chat completions
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message = Mock()
    mock_completion.choices[0].message.content = "Test response"
    mock_completion.choices[0].message.tool_calls = None
    client.chat.completions.create = Mock(return_value=mock_completion)

    return client


@pytest.fixture
def zai_session(mock_zai_client):
    """Create a ZAIChatSession instance for testing."""
    return ZAIChatSession(
        client=mock_zai_client,
        model_name="zai-model",
        system_instruction="You are helpful",
        temperature=0.7,
        top_p=0.9,
        max_messages=50,
        text_extractor=default_text_extractor,
    )


# =============================================================================
# Tests for encode_image_to_zai_format()
# =============================================================================

class TestEncodeImageToZaiFormat:
    """Tests for encode_image_to_zai_format() function."""

    def test_encodes_jpeg_to_base64_url(self):
        """encode_image_to_zai_format converts JPEG to base64 data URL."""
        # JPEG magic bytes
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        result = encode_image_to_zai_format(jpeg_data)

        assert result["type"] == "image_url"
        assert "url" in result["image_url"]
        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_encodes_png_to_base64_url(self):
        """encode_image_to_zai_format converts PNG to base64 data URL."""
        # PNG magic bytes
        png_data = b"\x89PNG\r\n\x1a\n"
        result = encode_image_to_zai_format(png_data)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_base64_encoding_is_correct(self):
        """encode_image_to_zai_format produces correct base64 encoding."""
        test_data = b"test"
        result = encode_image_to_zai_format(test_data)

        expected_b64 = base64.b64encode(test_data).decode("utf-8")
        assert result["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"

    def test_returns_correct_structure(self):
        """encode_image_to_zai_format returns correct dict structure."""
        test_data = b"\xff\xd8\xff\xe0"
        result = encode_image_to_zai_format(test_data)

        assert isinstance(result, dict)
        assert "type" in result
        assert "image_url" in result
        assert isinstance(result["image_url"], dict)
        assert "url" in result["image_url"]


# =============================================================================
# Tests for ZAIChatSession.__init__()
# =============================================================================

class TestZAIChatSessionInit:
    """Tests for ZAIChatSession.__init__()."""

    def test_initializes_with_required_params(self, mock_zai_client):
        """__init__ initializes with all required parameters."""
        session = ZAIChatSession(
            client=mock_zai_client,
            model_name="zai-model",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
        )

        assert session._client == mock_zai_client
        assert session._model_name == "zai-model"
        assert session._system_instruction == "You are helpful"
        assert session._temperature == 0.7
        assert session._top_p == 0.9
        assert session._max_messages == 50

    def test_initializes_with_custom_text_extractor(self, mock_zai_client):
        """__init__ accepts custom text extractor."""
        extractor = lambda x: "custom"
        session = ZAIChatSession(
            client=mock_zai_client,
            model_name="zai-model",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
            text_extractor=extractor,
        )

        assert session._text_extractor == extractor

    def test_service_tier_is_none(self, mock_zai_client):
        """__init__ sets service_tier to None for Z.AI compatibility."""
        session = ZAIChatSession(
            client=mock_zai_client,
            model_name="zai-model",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
        )

        assert session._service_tier is None

    def test_initializes_with_empty_history(self, mock_zai_client):
        """__init__ creates empty history list."""
        session = ZAIChatSession(
            client=mock_zai_client,
            model_name="zai-model",
            system_instruction="You are helpful",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
        )

        assert len(session.history) == 0
        assert isinstance(session.history, list)


# =============================================================================
# Tests for ZAIChatSession._build_messages()
# =============================================================================

class TestZAIChatSessionBuildMessages:
    """Tests for ZAIChatSession._build_messages()."""

    def test_builds_messages_with_system_instruction(self, zai_session):
        """_build_messages includes system instruction when set."""
        messages = zai_session._build_messages("Hello", None)

        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    def test_builds_messages_without_system_instruction(self, mock_zai_client):
        """_build_messages works without system instruction."""
        session = ZAIChatSession(
            client=mock_zai_client,
            model_name="zai-model",
            system_instruction="",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
            text_extractor=default_text_extractor,
        )

        messages = session._build_messages("Hello", None)

        assert len(messages) == 1  # Only user message
        assert messages[0]["role"] == "user"

    def test_builds_messages_with_history(self, zai_session):
        """_build_messages includes history messages."""
        # Add some history
        zai_session._history.append(ChatMessage(role="user", content="Previous message"))
        zai_session._history.append(ChatMessage(role="assistant", content="Previous response"))

        messages = zai_session._build_messages("Hello", None)

        assert len(messages) == 4  # system + 2 history + current user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Previous message"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Previous response"
        assert messages[3]["role"] == "user"

    def test_builds_messages_with_images_in_history(self, zai_session):
        """_build_messages handles images in history messages."""
        jpeg_data = b"\xff\xd8\xff\xe0"
        zai_session._history.append(
            ChatMessage(role="user", content="Look at this", images=[jpeg_data])
        )

        messages = zai_session._build_messages("Hello", None)

        assert len(messages) == 3  # system + history user + current user
        # History message should have content blocks
        assert isinstance(messages[1]["content"], list)
        assert messages[1]["content"][0]["type"] == "text"
        assert messages[1]["content"][0]["text"] == "Look at this"
        assert messages[1]["content"][1]["type"] == "image_url"

    def test_builds_current_message_with_text_only(self, zai_session):
        """_build_messages builds current user message with text."""
        messages = zai_session._build_messages("Hello world", None)

        # Last message should be user message
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][0]["text"] == "Hello world"

    def test_builds_current_message_with_images(self, zai_session):
        """_build_messages builds current user message with images."""
        jpeg_data = b"\xff\xd8\xff\xe0"
        messages = zai_session._build_messages("Look", [jpeg_data])

        # Last message should be user message
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        # Should have text + image_url
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][0]["text"] == "Look"
        assert user_msg["content"][1]["type"] == "image_url"

    def test_builds_current_message_with_images_only(self, zai_session):
        """_build_messages builds current user message with only images."""
        jpeg_data = b"\xff\xd8\xff\xe0"
        messages = zai_session._build_messages("", [jpeg_data])

        # Last message should be user message
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 1
        assert user_msg["content"][0]["type"] == "image_url"

    def test_builds_current_message_with_multiple_images(self, zai_session):
        """_build_messages builds current user message with multiple images."""
        jpeg_data = b"\xff\xd8\xff\xe0"
        png_data = b"\x89PNG\r\n\x1a\n"
        messages = zai_session._build_messages("Look at these", [jpeg_data, png_data])

        # Last message should be user message
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        # Should have text + 2 images
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"
        assert user_msg["content"][2]["type"] == "image_url"


# =============================================================================
# Tests for ZAIChatSession.send_message()
# =============================================================================

class TestZAIChatSessionSendMessage:
    """Tests for ZAIChatSession.send_message()."""

    def test_sends_message_and_returns_responses(self, zai_session):
        """send_message sends message and returns user/model messages."""
        user_msg, model_msg, response = zai_session.send_message(
            user_message="Hello",
            author_id=123
        )

        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert user_msg.author_id == 123
        assert model_msg.role == "assistant"
        assert model_msg.content == "Test response"
        assert response is not None

    def test_sends_with_optional_params(self, zai_session):
        """send_message includes optional parameters."""
        user_msg, model_msg, response = zai_session.send_message(
            user_message="Hello",
            author_id=123,
            author_name="TestUser",
            message_ids=["msg1", "msg2"]
        )

        assert user_msg.author_name == "TestUser"
        assert user_msg.message_ids == ["msg1", "msg2"]

    def test_sends_with_images(self, zai_session, mock_zai_client):
        """send_message includes images in the request."""
        jpeg_data = b"\xff\xd8\xff\xe0"
        user_msg, model_msg, response = zai_session.send_message(
            user_message="Look at this",
            author_id=123,
            images=[jpeg_data]
        )

        assert user_msg.images == [jpeg_data]

        # Verify API was called with correct parameters
        call_args = mock_zai_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "zai-model"
        assert "temperature" in call_args[1]
        assert "top_p" in call_args[1]

    def test_sends_with_tools(self, zai_session, mock_zai_client):
        """send_message passes tools to API."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        user_msg, model_msg, response = zai_session.send_message(
            user_message="Hello",
            author_id=123,
            tools=tools
        )

        # Verify tools were passed
        call_args = mock_zai_client.chat.completions.create.call_args
        assert call_args[1]["tools"] == tools

    def test_includes_history_in_messages(self, zai_session, mock_zai_client):
        """send_message includes history in API messages."""
        # Add to history
        zai_session._history.append(ChatMessage(role="user", content="Previous"))

        user_msg, model_msg, response = zai_session.send_message(
            user_message="Hello",
            author_id=123
        )

        # Verify messages include history
        call_args = mock_zai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        # Should have system + previous + current
        assert len(messages) >= 3

    def test_passes_temperature_and_top_p(self, zai_session, mock_zai_client):
        """send_message passes temperature and top_p to API."""
        user_msg, model_msg, response = zai_session.send_message(
            user_message="Hello",
            author_id=123
        )

        call_args = mock_zai_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["top_p"] == 0.9

    def test_does_not_pass_service_tier(self, zai_session, mock_zai_client):
        """send_message does not pass service_tier (Z.AI doesn't support it)."""
        user_msg, model_msg, response = zai_session.send_message(
            user_message="Hello",
            author_id=123
        )

        call_args = mock_zai_client.chat.completions.create.call_args
        assert "service_tier" not in call_args[1]


# =============================================================================
# Tests for ZAIChatSession.send_message_stream()
# =============================================================================

class TestZAIChatSessionSendMessageStream:
    """Tests for ZAIChatSession.send_message_stream()."""

    def test_returns_stream_and_user_message(self, zai_session, mock_zai_client):
        """send_message_stream returns stream and user message."""
        mock_stream = Mock()
        mock_zai_client.chat.completions.create.return_value = mock_stream

        stream, user_msg = zai_session.send_message_stream(
            user_message="Hello",
            author_id=123
        )

        assert stream == mock_stream
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert user_msg.author_id == 123

    def test_includes_optional_params(self, zai_session, mock_zai_client):
        """send_message_stream includes optional parameters."""
        mock_stream = Mock()
        mock_zai_client.chat.completions.create.return_value = mock_stream

        stream, user_msg = zai_session.send_message_stream(
            user_message="Hello",
            author_id=123,
            author_name="TestUser",
            message_ids=["msg1"]
        )

        assert user_msg.author_name == "TestUser"
        assert user_msg.message_ids == ["msg1"]

    def test_includes_images(self, zai_session, mock_zai_client):
        """send_message_stream includes images in the request."""
        mock_stream = Mock()
        mock_zai_client.chat.completions.create.return_value = mock_stream

        jpeg_data = b"\xff\xd8\xff\xe0"
        stream, user_msg = zai_session.send_message_stream(
            user_message="Look at this",
            author_id=123,
            images=[jpeg_data]
        )

        assert user_msg.images == [jpeg_data]

    def test_passes_stream_parameter(self, zai_session, mock_zai_client):
        """send_message_stream passes stream=True to API."""
        mock_stream = Mock()
        mock_zai_client.chat.completions.create.return_value = mock_stream

        stream, user_msg = zai_session.send_message_stream(
            user_message="Hello",
            author_id=123
        )

        call_args = mock_zai_client.chat.completions.create.call_args
        assert call_args[1]["stream"] is True

    def test_passes_tools_when_streaming(self, zai_session, mock_zai_client):
        """send_message_stream passes tools to API."""
        mock_stream = Mock()
        mock_zai_client.chat.completions.create.return_value = mock_stream

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        stream, user_msg = zai_session.send_message_stream(
            user_message="Hello",
            author_id=123,
            tools=tools
        )

        call_args = mock_zai_client.chat.completions.create.call_args
        assert call_args[1]["tools"] == tools


# =============================================================================
# Tests for ZAIChatSession.send_tool_results()
# =============================================================================

class TestZAIChatSessionSendToolResults:
    """Tests for ZAIChatSession.send_tool_results()."""

    def test_sends_tool_results(self, zai_session, mock_zai_client):
        """send_tool_results sends tool execution results."""
        # Create mock response with tool_calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Let me check"
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_func"
        mock_tc.function.arguments = '{"arg": "value"}'
        mock_response.choices[0].message.tool_calls = [mock_tc]

        # Tool results
        results = [{"name": "test_func", "result": "success"}]

        model_msg, response = zai_session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        assert model_msg.role == "assistant"
        assert model_msg.content == "Test response"
        assert response is not None

    def test_handles_multiple_tool_rounds(self, zai_session, mock_zai_client):
        """send_tool_results handles multiple rounds of tool calls."""
        # Create mock responses
        mock_response1 = Mock()
        mock_response1.choices = [Mock()]
        mock_response1.choices[0].message = Mock()
        mock_response1.choices[0].message.content = "Calling func1"
        mock_tc1 = Mock()
        mock_tc1.id = "call_1"
        mock_tc1.function.name = "func1"
        mock_tc1.function.arguments = "{}"
        mock_response1.choices[0].message.tool_calls = [mock_tc1]

        mock_response2 = Mock()
        mock_response2.choices = [Mock()]
        mock_response2.choices[0].message = Mock()
        mock_response2.choices[0].message.content = "Calling func2"
        mock_tc2 = Mock()
        mock_tc2.id = "call_2"
        mock_tc2.function.name = "func2"
        mock_tc2.function.arguments = "{}"
        mock_response2.choices[0].message.tool_calls = [mock_tc2]

        results1 = [{"name": "func1", "result": "success1"}]
        results2 = [{"name": "func2", "result": "success2"}]

        model_msg, response = zai_session.send_tool_results(
            tool_rounds=[
                (mock_response1, results1),
                (mock_response2, results2)
            ]
        )

        assert model_msg.role == "assistant"
        # Verify API was called
        assert mock_zai_client.chat.completions.create.called

    def test_removes_last_assistant_message(self, zai_session, mock_zai_client):
        """send_tool_results removes last assistant message from history."""
        # Add history with assistant message
        zai_session._history.append(ChatMessage(role="user", content="Hello"))
        zai_session._history.append(ChatMessage(role="assistant", content="Hi"))

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Let me check"
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_func"
        mock_tc.function.arguments = "{}"
        mock_response.choices[0].message.tool_calls = [mock_tc]

        results = [{"name": "test_func", "result": "success"}]

        model_msg, response = zai_session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        # Verify the call was made
        call_args = mock_zai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        # Should not have assistant message at the end (before tool calls)
        # The structure should be: system, user, assistant with tool_calls, tool results
        assert any(msg.get("role") == "assistant" and "tool_calls" in msg for msg in messages)

    def test_includes_system_instruction(self, zai_session, mock_zai_client):
        """send_tool_results includes system instruction."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Checking"
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_func"
        mock_tc.function.arguments = "{}"
        mock_response.choices[0].message.tool_calls = [mock_tc]

        results = [{"name": "test_func", "result": "success"}]

        model_msg, response = zai_session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        call_args = mock_zai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    def test_handles_tool_errors(self, zai_session, mock_zai_client):
        """send_tool_results handles tool results with errors."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Checking"
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_func"
        mock_tc.function.arguments = "{}"
        mock_response.choices[0].message.tool_calls = [mock_tc]

        # Tool results with error
        results = [{"name": "test_func", "error": "Failed"}]

        model_msg, response = zai_session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        assert model_msg.role == "assistant"
        # Verify the call was made
        assert mock_zai_client.chat.completions.create.called

    def test_passes_tools_parameter(self, zai_session, mock_zai_client):
        """send_tool_results passes tools to API."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Checking"
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_func"
        mock_tc.function.arguments = "{}"
        mock_response.choices[0].message.tool_calls = [mock_tc]

        results = [{"name": "test_func", "result": "success"}]
        tools = [{"type": "function", "function": {"name": "another_tool"}}]

        model_msg, response = zai_session.send_tool_results(
            tool_rounds=[(mock_response, results)],
            tools=tools
        )

        call_args = mock_zai_client.chat.completions.create.call_args
        assert call_args[1]["tools"] == tools

    def test_handles_images_in_history(self, zai_session, mock_zai_client):
        """send_tool_results handles images in history messages."""
        jpeg_data = b"\xff\xd8\xff\xe0"
        zai_session._history.append(
            ChatMessage(role="user", content="Look", images=[jpeg_data])
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Checking"
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_func"
        mock_tc.function.arguments = "{}"
        mock_response.choices[0].message.tool_calls = [mock_tc]

        results = [{"name": "test_func", "result": "success"}]

        model_msg, response = zai_session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        # Verify the call was made
        call_args = mock_zai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        # Should have message with content blocks (text + image_url)
        assert any(
            isinstance(msg.get("content"), list)
            for msg in messages
        )

    def test_formats_tool_calls_correctly(self, zai_session, mock_zai_client):
        """send_tool_results formats tool_calls correctly for Z.AI."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Checking"
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_func"
        mock_tc.function.arguments = '{"arg1": "value1"}'
        mock_response.choices[0].message.tool_calls = [mock_tc]

        results = [{"name": "test_func", "result": "success"}]

        model_msg, response = zai_session.send_tool_results(
            tool_rounds=[(mock_response, results)]
        )

        call_args = mock_zai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        # Find assistant message with tool_calls
        assistant_msg = next(
            (msg for msg in messages if msg.get("role") == "assistant" and "tool_calls" in msg),
            None
        )
        assert assistant_msg is not None
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["id"] == "call_123"
        assert assistant_msg["tool_calls"][0]["type"] == "function"
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "test_func"
        assert assistant_msg["tool_calls"][0]["function"]["arguments"] == '{"arg1": "value1"}'

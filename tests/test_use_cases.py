"""Tests for Use Cases."""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from persbot.use_cases.chat_use_case import ChatUseCase, ChatRequest, ChatResponse, StreamChunk
from persbot.use_cases.image_use_case import (
    ImageUseCase,
    VisionRequest,
    VisionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from persbot.use_cases.prompt_use_case import (
    PromptUseCase,
    PromptGenerationRequest,
    PromptGenerationResponse,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
)


class TestChatUseCase:
    """Test ChatUseCase business logic."""

    @pytest.mark.asyncio
    async def test_generate_chat_response_success(
        self, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test successful chat response generation."""
        # Set up mock session
        mock_chat_session = Mock()
        mock_chat_session.model_alias = "gemini-2.5-flash"
        mock_chat_session.session_key = "channel:111222333"
        mock_session_manager.get_or_create.return_value = (mock_chat_session, "channel:111222333")

        # Set up mock LLM response
        mock_llm_service.generate_chat_response.return_value = ("Hello! How can I help?", None)

        # Mock model usage service
        mock_llm_service.model_usage_service = Mock()
        mock_llm_service.model_usage_service.check_and_increment_usage = AsyncMock(
            return_value=(True, "gemini-2.5-flash", None)
        )
        mock_llm_service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-2.5-flash"
        mock_llm_service.model_usage_service.get_api_model_name = Mock(
            return_value="gemini-2.5-flash"
        )

        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        request = ChatRequest(
            user_message="Hello bot",
            discord_message=mock_message,
            resolution=Mock(session_key="channel:111222333", cleaned_message="Hello bot"),
        )

        response = await use_case.generate_chat_response(request)

        assert response is not None
        assert response.text == "Hello! How can I help?"
        assert response.session_key == "channel:111222333"
        assert len(response.images) == 0
        mock_llm_service.generate_chat_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_chat_response_stream_success(
        self, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test successful streaming chat response."""
        # Set up mock session
        mock_chat_session = Mock()
        mock_chat_session.model_alias = "gemini-2.5-flash"
        mock_chat_session.session_key = "channel:111222333"
        mock_session_manager.get_or_create.return_value = (mock_chat_session, "channel:111222333")

        # Set up mock LLM response
        mock_llm_service.generate_chat_response.return_value = ("Streaming response!", None)

        # Mock model usage service
        mock_llm_service.model_usage_service = Mock()
        mock_llm_service.model_usage_service.check_and_increment_usage = AsyncMock(
            return_value=(True, "gemini-2.5-flash", None)
        )
        mock_llm_service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-2.5-flash"

        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        request = ChatRequest(
            user_message="Stream this",
            discord_message=mock_message,
            resolution=Mock(session_key="channel:111222333", cleaned_message="Stream this"),
        )

        chunks = []
        async for chunk in use_case.generate_chat_response_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].text == "Streaming response!"
        assert chunks[0].is_final is True

    @pytest.mark.asyncio
    async def test_generate_chat_response_with_tool_calls(
        self, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test chat response with function calls."""
        # Set up mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled.return_value = True
        mock_tool_manager.get_enabled_tools.return_value = {"get_time": Mock()}
        mock_tool_manager.execute_tools = AsyncMock(return_value=[{"result": "17:00"}])

        # Set up mock session
        mock_chat_session = Mock()
        mock_chat_session.model_alias = "gemini-2.5-flash"
        mock_chat_session.session_key = "channel:111222333"
        mock_session_manager.get_or_create.return_value = (mock_chat_session, "channel:111222333")

        # Set up mock LLM response with tool call
        mock_response_obj = Mock()
        mock_llm_service.generate_chat_response.return_value = ("Thinking...", mock_response_obj)
        mock_llm_service.get_active_backend.return_value = Mock()
        mock_llm_service.extract_function_calls_from_response.return_value = [
            {"name": "get_time", "arguments": {}}
        ]
        mock_llm_service.send_tool_results = AsyncMock(return_value=("The time is 5 PM", Mock()))

        # Mock model usage service
        mock_llm_service.model_usage_service = Mock()
        mock_llm_service.model_usage_service.check_and_increment_usage = AsyncMock(
            return_value=(True, "gemini-2.5-flash", None)
        )
        mock_llm_service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-2.5-flash"

        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        request = ChatRequest(
            user_message="What time is it?",
            discord_message=mock_message,
            resolution=Mock(session_key="channel:111222333", cleaned_message="What time is it?"),
        )

        response = await use_case.generate_chat_response(request)

        assert response is not None
        assert "5 PM" in response.text
        assert response.tool_rounds > 0

    @pytest.mark.asyncio
    async def test_generate_chat_response_usage_limit_exceeded(
        self, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test chat response when usage limit is exceeded."""
        # Set up mock session
        mock_chat_session = Mock()
        mock_chat_session.model_alias = "gemini-2.5-flash"
        mock_chat_session.session_key = "channel:111222333"
        mock_session_manager.get_or_create.return_value = (mock_chat_session, "channel:111222333")

        # Mock usage limit exceeded
        mock_llm_service.model_usage_service = Mock()
        mock_llm_service.model_usage_service.check_and_increment_usage = AsyncMock(
            return_value=(False, "gemini-2.5-flash", "일일 사용량 초과")
        )
        mock_llm_service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-2.5-flash"

        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        request = ChatRequest(
            user_message="Hello",
            discord_message=mock_message,
            resolution=Mock(session_key="channel:111222333", cleaned_message="Hello"),
        )

        response = await use_case.generate_chat_response(request)

        assert response is not None
        assert response.notification is not None
        assert "초과" in response.notification
        assert "일일 사용량" in response.text

    @pytest.mark.asyncio
    async def test_regenerate_last_response_no_session(
        self, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test regeneration when session has no history."""
        mock_session_manager.undo_last_exchanges.return_value = []

        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        response = await use_case.regenerate_last_response(111222333, "channel:111222333")

        assert response is None

    @pytest.mark.asyncio
    async def test_get_or_create_session_new_session(
        self, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test creating a new session."""
        mock_session_manager.get_or_create.return_value = (
            Mock(session_key="channel:111222333"),
            "channel:111222333",
        )

        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        chat_session, session_key = await use_case._get_or_create_session(
            mock_message, Mock(session_key="channel:111222333", cleaned_message="test")
        )

        assert session_key == "channel:111222333"
        mock_session_manager.get_or_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_enabled_tools_with_tool_manager(
        self, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test getting enabled tools when tool manager is enabled."""
        mock_tool_manager = Mock()
        mock_tool_manager.is_enabled.return_value = True
        mock_tool_manager.get_enabled_tools.return_value = {"tool1": Mock(), "tool2": Mock()}

        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=mock_tool_manager,
        )

        tools = use_case._get_enabled_tools()

        assert tools is not None
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_get_enabled_tools_without_tool_manager(
        self, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test getting enabled tools when tool manager is disabled."""
        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
            tool_manager=None,
        )

        tools = use_case._get_enabled_tools()

        assert tools is None

    @pytest.mark.asyncio
    async def test_get_primary_message_single(
        self, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test getting primary message from single message."""
        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        result = use_case._get_primary_message(mock_message)
        assert result is mock_message

    @pytest.mark.asyncio
    async def test_get_primary_message_list(
        self, mock_app_config, mock_llm_service, mock_session_manager
    ):
        """Test getting primary message from list."""
        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        msg1 = Mock(id="1")
        msg2 = Mock(id="2")
        result = use_case._get_primary_message([msg1, msg2])
        assert result is msg1

    @pytest.mark.asyncio
    async def test_generate_chat_response_cancellation(
        self, mock_app_config, mock_llm_service, mock_session_manager, mock_message
    ):
        """Test cancellation before generation."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        use_case = ChatUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            session_manager=mock_session_manager,
        )

        request = ChatRequest(
            user_message="Hello",
            discord_message=mock_message,
            resolution=Mock(session_key="channel:111222333", cleaned_message="Hello"),
            cancel_event=cancel_event,
        )

        with pytest.raises(Exception):  # CancellationException
            await use_case.generate_chat_response(request)


class TestImageUseCase:
    """Test ImageUseCase business logic."""

    @pytest.mark.asyncio
    async def test_understand_images_success(
        self, mock_app_config, mock_llm_service, mock_usage_service, mock_message
    ):
        """Test successful image understanding."""
        # Mock vision backend
        mock_vision_backend = Mock()
        mock_vision_backend.generate_chat_response = AsyncMock(
            return_value=("This is an image of a cat", None)
        )
        mock_vision_session = Mock()
        mock_vision_backend.create_assistant_model.return_value = mock_vision_session

        mock_llm_service.get_backend_for_model.return_value = mock_vision_backend
        mock_llm_service.model_usage_service = Mock()
        mock_llm_service.model_usage_service.get_api_model_name = Mock(return_value="GLM 4.6V")

        # Mock usage service
        mock_usage_service.check_can_upload.return_value = True
        mock_usage_service.record_upload = AsyncMock()

        use_case = ImageUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_usage_service,
        )

        request = VisionRequest(
            images=[b"fake_image_data"],
            user_message="What's in this image?",
            discord_message=mock_message,
        )

        response = await use_case.understand_images(request)

        assert response is not None
        assert response.success is True
        assert "cat" in response.description

    @pytest.mark.asyncio
    async def test_understand_images_limit_exceeded(
        self, mock_app_config, mock_llm_service, mock_usage_service, mock_message
    ):
        """Test image understanding when limit exceeded."""
        # Disable permission checking so that user is not admin
        mock_app_config.no_check_permission = False

        # Mock vision backend for limit check
        mock_vision_backend = Mock()
        mock_vision_backend.generate_chat_response = AsyncMock(
            return_value=("This is an image of a cat", None)
        )
        mock_vision_backend.create_assistant_model.return_value = Mock()
        mock_llm_service.get_backend_for_model.return_value = mock_vision_backend
        mock_llm_service.model_usage_service = Mock()
        mock_llm_service.model_usage_service.get_api_model_name = Mock(return_value="GLM 4.6V")

        # Set up limit exceeded
        mock_usage_service.check_can_upload.return_value = False

        use_case = ImageUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_usage_service,
        )

        request = VisionRequest(
            images=[b"fake_image_data"],
            user_message="What's in this image?",
            discord_message=mock_message,
        )

        response = await use_case.understand_images(request)

        assert response is not None
        assert response.success is False
        assert response.error is not None
        assert "3개" in response.error

    @pytest.mark.asyncio
    async def test_generate_image_success(
        self, mock_app_config, mock_llm_service, mock_usage_service
    ):
        """Test successful image generation."""
        mock_image_service = Mock()
        mock_image_service.generate_image = AsyncMock(return_value=b"fake_generated_image")

        use_case = ImageUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_usage_service,
            image_service=mock_image_service,
        )

        request = ImageGenerationRequest(
            prompt="A beautiful sunset",
            channel_id=111222333,
        )

        response = await use_case.generate_image(request)

        assert response is not None
        assert response.success is True
        assert response.image_data == b"fake_generated_image"

    @pytest.mark.asyncio
    async def test_validate_prompt_empty(self, mock_app_config, mock_llm_service):
        """Test validation of empty prompt."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        is_valid, error = await use_case.validate_prompt("")

        assert is_valid is False
        assert error is not None
        assert "비어있습니다" in error


class TestPromptUseCase:
    """Test PromptUseCase business logic."""

    @pytest.mark.asyncio
    async def test_generate_prompt_from_concept_success(self, mock_app_config, mock_llm_service):
        """Test successful prompt generation from concept."""
        mock_backend = Mock()
        mock_response = Mock()
        mock_response.text = "You are a helpful AI assistant with expertise in Python programming."

        mock_backend.create_assistant_model.return_value = Mock()
        mock_backend.execute_with_retry = AsyncMock(return_value=mock_response)

        mock_llm_service.summarizer_backend = mock_backend

        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        request = PromptGenerationRequest(
            concept="A Python expert assistant",
        )

        response = await use_case.generate_prompt_from_concept(request)

        assert response is not None
        assert response.success is True
        assert "Python" in response.system_prompt
        assert "expert" in response.system_prompt

    @pytest.mark.asyncio
    async def test_generate_prompt_with_answers(self, mock_app_config, mock_llm_service):
        """Test prompt generation with Q&A incorporated."""
        mock_backend = Mock()
        mock_response = Mock()
        mock_response.text = "You are a helpful assistant specialized in data science."

        mock_backend.create_assistant_model.return_value = Mock()
        mock_backend.execute_with_retry = AsyncMock(return_value=mock_response)

        mock_llm_service.summarizer_backend = mock_backend

        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        request = PromptGenerationRequest(
            concept="Data science assistant",
            questions_and_answers="Q: What tools should I use? A: Python, pandas, scikit-learn",
        )

        response = await use_case.generate_prompt_from_concept(request)

        assert response is not None
        assert response.success is True

    @pytest.mark.asyncio
    async def test_generate_questions_success(self, mock_app_config, mock_llm_service):
        """Test successful question generation."""
        mock_backend = Mock()
        mock_response = Mock()
        mock_response.text = (
            '{"questions": [{"question": "What is your expertise?", "sample_answer": "AI and ML"}]}'
        )

        mock_backend.create_assistant_model.return_value = Mock()
        mock_backend.generate_content = AsyncMock(return_value=mock_response)
        mock_backend._summary_model_name = "test-model"

        mock_llm_service.summarizer_backend = mock_backend

        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        request = QuestionGenerationRequest(
            concept="A technical writing assistant",
            max_questions=5,
        )

        # Patch isinstance to use Gemini path (simpler than mocking OpenAI)
        with patch("persbot.use_cases.prompt_use_case.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: False  # Force Gemini path

            response = await use_case.generate_questions(request)

        assert response is not None
        assert response.success is True
        assert len(response.questions) > 0

    @pytest.mark.asyncio
    async def test_validate_prompt_valid(self, mock_app_config, mock_llm_service):
        """Test validation of a valid prompt."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        prompt = """
        You are a helpful AI assistant.
        Your role is to provide accurate information and help users.
        You should always be polite and respectful.
        """

        is_valid, error = await use_case.validate_prompt(prompt)

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_prompt_too_short(self, mock_app_config, mock_llm_service):
        """Test validation of a too-short prompt."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        prompt = "Short"

        is_valid, error = await use_case.validate_prompt(prompt)

        assert is_valid is False
        assert error is not None
        assert "짧습니다" in error

    @pytest.mark.asyncio
    async def test_validate_prompt_missing_role(self, mock_app_config, mock_llm_service):
        """Test validation of prompt without role description."""
        mock_app_config.no_check_permission = False

        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Long enough but missing role keywords (at least 100 characters)
        prompt = (
            "This is a very long prompt without any specific role description or instructions about what the assistant should do in various situations to help users with their tasks and provide helpful information. "
            * 2
        )

        is_valid, error = await use_case.validate_prompt(prompt)

        assert is_valid is False
        assert error is not None
        assert "역할" in error

    @pytest.mark.asyncio
    async def test_validate_prompt_empty(self, mock_app_config, mock_llm_service):
        """Test validation of empty prompt."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        is_valid, error = await use_case.validate_prompt("")

        assert is_valid is False
        assert error is not None
        assert "비어있습니다" in error

    @pytest.mark.asyncio
    async def test_optimize_prompt_for_cache(self, mock_app_config, mock_llm_service):
        """Test prompt optimization for caching (currently returns as-is)."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        prompt = "This is a test prompt for optimization."

        optimized = await use_case.optimize_prompt_for_cache(prompt, target_tokens=32768)

        # Currently returns prompt as-is (TODO: implement actual optimization)
        assert optimized == prompt

    @pytest.mark.asyncio
    async def test_extract_response_text_from_text_response(
        self, mock_app_config, mock_llm_service
    ):
        """Test extracting text from string response."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        result = use_case._extract_response_text("Direct text response")

        assert result == "Direct text response"

    @pytest.mark.asyncio
    async def test_extract_response_text_from_object_with_text(
        self, mock_app_config, mock_llm_service
    ):
        """Test extracting text from response object with text attribute."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        mock_response = Mock()
        mock_response.text = "Text from object"

        result = use_case._extract_response_text(mock_response)

        assert result == "Text from object"

    @pytest.mark.asyncio
    async def test_extract_response_text_from_object_with_choices(
        self, mock_app_config, mock_llm_service
    ):
        """Test extracting text from response object with choices."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock response with choices but without text attribute
        class MockResponse:
            def __init__(self):
                self.choices = [Mock(message=Mock(content="Content from choices"))]

        mock_response = MockResponse()

        # Mock hasattr check to return False for 'text' attribute
        with patch.object(
            use_case, "_extract_response_text", wraps=use_case._extract_response_text
        ):
            # Directly test the internal logic
            result = use_case._extract_response_text(mock_response)

        # Should return the content since choices is present and has content
        # Note: Since we can't control hasattr, let's test with a different approach
        mock_resp = Mock(spec=[])  # Create mock without text attribute
        mock_resp.choices = [Mock(message=Mock(content="Content from choices"))]

        result = use_case._extract_response_text(mock_resp)
        assert result == "Content from choices"

    @pytest.mark.asyncio
    async def test_parse_questions_response_valid_json(self, mock_app_config, mock_llm_service):
        """Test parsing valid JSON questions response."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        response_text = '{"questions": [{"question": "Q1?", "sample_answer": "A1"}]}'

        result = use_case._parse_questions_response(response_text)

        assert len(result) == 1
        assert result[0]["question"] == "Q1?"

    @pytest.mark.asyncio
    async def test_parse_questions_response_markdown_json(self, mock_app_config, mock_llm_service):
        """Test parsing JSON wrapped in markdown code blocks."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        response_text = """```json
{
  "questions": [
    {"question": "Q1?", "sample_answer": "A1"},
    {"question": "Q2?", "sample_answer": "A2"}
  ]
}
```"""

        result = use_case._parse_questions_response(response_text)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_parse_questions_response_invalid_json(self, mock_app_config, mock_llm_service):
        """Test parsing invalid JSON returns empty list."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        result = use_case._parse_questions_response("not valid json")

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_prompt_error_handling(self, mock_app_config, mock_llm_service):
        """Test error handling in prompt generation."""
        mock_backend = Mock()
        mock_backend.create_assistant_model.side_effect = Exception("API Error")
        mock_llm_service.summarizer_backend = mock_backend

        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        request = PromptGenerationRequest(
            concept="Test concept",
        )

        response = await use_case.generate_prompt_from_concept(request)

        assert response.success is False
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_generate_questions_error_handling(self, mock_app_config, mock_llm_service):
        """Test error handling in question generation."""
        mock_backend = Mock()
        mock_backend.execute_with_retry.side_effect = Exception("API Error")
        mock_llm_service.summarizer_backend = mock_backend

        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        request = QuestionGenerationRequest(
            concept="Test concept",
        )

        response = await use_case.generate_questions(request)

        assert response.success is False
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_extract_response_text_from_text_response(
        self, mock_app_config, mock_llm_service
    ):
        """Test extracting text from string response."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        result = use_case._extract_response_text("Direct text response")

        assert result == "Direct text response"

    @pytest.mark.asyncio
    async def test_extract_response_text_from_object_with_text(
        self, mock_app_config, mock_llm_service
    ):
        """Test extracting text from response object with text attribute."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        mock_response = Mock()
        mock_response.text = "Text from object"

        result = use_case._extract_response_text(mock_response)

        assert result == "Text from object"

    @pytest.mark.asyncio
    async def test_extract_response_text_from_object_with_choices(
        self, mock_app_config, mock_llm_service
    ):
        """Test extracting text from response object with choices."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        # Mock response with choices but without text attribute
        class MockResponse:
            def __init__(self):
                self.choices = [Mock(message=Mock(content="Content from choices"))]

        mock_response = MockResponse()

        # Mock hasattr check to return False for 'text' attribute
        with patch.object(
            use_case, "_extract_response_text", wraps=use_case._extract_response_text
        ):
            # Directly test the internal logic
            result = use_case._extract_response_text(mock_response)

        # Should return the content since choices is present and has content
        # Note: Since we can't control hasattr, let's test with a different approach
        mock_resp = Mock(spec=[])  # Create mock without text attribute
        mock_resp.choices = [Mock(message=Mock(content="Content from choices"))]

        result = use_case._extract_response_text(mock_resp)
        assert result == "Content from choices"

    @pytest.mark.asyncio
    async def test_parse_questions_response_valid_json(self, mock_app_config, mock_llm_service):
        """Test parsing valid JSON questions response."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        response_text = '{"questions": [{"question": "Q1?", "sample_answer": "A1"}]}'

        result = use_case._parse_questions_response(response_text)

        assert len(result) == 1
        assert result[0]["question"] == "Q1?"

    @pytest.mark.asyncio
    async def test_parse_questions_response_markdown_json(self, mock_app_config, mock_llm_service):
        """Test parsing JSON wrapped in markdown code blocks."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        response_text = """```json
{
  "questions": [
    {"question": "Q1?", "sample_answer": "A1"},
    {"question": "Q2?", "sample_answer": "A2"}
  ]
}
```"""

        result = use_case._parse_questions_response(response_text)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_parse_questions_response_invalid_json(self, mock_app_config, mock_llm_service):
        """Test parsing invalid JSON returns empty list."""
        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        result = use_case._parse_questions_response("not valid json")

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_prompt_error_handling(self, mock_app_config, mock_llm_service):
        """Test error handling in prompt generation."""
        mock_backend = Mock()
        mock_backend.create_assistant_model.side_effect = Exception("API Error")
        mock_llm_service.summarizer_backend = mock_backend

        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        request = PromptGenerationRequest(
            concept="Test concept",
        )

        response = await use_case.generate_prompt_from_concept(request)

        assert response.success is False
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_generate_questions_error_handling(self, mock_app_config, mock_llm_service):
        """Test error handling in question generation."""
        mock_backend = Mock()
        mock_backend.execute_with_retry.side_effect = Exception("API Error")
        mock_llm_service.summarizer_backend = mock_backend

        use_case = PromptUseCase(
            config=mock_app_config,
            llm_service=mock_llm_service,
        )

        request = QuestionGenerationRequest(
            concept="Test concept",
        )

        response = await use_case.generate_questions(request)

        assert response.success is False
        assert response.error is not None

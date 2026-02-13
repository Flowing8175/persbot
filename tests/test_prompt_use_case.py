"""Feature tests for prompt use case module.

Tests focus on behavior using mocking:
- PromptGenerationRequest: request for generating prompt
- PromptGenerationResponse: response from generation
- QuestionGenerationRequest: request for generating questions
- QuestionGenerationResponse: response from question generation
- Question: a clarifying question
- PromptUseCase: use case for prompt operations
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


class TestPromptGenerationRequest:
    """Tests for PromptGenerationRequest dataclass."""

    def test_request_exists(self):
        """PromptGenerationRequest class exists."""
        from persbot.use_cases.prompt_use_case import PromptGenerationRequest
        assert PromptGenerationRequest is not None

    def test_request_has_required_fields(self):
        """PromptGenerationRequest has required fields."""
        from persbot.use_cases.prompt_use_case import PromptGenerationRequest

        request = PromptGenerationRequest(concept="A helpful assistant")

        assert request.concept == "A helpful assistant"

    def test_request_defaults(self):
        """PromptGenerationRequest has correct defaults."""
        from persbot.use_cases.prompt_use_case import PromptGenerationRequest

        request = PromptGenerationRequest(concept="test")

        assert request.questions_and_answers is None
        assert request.use_cache is False


class TestPromptGenerationResponse:
    """Tests for PromptGenerationResponse dataclass."""

    def test_response_exists(self):
        """PromptGenerationResponse class exists."""
        from persbot.use_cases.prompt_use_case import PromptGenerationResponse
        assert PromptGenerationResponse is not None

    def test_response_has_required_fields(self):
        """PromptGenerationResponse has required fields."""
        from persbot.use_cases.prompt_use_case import PromptGenerationResponse

        response = PromptGenerationResponse(
            system_prompt="You are helpful",
            success=True,
        )

        assert response.system_prompt == "You are helpful"
        assert response.success is True

    def test_response_defaults(self):
        """PromptGenerationResponse has correct defaults."""
        from persbot.use_cases.prompt_use_case import PromptGenerationResponse

        response = PromptGenerationResponse(
            system_prompt="",
            success=False,
            error="Something went wrong",
        )

        assert response.error == "Something went wrong"


class TestQuestionGenerationRequest:
    """Tests for QuestionGenerationRequest dataclass."""

    def test_request_exists(self):
        """QuestionGenerationRequest class exists."""
        from persbot.use_cases.prompt_use_case import QuestionGenerationRequest
        assert QuestionGenerationRequest is not None

    def test_request_has_required_fields(self):
        """QuestionGenerationRequest has required fields."""
        from persbot.use_cases.prompt_use_case import QuestionGenerationRequest

        request = QuestionGenerationRequest(concept="A friendly bot")

        assert request.concept == "A friendly bot"

    def test_request_defaults(self):
        """QuestionGenerationRequest has correct defaults."""
        from persbot.use_cases.prompt_use_case import QuestionGenerationRequest

        request = QuestionGenerationRequest(concept="test")

        assert request.max_questions == 5


class TestQuestionGenerationResponse:
    """Tests for QuestionGenerationResponse dataclass."""

    def test_response_exists(self):
        """QuestionGenerationResponse class exists."""
        from persbot.use_cases.prompt_use_case import QuestionGenerationResponse
        assert QuestionGenerationResponse is not None

    def test_response_has_required_fields(self):
        """QuestionGenerationResponse has required fields."""
        from persbot.use_cases.prompt_use_case import QuestionGenerationResponse

        response = QuestionGenerationResponse(
            questions=[{"question": "Q1", "sample_answer": "A1"}],
            success=True,
        )

        assert len(response.questions) == 1
        assert response.success is True


class TestQuestion:
    """Tests for Question dataclass."""

    def test_question_exists(self):
        """Question class exists."""
        from persbot.use_cases.prompt_use_case import Question
        assert Question is not None

    def test_question_has_required_fields(self):
        """Question has required fields."""
        from persbot.use_cases.prompt_use_case import Question

        question = Question(
            question="What is your purpose?",
            sample_answer="To help users",
        )

        assert question.question == "What is your purpose?"
        assert question.sample_answer == "To help users"


class TestPromptUseCase:
    """Tests for PromptUseCase class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.temperature = 1.0
        config.top_p = 1.0
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        service.summarizer_backend = Mock()
        service.summarizer_backend._summary_model_name = "test-model"
        return service

    def test_prompt_use_case_exists(self):
        """PromptUseCase class exists."""
        from persbot.use_cases.prompt_use_case import PromptUseCase
        assert PromptUseCase is not None

    def test_creates_with_dependencies(self, mock_config, mock_llm_service):
        """PromptUseCase creates with dependencies."""
        from persbot.use_cases.prompt_use_case import PromptUseCase

        use_case = PromptUseCase(mock_config, mock_llm_service)

        assert use_case.config == mock_config
        assert use_case.llm_service == mock_llm_service


class TestPromptUseCaseValidatePrompt:
    """Tests for PromptUseCase.validate_prompt method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return Mock()

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        service.summarizer_backend = Mock()
        return service

    @pytest.mark.asyncio
    async def test_validate_prompt_rejects_empty(self, mock_config, mock_llm_service):
        """validate_prompt rejects empty prompt."""
        from persbot.use_cases.prompt_use_case import PromptUseCase

        use_case = PromptUseCase(mock_config, mock_llm_service)
        is_valid, error = await use_case.validate_prompt("")

        assert is_valid is False
        assert error is not None

    @pytest.mark.asyncio
    async def test_validate_prompt_rejects_whitespace(self, mock_config, mock_llm_service):
        """validate_prompt rejects whitespace-only prompt."""
        from persbot.use_cases.prompt_use_case import PromptUseCase

        use_case = PromptUseCase(mock_config, mock_llm_service)
        is_valid, error = await use_case.validate_prompt("   ")

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_prompt_rejects_short(self, mock_config, mock_llm_service):
        """validate_prompt rejects short prompt."""
        from persbot.use_cases.prompt_use_case import PromptUseCase

        use_case = PromptUseCase(mock_config, mock_llm_service)
        is_valid, error = await use_case.validate_prompt("Short")

        assert is_valid is False
        assert "짧" in error  # "short" in Korean

    @pytest.mark.asyncio
    async def test_validate_prompt_accepts_valid(self, mock_config, mock_llm_service):
        """validate_prompt accepts valid prompt."""
        from persbot.use_cases.prompt_use_case import PromptUseCase

        use_case = PromptUseCase(mock_config, mock_llm_service)
        valid_prompt = "You are a helpful assistant. Your role is to assist users with their questions and provide accurate information."
        is_valid, error = await use_case.validate_prompt(valid_prompt)

        assert is_valid is True
        assert error is None


class TestPromptUseCaseExtractResponseText:
    """Tests for PromptUseCase._extract_response_text method."""

    @pytest.fixture
    def use_case(self):
        """Create a PromptUseCase instance."""
        from persbot.use_cases.prompt_use_case import PromptUseCase

        mock_config = Mock()
        mock_llm_service = Mock()
        return PromptUseCase(mock_config, mock_llm_service)

    def test_extracts_from_string(self, use_case):
        """_extract_response_text extracts from string."""
        result = use_case._extract_response_text("Hello world")
        assert result == "Hello world"

    def test_extracts_from_text_attribute(self, use_case):
        """_extract_response_text extracts from .text attribute."""
        mock_response = Mock()
        mock_response.text = "Hello from text"

        result = use_case._extract_response_text(mock_response)
        assert result == "Hello from text"

    def test_extracts_from_choices(self, use_case):
        """_extract_response_text extracts from choices."""
        mock_response = Mock()
        # Remove default .text attribute so choices path is taken
        del mock_response.text
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello from choices"

        result = use_case._extract_response_text(mock_response)
        assert result == "Hello from choices"


class TestPromptUseCaseParseQuestionsResponse:
    """Tests for PromptUseCase._parse_questions_response method."""

    @pytest.fixture
    def use_case(self):
        """Create a PromptUseCase instance."""
        from persbot.use_cases.prompt_use_case import PromptUseCase

        mock_config = Mock()
        mock_llm_service = Mock()
        return PromptUseCase(mock_config, mock_llm_service)

    def test_parses_json_with_questions_key(self, use_case):
        """_parse_questions_response parses JSON with questions key."""
        json_response = '{"questions": [{"question": "Q1", "sample_answer": "A1"}]}'

        result = use_case._parse_questions_response(json_response)

        assert len(result) == 1
        assert result[0]["question"] == "Q1"

    def test_parses_json_list(self, use_case):
        """_parse_questions_response parses JSON list."""
        json_response = '[{"question": "Q1", "sample_answer": "A1"}]'

        result = use_case._parse_questions_response(json_response)

        assert len(result) == 1

    def test_returns_empty_for_invalid_json(self, use_case):
        """_parse_questions_response returns empty for invalid JSON."""
        result = use_case._parse_questions_response("not valid json")
        assert result == []

    def test_parses_markdown_code_block(self, use_case):
        """_parse_questions_response parses markdown code block."""
        markdown_response = '''```json
{"questions": [{"question": "Q1", "sample_answer": "A1"}]}
```'''

        result = use_case._parse_questions_response(markdown_response)

        assert len(result) == 1

"""Feature tests for image use case module.

Tests focus on behavior using mocking:
- VisionRequest: request for vision understanding
- VisionResponse: response from vision understanding
- ImageGenerationRequest: request for image generation
- ImageGenerationResponse: response from image generation
- ImageUseCase: use case for image operations
"""

import sys
from unittest.mock import Mock, MagicMock, AsyncMock, patch

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


class TestVisionRequest:
    """Tests for VisionRequest dataclass."""

    def test_vision_request_exists(self):
        """VisionRequest class exists."""
        from persbot.use_cases.image_use_case import VisionRequest
        assert VisionRequest is not None

    def test_vision_request_has_required_fields(self):
        """VisionRequest has required fields."""
        from persbot.use_cases.image_use_case import VisionRequest

        mock_message = MagicMock()
        request = VisionRequest(
            images=[b"image_data"],
            user_message="What is in this image?",
            discord_message=mock_message,
        )

        assert request.images == [b"image_data"]
        assert request.user_message == "What is in this image?"
        assert request.discord_message == mock_message

    def test_vision_request_defaults(self):
        """VisionRequest has correct defaults."""
        from persbot.use_cases.image_use_case import VisionRequest

        mock_message = MagicMock()
        request = VisionRequest(
            images=[],
            user_message="test",
            discord_message=mock_message,
        )

        assert request.cancel_event is None


class TestVisionResponse:
    """Tests for VisionResponse dataclass."""

    def test_vision_response_exists(self):
        """VisionResponse class exists."""
        from persbot.use_cases.image_use_case import VisionResponse
        assert VisionResponse is not None

    def test_vision_response_has_required_fields(self):
        """VisionResponse has required fields."""
        from persbot.use_cases.image_use_case import VisionResponse

        response = VisionResponse(
            description="A cat sitting on a couch",
            success=True,
        )

        assert response.description == "A cat sitting on a couch"
        assert response.success is True

    def test_vision_response_defaults(self):
        """VisionResponse has correct defaults."""
        from persbot.use_cases.image_use_case import VisionResponse

        response = VisionResponse(
            description="",
            success=False,
            error="Failed to process image",
        )

        assert response.error == "Failed to process image"


class TestImageGenerationRequest:
    """Tests for ImageGenerationRequest dataclass."""

    def test_image_generation_request_exists(self):
        """ImageGenerationRequest class exists."""
        from persbot.use_cases.image_use_case import ImageGenerationRequest
        assert ImageGenerationRequest is not None

    def test_image_generation_request_has_required_fields(self):
        """ImageGenerationRequest has required fields."""
        from persbot.use_cases.image_use_case import ImageGenerationRequest

        request = ImageGenerationRequest(
            prompt="A beautiful sunset",
            channel_id=123,
        )

        assert request.prompt == "A beautiful sunset"
        assert request.channel_id == 123

    def test_image_generation_request_defaults(self):
        """ImageGenerationRequest has correct defaults."""
        from persbot.use_cases.image_use_case import ImageGenerationRequest

        request = ImageGenerationRequest(
            prompt="test",
            channel_id=123,
        )

        assert request.cancel_event is None
        assert request.model is None


class TestImageGenerationResponse:
    """Tests for ImageGenerationResponse dataclass."""

    def test_image_generation_response_exists(self):
        """ImageGenerationResponse class exists."""
        from persbot.use_cases.image_use_case import ImageGenerationResponse
        assert ImageGenerationResponse is not None

    def test_image_generation_response_has_required_fields(self):
        """ImageGenerationResponse has required fields."""
        from persbot.use_cases.image_use_case import ImageGenerationResponse

        response = ImageGenerationResponse(
            image_data=b"png_data",
            success=True,
            model_used="dalle-3",
        )

        assert response.image_data == b"png_data"
        assert response.success is True
        assert response.model_used == "dalle-3"

    def test_image_generation_response_defaults(self):
        """ImageGenerationResponse has correct defaults."""
        from persbot.use_cases.image_use_case import ImageGenerationResponse

        response = ImageGenerationResponse(
            image_data=b"",
            success=False,
            error="Generation failed",
        )

        assert response.error == "Generation failed"
        assert response.model_used is None


class TestImageUseCase:
    """Tests for ImageUseCase class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.no_check_permission = False
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        service.get_backend_for_model = Mock(return_value=None)
        service.model_usage_service = Mock()
        service.model_usage_service.get_api_model_name = Mock(return_value="test-model")
        return service

    @pytest.fixture
    def mock_image_usage_service(self):
        """Create a mock image usage service."""
        service = Mock()
        service.check_can_upload = Mock(return_value=True)
        service.record_upload = AsyncMock()
        return service

    def test_image_use_case_exists(self):
        """ImageUseCase class exists."""
        from persbot.use_cases.image_use_case import ImageUseCase
        assert ImageUseCase is not None

    def test_creates_with_dependencies(
        self, mock_config, mock_llm_service, mock_image_usage_service
    ):
        """ImageUseCase creates with dependencies."""
        from persbot.use_cases.image_use_case import ImageUseCase

        use_case = ImageUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_image_usage_service,
        )

        assert use_case.config == mock_config
        assert use_case.llm_service == mock_llm_service
        assert use_case.image_usage_service == mock_image_usage_service

    def test_has_vision_model_alias(
        self, mock_config, mock_llm_service, mock_image_usage_service
    ):
        """ImageUseCase has vision model alias."""
        from persbot.use_cases.image_use_case import ImageUseCase

        use_case = ImageUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_image_usage_service,
        )

        assert hasattr(use_case, '_vision_model_alias')
        assert use_case._vision_model_alias is not None


class TestImageUseCaseUnderstandImages:
    """Tests for ImageUseCase.understand_images method."""

    @pytest.fixture
    def use_case(self):
        """Create an ImageUseCase instance."""
        from persbot.use_cases.image_use_case import ImageUseCase

        mock_config = Mock()
        mock_config.no_check_permission = True

        mock_llm_service = Mock()
        mock_llm_service.get_backend_for_model = Mock(return_value=None)
        mock_llm_service.model_usage_service = Mock()
        mock_llm_service.model_usage_service.get_api_model_name = Mock(return_value="test-model")

        mock_image_usage_service = Mock()
        mock_image_usage_service.check_can_upload = Mock(return_value=True)
        mock_image_usage_service.record_upload = AsyncMock()

        return ImageUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_image_usage_service,
        )

    @pytest.mark.asyncio
    async def test_returns_none_without_images(self, use_case):
        """understand_images returns None without images."""
        from persbot.use_cases.image_use_case import VisionRequest

        mock_message = MagicMock()
        request = VisionRequest(
            images=[],
            user_message="test",
            discord_message=mock_message,
        )

        result = await use_case.understand_images(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_vision_backend(self, use_case):
        """understand_images returns None when no vision backend."""
        from persbot.use_cases.image_use_case import VisionRequest

        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = 123

        request = VisionRequest(
            images=[b"image_data"],
            user_message="test",
            discord_message=mock_message,
        )

        result = await use_case.understand_images(request)
        assert result is None


class TestImageUseCaseGenerateImage:
    """Tests for ImageUseCase.generate_image method."""

    @pytest.fixture
    def use_case_without_image_service(self):
        """Create an ImageUseCase instance without image service."""
        from persbot.use_cases.image_use_case import ImageUseCase

        mock_config = Mock()
        mock_config.no_check_permission = True

        mock_llm_service = Mock()
        mock_image_usage_service = Mock()

        return ImageUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_image_usage_service,
            image_service=None,
        )

    @pytest.mark.asyncio
    async def test_returns_error_without_image_service(self, use_case_without_image_service):
        """generate_image returns error without image service."""
        from persbot.use_cases.image_use_case import ImageGenerationRequest

        request = ImageGenerationRequest(
            prompt="test image",
            channel_id=123,
        )

        result = await use_case_without_image_service.generate_image(request)

        assert result is not None
        assert result.success is False
        assert "error" in result.error.lower() or "서비스" in result.error


class TestImageUseCaseCheckImageLimit:
    """Tests for ImageUseCase.check_image_limit method."""

    @pytest.fixture
    def use_case(self):
        """Create an ImageUseCase instance."""
        from persbot.use_cases.image_use_case import ImageUseCase

        mock_config = Mock()
        mock_config.no_check_permission = True

        mock_llm_service = Mock()
        mock_image_usage_service = Mock()
        mock_image_usage_service.check_can_upload = Mock(return_value=True)

        return ImageUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_image_usage_service,
        )

    def test_returns_none_when_allowed(self, use_case):
        """check_image_limit returns None when allowed."""
        mock_author = MagicMock()

        result = use_case.check_image_limit(mock_author, 1)
        assert result is None


class TestImageUseCaseIsAdmin:
    """Tests for ImageUseCase._is_admin method."""

    @pytest.fixture
    def use_case_no_permission_check(self):
        """Create an ImageUseCase instance with no permission check."""
        from persbot.use_cases.image_use_case import ImageUseCase

        mock_config = Mock()
        mock_config.no_check_permission = True

        mock_llm_service = Mock()
        mock_image_usage_service = Mock()

        return ImageUseCase(
            config=mock_config,
            llm_service=mock_llm_service,
            image_usage_service=mock_image_usage_service,
        )

    def test_returns_true_when_no_check_permission(self, use_case_no_permission_check):
        """_is_admin returns True when no_check_permission is True."""
        mock_author = MagicMock()

        result = use_case_no_permission_check._is_admin(mock_author)
        assert result is True

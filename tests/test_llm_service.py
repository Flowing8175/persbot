"""Comprehensive tests for LLMService."""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import discord
import pytest
import pytest_asyncio

from persbot.services.gemini_service import GeminiService
from persbot.services.llm_service import LLMService
from persbot.services.model_usage_service import ModelDefinition, ModelUsageService
from persbot.services.openai_service import OpenAIService
from persbot.services.prompt_service import PromptService
from persbot.services.usage_service import ImageUsageService
from persbot.services.zai_service import ZAIService


# Helper function to create test model definitions
def _create_test_model_definitions():
    """Create test ModelDefinition instances for use in tests."""
    return {
        "gemini-2.5-flash": ModelDefinition(
            display_name="Gemini 2.5 Flash",
            api_model_name="gemini-2.5-flash",
            daily_limit=100,
            scope="guild",
            provider="gemini",
            fallback_alias=None,
        ),
        "gpt-4": ModelDefinition(
            display_name="GPT-4",
            api_model_name="gpt-4",
            daily_limit=100,
            scope="guild",
            provider="openai",
            fallback_alias="gemini-2.5-flash",
        ),
        "glm-4.7": ModelDefinition(
            display_name="GLM 4.7",
            api_model_name="glm-4.7",
            daily_limit=100,
            scope="guild",
            provider="zai",
            fallback_alias=None,
        ),
    }


# Helper to create a proper mock config that returns strings for attributes
def _create_mock_config(
    assistant_provider="gemini",
    summarizer_provider="gemini",
    assistant_model="gemini-2.5-flash",
    summarizer_model="gemini-2.5-flash",
    gemini_key="test_gemini_key",
    openai_key="test_openai_key",
    zai_key="test_zai_key",
):
    """Create a mock AppConfig that returns proper values for attributes.

    Using a simple object instead of Mock to avoid issues with string methods
    like .lower() returning Mock objects instead of strings.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        assistant_llm_provider=assistant_provider,
        summarizer_llm_provider=summarizer_provider,
        assistant_model_name=assistant_model,
        summarizer_model_name=summarizer_model,
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        zai_api_key=zai_key,
        no_check_permission=True,
    )


class TestLLMServiceInit:
    """Test LLMService.__init__() - Test initialization."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            return LLMService(mock_config)

    def test_create_openai_backend(self, llm_service):
        """Test OpenAI backend creation."""
        backend = llm_service._create_backend(
            "openai", assistant_model_name="gpt-4", summary_model_name="gpt-3.5-turbo"
        )

        assert isinstance(backend, OpenAIService)

    def test_create_zai_backend(self, llm_service):
        """Test Z.AI backend creation."""
        backend = llm_service._create_backend(
            "zai", assistant_model_name="glm-4.7", summary_model_name="glm-4-flash"
        )

        assert isinstance(backend, ZAIService)

    def test_create_gemini_backend(self, llm_service):
        """Test Gemini backend creation (default)."""
        backend = llm_service._create_backend(
            "gemini",
            assistant_model_name="gemini-2.5-flash",
            summary_model_name="gemini-2.5-pro",
        )

        assert isinstance(backend, GeminiService)

    def test_passing_correct_model_names(self, llm_service):
        """Test passing correct model names to backend."""
        backend = llm_service._create_backend(
            "gemini",
            assistant_model_name="gemini-2.5-flash-exp",
            summary_model_name="gemini-2.5-pro-exp",
        )

        # Verify model names are set correctly
        assert backend._assistant_model_name == "gemini-2.5-flash-exp"
        assert backend._summary_model_name == "gemini-2.5-pro-exp"

    def test_create_backend_without_summary_model(self, llm_service):
        """Test creating backend without summary model name."""
        backend = llm_service._create_backend(
            "gemini", assistant_model_name="gemini-2.5-flash", summary_model_name=None
        )

        assert isinstance(backend, GeminiService)


class TestGetBackendForModel:
    """Test get_backend_for_model() - Test backend retrieval."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    def test_delegating_to_assistant_backend(self, llm_service):
        """Test delegating to assistant_backend."""
        mock_model = Mock()
        with patch.object(
            llm_service.assistant_backend,
            "create_assistant_model",
            return_value=mock_model,
        ) as mocked_method:
            result = llm_service.create_assistant_model("Test instruction")

        assert result == mock_model
        mocked_method.assert_called_once_with("Test instruction", use_cache=True)

    def test_use_cache_parameter(self, llm_service):
        """Test use_cache parameter is passed correctly."""
        mock_model = Mock()
        with patch.object(
            llm_service.assistant_backend,
            "create_assistant_model",
            return_value=mock_model,
        ) as mocked_method:
            result = llm_service.create_assistant_model("Test instruction", use_cache=False)

        assert result == mock_model
        mocked_method.assert_called_once_with("Test instruction", use_cache=False)

    def test_return_value(self, llm_service):
        """Test return value from backend."""
        mock_model = Mock()
        with patch.object(
            llm_service.assistant_backend,
            "create_assistant_model",
            return_value=mock_model,
        ):
            result = llm_service.create_assistant_model("Test instruction")

        assert result is mock_model


class TestSummarizeText:
    """Test summarize_text() - Test summarization."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    @pytest.mark.asyncio
    async def test_delegating_to_summarizer_backend(self, llm_service):
        """Test delegating to summarizer_backend."""
        with patch.object(
            llm_service.summarizer_backend, "summarize_text", new_callable=AsyncMock
        ) as mock_summarize:
            mock_summarize.return_value = "Summary text"
            result = await llm_service.summarize_text("Original text")
            assert result == "Summary text"
            mock_summarize.assert_called_once_with("Original text")

    @pytest.mark.asyncio
    async def test_async_call(self, llm_service):
        """Test async call to backend."""
        with patch.object(
            llm_service.summarizer_backend, "summarize_text", new_callable=AsyncMock
        ) as mock_summarize:
            mock_summarize.return_value = "Async summary"
            result = await llm_service.summarize_text("Test text")

        assert result == "Async summary"

    @pytest.mark.asyncio
    async def test_return_value(self, llm_service):
        """Test return value from async call."""
        expected_summary = "This is a summary"
        with patch.object(
            llm_service.summarizer_backend, "summarize_text", new_callable=AsyncMock
        ) as mock_summarize:
            mock_summarize.return_value = expected_summary
            result = await llm_service.summarize_text("Long text here")

        assert result == expected_summary


class TestGeneratePromptFromConcept:
    """Test generate_prompt_from_concept() - Test prompt generation."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    @pytest.mark.asyncio
    async def test_creating_meta_model_with_meta_prompt(self, llm_service):
        """Test creating meta model with META_PROMPT."""
        from persbot.prompts import META_PROMPT

        mock_model = Mock()
        mock_model.generate_content = AsyncMock(return_value="Generated prompt")
        mock_backend = Mock()
        mock_backend.assistant_model = mock_model
        mock_backend.execute_with_retry = AsyncMock(return_value="Generated prompt")
        mock_backend.create_assistant_model = Mock(return_value=mock_model)

        llm_service.summarizer_backend = mock_backend

        result = await llm_service.generate_prompt_from_concept("Exciting Boyfriend")

        mock_backend.create_assistant_model.assert_called_once_with(META_PROMPT, use_cache=False)

    @pytest.mark.asyncio
    async def test_calling_generate_content(self, llm_service):
        """Test calling generate_content on meta model."""
        mock_model = Mock()
        mock_model.generate_content = AsyncMock(return_value="Generated content")
        mock_backend = Mock()
        mock_backend.assistant_model = mock_model
        mock_backend.execute_with_retry = AsyncMock(return_value="Generated content")
        mock_backend.create_assistant_model = Mock(return_value=mock_model)

        llm_service.summarizer_backend = mock_backend

        result = await llm_service.generate_prompt_from_concept("Test concept")

        assert mock_backend.execute_with_retry.called

    @pytest.mark.asyncio
    async def test_async_execution(self, llm_service):
        """Test async execution of prompt generation."""
        mock_model = Mock()
        mock_model.generate_content = AsyncMock(return_value="Async generated")
        mock_backend = Mock()
        mock_backend.assistant_model = mock_model
        mock_backend.execute_with_retry = AsyncMock(return_value="Async generated")
        mock_backend.create_assistant_model = Mock(return_value=mock_model)

        llm_service.summarizer_backend = mock_backend

        result = await llm_service.generate_prompt_from_concept("Test")

        assert result == "Async generated"

    @pytest.mark.asyncio
    async def test_return_value(self, llm_service):
        """Test return value from async generation."""
        expected = "Detailed system prompt here"
        mock_model = Mock()
        mock_model.generate_content = AsyncMock(return_value=expected)
        mock_backend = Mock()
        mock_backend.assistant_model = mock_model
        mock_backend.execute_with_retry = AsyncMock(return_value=expected)
        mock_backend.create_assistant_model = Mock(return_value=mock_model)

        llm_service.summarizer_backend = mock_backend

        result = await llm_service.generate_prompt_from_concept("Concept")

        assert result == expected

    @pytest.mark.asyncio
    async def test_none_when_no_assistant_model_attribute(self, llm_service):
        """Test returning None when backend has no assistant_model attribute."""
        mock_backend = Mock()
        del mock_backend.assistant_model

        llm_service.summarizer_backend = mock_backend

        result = await llm_service.generate_prompt_from_concept("Concept")

        assert result is None


class TestGenerateChatResponse:
    """Test generate_chat_response() - Test chat response generation."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    @pytest.fixture
    def mock_message(self):
        """Create a mock Discord message."""
        message = Mock()
        message.author = Mock(id=123456, name="TestUser")
        message.channel = Mock(id=789012)
        message.guild = Mock(id=456789)
        return message

    def test_single_message(self, llm_service, mock_message):
        """Test single message."""
        user_id, channel_id, guild_id, primary_author = llm_service._extract_message_metadata(
            mock_message
        )

        assert user_id == 123456
        assert channel_id == 789012
        assert guild_id == 456789
        assert primary_author == mock_message.author

    def test_message_list(self, llm_service, mock_message):
        """Test message list."""
        message_list = [mock_message, Mock()]
        user_id, channel_id, guild_id, primary_author = llm_service._extract_message_metadata(
            message_list
        )

        # Should use first message
        assert user_id == 123456
        assert primary_author == mock_message.author

    def test_user_id_extraction(self, llm_service, mock_message):
        """Test user_id extraction."""
        user_id, _, _, _ = llm_service._extract_message_metadata(mock_message)

        assert user_id == 123456

    def test_channel_id_extraction(self, llm_service, mock_message):
        """Test channel_id extraction."""
        _, channel_id, _, _ = llm_service._extract_message_metadata(mock_message)

        assert channel_id == 789012

    def test_guild_id_extraction(self, llm_service, mock_message):
        """Test guild_id extraction."""
        _, _, guild_id, _ = llm_service._extract_message_metadata(mock_message)

        assert guild_id == 456789

    def test_primary_author_extraction(self, llm_service, mock_message):
        """Test primary_author extraction."""
        _, _, _, primary_author = llm_service._extract_message_metadata(mock_message)

        assert primary_author == mock_message.author

    def test_handling_dm_no_guild(self, llm_service):
        """Test handling DM (no guild)."""
        message = Mock()
        message.author = Mock(id=123456)
        message.channel = Mock(id=789012)
        message.guild = None

        _, _, guild_id, _ = llm_service._extract_message_metadata(message)

        # Should use user_id as guild_id
        assert guild_id == 123456


class TestCountImagesInMessage:
    """Test _count_images_in_message() - Test image counting."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    def test_no_images(self, llm_service):
        """Test no images."""
        message = Mock()
        message.attachments = []

        count = llm_service._count_images_in_message(message)

        assert count == 0

    def test_single_image(self, llm_service):
        """Test single image."""
        attachment = Mock()
        attachment.content_type = "image/png"
        message = Mock()
        message.attachments = [attachment]

        count = llm_service._count_images_in_message(message)

        assert count == 1

    def test_multiple_images(self, llm_service):
        """Test multiple images."""
        attachment1 = Mock()
        attachment1.content_type = "image/png"
        attachment2 = Mock()
        attachment2.content_type = "image/jpeg"
        attachment3 = Mock()
        attachment3.content_type = "image/gif"
        message = Mock()
        message.attachments = [attachment1, attachment2, attachment3]

        count = llm_service._count_images_in_message(message)

        assert count == 3

    def test_filtering_non_image_content_types(self, llm_service):
        """Test filtering non-image content types."""
        image = Mock()
        image.content_type = "image/png"
        video = Mock()
        video.content_type = "video/mp4"
        document = Mock()
        document.content_type = "application/pdf"
        message = Mock()
        message.attachments = [image, video, document]

        count = llm_service._count_images_in_message(message)

        assert count == 1

    def test_message_list(self, llm_service):
        """Test message list."""
        message1 = Mock()
        attachment1 = Mock()
        attachment1.content_type = "image/png"
        message1.attachments = [attachment1]

        message2 = Mock()
        attachment2 = Mock()
        attachment2.content_type = "image/jpeg"
        message2.attachments = [attachment2]

        count = llm_service._count_images_in_message([message1, message2])

        assert count == 2


class TestCheckImageUsageLimit:
    """Test _check_image_usage_limit() - Test image limit checking."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    @pytest.fixture
    def mock_admin_author(self):
        """Create a mock admin author."""
        author = Mock()
        author.guild_permissions = Mock(manage_guild=True)
        return author

    @pytest.fixture
    def mock_regular_author(self):
        """Create a mock regular author."""
        author = Mock()
        author.guild_permissions = Mock(manage_guild=False)
        return author

    def test_admin_bypass(self, llm_service, mock_admin_author):
        """Test admin bypass."""
        result = llm_service._check_image_usage_limit(mock_admin_author, 3)

        assert result is None

    def test_no_check_permission_bypass(self, mock_config):
        """Test NO_CHECK_PERMISSION bypass."""
        mock_config.no_check_permission = False

        with patch("persbot.services.llm_service.ModelUsageService"):
            service = LLMService(mock_config)
            service.config.no_check_permission = True

            author = Mock()
            author.guild_permissions = Mock(manage_guild=False)

            result = service._check_image_usage_limit(author, 3)

            assert result is None

    def test_limit_exceeded(self, llm_service, mock_regular_author):
        """Test limit exceeded."""
        llm_service.config.no_check_permission = False
        llm_service.image_usage_service.check_can_upload = Mock(return_value=False)

        result = llm_service._check_image_usage_limit(mock_regular_author, 3)

        assert result is not None
        assert result[0] == "❌ 이미지는 하루에 최대 3개 업로드하실 수 있습니다."

    def test_limit_not_exceeded(self, llm_service, mock_regular_author):
        """Test limit not exceeded."""
        llm_service.config.no_check_permission = False
        llm_service.image_usage_service.check_can_upload = Mock(return_value=True)

        result = llm_service._check_image_usage_limit(mock_regular_author, 2)

        assert result is None

    def test_limit_parameter(self, llm_service, mock_regular_author):
        """Test limit parameter (3 images)."""
        llm_service.config.no_check_permission = False
        llm_service.image_usage_service.check_can_upload = Mock(return_value=True)

        result = llm_service._check_image_usage_limit(mock_regular_author, 3)

        # Should call check_can_upload with limit=3
        llm_service.image_usage_service.check_can_upload.assert_called_once()
        call_args = llm_service.image_usage_service.check_can_upload.call_args
        assert call_args[1] == {"limit": 3}

    def test_returning_error_tuple(self, llm_service, mock_regular_author):
        """Test returning error tuple."""
        llm_service.config.no_check_permission = False
        llm_service.image_usage_service.check_can_upload = Mock(return_value=False)

        result = llm_service._check_image_usage_limit(mock_regular_author, 3)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is not None
        assert result[1] is None

    def test_returning_none(self, llm_service, mock_regular_author):
        """Test returning None."""
        llm_service.config.no_check_permission = False
        llm_service.image_usage_service.check_can_upload = Mock(return_value=True)

        result = llm_service._check_image_usage_limit(mock_regular_author, 1)

        assert result is None


class TestRecordImageUsageIfNeeded:
    """Test _record_image_usage_if_needed() - Test usage recording."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    @pytest.fixture
    def mock_admin_author(self):
        """Create a mock admin author."""
        author = Mock()
        author.guild_permissions = Mock(manage_guild=True)
        return author

    @pytest.fixture
    def mock_regular_author(self):
        """Create a mock regular author."""
        author = Mock()
        author.guild_permissions = Mock(manage_guild=False)
        return author

    @pytest.mark.asyncio
    async def test_admin_bypass(self, llm_service, mock_admin_author):
        """Test admin bypass."""
        llm_service.image_usage_service.record_upload = AsyncMock()

        await llm_service._record_image_usage_if_needed(mock_admin_author, 2)

        llm_service.image_usage_service.record_upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_check_permission_bypass(self, mock_config):
        """Test NO_CHECK_PERMISSION bypass."""
        mock_config.no_check_permission = False

        with patch("persbot.services.llm_service.ModelUsageService"):
            service = LLMService(mock_config)
            service.config.no_check_permission = True

            author = Mock()
            author.guild_permissions = Mock(manage_guild=False)
            service.image_usage_service.record_upload = AsyncMock()

            await service._record_image_usage_if_needed(author, 2)

            service.image_usage_service.record_upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_recording_for_non_admin(self, llm_service, mock_regular_author):
        """Test recording for non-admin."""
        llm_service.config.no_check_permission = False
        llm_service.image_usage_service.record_upload = AsyncMock()

        await llm_service._record_image_usage_if_needed(mock_regular_author, 2)

        llm_service.image_usage_service.record_upload.assert_called_once_with(
            mock_regular_author.id, 2
        )

    @pytest.mark.asyncio
    async def test_skipping_for_admin(self, llm_service, mock_admin_author):
        """Test skipping for admin."""
        llm_service.config.no_check_permission = True
        llm_service.image_usage_service.record_upload = AsyncMock()

        await llm_service._record_image_usage_if_needed(mock_admin_author, 2)

        llm_service.image_usage_service.record_upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_call_to_record_upload(self, llm_service, mock_regular_author):
        """Test async call to record_upload."""
        llm_service.config.no_check_permission = False
        llm_service.image_usage_service.record_upload = AsyncMock()

        await llm_service._record_image_usage_if_needed(mock_regular_author, 3)

        # Should have been called
        assert llm_service.image_usage_service.record_upload.called


class TestPrepareResponseWithNotification:
    """Test _prepare_response_with_notification() - Test response preparation."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    def test_prepending_notification(self, llm_service):
        """Test prepending notification."""
        response = ("Original response", None)
        notification = "Important notice"

        result = llm_service._prepare_response_with_notification(response, notification)

        text, obj = result
        assert text.startswith("📢 Important notice")
        assert "Original response" in text

    def test_no_notification(self, llm_service):
        """Test no notification."""
        response = ("Original response", None)
        notification = None

        result = llm_service._prepare_response_with_notification(response, notification)

        assert result == response

    def test_with_response_and_notification(self, llm_service):
        """Test with response and notification."""
        response = ("AI response text", {"data": "object"})
        notification = "Model switched"

        result = llm_service._prepare_response_with_notification(response, notification)

        text, obj = result
        assert "📢 Model switched" in text
        assert "AI response text" in text
        assert obj == {"data": "object"}

    def test_with_none_response(self, llm_service):
        """Test with None response."""
        result = llm_service._prepare_response_with_notification(None, "Notification")

        assert result is None

    def test_returning_tuple_correctly(self, llm_service):
        """Test returning tuple correctly."""
        response = ("Response text", None)
        notification = "Test"

        result = llm_service._prepare_response_with_notification(response, notification)

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestGetUserRoleName:
    """Test get_user_role_name() - Test role name."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    def test_delegating_to_assistant_backend(self, llm_service):
        """Test delegating to assistant_backend."""
        llm_service.assistant_backend.get_user_role_name = Mock(return_value="user")

        result = llm_service.get_user_role_name()

        assert result == "user"
        llm_service.assistant_backend.get_user_role_name.assert_called_once()


class TestGetAssistantRoleName:
    """Test get_assistant_role_name() - Test role name."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    def test_delegating_to_assistant_backend(self, llm_service):
        """Test delegating to assistant_backend."""
        llm_service.assistant_backend.get_assistant_role_name = Mock(return_value="model")

        result = llm_service.get_assistant_role_name()

        assert result == "model"
        llm_service.assistant_backend.get_assistant_role_name.assert_called_once()


class TestUpdateParameters:
    """Test update_parameters() - Test parameter updates."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

    @pytest.fixture
    def llm_service(self, mock_config):
        """Create an LLMService instance."""
        with patch("persbot.services.llm_service.ModelUsageService"), \
             patch("persbot.services.llm_service.GeminiService") as mock_gemini, \
             patch("persbot.services.llm_service.OpenAIService") as mock_openai, \
             patch("persbot.services.llm_service.ZAIService") as mock_zai:
            mock_gemini.return_value = Mock()
            mock_openai.return_value = Mock()
            mock_zai.return_value = Mock()
            service = LLMService(mock_config)
            return service

    def test_temperature_update(self, llm_service):
        """Test temperature update."""
        llm_service.update_parameters(temperature=0.7)

        assert llm_service.config.temperature == 0.7

    def test_top_p_update(self, llm_service):
        """Test top_p update."""
        llm_service.update_parameters(top_p=0.9)

        assert llm_service.config.top_p == 0.9

    def test_thinking_budget_update(self, llm_service):
        """Test thinking_budget update."""
        llm_service.update_parameters(thinking_budget=5000)

        assert llm_service.config.thinking_budget == 5000

    def test_updating_config_object(self, llm_service):
        """Test updating config object."""
        llm_service.update_parameters(temperature=0.5, top_p=0.8, thinking_budget=3000)

        assert llm_service.config.temperature == 0.5
        assert llm_service.config.top_p == 0.8
        assert llm_service.config.thinking_budget == 3000

    def test_reloading_assistant_backend(self, llm_service):
        """Test reloading assistant_backend."""
        llm_service.assistant_backend.reload_parameters = Mock()

        llm_service.update_parameters(temperature=0.5)

        llm_service.assistant_backend.reload_parameters.assert_called_once()

    def test_reloading_summarizer_backend(self, mock_config):
        """Test reloading summarizer_backend when different."""
        mock_config.assistant_llm_provider = "gemini"
        mock_config.summarizer_llm_provider = "openai"

        with patch("persbot.services.llm_service.ModelUsageService"):
            service = LLMService(mock_config)
            service.assistant_backend.reload_parameters = Mock()
            service.summarizer_backend.reload_parameters = Mock()

            service.update_parameters(temperature=0.5)

            service.assistant_backend.reload_parameters.assert_called_once()
            service.summarizer_backend.reload_parameters.assert_called_once()

    def test_reloading_auxiliary_backends(self, llm_service):
        """Test reloading auxiliary backends."""
        mock_backend = Mock()
        mock_backend.reload_parameters = Mock()
        llm_service._aux_backends = {"openai": mock_backend}

        llm_service.update_parameters(temperature=0.5)

        mock_backend.reload_parameters.assert_called_once()

    def test_parameter_validation(self, llm_service):
        """Test parameter validation (should accept valid values)."""
        # Should not raise exception for valid values
        llm_service.update_parameters(temperature=0.0, top_p=0.0, thinking_budget=0)

        assert llm_service.config.temperature == 0.0
        assert llm_service.config.top_p == 0.0
        assert llm_service.config.thinking_budget == 0

    def test_no_reload_when_no_change(self, llm_service):
        """Test that reload is called even when no parameters change."""
        llm_service.assistant_backend.reload_parameters = Mock()

        llm_service.update_parameters()

        llm_service.assistant_backend.reload_parameters.assert_called_once()


class TestProviderSwitching:
    """Test Provider switching - Test model provider changes."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        return _create_mock_config()

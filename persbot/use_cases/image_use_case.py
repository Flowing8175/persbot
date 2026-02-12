"""Image use case for handling image-related operations.

This use case handles image generation, vision understanding,
and image usage tracking.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import discord

from persbot.config import AppConfig
from persbot.services.image_service import ImageService
from persbot.services.image_model_service import get_channel_image_model
from persbot.services.llm_service import LLMService
from persbot.services.usage_service import ImageUsageService

logger = logging.getLogger(__name__)


@dataclass
class VisionRequest:
    """Request for vision understanding."""

    images: list[bytes]
    user_message: str
    discord_message: discord.Message
    cancel_event: Optional[asyncio.Event] = None


@dataclass
class VisionResponse:
    """Response from vision understanding."""

    description: str
    success: bool
    error: Optional[str] = None


@dataclass
class ImageGenerationRequest:
    """Request for image generation."""

    prompt: str
    channel_id: int
    cancel_event: Optional[asyncio.Event] = None
    model: Optional[str] = None


@dataclass
class ImageGenerationResponse:
    """Response from image generation."""

    image_data: bytes
    success: bool
    error: Optional[str] = None
    model_used: Optional[str] = None


class ImageUseCase:
    """Use case for handling image-related operations."""

    def __init__(
        self,
        config: AppConfig,
        llm_service: LLMService,
        image_usage_service: ImageUsageService,
        image_service: Optional[ImageService] = None,
    ) -> None:
        """Initialize the image use case.

        Args:
            config: Application configuration.
            llm_service: LLM service for vision understanding.
            image_usage_service: Service for tracking image usage.
            image_service: Optional service for image generation.
        """
        self.config = config
        self.llm_service = llm_service
        self.image_usage_service = image_usage_service
        self.image_service = image_service

        # Vision model configuration
        self._vision_model_alias = "GLM 4.6V"

    async def understand_images(self, request: VisionRequest) -> Optional[VisionResponse]:
        """Understand the content of images using vision models.

        Args:
            request: The vision request containing images and context.

        Returns:
            VisionResponse with description, or None if failed.
        """
        if not request.images:
            return None

        # Check if user has permission
        primary_author = request.discord_message.author
        if not self._can_upload_images(primary_author):
            return VisionResponse(
                success=False,
                description="",
                error="이미지는 하루에 최대 3개 업로드하실 수 있습니다.",
            )

        try:
            # Get vision backend
            vision_backend = self.llm_service.get_backend_for_model(self._vision_model_alias)
            if not vision_backend:
                logger.warning(f"Vision model {self._vision_model_alias} unavailable")
                return None

            # Create temporary vision session
            vision_session = self._create_vision_session(vision_backend)

            # Create a mock message with images for the vision API
            mock_message = self._create_mock_message_with_images(
                request.images, request.discord_message
            )

            # Get vision understanding
            result = await vision_backend.generate_chat_response(
                vision_session,
                request.user_message,
                mock_message,
                model_name=self.llm_service.model_usage_service.get_api_model_name(
                    self._vision_model_alias
                ),
                tools=None,
                cancel_event=request.cancel_event,
            )

            if result:
                vision_text, _ = result

                # Record usage
                await self._record_image_usage(primary_author, len(request.images))

                return VisionResponse(
                    success=True,
                    description=vision_text,
                )

            return None

        except Exception as e:
            logger.error(f"Vision understanding failed: {e}", exc_info=True)
            return VisionResponse(
                success=False,
                description="",
                error=str(e),
            )

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> Optional[ImageGenerationResponse]:
        """Generate an image from a text prompt.

        Args:
            request: The image generation request.

        Returns:
            ImageGenerationResponse with generated image, or None if failed.
        """
        if not self.image_service:
            return ImageGenerationResponse(
                success=False,
                image_data=b"",
                error="이미지 생성 서비스가 활성화되지 않았습니다.",
            )

        try:
            # Get the model to use
            model = request.model or get_channel_image_model(request.channel_id)

            # Generate the image
            image_data = await self.image_service.generate_image(
                prompt=request.prompt,
                model=model,
                cancel_event=request.cancel_event,
            )

            if image_data:
                return ImageGenerationResponse(
                    success=True,
                    image_data=image_data,
                    model_used=model,
                )

            return None

        except asyncio.CancelledError:
            logger.info("Image generation cancelled")
            return None
        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            return ImageGenerationResponse(
                success=False,
                image_data=b"",
                error=str(e),
            )

    def check_image_limit(self, author: discord.abc.User, image_count: int) -> Optional[str]:
        """Check if user can upload the given number of images.

        Args:
            author: The Discord user.
            image_count: Number of images to upload.

        Returns:
            Error message if limit exceeded, None otherwise.
        """
        if self._can_upload_images(author, image_count):
            return None
        return "이미지는 하루에 최대 3개 업로드하실 수 있습니다."

    async def record_image_usage(self, author: discord.abc.User, count: int) -> None:
        """Record image usage for rate limiting.

        Args:
            author: The Discord user.
            count: Number of images uploaded.
        """
        if not self._is_admin(author):
            await self.image_usage_service.record_upload(author.id, count)

    def get_channel_image_model(self, channel_id: int) -> str:
        """Get the configured image model for a channel.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            The model name to use for image generation.
        """
        return get_channel_image_model(channel_id)

    def _can_upload_images(self, author: discord.abc.User, count: int = 1) -> bool:
        """Check if user can upload images.

        Args:
            author: The Discord user.
            count: Number of images to upload.

        Returns:
            True if allowed, False otherwise.
        """
        if self._is_admin(author):
            return True
        return self.image_usage_service.check_can_upload(author.id, count, limit=3)

    def _is_admin(self, author: discord.abc.User) -> bool:
        """Check if user is an admin.

        Args:
            author: The Discord user.

        Returns:
            True if user has admin permissions.
        """
        if self.config.no_check_permission:
            return True
        return isinstance(author, discord.Member) and author.guild_permissions.manage_guild

    async def _record_image_usage(self, author: discord.abc.User, count: int) -> None:
        """Record image usage for non-admin users.

        Args:
            author: The Discord user.
            count: Number of images uploaded.
        """
        if not self._is_admin(author):
            await self.image_usage_service.record_upload(author.id, count)

    def _create_vision_session(self, backend: Any) -> Any:
        """Create a temporary session for vision understanding.

        Args:
            backend: The LLM backend.

        Returns:
            A chat session configured for vision.
        """
        system_prompt = (
            "You are a vision understanding assistant. "
            "Describe the content of images concisely and accurately "
            "in the language of the user's message."
        )
        return backend.create_assistant_model(system_prompt, use_cache=False)

    def _create_mock_message_with_images(
        self, images: list[bytes], original_message: discord.Message
    ) -> discord.Message:
        """Create a mock message with image attachments.

        Args:
            images: List of image bytes.
            original_message: Original Discord message.

        Returns:
            A mock Discord.Message with image attachments.
        """
        # This is a simplified version - full implementation would
        # create a proper mock message with attachments
        return original_message

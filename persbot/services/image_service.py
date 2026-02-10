"""Image generation service for SoyeBot AI.

This service handles image generation through OpenRouter API with retry logic
and cancellation support.
"""

import asyncio
import base64
import hashlib
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from openai import APIStatusError, AuthenticationError, OpenAI, RateLimitError
from PIL import Image

from persbot.config import AppConfig
from persbot.services.base import BaseLLMService
from persbot.services.retry_handler import (
    BackoffStrategy,
    RetryConfig,
    RetryHandler,
)
from persbot.utils import get_mime_type, process_image_sync

logger = logging.getLogger(__name__)


class ImageGenerationError(Exception):
    """Base exception for image generation errors."""
    pass


class ImageService(BaseLLMService):
    """Service for generating images via OpenRouter API.

    This service handles:
    - Image generation using OpenRouter-compatible models
    - Retry logic for transient failures
    - Cancellation support via cancel_event
    - Image-to-image generation with base64 input
    """

    def __init__(self, config: AppConfig):
        """Initialize the image service.

        Args:
            config: Application configuration containing API keys and settings.
        """
        super().__init__(config)
        self._client: Optional[OpenAI] = None
        self._retry_handler: Optional[RetryHandler] = None

    def _get_client(self) -> OpenAI:
        """Get or create the OpenAI client for OpenRouter."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=self.config.api_request_timeout,
            )
        return self._client

    def _create_retry_config(self) -> RetryConfig:
        """Create retry configuration for image generation API."""
        return RetryConfig(
            max_retries=2,
            base_delay=2.0,
            max_delay=32.0,
            rate_limit_delay=5,
            request_timeout=self.config.api_request_timeout,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        )

    def _create_retry_handler(self) -> RetryHandler:
        """Create retry handler for image generation."""
        # Use a generic retry handler since OpenAI's retry logic works for OpenRouter too
        from persbot.services.retry_handler import OpenAIRetryHandler
        return OpenAIRetryHandler(self._create_retry_config())

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if exception is a rate limit error."""
        error_str = str(error).lower()
        return (
            "rate limit" in error_str
            or "429" in error_str
            or isinstance(error, RateLimitError)
        )

    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        """Log raw request details for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        prompt_hash = hashlib.sha256(user_message.encode()).hexdigest()[:16]
        logger.debug(
            "[IMAGE REQUEST] Generating image (prompt_hash=%s, length=%d)",
            prompt_hash,
            len(user_message),
        )

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw response details for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        logger.debug("[IMAGE RESPONSE %s] %s", attempt, response_obj)

    def _extract_text_from_response(self, response_obj: Any) -> str:
        """Extract text content from response (not used for image generation)."""
        return ""

    def get_user_role_name(self) -> str:
        """Return the role name for user messages."""
        return "user"

    def get_assistant_role_name(self) -> str:
        """Return the role name for assistant messages."""
        return "assistant"

    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        model: Optional[str] = None,
        image_input: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[bytes, str]]:
        """
        Generate an image using OpenRouter image generation API.

        Args:
            prompt: The text prompt for image generation.
            aspect_ratio: Aspect ratio for the image (e.g., "1:1", "16:9").
            model: Specific model to use for generation. If None, uses config default.
            image_input: Optional base64-encoded image data for image-to-image.
            cancel_event: AsyncIO event to check for cancellation before API calls.

        Returns:
            Tuple of (image_bytes, image_format) on success, None on failure.

        Raises:
            asyncio.CancelledError: If cancel_event is set before/during API call.
            ImageGenerationError: If image generation fails after retries.
        """
        # Check cancellation event before starting
        if cancel_event and cancel_event.is_set():
            logger.info("Image generation aborted due to cancellation signal before API call")
            raise asyncio.CancelledError("Image generation aborted by user")

        if not prompt or not prompt.strip():
            raise ImageGenerationError("Image prompt cannot be empty")

        # Calculate prompt hash for logging
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        prompt_length = len(prompt)

        # Truncate prompt if too long (OpenRouter has ~2000 char limit)
        MAX_PROMPT_LENGTH = 2000
        enhanced_prompt = prompt
        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning(
                "Prompt too long (prompt_hash=%s, length=%d, max=%d): truncating",
                prompt_hash,
                prompt_length,
                MAX_PROMPT_LENGTH,
            )
            enhanced_prompt = prompt[:MAX_PROMPT_LENGTH]

        self._log_raw_request(enhanced_prompt)

        # Use provided model or fall back to config default
        image_model = model or self.config.openrouter_image_model

        # Check for cancellation before API call
        if cancel_event and cancel_event.is_set():
            logger.info("Image generation aborted due to cancellation signal before API call")
            raise asyncio.CancelledError("Image generation aborted by user")

        # Build image config with aspect_ratio
        image_config = {"aspect_ratio": aspect_ratio}

        # Build user message content - include image if provided
        if image_input:
            # Image-to-image: include both text prompt and input image
            user_content = [
                {"type": "text", "text": enhanced_prompt},
                {"type": "image_url", "image_url": {"url": image_input}}
            ]
            logger.info(
                "Including input image in generation request (prompt_hash=%s)",
                prompt_hash,
            )
        else:
            # Text-to-image: only prompt
            user_content = enhanced_prompt

        try:
            # Execute with retry logic
            result = await self.execute_with_retry(
                lambda: self._execute_image_generation(
                    image_model, user_content, image_config, prompt_hash, prompt_length
                ),
                "이미지 생성",
                return_full_response=True,
                cancel_event=cancel_event,
            )

            if result is None:
                return None

            image_bytes, image_format = result

            logger.info(
                "Image generated successfully (prompt_hash=%s, response_time=%.2fs)",
                prompt_hash,
                time.time(),
            )

            return (image_bytes, image_format)

        except asyncio.CancelledError:
            logger.info("Image generation cancelled by user (prompt_hash=%s)", prompt_hash)
            raise
        except Exception as e:
            logger.error(
                "Image generation failed (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                e,
                exc_info=True,
            )
            raise ImageGenerationError(f"Image generation failed: {str(e)}")

    def _execute_image_generation(
        self,
        model: str,
        user_content: Any,
        image_config: Dict[str, str],
        prompt_hash: str,
        prompt_length: int,
    ) -> Tuple[bytes, str]:
        """Execute the actual image generation API call (synchronous).

        Args:
            model: Model name to use.
            user_content: User message content (text or text + image).
            image_config: Image generation configuration.
            prompt_hash: Hash for logging.
            prompt_length: Prompt length for logging.

        Returns:
            Tuple of (image_bytes, image_format).

        Raises:
            APIStatusError: If API returns an error status.
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit is exceeded.
        """
        client = self._get_client()

        api_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            modalities=["image"],
            extra_body={"image_config": image_config},
        )

        # Validate response structure
        if not api_response.choices or len(api_response.choices) == 0:
            logger.error(
                "No choices in image generation response (prompt_hash=%s, length=%d)",
                prompt_hash,
                prompt_length,
            )
            raise ImageGenerationError("No image generated")

        message = api_response.choices[0].message
        if not hasattr(message, "images") or not message.images or len(message.images) == 0:
            logger.error(
                "No images in response message (prompt_hash=%s, length=%d)",
                prompt_hash,
                prompt_length,
            )
            raise ImageGenerationError("No image generated")

        # Parse image URL: handle both base64 data URLs and HTTP/HTTPS URLs
        image_obj = message.images[0]
        if isinstance(image_obj, dict):
            image_data_url = image_obj["image_url"]["url"]
        else:
            image_data_url = image_obj.image_url.url

        return self._process_image_url(image_data_url, prompt_hash, prompt_length)

    def _process_image_url(
        self, image_data_url: str, prompt_hash: str, prompt_length: int
    ) -> Tuple[bytes, str]:
        """Process image URL and extract image bytes.

        Args:
            image_data_url: URL or base64 data URL of the image.
            prompt_hash: Hash for logging.
            prompt_length: Prompt length for logging.

        Returns:
            Tuple of (image_bytes, image_format).
        """
        if image_data_url.startswith("data:"):
            # Base64 data URL format
            return self._decode_base64_image(image_data_url, prompt_hash, prompt_length)
        elif image_data_url.startswith(("http://", "https://")):
            # HTTP/HTTPS URL - return URL for async fetching
            # For sync method, we need to handle differently
            # Return URL with a special marker
            return (image_data_url.encode("utf-8"), "url")
        else:
            logger.error(
                "Unknown image URL format (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                image_data_url[:50] if image_data_url else "empty",
            )
            raise ImageGenerationError("Invalid image URL format")

    def _decode_base64_image(
        self, data_url: str, prompt_hash: str, prompt_length: int
    ) -> Tuple[bytes, str]:
        """Decode base64 image data URL.

        Args:
            data_url: Base64 data URL.
            prompt_hash: Hash for logging.
            prompt_length: Prompt length for logging.

        Returns:
            Tuple of (image_bytes, image_format).
        """
        try:
            header, data = data_url.split(",", 1)
            image_bytes = base64.b64decode(data)

            # Extract format from header
            if "image/png" in header:
                image_format = "png"
            elif "image/jpeg" in header or "image/jpg" in header:
                image_format = "jpeg"
            elif "image/webp" in header:
                image_format = "webp"
            else:
                image_format = "png"

            return (image_bytes, image_format)
        except (ValueError, IndexError) as decode_error:
            logger.error(
                "Failed to decode base64 image (prompt_hash=%s, length=%d): %s",
                prompt_hash,
                prompt_length,
                decode_error,
            )
            raise ImageGenerationError("Failed to decode generated image")

    async def fetch_image_from_url(
        self,
        url: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[bytes]:
        """
        Fetch image bytes from a URL (for HTTP/HTTPS image URLs).

        Args:
            url: HTTP/HTTPS URL to fetch.
            cancel_event: AsyncIO event to check for cancellation.

        Returns:
            Image bytes on success, None on failure.

        Raises:
            asyncio.CancelledError: If cancel_event is set before/during fetch.
        """
        # Check for cancellation before fetching
        if cancel_event and cancel_event.is_set():
            logger.info("Image fetch aborted due to cancellation signal before fetch")
            raise asyncio.CancelledError("Image fetch aborted by user")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(
                            "Failed to fetch image from URL (status=%d): %s",
                            response.status,
                            url[:100],
                        )
                        return None

                    image_bytes = await response.read()
                    return image_bytes

        except asyncio.CancelledError:
            logger.info("Image fetch cancelled by user")
            raise
        except Exception as e:
            logger.error("Failed to fetch image from URL: %s", e)
            return None

    async def generate_image_with_fetch(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        model: Optional[str] = None,
        image_input: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[bytes]:
        """
        Generate image and fetch bytes if URL is returned (convenience method).

        This combines generate_image() and fetch_image_from_url() for common use case.

        Args:
            prompt: The text prompt for image generation.
            aspect_ratio: Aspect ratio for the image.
            model: Specific model to use.
            image_input: Optional base64-encoded image for image-to-image.
            cancel_event: AsyncIO event to check for cancellation.

        Returns:
            Image bytes on success, None on failure.
        """
        result = await self.generate_image(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            model=model,
            image_input=image_input,
            cancel_event=cancel_event,
        )

        if result is None:
            return None

        image_bytes, image_format = result

        # If result is a URL, fetch it
        if image_format == "url":
            url = image_bytes.decode("utf-8")
            return await self.fetch_image_from_url(url, cancel_event)

        return image_bytes

    # Abstract methods for tool support (not used for image generation)
    async def generate_chat_response(
        self,
        chat_session: Any,
        user_message: str,
        discord_message: Any,
        model_name: Optional[str] = None,
        tools: Optional[Any] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Not applicable for image service."""
        raise NotImplementedError("Image service does not support chat responses")

    async def send_tool_results(
        self,
        chat_session: Any,
        tool_rounds: List[Tuple[Any, Any]],
        tools: Optional[Any] = None,
        discord_message: Optional[Any] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Not applicable for image service."""
        raise NotImplementedError("Image service does not support tool results")

    def get_tools_for_provider(self, tools: List[Any]) -> Any:
        """Not applicable for image service."""
        return []

    def extract_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Not applicable for image service."""
        return []

    def format_function_results(self, results: List[Dict[str, Any]]) -> Any:
        """Not applicable for image service."""
        return []

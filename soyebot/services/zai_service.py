"""Z.AI API service for SoyeBot.

Supports both Standard API and Coding Plan API endpoints:
- Standard API: https://api.z.ai/api/paas/v4/ (pay-as-you-go, token-based)
- Coding Plan API: https://api.z.ai/api/coding/paas/v4/ (subscription-based, prompt-based)

Enable Coding Plan API by setting ZAI_CODING_PLAN=true in environment.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Tuple, Union, List, Dict

import discord
from openai import OpenAI, RateLimitError

from soyebot.config import AppConfig
from soyebot.services.base import BaseLLMService, ChatMessage
from soyebot.services.prompt_service import PromptService
from soyebot.utils import get_mime_type
from soyebot.tools.adapters.zai_adapter import ZAIToolAdapter

logger = logging.getLogger(__name__)


class ZAIChatSession:
    """Z.AI chat session with history management."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int,
        text_extractor,
    ):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._top_p = top_p
        self._max_messages = max_messages
        self._text_extractor = text_extractor
        self._history: Deque[ChatMessage] = deque(maxlen=max_messages)

    @property
    def history(self) -> list[ChatMessage]:
        """Get list of chat messages in history."""
        return list(self._history)

    @history.setter
    def history(self, new_history: list[ChatMessage]) -> None:
        """Replace history with a new list."""
        self._history.clear()
        self._history.extend(new_history)

    def _append_history(
        self,
        role: str,
        content: str,
        author_id: Optional[int] = None,
        author_name: Optional[str] = None,
        message_ids: list[str] = None,
    ) -> None:
        """Append a message to history if content is not empty."""
        if not content:
            return
        self._history.append(
            ChatMessage(
                role=role,
                content=content,
                author_id=author_id,
                author_name=author_name,
                message_ids=message_ids or [],
            )
        )

    def _create_user_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[list[str]] = None,
        images: list[bytes] = None,
    ) -> ChatMessage:
        """Create a ChatMessage for user input."""
        return ChatMessage(
            role="user",
            content=user_message,
            author_id=author_id,
            author_name=author_name,
            message_ids=message_ids or [],
            images=images or [],
        )

    def _build_messages_list(self, user_message: str, images: list[bytes]) -> list:
        """Build messages list for API call."""
        messages = []

        # Add system instruction
        if self._system_instruction:
            messages.append({"role": "system", "content": self._system_instruction})

        # Add history
        for msg in self._history:
            if msg.images:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for img_bytes in msg.images:
                    import base64

                    mime_type = get_mime_type(img_bytes)
                    b64_str = base64.b64encode(img_bytes).decode("utf-8")
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_str}"},
                        }
                    )
                messages.append({"role": msg.role, "content": content_blocks})
            else:
                messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        user_content = []
        if user_message:
            user_content.append({"type": "text", "text": user_message})

        if images:
            import base64

            for img_bytes in images:
                mime_type = get_mime_type(img_bytes)
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_str}"},
                    }
                )

        if user_content:
            messages.append({"role": "user", "content": user_content})

        return messages

    def generate_content(self, prompt: str):
        """Generate content for a single prompt without history (for meta-prompt generation)."""
        messages = []

        # Add system instruction
        if self._system_instruction:
            messages.append({"role": "system", "content": self._system_instruction})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        # Call API
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
        )

        # Extract and return response content
        return self._text_extractor(response)

    def send_message(
        self,
        user_message: str,
        author_id: int,
        author_name: Optional[str] = None,
        message_ids: Optional[list[str]] = None,
        images: list[bytes] = None,
    ):
        """Send message to Z.AI API and get response."""
        user_msg = self._create_user_message(
            user_message, author_id, author_name, message_ids, images
        )

        # Build messages list
        messages = self._build_messages_list(user_message, images)

        # Call API
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
        )

        # Extract response content
        message_content = self._text_extractor(response)
        model_msg = ChatMessage(role="assistant", content=message_content)

        return user_msg, model_msg, response


class ZAIService(BaseLLMService):
    """Z.AI Coding Plan API와의 모든 상호작용을 관리합니다."""

    def __init__(
        self,
        config: AppConfig,
        *,
        assistant_model_name: str,
        summary_model_name: Optional[str] = None,
        prompt_service: PromptService,
    ):
        super().__init__(config)
        # Initialize OpenAI client with Z.AI base URL and timeout
        self.client = OpenAI(
            api_key=config.zai_api_key,
            base_url=config.zai_base_url,
            timeout=config.api_request_timeout,
        )
        self._assistant_cache: dict[int, ZAIChatSession] = {}
        self._max_messages = 7
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

        # Preload default assistant model
        self.assistant_model = self._get_or_create_assistant(
            self._assistant_model_name,
            self.prompt_service.get_active_assistant_prompt(),
        )
        api_type = "Coding Plan" if self.config.zai_coding_plan else "Standard"
        logger.info(
            "Z.AI %s API 모델 '%s' 준비 완료 (endpoint: %s)",
            api_type,
            self._assistant_model_name,
            self.config.zai_base_url,
        )

    def _get_or_create_assistant(self, model_name: str, system_instruction: str):
        """Get or create a chat session for the given model."""
        key = hash((model_name, system_instruction))
        if key not in self._assistant_cache:
            self._assistant_cache[key] = ZAIChatSession(
                self.client,
                model_name,
                system_instruction,
                getattr(self.config, "temperature", 1.0),
                getattr(self.config, "top_p", 1.0),
                self._max_messages,
                self._extract_text_from_response,
            )
        return self._assistant_cache[key]

    def create_assistant_model(self, system_instruction: str, use_cache: bool = True):
        """Create a chat session with the given system instruction."""
        return self._get_or_create_assistant(
            self._assistant_model_name, system_instruction
        )

    def reload_parameters(self) -> None:
        """Reload parameters by clearing assistant cache."""
        self._assistant_cache.clear()
        api_type = "Coding Plan" if self.config.zai_coding_plan else "Standard"
        logger.info(
            "Z.AI %s API assistant cache cleared to apply new parameters.", api_type
        )

    def get_user_role_name(self) -> str:
        """Return role name for user messages."""
        return "user"

    def get_assistant_role_name(self) -> str:
        """Return role name for assistant messages."""
        return "assistant"

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if exception is a rate limit error."""
        if isinstance(error, RateLimitError):
            return True
        error_str = str(error).lower()
        return "rate limit" in error_str or "429" in error_str

    def _extract_retry_delay(self, error: Exception) -> Optional[float]:
        """Extract retry delay from error, if available."""
        error_str = str(error)
        # Try to find retry delay in error message
        import re

        match = re.search(r"please retry in (\d+)s", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        """Log raw request for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug(
                "[RAW Z.AI API REQUEST] User message preview: %r", user_message[:200]
            )
            if chat_session and hasattr(chat_session, "history"):
                history = chat_session.history
                formatted = []
                for msg in history[-5:]:
                    role = msg.role
                    content = str(msg.content)

                    author_label = str(msg.author_name or msg.author_id or "bot")
                    display_content = content
                    if msg.author_name and content.startswith(f"{msg.author_name}:"):
                        display_content = content[len(msg.author_name) + 1 :].strip()

                    truncated = display_content[:100].replace("\n", " ")
                    formatted.append(f"{role} (author:{author_label}) {truncated}")
                if formatted:
                    logger.debug(
                        "[RAW Z.AI API REQUEST] Recent history:\n%s",
                        "\n".join(formatted),
                    )
        except Exception:
            logger.exception("[RAW Z.AI API REQUEST] Error logging raw request")

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw response for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug("[RAW Z.AI API RESPONSE %s] %s", attempt, response_obj)
        except Exception:
            logger.exception(
                "[RAW Z.AI API RESPONSE %s] Error logging raw response", attempt
            )

    def _extract_text_from_response(self, response_obj: Any) -> str:
        """Extract text content from response object."""
        try:
            choices = getattr(response_obj, "choices", []) or []
            for choice in choices:
                message = getattr(choice, "message", None)
                if message and hasattr(message, "content") and message.content:
                    # Handle both string content and list of content blocks
                    content = message.content
                    if isinstance(content, str):
                        return content.strip()
                    elif isinstance(content, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                text_parts.append(block["text"])
                            elif hasattr(block, "text"):
                                text_parts.append(str(block.text))
                        return " ".join(text_parts).strip()
        except Exception:
            logger.exception("Failed to extract text from Z.AI response")
        return ""

    async def summarize_text(self, text: str) -> Optional[str]:
        """Summarize text using Z.AI model."""
        if not text.strip():
            return "요약할 메시지가 없습니다."

        prompt = f"Discord 대화 내용:\n{text}"
        return await self.execute_with_retry(
            lambda: self.client.chat.completions.create(
                model=self._summary_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompt_service.get_summary_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=getattr(self.config, "temperature", 1.0),
                top_p=getattr(self.config, "top_p", 1.0),
            ),
            "요약",
        )

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        discord_message: Union[discord.Message, list[discord.Message]],
        model_name: Optional[str] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Generate chat response."""
        self._log_raw_request(user_message, chat_session)

        if isinstance(discord_message, list):
            primary_msg = discord_message[0]
            message_ids = [str(m.id) for m in discord_message]
        else:
            primary_msg = discord_message
            message_ids = [str(discord_message.id)]

        author_id = primary_msg.author.id
        author_name = getattr(primary_msg.author, "name", str(author_id))

        # Check for model switch
        current_model_name = getattr(chat_session, "_model_name", None)
        if model_name and current_model_name != model_name:
            logger.info(
                "Switching Z.AI chat session model from %s to %s",
                current_model_name,
                model_name,
            )
            chat_session._model_name = model_name

        # Extract images from messages
        images = []
        if isinstance(discord_message, list):
            for msg in discord_message:
                imgs = await self._extract_images_from_message(msg)
                images.extend(imgs)
        else:
            images = await self._extract_images_from_message(discord_message)

        result = await self.execute_with_retry(
            lambda: chat_session.send_message(
                user_message,
                author_id,
                author_name=author_name,
                message_ids=message_ids,
                images=images,
            ),
            "응답 생성",
            return_full_response=True,
            discord_message=primary_msg,
        )

        if result is None:
            return None

        user_msg, model_msg, response = result

        # Update history safely
        chat_session._history.append(user_msg)
        chat_session._history.append(model_msg)

        return model_msg.content, response

    # Tool support methods
    def get_tools_for_provider(self, tools: List[Any]) -> Any:
        """Convert tool definitions to Z.AI format.

        Args:
            tools: List of ToolDefinition objects to convert.

        Returns:
            List of tool dictionaries in Z.AI function calling format.
        """
        return ZAIToolAdapter.convert_tools(tools)

    def extract_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract function calls from Z.AI response.

        Args:
            response: Z.AI response object (from chat.completions.create).

        Returns:
            List of function call dictionaries with 'id', 'name', and 'parameters'.
        """
        return ZAIToolAdapter.extract_function_calls(response)

    def format_function_results(self, results: List[Dict[str, Any]]) -> Any:
        """Format function results for sending back to Z.AI.

        Args:
            results: List of dicts with 'id', 'name', 'result', and optionally 'error'.

        Returns:
            List of message dictionaries in Z.AI tool format.
        """
        return ZAIToolAdapter.create_tool_messages(results)

"""OpenAI API service for SoyeBot."""

import json
import logging
import base64
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Tuple, Union

import discord
from openai import OpenAI, RateLimitError

from soyebot.config import AppConfig
from soyebot.services.base import BaseLLMService, ChatMessage
from soyebot.services.prompt_service import PromptService
from soyebot.utils import get_mime_type

logger = logging.getLogger(__name__)


class BaseOpenAISession:
    """Base class for OpenAI chat sessions."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int,
        service_tier: str,
        text_extractor,
    ):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._top_p = top_p
        self._max_messages = max_messages
        self._service_tier = service_tier
        self._text_extractor = text_extractor
        self._history: Deque[ChatMessage] = deque(maxlen=max_messages)

    @property
    def history(self):
        return list(self._history)

    @history.setter
    def history(self, new_history: list[ChatMessage]):
        """Setter to allow replacing the history."""
        self._history.clear()
        self._history.extend(new_history)

    def _append_history(self, role: str, content: str, author_id: Optional[int] = None, author_name: Optional[str] = None, message_ids: list[str] = None) -> None:
        if not content:
            return
        self._history.append(ChatMessage(role=role, content=content, author_id=author_id, author_name=author_name, message_ids=message_ids or []))

    def _create_user_message(self, user_message: str, author_id: int, author_name: Optional[str] = None, message_ids: Optional[list[str]] = None, images: list[bytes] = None) -> ChatMessage:
        return ChatMessage(
            role="user",
            content=user_message,
            author_id=author_id,
            author_name=author_name,
            message_ids=message_ids or [],
            images=images or []
        )


class ResponseChatSession(BaseOpenAISession):
    """Response API-backed chat session with a bounded context window."""

    def _build_input_payload(self) -> list:
        payload = []
        if self._system_instruction:
            payload.append(
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._system_instruction,
                        }
                    ],
                }
            )

        for entry in self._history:
            content_type = "output_text" if entry.role == "assistant" else "input_text"
            payload.append(
                {
                    "type": "message",
                    "role": entry.role,
                    "content": [
                        {
                            "type": content_type,
                            "text": entry.content,
                        }
                    ],
                }
            )
        return payload

    def send_message(self, user_message: str, author_id: int, author_name: Optional[str] = None, message_ids: Optional[list[str]] = None, images: list[bytes] = None):
        user_msg = self._create_user_message(user_message, author_id, author_name, message_ids, images)

        # We need to temporarily simulate the history having the user message
        # for the input payload construction, without mutating the permanent history state.
        current_payload = self._build_input_payload()

        # Build current message content
        content_list = []
        if user_message:
            content_list.append({
                "type": "input_text",
                "text": user_message
            })

        if images:
            logger.warning("Images provided to ResponseChatSession (OpenAI Responses API), but image support is not fully implemented for this endpoint. Ignoring images.")

        # Append user message to payload
        current_payload.append({
            "type": "message",
            "role": "user",
            "content": content_list
        })

        response = self._client.responses.create(
            model=self._model_name,
            input=current_payload,
            temperature=self._temperature,
            top_p=self._top_p,
            service_tier=self._service_tier,
        )

        message_content = self._text_extractor(response)
        model_msg = ChatMessage(role="assistant", content=message_content)

        return user_msg, model_msg, response


class ChatCompletionSession(BaseOpenAISession):
    """Chat Completion API-backed chat session for fine-tuned models."""

    def send_message(self, user_message: str, author_id: int, author_name: Optional[str] = None, message_ids: Optional[list[str]] = None, images: list[bytes] = None):
        user_msg = self._create_user_message(user_message, author_id, author_name, message_ids, images)

        # Build messages list for chat completions API
        messages = []
        if self._system_instruction:
            messages.append({"role": "system", "content": self._system_instruction})

        # Convert existing history to dicts
        api_history = []
        for msg in self._history:
            # Reconstruct content with images if present
            if msg.images:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})

                for img_bytes in msg.images:
                     b64_str = base64.b64encode(img_bytes).decode('utf-8')
                     mime_type = get_mime_type(img_bytes)
                     content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_str}"
                        }
                    })
                api_history.append({"role": msg.role, "content": content_blocks})
            else:
                api_history.append({"role": msg.role, "content": msg.content})

        messages.extend(api_history)

        # Add current message to API payload
        if images:
            content_list = []
            if user_message:
                content_list.append({"type": "text", "text": user_message})

            for img_bytes in images:
                # Convert bytes to base64
                b64_str = base64.b64encode(img_bytes).decode('utf-8')
                # Determine mime type
                mime_type = get_mime_type(img_bytes)

                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_str}"
                    }
                })
            messages.append({"role": "user", "content": content_list})
        else:
            messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            service_tier=self._service_tier,
        )

        # Extract text using the provided text extractor (but first try extracting from choice)
        message_content = ""
        if response.choices and response.choices[0].message.content:
             message_content = response.choices[0].message.content.strip()
        else:
             # Fallback to the service's text extractor if standard structure is missing
             message_content = self._text_extractor(response)

        model_msg = ChatMessage(role="assistant", content=message_content)

        return user_msg, model_msg, response


class _ChatCompletionModel:
    """Chat Completion API wrapper."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int,
        service_tier: str,
        text_extractor,
    ):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._top_p = top_p
        self._max_messages = max_messages
        self._service_tier = service_tier
        self._text_extractor = text_extractor

    def start_chat(self, system_instruction: str = None):
        return ChatCompletionSession(
            self._client,
            self._model_name,
            system_instruction or self._system_instruction,
            self._temperature,
            self._top_p,
            self._max_messages,
            self._service_tier,
            self._text_extractor,
        )


class _ResponseModel:
    """Response API wrapper mirroring the cached model API."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        temperature: float,
        top_p: float,
        max_messages: int,
        service_tier: str,
        text_extractor,
    ):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._top_p = top_p
        self._max_messages = max_messages
        self._service_tier = service_tier
        self._text_extractor = text_extractor

    def start_chat(self, system_instruction: str = None):
        return ResponseChatSession(
            self._client,
            self._model_name,
            system_instruction or self._system_instruction,
            self._temperature,
            self._top_p,
            self._max_messages,
            self._service_tier,
            self._text_extractor,
        )


class OpenAIService(BaseLLMService):
    """OpenAI API와의 모든 상호작용을 관리합니다."""

    def __init__(
        self,
        config: AppConfig,
        *,
        assistant_model_name: str,
        summary_model_name: Optional[str] = None,
        prompt_service: PromptService
    ):
        super().__init__(config)
        self.client = OpenAI(api_key=config.openai_api_key)
        self._assistant_cache: dict[int, Union[_ResponseModel, _ChatCompletionModel]] = {}
        self._max_messages = 7
        self._assistant_model_name = assistant_model_name
        self._summary_model_name = summary_model_name or assistant_model_name
        self.prompt_service = prompt_service

        # Preload default response model
        self.assistant_model = self._get_or_create_assistant(
            self._assistant_model_name, self.prompt_service.get_active_assistant_prompt()
        )
        logger.info("OpenAI Response 모델 '%s' 준비 완료.", self._assistant_model_name)

    def _get_or_create_assistant(self, model_name: str, system_instruction: str):
        key = hash((model_name, system_instruction))
        if key not in self._assistant_cache:
            # Select model wrapper based on configuration (Fine-tuned models use Chat Completions)
            # Check if the requested model matches the configured fine-tuned model
            use_finetuned_logic = (
                self.config.openai_finetuned_model and
                model_name == self.config.openai_finetuned_model
            )

            if use_finetuned_logic:
                 self._assistant_cache[key] = _ChatCompletionModel(
                    self.client,
                    model_name,
                    system_instruction,
                    getattr(self.config, 'temperature', 1.0),
                    getattr(self.config, 'top_p', 1.0),
                    self._max_messages,
                    "default",
                    self._extract_text_from_response_output,
                )
            else:
                self._assistant_cache[key] = _ResponseModel(
                    self.client,
                    model_name,
                    system_instruction,
                    getattr(self.config, 'temperature', 1.0),
                    getattr(self.config, 'top_p', 1.0),
                    self._max_messages,
                    getattr(self.config, 'service_tier', 'flex'),
                    self._extract_text_from_response_output,
                )
        return self._assistant_cache[key]

    def create_assistant_model(self, system_instruction: str):
        return self._get_or_create_assistant(self._assistant_model_name, system_instruction)

    def reload_parameters(self) -> None:
        """Reload parameters by clearing the assistant cache."""
        self._assistant_cache.clear()
        logger.info("OpenAI assistant cache cleared to apply new parameters.")

    def get_user_role_name(self) -> str:
        """Return the role name for user messages."""
        return "user"

    def get_assistant_role_name(self) -> str:
        """Return the role name for assistant messages."""
        return "assistant"

    def _is_rate_limit_error(self, error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True
        error_str = str(error).lower()
        return 'rate limit' in error_str or '429' in error_str

    def _log_raw_request(self, user_message: str, chat_session: Any = None) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug("[RAW API REQUEST] User message preview: %r", user_message[:200])
            if chat_session and hasattr(chat_session, 'history'):
                history = chat_session.history
                formatted = []
                for msg in history[-5:]:
                    role = msg.role
                    content = str(msg.content)
                    
                    # Clean up content display if it starts with "Name: "
                    author_label = str(msg.author_name or msg.author_id or "bot")
                    display_content = content
                    if msg.author_name and content.startswith(f"{msg.author_name}:"):
                        display_content = content[len(msg.author_name)+1:].strip()
                    
                    truncated = display_content[:100].replace("\n", " ")
                    formatted.append(f"{role} (author:{author_label}) {truncated}")
                if formatted:
                    logger.debug("[RAW API REQUEST] Recent history:\n%s", "\n".join(formatted))
        except Exception:
            logger.exception("[RAW API REQUEST] Error logging raw request")

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug("[RAW API RESPONSE %s] %s", attempt, response_obj)
        except Exception:
            logger.exception("[RAW API RESPONSE %s] Error logging raw response", attempt)

    def _extract_text_from_response(self, response_obj: Any) -> str:
        try:
            choices = getattr(response_obj, 'choices', []) or []
            for choice in choices:
                message = getattr(choice, 'message', None)
                if message and getattr(message, 'content', None):
                    return str(message.content).strip()
        except Exception:
            logger.exception("Failed to extract text from OpenAI response")

        return self._extract_text_from_response_output(response_obj)

    def _extract_text_from_response_output(self, response_obj) -> str:
        try:
            text_fragments = []
            seen_fragments = set()
            output_text = getattr(response_obj, 'output_text', None)
            if output_text:
                if isinstance(output_text, str):
                    normalized = str(output_text).strip()
                    if normalized and normalized not in seen_fragments:
                        text_fragments.append(normalized)
                        seen_fragments.add(normalized)
                else:
                    try:
                        for part in output_text:
                            normalized = str(part).strip()
                            if normalized and normalized not in seen_fragments:
                                text_fragments.append(normalized)
                                seen_fragments.add(normalized)
                    except TypeError:
                        normalized = str(output_text).strip()
                        if normalized and normalized not in seen_fragments:
                            text_fragments.append(normalized)
                            seen_fragments.add(normalized)

            output_items = getattr(response_obj, 'output', None) or []
            for item in output_items:
                content_list = getattr(item, 'content', None) or []
                for content in content_list:
                    text_value = getattr(content, 'text', None)
                    if text_value:
                        normalized = str(text_value).strip()
                        if normalized and normalized not in seen_fragments:
                            text_fragments.append(normalized)
                            seen_fragments.add(normalized)
            return "\n".join(text_fragments).strip()
        except Exception:
            logger.exception("Failed to extract text from response output")
        return ""


    async def summarize_text(self, text: str) -> Optional[str]:
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
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=getattr(self.config, 'temperature', 1.0),
                top_p=getattr(self.config, 'top_p', 1.0),
                service_tier=getattr(self.config, 'service_tier', 'flex'),
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
        self._log_raw_request(user_message, chat_session)

        if isinstance(discord_message, list):
            primary_msg = discord_message[0]
            message_ids = [str(m.id) for m in discord_message]
        else:
            primary_msg = discord_message
            message_ids = [str(discord_message.id)]

        author_id = primary_msg.author.id
        author_name = getattr(primary_msg.author, 'name', str(author_id))

        # Check for model switch
        # OpenAI sessions don't wrap a "factory" as cleanly as GeminiService (which has _ChatSession holding _factory)
        # In OpenAI service, we create a `ResponseChatSession` which holds `self._model_name`.
        # We need to recreate the session or update its internal model name if it changed.
        # But wait, `chat_session` is the OBJECT returned by `start_chat`.
        # `ResponseChatSession` has `self._model_name`.
        # If `model_name` is provided and differs, we should update it.

        current_model_name = getattr(chat_session, '_model_name', None)
        if model_name and current_model_name != model_name:
             logger.info("Switching OpenAI chat session model from %s to %s", current_model_name, model_name)
             chat_session._model_name = model_name
             # We might need to update other params if they depend on model, but usually shared.

        # Extract images from messages
        images = []
        if isinstance(discord_message, list):
            for msg in discord_message:
                imgs = await self._extract_images_from_message(msg)
                images.extend(imgs)
        else:
            images = await self._extract_images_from_message(discord_message)

        result = await self.execute_with_retry(
            lambda: chat_session.send_message(user_message, author_id, author_name=author_name, message_ids=message_ids, images=images),
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

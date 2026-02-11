"""Chat use case for handling chat generation logic.

This use case orchestrates the flow of generating chat responses,
including session management, tool execution, and response handling.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional, Union

import discord

from persbot.bot.session import ResolvedSession, SessionManager
from persbot.config import AppConfig
from persbot.domain import ModelAlias, Provider
from persbot.exceptions import (
    AbortSignalException,
    CancellationException,
    SessionException,
)
from persbot.services.cache_service import CacheService
from persbot.services.llm_service import LLMService
from persbot.services.model_usage_service import ModelUsageService
from persbot.services.prompt_service import PromptService
from persbot.tools import ToolManager

logger = logging.getLogger(__name__)


@dataclass
class ChatRequest:
    """Request for chat generation."""

    user_message: str
    discord_message: Union[discord.Message, list[discord.Message]]
    resolution: ResolvedSession
    use_summarizer_backend: bool = False
    cancel_event: Optional[asyncio.Event] = None


@dataclass
class ChatResponse:
    """Response from chat generation."""

    text: str
    session_key: str
    response_obj: Any
    images: list[bytes]
    notification: Optional[str] = None
    tool_rounds: int = 0


@dataclass
class StreamChunk:
    """A chunk of streamed response."""

    text: str
    is_final: bool = False
    metadata: dict[str, Any] = None


class ChatUseCase:
    """Use case for handling chat generation operations."""

    def __init__(
        self,
        config: AppConfig,
        llm_service: LLMService,
        session_manager: SessionManager,
        tool_manager: Optional[ToolManager] = None,
        prompt_service: Optional[PromptService] = None,
        cache_service: Optional[CacheService] = None,
    ):
        """Initialize the chat use case.

        Args:
            config: Application configuration.
            llm_service: LLM service for generating responses.
            session_manager: Session manager for chat sessions.
            tool_manager: Optional tool manager for function calling.
            prompt_service: Optional prompt service for prompts.
            cache_service: Optional cache service for context caching.
        """
        self.config = config
        self.llm_service = llm_service
        self.session_manager = session_manager
        self.tool_manager = tool_manager
        self.prompt_service = prompt_service
        self.cache_service = cache_service

        # Tool name translations for UI
        self._tool_labels: dict[str, str] = {
            "generate_image": "이미지 생성 도구",
            "send_image": "이미지 전송 도구",
            "get_time": "시간 확인 도구",
            "web_search": "웹 검색 도구",
            "get_weather": "날씨 확인 도구",
            "get_guild_info": "서버 정보 도구",
            "get_guild_roles": "서버 역할 도구",
            "get_guild_emojis": "서버 이모지 도구",
            "search_episodic_memory": "기억 검색 도구",
            "save_episodic_memory": "기억 저장 도구",
            "remove_episodic_memory": "기억 삭제 도구",
            "get_user_info": "사용자 정보 도구",
            "get_member_info": "멤버 정보 도구",
            "get_member_roles": "멤버 역할 도구",
            "inspect_external_content": "웹 콘텐츠 확인 도구",
            "get_channel_info": "채널 정보 도구",
            "get_channel_history": "채널 기록 도구",
            "get_message": "메시지 확인 도구",
            "list_channels": "채널 목록 도구",
            "check_virtual_routine_status": "루틴 상태 확인 도구",
            "get_routine_schedule": "루틴 일정 확인 도구",
            "generate_situational_snapshot": "상황 스냅샷 도구",
            "describe_scene_atmosphere": "장면 분위기 묘사 도구",
        }

    async def generate_chat_response(self, request: ChatRequest) -> Optional[ChatResponse]:
        """Generate a chat response for the given request.

        Args:
            request: The chat request containing message and context.

        Returns:
            ChatResponse if successful, None if generation failed.

        Raises:
            AbortSignalException: If operation is cancelled by user.
            SessionException: If session operations fail.
        """
        # Resolve message
        primary_message = self._get_primary_message(request.discord_message)

        # Get or create session
        chat_session, session_key = await self._get_or_create_session(
            primary_message, request.resolution
        )

        # Check for cancellation
        if request.cancel_event and request.cancel_event.is_set():
            raise CancellationException("Operation cancelled before generation")

        # Get tools if enabled
        tools = self._get_enabled_tools()

        # Check usage limits
        notification = await self._check_usage_limits(
            primary_message, chat_session
        )
        if notification and "초과" in notification:
            return ChatResponse(
                text=notification,
                session_key=session_key,
                response_obj=None,
                images=[],
                notification=notification,
            )

        # Generate response with tools
        response = await self._generate_with_tools(
            chat_session,
            request.user_message,
            request.discord_message,
            tools,
            request.use_summarizer_backend,
            request.cancel_event,
            primary_message,
        )

        if response:
            response.notification = notification

        return response

    async def generate_chat_response_stream(
        self, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming chat response.

        Args:
            request: The chat request containing message and context.

        Yields:
            StreamChunk objects as they are generated.

        Raises:
            AbortSignalException: If operation is cancelled by user.
        """
        # TODO: Implement streaming response
        # For now, fall back to non-streaming
        response = await self.generate_chat_response(request)
        if response:
            yield StreamChunk(text=response.text, is_final=True)

    async def regenerate_last_response(
        self,
        channel_id: int,
        session_key: str,
    ) -> Optional[ChatResponse]:
        """Regenerate the last response in a session.

        Args:
            channel_id: The Discord channel ID.
            session_key: The session key.

        Returns:
            ChatResponse if successful, None otherwise.
        """
        # Undo last exchange
        removed = self.session_manager.undo_last_exchanges(session_key, 1)
        if not removed:
            return None

        # Extract user content from removed messages
        user_content = await self._extract_user_content_from_removed(removed)
        if not user_content:
            return None

        # This is a simplified version - full implementation would need
        # the original Discord message context
        return None

    def _get_primary_message(
        self, discord_message: Union[discord.Message, list[discord.Message]]
    ) -> discord.Message:
        """Get the primary message from a message or list.

        Args:
            discord_message: Single message or list of messages.

        Returns:
            The primary (first) message.
        """
        if isinstance(discord_message, list):
            return discord_message[0]
        return discord_message

    async def _get_or_create_session(
        self,
        primary_message: discord.Message,
        resolution: ResolvedSession,
    ) -> tuple:
        """Get or create a chat session.

        Args:
            primary_message: The primary Discord message.
            resolution: The resolved session information.

        Returns:
            Tuple of (chat_session, session_key).
        """
        return await self.session_manager.get_or_create(
            user_id=primary_message.author.id,
            username=primary_message.author.name,
            session_key=resolution.session_key,
            channel_id=primary_message.channel.id,
            message_content=resolution.cleaned_message,
            message_ts=primary_message.created_at,
            message_id=str(primary_message.id),
        )

    def _get_enabled_tools(self) -> Optional[list]:
        """Get list of enabled tools.

        Returns:
            List of enabled tool definitions, or None if tools disabled.
        """
        if self.tool_manager and self.tool_manager.is_enabled():
            return list(self.tool_manager.get_enabled_tools().values())
        return None

    async def _check_usage_limits(
        self, message: discord.Message, chat_session: Any
    ) -> Optional[str]:
        """Check usage limits and return notification if needed.

        Args:
            message: The Discord message.
            chat_session: The chat session.

        Returns:
            Notification message if limit reached, None otherwise.
        """
        model_alias = getattr(
            chat_session, "model_alias",
            self.llm_service.model_usage_service.DEFAULT_MODEL_ALIAS
        )

        user_id, channel_id, guild_id, primary_author = self._extract_message_metadata(
            message
        )

        is_allowed, final_alias, notification = (
            await self.llm_service.model_usage_service.check_and_increment_usage(
                guild_id, model_alias
            )
        )

        if final_alias != model_alias:
            chat_session.model_alias = final_alias

        return notification if not is_allowed else None

    def _extract_message_metadata(self, message: discord.Message) -> tuple:
        """Extract metadata from a Discord message.

        Args:
            message: The Discord message.

        Returns:
            Tuple of (user_id, channel_id, guild_id, author).
        """
        primary_author = message.author
        user_id = primary_author.id
        channel_id = message.channel.id
        guild_id = message.guild.id if message.guild else user_id
        return user_id, channel_id, guild_id, primary_author

    async def _generate_with_tools(
        self,
        chat_session: Any,
        user_message: str,
        discord_message: Union[discord.Message, list[discord.Message]],
        tools: Optional[list],
        use_summarizer: bool,
        cancel_event: Optional[asyncio.Event],
        primary_message: discord.Message,
    ) -> Optional[ChatResponse]:
        """Generate response with tool execution if needed.

        Args:
            chat_session: The chat session.
            user_message: The user's message.
            discord_message: The Discord message(s).
            tools: List of available tools.
            use_summarizer: Whether to use summarizer backend.
            cancel_event: Cancellation event.
            primary_message: Primary Discord message.

        Returns:
            ChatResponse if successful, None otherwise.
        """
        session_key = getattr(chat_session, "session_key", "unknown")

        # Generate initial response
        result = await self.llm_service.generate_chat_response(
            chat_session,
            user_message,
            discord_message,
            use_summarizer_backend=use_summarizer,
            tools=tools,
            cancel_event=cancel_event,
        )

        if not result:
            return None

        response_text, response_obj = result

        # Handle tool calls if present
        if tools and self.tool_manager:
            return await self._handle_tool_calls(
                chat_session,
                response_text,
                response_obj,
                tools,
                discord_message,
                use_summarizer,
                cancel_event,
                primary_message,
                session_key,
            )

        return ChatResponse(
            text=response_text or "",
            session_key=session_key,
            response_obj=response_obj,
            images=[],
        )

    async def _handle_tool_calls(
        self,
        chat_session: Any,
        response_text: str,
        response_obj: Any,
        tools: list,
        discord_message: Union[discord.Message, list[discord.Message]],
        use_summarizer: bool,
        cancel_event: Optional[asyncio.Event],
        primary_message: discord.Message,
        session_key: str,
    ) -> ChatResponse:
        """Handle function calls in the response.

        Args:
            chat_session: The chat session.
            response_text: Initial response text.
            response_obj: Response object.
            tools: Available tools.
            discord_message: Discord message(s).
            use_summarizer: Whether to use summarizer.
            cancel_event: Cancellation event.
            primary_message: Primary Discord message.
            session_key: Session key.

        Returns:
            ChatResponse with final text and any generated images.
        """
        max_tool_rounds = 10
        tool_rounds = 0
        generated_images: list[bytes] = []

        active_backend = self.llm_service.get_active_backend(
            chat_session, use_summarizer_backend=use_summarizer
        )

        function_calls = self.llm_service.extract_function_calls_from_response(
            active_backend, response_obj
        )

        while function_calls and tool_rounds < max_tool_rounds:
            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                raise CancellationException("Tool execution cancelled")

            logger.info(
                "Tool round %d: %d calls",
                tool_rounds + 1,
                len(function_calls),
            )

            # Send progress notification
            progress_msg = await self._send_tool_progress(
                function_calls, primary_message.channel
            )

            try:
                # Execute tools
                results = await self.tool_manager.execute_tools(
                    function_calls, primary_message, cancel_event
                )

                # Collect images
                for result_item in results:
                    if result_item.get("image_bytes"):
                        generated_images.append(result_item["image_bytes"])

                # Send results back to LLM
                tool_rounds_list = [(response_obj, results)]
                continuation = await self.llm_service.send_tool_results(
                    chat_session,
                    tool_rounds=tool_rounds_list,
                    tools=tools,
                    discord_message=primary_message,
                    cancel_event=cancel_event,
                )

                if not continuation:
                    break

                response_text, response_obj = continuation

                # Check for more function calls
                function_calls = self.llm_service.extract_function_calls_from_response(
                    active_backend, response_obj
                )
                tool_rounds += 1

            except Exception as e:
                logger.error("Tool execution error: %s", e, exc_info=True)
                break
            finally:
                # Clean up progress message
                if progress_msg:
                    try:
                        await progress_msg.delete()
                    except Exception:
                        pass

        return ChatResponse(
            text=response_text or "",
            session_key=session_key,
            response_obj=response_obj,
            images=generated_images,
            tool_rounds=tool_rounds,
        )

    async def _send_tool_progress(
        self, function_calls: list, channel: discord.abc.Messageable
    ) -> Optional[discord.Message]:
        """Send a progress notification for tool execution.

        Args:
            function_calls: List of function calls being executed.
            channel: Discord channel to send to.

        Returns:
            The sent message, or None if sending failed.
        """
        try:
            names = [
                self._tool_labels.get(call.get("name", "unknown"), call.get("name", "unknown"))
                for call in function_calls
            ]
            notification = f"🔧 {', '.join(names)} 사용 중..."
            return await channel.send(notification)
        except Exception:
            return None

    async def _extract_user_content_from_removed(self, removed: list) -> Optional[str]:
        """Extract user message content from removed session history.

        Args:
            removed: List of removed history entries.

        Returns:
            Extracted user content, or None if not found.
        """
        # TODO: Implement content extraction from removed history
        return None

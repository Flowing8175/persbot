"""Shared helpers for Discord chat-driven cogs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import discord

from persbot.bot.session import ResolvedSession, SessionManager
from persbot.services.llm_service import LLMService
from persbot.utils import smart_split

if TYPE_CHECKING:
    from persbot.tools import ToolManager

logger = logging.getLogger(__name__)

__all__ = [
    "ChatReply",
    "resolve_session_for_message",
    "create_chat_reply",
    "send_split_response",
]


@dataclass(frozen=True)
class ChatReply:
    """Container for an LLM response tied to a session."""

    text: str
    session_key: str
    response: object
    images: list[bytes] = field(default_factory=list)


async def resolve_session_for_message(
    message: discord.Message,
    content: str,
    *,
    session_manager: SessionManager,
    cancel_event: Optional[asyncio.Event] = None,
) -> Optional[ResolvedSession]:
    """Resolve the logical chat session for a Discord message."""

    is_reply_to_summary = False
    # Handle replies by adding context to the message content
    # We no longer branch sessions for replies; instead, we just provide context to the LLM.
    if message.reference and message.reference.message_id:
        ref_msg = message.reference.resolved

        # If not resolved or deleted, try to fetch it
        if ref_msg is None or isinstance(ref_msg, discord.DeletedReferencedMessage):
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
            except (discord.NotFound, discord.HTTPException):
                ref_msg = None

        if ref_msg:
            # Add reply context to the content
            # Using clean_content to get readable names instead of mention IDs
            ref_text = ref_msg.clean_content
            reply_context = f'(답장 대상: {ref_msg.author.id}, 내용: "{ref_text}")\n'
            content = reply_context + content

            # Check if this is a reply to a summary message
            # Summary messages typically start with "**... 요약:**"
            # We also check if it's from the bot itself
            if ref_msg.author.bot and "요약:**" in ref_text:
                is_reply_to_summary = True

    resolution = await session_manager.resolve_session(
        channel_id=message.channel.id,
        author_id=message.author.id,
        username=message.author.name,
        message_id=str(message.id),
        message_content=content,
        # Pass None for reference_message_id to ensure we stick to the channel session
        reference_message_id=None,
        created_at=message.created_at,
        cancel_event=cancel_event,
    )
    if resolution:
        resolution.is_reply_to_summary = is_reply_to_summary
    return resolution if resolution.cleaned_message else None


async def create_chat_reply(
    message: Union[discord.Message, list[discord.Message]],
    *,
    resolution: ResolvedSession,
    llm_service: LLMService,
    session_manager: SessionManager,
    tool_manager: Optional["ToolManager"] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> Optional[ChatReply]:
    """Create or reuse a chat session and fetch an LLM reply."""

    # Determine primary message for metadata
    if isinstance(message, list):
        primary_msg = message[0]
        # We don't link message_id here for get_or_create because it's for session resumption
        # The actual message linking happens in generate_chat_response
        msg_id_for_session = str(primary_msg.id)
    else:
        primary_msg = message
        msg_id_for_session = str(message.id)

    chat_session, session_key = await session_manager.get_or_create(
        user_id=primary_msg.author.id,
        username=primary_msg.author.name,
        session_key=resolution.session_key,
        channel_id=primary_msg.channel.id,
        message_content=resolution.cleaned_message,
        message_ts=primary_msg.created_at,
        message_id=msg_id_for_session,
    )

    # Note: chat_session now has .model_alias set by get_or_create (via session persistence or default)
    # llm_service.generate_chat_response will read this alias from chat_session.

    # Get tools from ToolManager if available
    tools = None
    if tool_manager and tool_manager.is_enabled():
        # get_enabled_tools() returns a dict, but we need a list
        tools = list(tool_manager.get_enabled_tools().values())

    response_result = await llm_service.generate_chat_response(
        chat_session,
        resolution.cleaned_message,
        message,  # This can be a list of messages if AutoChannelCog passes it, but type hint says discord.Message
        use_summarizer_backend=resolution.is_reply_to_summary,
        tools=tools,
        cancel_event=cancel_event,
    )

    if not response_result:
        return None

    response_text, response_obj = response_result

    # Check for function calls in the response
    if tool_manager and tool_manager.is_enabled():
        function_calls = llm_service.extract_function_calls_from_response(
            llm_service.assistant_backend, response_obj
        )

        # Loop to handle function calls until LLM stops calling tools
        max_tool_rounds = 10  # Prevent infinite loops
        tool_rounds = 0
        generated_images = []  # Collect all images across tool rounds

        while function_calls and tool_rounds < max_tool_rounds:
            logger.info(
                "Detected %d function calls in response (round %d)",
                len(function_calls),
                tool_rounds + 1,
            )

            try:
                # Execute tools in parallel
                results = await tool_manager.execute_tools(function_calls, primary_msg)
                logger.info(
                    "Executed %d tools: %s",
                    len(results),
                    [r.get("name") for r in results],
                )

                # Collect any generated images to send after LLM response
                for result_item in results:
                    image_bytes = result_item.get("image_bytes")
                    if image_bytes:
                        generated_images.append(image_bytes)

                # Send tool results back to LLM and get continuation
                # Create a new round with (response_obj, tool_results)
                tool_results_list = [(response_obj, results)]

                # Use send_tool_results to get continuation response
                continuation = await llm_service.send_tool_results(
                    chat_session,
                    tool_rounds=tool_results_list,
                    tools=tools,
                    discord_message=primary_msg,
                    cancel_event=cancel_event,
                )

                if not continuation:
                    logger.warning("Tool results sent but no continuation received from LLM")
                    break

                # Update response_text and response_obj from continuation
                response_text, response_obj = continuation

                # Check for more function calls in the continuation
                function_calls = llm_service.extract_function_calls_from_response(
                    llm_service.get_active_backend(
                        chat_session,
                        use_summarizer_backend=resolution.is_reply_to_summary,
                    ),
                    response_obj,
                )

                tool_rounds += 1

            except Exception as e:
                logger.error("Error executing tools: %s", e, exc_info=True)
                # Return original response on tool execution error
                break

        if tool_rounds > 0:
            # We executed tools, use the final response from the LLM
            return ChatReply(
                text=response_text or "",
                session_key=session_key,
                response=response_obj,
                images=generated_images,
            )

    return ChatReply(
        text=response_text or "",
        session_key=session_key,
        response=response_obj,
        images=[],
    )


async def send_split_response(
    channel: discord.abc.Messageable, reply: ChatReply, session_manager: SessionManager
):
    """
    Shared utility to split and send a response line by line.
    Handles cancellation by undoing the last exchange in session history.
    """
    try:
        # First, split by existing newlines to preserve the "line by line" feel
        initial_lines = reply.text.split("\n")
        final_lines = []

        for line in initial_lines:
            if not line.strip():
                final_lines.append("")
                continue

            # If a line is too long, smart_split it
            if len(line) > 1900:
                final_lines.extend(smart_split(line))
            else:
                final_lines.append(line)

        for line in final_lines:
            if not line.strip():
                continue

            # Calculate delay: proportional to length, clamped 0.1s - 1.7s
            delay = max(0.1, min(1.7, len(line) * 0.05))

            # Send typing status while waiting
            async with channel.typing():
                await asyncio.sleep(delay)
                sent_msg = await channel.send(line)

                # Link message to session
                session_manager.link_message_to_session(str(sent_msg.id), reply.session_key)

        # Send any generated images as attachments
        if reply.images:
            import io

            for img_bytes in reply.images:
                async with channel.typing():
                    img_file = discord.File(io.BytesIO(img_bytes), filename="generated_image.png")
                    img_msg = await channel.send(file=img_file)

                    # Link image message to session
                    session_manager.link_message_to_session(str(img_msg.id), reply.session_key)

    except asyncio.CancelledError:
        logger.info(f"Sending interrupted for channel {channel.id}.")
        raise  # Re-raise to signal cancellation

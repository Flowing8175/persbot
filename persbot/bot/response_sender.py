"""Response sending for Discord bot interactions.

This module handles sending responses to Discord channels with proper
formatting, splitting, and session tracking.
"""

import asyncio
import io
import logging

import discord

from persbot.bot.chat_models import ChatReply
from persbot.bot.session import SessionManager
from persbot.constants import MessageConfig
from persbot.utils import smart_split

logger = logging.getLogger(__name__)


async def send_split_response(
    channel: discord.abc.Messageable,
    reply: ChatReply,
    session_manager: SessionManager,
):
    """Send a response line by line with proper formatting.

    This function respects asyncio cancellation and will stop sending
    immediately when a new message arrives (like pressing STOP).

    Args:
        channel: The Discord channel to send to.
        reply: The chat reply to send.
        session_manager: The session manager for linking messages.

    Raises:
        asyncio.CancelledError: If sending is interrupted.
    """
    try:
        # Get the display text (with notification if present)
        text = reply.display_text

        # First, split by existing newlines to preserve the "line by line" feel
        initial_lines = text.split("\n")
        final_lines = []

        for line in initial_lines:
            if not line.strip():
                final_lines.append("")
                continue

            # If a line is too long, smart_split it
            if len(line) > MessageConfig.MAX_SPLIT_LENGTH:
                final_lines.extend(smart_split(line))
            else:
                final_lines.append(line)

        for line in final_lines:
            if not line.strip():
                continue

            # Calculate delay: proportional to length, clamped 0.1s - 1.7s
            delay = max(
                MessageConfig.TYPING_DELAY_MIN,
                min(
                    MessageConfig.TYPING_DELAY_MAX,
                    len(line) * MessageConfig.TYPING_DELAY_MULTIPLIER,
                ),
            )

            # Send typing status while waiting
            async with channel.typing():
                await asyncio.sleep(delay)
                sent_msg = await channel.send(line)

                # Link message to session
                session_manager.link_message_to_session(str(sent_msg.id), reply.session_key)

        # Send any generated images as attachments
        if reply.images:
            for img_bytes in reply.images:
                async with channel.typing():
                    img_file = discord.File(fp=io.BytesIO(img_bytes), filename="generated_image.png")
                    img_msg = await channel.send(file=img_file)

                    # Link image message to session
                    session_manager.link_message_to_session(str(img_msg.id), reply.session_key)

    except asyncio.CancelledError:
        raise  # Re-raise to signal cancellation


async def send_immediate_response(
    channel: discord.abc.Messageable,
    text: str,
    session_key: str,
    session_manager: SessionManager,
) -> discord.Message:
    """Send an immediate response without line-by-line delays.

    Args:
        channel: The Discord channel to send to.
        text: The text to send.
        session_key: The session key for linking.
        session_manager: The session manager.

    Returns:
        The sent message.
    """
    sent_msg = await channel.send(text)
    session_manager.link_message_to_session(str(sent_msg.id), session_key)
    return sent_msg


async def send_with_images(
    channel: discord.abc.Messageable,
    text: str,
    images: list[bytes],
    session_key: str,
    session_manager: SessionManager,
) -> list[discord.Message]:
    """Send a response with image attachments.

    Args:
        channel: The Discord channel to send to.
        text: The text content.
        images: List of image bytes.
        session_key: The session key for linking.
        session_manager: The session manager.

    Returns:
        List of sent messages.
    """
    sent_messages = []

    # Send text first
    if text:
        msg = await channel.send(text)
        session_manager.link_message_to_session(str(msg.id), session_key)
        sent_messages.append(msg)

    # Send images
    for i, img_bytes in enumerate(images):
        async with channel.typing():
            img_file = discord.File(fp=io.BytesIO(img_bytes), filename="generated_image_{}.png".format(i))
            img_msg = await channel.send(file=img_file)
            session_manager.link_message_to_session(str(img_msg.id), session_key)
            sent_messages.append(img_msg)

    return sent_messages

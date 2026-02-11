"""Utility classes and functions for SoyeBot."""

import io
import logging
import re
from typing import Any, Optional, Union

import discord
from PIL import Image

GENERIC_ERROR_MESSAGE = "❌ 봇 내부에서 예상치 못한 오류가 발생했어요. 개발자에게 문의해주세요."

# Error Constants
ERROR_API_TIMEOUT = "❌ API 요청 시간이 초과되었습니다."
ERROR_API_QUOTA_EXCEEDED = "❌ API 사용량이 초과되었습니다."
ERROR_RATE_LIMIT = "⏳ 뇌 과부하! 잠시만 기다려주세요."
ERROR_PERMISSION_DENIED = "❌ 권한이 없습니다."
ERROR_INVALID_ARGUMENT = "❌ 잘못된 인자입니다."

logger = logging.getLogger(__name__)


def smart_split(text: str, max_length: int = 1900) -> list[str]:
    """
    Intelligently split text into chunks of at most max_length.
    Prefers splitting at double newlines, then single newlines, then spaces.
    """
    if len(text) <= max_length:
        return [text]

    def find_split_point(text_to_split: str, length: int) -> int:
        """Find the best split point in text using early returns."""
        # Try double newline first
        split_pos = text_to_split.rfind("\n\n", 0, length)
        if split_pos != -1:
            return split_pos + 2

        # Try single newline
        split_pos = text_to_split.rfind("\n", 0, length)
        if split_pos != -1:
            return split_pos + 1

        # Try space
        split_pos = text_to_split.rfind(" ", 0, length)
        if split_pos != -1:
            return split_pos + 1

        # Hard cut at max_length
        return length

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        split_at = find_split_point(text, max_length)
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    return chunks


async def _send_to_interaction(
    target: discord.Interaction,
    chunk: str,
    chunk_index: int,
    current_kwargs: dict[str, Any],
) -> discord.Message:
    """Send message to Discord Interaction.

    Args:
        target: The Discord interaction to send to.
        chunk: The message content to send.
        chunk_index: The index of this chunk in the message.
        current_kwargs: Additional keyword arguments for the send.

    Returns:
        The sent message.
    """
    if chunk_index == 0:
        if target.response.is_done():
            return await target.followup.send(chunk, **current_kwargs)
        await target.response.send_message(chunk, **current_kwargs)
        return await target.original_response()
    return await target.followup.send(chunk, **current_kwargs)


async def _send_to_message(
    target: discord.Message,
    chunk: str,
    chunk_index: int,
    mention_author: bool,
    current_kwargs: dict[str, Any],
) -> discord.Message:
    """Send message to Discord Message with reply fallback.

    Args:
        target: The Discord message to reply to.
        chunk: The message content to send.
        chunk_index: The index of this chunk in the message.
        mention_author: Whether to mention the author.
        current_kwargs: Additional keyword arguments for the send.

    Returns:
        The sent message.
    """
    if chunk_index == 0:
        try:
            return await target.reply(chunk, mention_author=mention_author, **current_kwargs)
        except (discord.NotFound, discord.HTTPException) as reply_error:
            if "Unknown message" in str(reply_error):
                logger.debug(f"Original message not found, sending to channel instead")
                return await target.channel.send(chunk, **current_kwargs)
            raise
    return await target.channel.send(chunk, **current_kwargs)


async def _send_to_messageable(
    target: Any,
    chunk: str,
    current_kwargs: dict[str, Any],
) -> discord.Message:
    """Send message to any target with a send() method.

    Args:
        target: Any object with a send() method.
        chunk: The message content to send.
        current_kwargs: Additional keyword arguments for the send.

    Returns:
        The sent message.
    """
    return await target.send(chunk, **current_kwargs)


async def send_discord_message(
    target: Union[
        discord.abc.Messageable,
        discord.Message,
        discord.Interaction,
        Any,  # commands.Context
    ],
    content: str,
    **kwargs: Any,
) -> list[discord.Message]:
    """
    Unified method to send Discord messages with automatic splitting.

    Args:
        target: discord.abc.Messageable, discord.Message, discord.Interaction,
                or commands.Context.
        content: The message content to send.
        **kwargs: Additional arguments to pass to the send method.

    Returns:
        List of sent messages.
    """
    if not content and not any(k in kwargs for k in ("embed", "embeds", "file", "files", "view")):
        return []

    chunks = smart_split(content) if content else [""]
    sent_messages = []
    mention_author = kwargs.pop("mention_author", False)
    last_index = len(chunks) - 1

    for i, chunk in enumerate(chunks):
        current_kwargs = kwargs.copy()

        if i > 0:
            current_kwargs.pop("reference", None)

        if i < last_index:
            for key in ["embed", "embeds", "view", "file", "files", "delete_after"]:
                current_kwargs.pop(key, None)

        try:
            if isinstance(target, discord.Interaction):
                msg = await _send_to_interaction(target, chunk, i, current_kwargs)
            elif isinstance(target, discord.Message):
                msg = await _send_to_message(target, chunk, i, mention_author, current_kwargs)
            elif hasattr(target, "send"):
                msg = await _send_to_messageable(target, chunk, current_kwargs)
            else:
                logger.error(f"Unsupported target type for send_discord_message: {type(target)}")
                break

            if msg:
                sent_messages.append(msg)
        except Exception as e:
            logger.error(f"Error sending message chunk {i}: {e}")
            break

    return sent_messages


_TIME_TOKEN_PATTERN = re.compile(r"(\d+)\s*(시간|분)")


class DiscordUI:
    """Discord UI 상호작용을 위한 헬퍼 클래스"""

    @staticmethod
    async def safe_send(channel: discord.abc.Messageable, content: str) -> Optional[discord.Message]:
        """Send a message to a channel with error handling.

        Args:
            channel: The Discord channel or messageable to send to.
            content: The message content to send.

        Returns:
            The sent message, or None if sending failed.
        """
        try:
            return await channel.send(content)
        except discord.Forbidden:
            logger.warning(f"{getattr(channel, 'name', 'channel')}에 메시지 전송 실패 (권한 부족)")
            return None
        except Exception as e:
            logger.error(f"{getattr(channel, 'name', 'channel')}에 메시지 전송 중 에러: {e}")
            return None


def parse_korean_time(time_str: str) -> Optional[int]:
    """Convert strings such as '1시간30분' into minutes.

    Args:
        time_str: A string containing time units like '1시간30분'.

    Returns:
        The total number of minutes, or None if parsing failed.
    """

    if not time_str:
        return None

    units = {"시간": 60, "분": 1}
    total_minutes = 0

    for value, unit in _TIME_TOKEN_PATTERN.findall(time_str):
        total_minutes += int(value) * units[unit]

    return total_minutes or None


def extract_message_content(message: discord.Message) -> str:
    """메시지에서 봇 mention을 제거한 내용을 추출합니다.

    Args:
        message: The Discord message to extract content from.

    Returns:
        The message content with bot mentions removed.
    """
    user_message = message.content
    for mention in message.mentions:
        user_message = user_message.replace(f"<@{mention.id}>", "")
        user_message = user_message.replace(f"<@!{mention.id}>", "")
    return user_message.strip()


def get_mime_type(data: bytes) -> str:
    """Detect basic image mime type from bytes.

    Args:
        data: Raw image bytes.

    Returns:
        The detected MIME type (e.g., 'image/jpeg').
    """
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    elif data.startswith(b"\x89PNG"):
        return "image/png"
    elif data.startswith(b"GIF8"):
        return "image/gif"
    elif data.startswith(b"RIFF") and b"WEBP" in data[:20]:
        return "image/webp"
    return "image/jpeg"  # Default fallback


def process_image_sync(image_data: bytes, filename: str) -> bytes:
    """Process image synchronously with Pillow (CPU-bound).

    Downscale images to ~1MP to reduce API payload size while maintaining quality.

    Args:
        image_data: Raw image bytes to process.
        filename: Original filename for logging purposes.

    Returns:
        Processed image bytes (JPEG format, downscaled if needed).

    Raises:
        Exception: If image processing fails (returns original data on error).
    """
    target_pixels = 1_000_000  # 1 Megapixel
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            # Check current dimensions
            width, height = img.size
            pixels = width * height

            if pixels > target_pixels:
                # Calculate scaling factor
                ratio = (target_pixels / pixels) ** 0.5
                new_width = int(width * ratio)
                new_height = int(height * ratio)

                logger.info(
                    "Downscaling image %s from %dx%d to %dx%d",
                    filename,
                    width,
                    height,
                    new_width,
                    new_height,
                )

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to JPEG (compatible and efficient)
            output_buffer = io.BytesIO()
            # Convert to RGB if needed (e.g. RGBA -> RGB for JPEG)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img.save(output_buffer, format="JPEG", quality=85)
            return output_buffer.getvalue()
    except Exception as img_err:
        logger.error("Failed to process image %s: %s", filename, img_err)
        # Fallback to original if processing fails
        return image_data

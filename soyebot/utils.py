"""Utility classes and functions for SoyeBot."""

import logging
import re
from typing import Optional

import discord

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

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try to find a good split point
        split_at = -1
        # 1. Double newline
        split_at = text.rfind("\n\n", 0, max_length)
        if split_at != -1:
            split_at += 2 # Include the newlines in the current chunk or just split after them?
            # Actually, splitting AFTER the newlines is cleaner.
        else:
            # 2. Single newline
            split_at = text.rfind("\n", 0, max_length)
            if split_at != -1:
                split_at += 1
            else:
                # 3. Space
                split_at = text.rfind(" ", 0, max_length)
                if split_at != -1:
                    split_at += 1
                else:
                    # 4. Hard cut
                    split_at = max_length

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    return chunks

async def send_discord_message(target, content: str, **kwargs) -> list[discord.Message]:
    """
    Unified method to send Discord messages with automatic splitting.
    target: discord.abc.Messageable, discord.Message, discord.Interaction, or commands.Context
    """
    # If no content and no other sendable items, return empty list
    if not content and not any(k in kwargs for k in ('embed', 'embeds', 'file', 'files', 'view')):
        return []

    chunks = smart_split(content) if content else [""]
    sent_messages = []

    # Filter kwargs for specific send methods
    is_reply = isinstance(target, discord.Message)
    mention_author = kwargs.pop("mention_author", False)
    
    last_index = len(chunks) - 1
    
    for i, chunk in enumerate(chunks):
        # Create a copy of kwargs for the current chunk
        current_kwargs = kwargs.copy()
        
        # 1. References/Reply context apply only to the FIRST chunk
        if i > 0:
            current_kwargs.pop('reference', None)

        # 2. Visuals/Actions (Embeds, Views, Files) apply only to the LAST chunk for better UX
        if i < last_index:
            for key in ['embed', 'embeds', 'view', 'file', 'files', 'delete_after']:
                current_kwargs.pop(key, None)

        try:
            if isinstance(target, discord.Interaction):
                if i == 0:
                    if target.response.is_done():
                        msg = await target.followup.send(chunk, **current_kwargs)
                    else:
                        await target.response.send_message(chunk, **current_kwargs)
                        msg = await target.original_response()
                else:
                    msg = await target.followup.send(chunk, **current_kwargs)
            elif isinstance(target, discord.Message):
                if i == 0:
                    msg = await target.reply(chunk, mention_author=mention_author, **current_kwargs)
                else:
                    msg = await target.channel.send(chunk, **current_kwargs)
            elif hasattr(target, "send"): # Context or Messageable
                msg = await target.send(chunk, **current_kwargs)
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
    async def safe_send(channel: discord.TextChannel, content: str) -> Optional[discord.Message]:
        try:
            return await channel.send(content)
        except discord.Forbidden:
            logger.warning(f"{channel.name}에 메시지 전송 실패 (권한 부족)")
            return None
        except Exception as e:
            logger.error(f"{channel.name}에 메시지 전송 중 에러: {e}")
            return None

def parse_korean_time(time_str: str) -> Optional[int]:
    """Convert strings such as '1시간30분' into minutes."""

    if not time_str:
        return None

    units = {"시간": 60, "분": 1}
    total_minutes = 0

    for value, unit in _TIME_TOKEN_PATTERN.findall(time_str):
        total_minutes += int(value) * units[unit]

    return total_minutes or None

def extract_message_content(message: discord.Message) -> str:
    """메시지에서 봇 mention을 제거한 내용을 추출합니다."""
    user_message = message.content
    for mention in message.mentions:
        user_message = user_message.replace(f"<@{mention.id}>", "")
        user_message = user_message.replace(f"<@!{mention.id}>", "")
    return user_message.strip()

def get_mime_type(data: bytes) -> str:
    """Detect basic image mime type from bytes."""
    if data.startswith(b'\xff\xd8'):
        return "image/jpeg"
    elif data.startswith(b'\x89PNG'):
        return "image/png"
    elif data.startswith(b'GIF8'):
        return "image/gif"
    elif data.startswith(b'RIFF') and b'WEBP' in data[:20]:
        return "image/webp"
    return "image/jpeg" # Default fallback

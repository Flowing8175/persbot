"""Utility classes and functions for SoyeBot."""

import discord
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DiscordUI:
    """Discord UI 상호작용을 위한 헬퍼 클래스"""
    @staticmethod
    async def safe_edit(message: Optional[discord.Message], content: str) -> bool:
        if not message:
            return False
        try:
            await message.edit(content=content)
            return True
        except (discord.Forbidden, discord.NotFound) as e:
            logger.warning(f"메시지 수정 실패: {e}")
            return False
        except Exception as e:
            logger.warning(f"메시지 수정 중 에러: {e}")
            return False

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
    """'1시간30분' 같은 한국어 시간을 분으로 변환합니다."""
    time_str = time_str.strip()
    total_minutes = 0
    hour_match = re.search(r'(\d+)\s*시간', time_str)
    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60
    minute_match = re.search(r'(\d+)\s*분', time_str)
    if minute_match:
        total_minutes += int(minute_match.group(1))
    return total_minutes if total_minutes > 0 else None

def extract_message_content(message: discord.Message) -> str:
    """메시지에서 봇 mention을 제거한 내용을 추출합니다."""
    user_message = message.content
    for mention in message.mentions:
        user_message = user_message.replace(f"<@{mention.id}>", "")
        user_message = user_message.replace(f"<@!{mention.id}>", "")
    return user_message.strip()

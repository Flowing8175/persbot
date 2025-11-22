"""Utility classes and functions for SoyeBot."""

import logging
import re
from typing import Optional

import discord

GENERIC_ERROR_MESSAGE = "❌ 봇 내부에서 예상치 못한 오류가 발생했어요. 개발자에게 문의해주세요."

logger = logging.getLogger(__name__)

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

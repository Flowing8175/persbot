"""Configuration loader for SoyeBot."""

import os
import sys
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

# --- 로딩 및 기본 설정 ---
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- 설정 ---
@dataclass(frozen=True)
class AppConfig:
    """애플리케이션 설정"""
    discord_token: str
    gemini_api_key: str
    model_name: str = 'gemini-2.5-flash-lite'
    max_messages_per_fetch: int = 300
    api_max_retries: int = 2
    api_rate_limit_retry_after: int = 5
    api_request_timeout: int = 30
    api_retry_backoff_base: float = 2.0  # Exponential backoff base
    api_retry_backoff_max: float = 32.0  # Max backoff cap (seconds)
    progress_update_interval: float = 0.5
    countdown_update_interval: int = 5
    max_session_records: int = 2
    max_tracked_message_ids: int = 800
    command_prefix: str = '!'

    # --- Database Configuration ---
    database_path: str = 'soyebot.db'
    # Gemini/LLM model tuning
    # Temperature controls creativity (0.0 = deterministic, higher = more creative)
    temperature: float = 1.0

def load_config() -> AppConfig:
    """환경 변수에서 설정을 로드합니다."""
    discord_token = os.environ.get('DISCORD_TOKEN')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

    if not discord_token or not gemini_api_key:
        logger.error("에러: DISCORD_TOKEN 또는 GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    return AppConfig(discord_token=discord_token, gemini_api_key=gemini_api_key)

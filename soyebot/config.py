"""Configuration loader for SoyeBot."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# --- 로딩 및 기본 설정 ---
# Load the local .env file from the project root to populate environment
# variables. We intentionally avoid find_dotenv because the project is typically
# run from the repository root and we want predictable loading behavior.
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)
    logging.getLogger(__name__).debug("Loaded environment variables from %s", _dotenv_path)
else:
    logging.getLogger(__name__).debug("No .env file found; relying on existing environment")

logging.basicConfig(
    level=logging.INFO,
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
    model_name: str = 'gemini-2.5-flash'
    eval_model_name: str = 'gemini-2.5-flash-lite'
    max_messages_per_fetch: int = 300
    api_max_retries: int = 2
    api_rate_limit_retry_after: int = 5
    api_request_timeout: int = 30
    api_retry_backoff_base: float = 2.0  # Exponential backoff base
    api_retry_backoff_max: float = 32.0  # Max backoff cap (seconds)
    progress_update_interval: float = 0.5
    countdown_update_interval: int = 5
    command_prefix: str = '!'

    # --- Database Configuration ---
    database_path: str = 'soyebot.db'
    # Gemini/LLM model tuning
    # Temperature controls creativity (0.0 = deterministic, higher = more creative)
    temperature: float = 1.0
    # Channels where every message should be auto-processed by Gemini
    auto_reply_channel_ids: Tuple[int, ...] = ()
    # --- Session Management ---
    session_cache_limit: int = 200
    session_inactive_minutes: int = 30
    session_similarity_threshold: float = 0.2

def load_config() -> AppConfig:
    """환경 변수에서 설정을 로드합니다."""
    discord_token = os.environ.get('DISCORD_TOKEN')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

    if not discord_token or not gemini_api_key:
        logger.error("에러: DISCORD_TOKEN 또는 GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    auto_channel_env = os.environ.get('AUTO_REPLY_CHANNEL_IDS', '')
    auto_reply_channel_ids: Tuple[int, ...] = ()
    if auto_channel_env.strip():
        valid_ids = []
        invalid_entries = []
        for cid in auto_channel_env.split(','):
            stripped = cid.strip()
            if not stripped:
                continue

            try:
                valid_ids.append(int(stripped))
            except ValueError:
                invalid_entries.append(stripped)

        if invalid_entries:
            logger.warning(
                "AUTO_REPLY_CHANNEL_IDS 환경 변수에 잘못된 값이 있어 무시되었습니다: %s",
                invalid_entries,
            )

        auto_reply_channel_ids = tuple(valid_ids)

    return AppConfig(
        discord_token=discord_token,
        gemini_api_key=gemini_api_key,
        auto_reply_channel_ids=auto_reply_channel_ids,
    )

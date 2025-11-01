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
    llm_mode: str  # 'internal' or 'external'
    llm_endpoint_url: str = None  # Only used for external mode
    model_path: str = None  # Only used for internal mode
    model_name: str = 'local-model'
    max_messages_per_fetch: int = 300
    api_max_retries: int = 2
    api_rate_limit_retry_after: int = 5
    api_request_timeout: int = 30
    progress_update_interval: float = 0.5
    countdown_update_interval: int = 5
    session_ttl_minutes: int = 10
    session_cleanup_interval: int = 300
    command_prefix: str = '!'

    # LLM Generation Settings
    max_tokens: int = 1024  # Maximum tokens to generate in responses
    summary_max_tokens: int = 512  # Maximum tokens for summaries

    # --- Database Configuration ---
    database_path: str = 'soyebot.db'

def load_config() -> AppConfig:
    """환경 변수에서 설정을 로드합니다."""
    discord_token = os.environ.get('DISCORD_TOKEN')
    llm_mode = os.environ.get('LLM_MODE', 'internal').lower()

    if not discord_token:
        logger.error("에러: DISCORD_TOKEN 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    if llm_mode == 'internal':
        model_path = os.environ.get('MODEL_PATH')
        if not model_path:
            logger.error("에러: LLM_MODE가 'internal'일 때 MODEL_PATH 환경 변수가 필요합니다.")
            sys.exit(1)
        logger.info(f"Using internal LLM mode with model at: {model_path}")
        return AppConfig(
            discord_token=discord_token,
            llm_mode='internal',
            model_path=model_path
        )
    elif llm_mode == 'external':
        llm_endpoint_url = os.environ.get('LLM_ENDPOINT_URL')
        if not llm_endpoint_url:
            logger.error("에러: LLM_MODE가 'external'일 때 LLM_ENDPOINT_URL 환경 변수가 필요합니다.")
            sys.exit(1)
        logger.info(f"Using external LLM mode with endpoint: {llm_endpoint_url}")
        return AppConfig(
            discord_token=discord_token,
            llm_mode='external',
            llm_endpoint_url=llm_endpoint_url
        )
    else:
        logger.error(f"에러: 잘못된 LLM_MODE '{llm_mode}'. 'internal' 또는 'external'만 지원됩니다.")
        sys.exit(1)

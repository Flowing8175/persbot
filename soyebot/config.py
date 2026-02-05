"""Configuration loader for SoyeBot."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

# --- 로딩 및 기본 설정 ---
# Load the local .env file from the project root to populate environment
# variables. We intentionally avoid find_dotenv because the project is typically
# run from the repository root and we want predictable loading behavior.
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)
    logging.getLogger(__name__).debug(
        "Loaded environment variables from %s", _dotenv_path
    )
else:
    logging.getLogger(__name__).debug(
        "No .env file found; relying on existing environment"
    )


def _resolve_log_level(raw_level: str) -> int:
    """Return a logging level constant from a string, defaulting to INFO."""

    if not raw_level:
        return logging.INFO

    normalized = raw_level.strip().upper()
    if normalized in logging._nameToLevel:
        return logging._nameToLevel[normalized]

    logging.getLogger(__name__).warning(
        "Unknown LOG_LEVEL '%s'; defaulting to INFO", raw_level
    )
    return logging.INFO


logging.basicConfig(
    level=_resolve_log_level(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- 설정 ---
DEFAULT_GEMINI_ASSISTANT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-2.5-pro"
DEFAULT_OPENAI_ASSISTANT_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_SUMMARY_MODEL = "gpt-5-mini"


@dataclass
class AppConfig:
    """애플리케이션 설정"""

    discord_token: str
    assistant_llm_provider: str = "gemini"
    summarizer_llm_provider: str = "gemini"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    assistant_model_name: str = DEFAULT_GEMINI_ASSISTANT_MODEL
    summarizer_model_name: str = DEFAULT_GEMINI_SUMMARY_MODEL
    max_messages_per_fetch: int = 300
    api_max_retries: int = 2
    api_rate_limit_retry_after: int = 5
    api_request_timeout: int = 30
    api_retry_backoff_base: float = 2.0  # Exponential backoff base
    api_retry_backoff_max: float = 32.0  # Max backoff cap (seconds)
    progress_update_interval: float = 0.5
    countdown_update_interval: int = 5
    command_prefix: str = "!"
    service_tier: str = "flex"
    openai_finetuned_model: Optional[str] = None

    # Gemini/LLM model tuning
    # Temperature controls creativity (0.0 = deterministic, higher = more creative)
    temperature: float = 1.0
    # Nucleus sampling (Top-p) controls diversity
    top_p: float = 1.0
    # Gemini Context Caching
    gemini_cache_min_tokens: int = 32768
    gemini_cache_ttl_minutes: int = 60
    # Gemini Thinking Budget (in tokens)
    thinking_budget: Optional[int] = None
    max_history: int = 50

    # Channels where every message should be auto-processed by Gemini
    auto_reply_channel_ids: Tuple[int, ...] = ()
    log_level: int = logging.INFO
    # --- Session Management ---
    session_cache_limit: int = 200
    session_inactive_minutes: int = 30
    message_buffer_delay: float = 0.1
    break_cut_mode: bool = True
    no_check_permission: bool = False


def _normalize_provider(raw_provider: Optional[str], default: str) -> str:
    if raw_provider is None or not raw_provider.strip():
        return default
    return raw_provider.strip().lower()


def _validate_provider(provider: str) -> str:
    if provider not in {"gemini", "openai"}:
        logger.error(
            "에러: LLM 공급자는 'gemini' 또는 'openai'여야 합니다. (입력값: %s)",
            provider,
        )
        sys.exit(1)
    return provider


def _first_nonempty_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _parse_float_env(name: str, default: float) -> float:
    """Parse float from environment variable with fallback."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "%s 설정이 숫자가 아닙니다. 기본값 %s을 사용합니다.", name, default
        )
        return default


def _parse_int_env(name: str, default: int) -> int:
    """Parse int from environment variable with fallback."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "%s 설정이 숫자가 아닙니다. 기본값 %s을 사용합니다.", name, default
        )
        return default


def _parse_thinking_budget() -> Optional[int]:
    """Parse THINKING_BUDGET with special 'off' handling."""
    raw = os.environ.get("THINKING_BUDGET", "off").strip().lower()
    if raw == "off":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "THINKING_BUDGET 설정이 올바르지 않습니다. 기본값 'off'를 사용합니다."
        )
        return None


def _parse_auto_channel_ids() -> Tuple[int, ...]:
    """Parse comma-separated channel IDs from environment."""
    raw = os.environ.get("AUTO_REPLY_CHANNEL_IDS", "").strip()
    if not raw:
        return ()

    valid_ids, invalid_entries = [], []
    for cid in raw.split(","):
        stripped = cid.strip()
        if not stripped:
            continue
        try:
            valid_ids.append(int(stripped))
        except ValueError:
            invalid_entries.append(stripped)

    if invalid_entries:
        logger.warning(
            "AUTO_REPLY_CHANNEL_IDS에 잘못된 값이 있어 무시됨: %s", invalid_entries
        )

    return tuple(valid_ids)


def _resolve_model_name(provider: str, *, role: str) -> str:
    """Return the model name for the given provider/role using clear priority.

    Priority order:
    1. Role-specific override (e.g., OPENAI_ASSISTANT_MODEL_NAME)
    2. Sensible provider defaults
    """

    if provider == "openai":
        if role == "assistant":
            return (
                _first_nonempty_env("OPENAI_ASSISTANT_MODEL_NAME")
                or DEFAULT_OPENAI_ASSISTANT_MODEL
            )
        return (
            _first_nonempty_env("OPENAI_SUMMARY_MODEL_NAME")
            or DEFAULT_OPENAI_SUMMARY_MODEL
        )

    # Gemini
    if role == "assistant":
        return (
            _first_nonempty_env("GEMINI_ASSISTANT_MODEL_NAME")
            or DEFAULT_GEMINI_ASSISTANT_MODEL
        )
    return (
        _first_nonempty_env("GEMINI_SUMMARY_MODEL_NAME") or DEFAULT_GEMINI_SUMMARY_MODEL
    )


def load_config() -> AppConfig:
    """환경 변수에서 설정을 로드합니다."""
    discord_token = os.environ.get("DISCORD_TOKEN")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    service_tier = os.environ.get("SERVICE_TIER", "flex")

    # Provider별 설정 (어시스턴트/요약 분리)
    default_assistant_provider = AppConfig.__dataclass_fields__[
        "assistant_llm_provider"
    ].default
    assistant_llm_provider = _validate_provider(
        _normalize_provider(
            os.environ.get("ASSISTANT_LLM_PROVIDER"),
            default_assistant_provider,
        )
    )
    summarizer_llm_provider = _validate_provider(
        _normalize_provider(
            os.environ.get("SUMMARIZER_LLM_PROVIDER"),
            assistant_llm_provider,
        )
    )

    # Provider별 모델 설정 (역할별 우선순위 명확화)
    openai_finetuned_model = _first_nonempty_env("OPENAI_FINETUNED_MODEL")

    assistant_model_name = _resolve_model_name(assistant_llm_provider, role="assistant")

    # If using OpenAI and a fine-tuned model is specified, override the assistant model
    if assistant_llm_provider == "openai" and openai_finetuned_model:
        assistant_model_name = openai_finetuned_model
        logger.info("OpenAI Fine-tuned model selected: %s", assistant_model_name)

    summarizer_model_name = _resolve_model_name(summarizer_llm_provider, role="summary")

    # 필수 키 검증
    if not discord_token:
        logger.error("에러: DISCORD_TOKEN 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    if (
        "gemini" in {assistant_llm_provider, summarizer_llm_provider}
        and not gemini_api_key
    ):
        logger.error("에러: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    uses_openai = "openai" in {assistant_llm_provider, summarizer_llm_provider}

    if uses_openai and not openai_api_key:
        logger.error("에러: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    # Parse simple environment values using helpers
    auto_reply_channel_ids = _parse_auto_channel_ids()
    message_buffer_delay = _parse_float_env("MESSAGE_BUFFER_DELAY", 2.5)
    temperature = _parse_float_env("TEMPERATURE", 1.0)
    top_p = _parse_float_env("TOP_P", 1.0)
    thinking_budget = _parse_thinking_budget()
    max_history = _parse_int_env("MAX_HISTORY", 50)

    logger.info(
        "LLM_PROVIDER(assistant)=%s, LLM_PROVIDER(summarizer)=%s, assistant_model=%s, summarizer_model=%s",
        assistant_llm_provider,
        summarizer_llm_provider,
        assistant_model_name,
        summarizer_model_name,
    )

    return AppConfig(
        discord_token=discord_token,
        assistant_llm_provider=assistant_llm_provider,
        summarizer_llm_provider=summarizer_llm_provider,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        assistant_model_name=assistant_model_name,
        summarizer_model_name=summarizer_model_name,
        auto_reply_channel_ids=auto_reply_channel_ids,
        log_level=_resolve_log_level(os.environ.get("LOG_LEVEL", "INFO")),
        service_tier=service_tier,
        openai_finetuned_model=openai_finetuned_model,
        message_buffer_delay=message_buffer_delay,
        temperature=temperature,
        top_p=top_p,
        gemini_cache_min_tokens=_parse_int_env("GEMINI_CACHE_MIN_TOKENS", 32768),
        gemini_cache_ttl_minutes=_parse_int_env("GEMINI_CACHE_TTL_MINUTES", 60),
        thinking_budget=thinking_budget,
        max_history=max_history,
        no_check_permission=os.environ.get("NO_CHECK_PERMISSION", "").lower()
        in ("true", "1", "yes"),
    )

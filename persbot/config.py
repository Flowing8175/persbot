"""Configuration loader for SoyeBot."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --- Load Environment & Default Configuration ---
# Load the local .env file from the project root to populate environment
# variables. We intentionally avoid find_dotenv because the project is typically
# run from the repository root and we want predictable loading behavior.
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if _dotenv_path.exists():
    _ = load_dotenv(_dotenv_path)


def _resolve_log_level(raw_level: str) -> int:
    """Return a logging level constant from a string, defaulting to INFO."""

    if not raw_level:
        return logging.INFO

    normalized = raw_level.strip().upper()

    # Use getattr to access logging level constants (public API)
    level = getattr(logging, normalized, None)
    if isinstance(level, int) and level > 0:
        return level

    logging.getLogger(__name__).warning("Unknown LOG_LEVEL '%s'; defaulting to INFO", raw_level)
    return logging.INFO


logging.basicConfig(
    level=_resolve_log_level(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Configuration Defaults ---
DEFAULT_GEMINI_ASSISTANT_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-2.5-pro"
DEFAULT_OPENAI_ASSISTANT_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_SUMMARY_MODEL = "gpt-5-mini"
DEFAULT_ZAI_ASSISTANT_MODEL = "glm-4.7"
DEFAULT_ZAI_SUMMARY_MODEL = "glm-4-flash"


@dataclass
class AppConfig:
    """Application configuration"""

    discord_token: str
    assistant_llm_provider: str = "gemini"
    summarizer_llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    zai_api_key: str | None = None
    zai_base_url: str = "https://api.z.ai/api/paas/v4/"
    zai_coding_plan: bool = False
    openrouter_api_key: str | None = None
    openrouter_image_model: str = "black-forest-labs/flux.2-klein-4b"
    assistant_model_name: str = DEFAULT_GEMINI_ASSISTANT_MODEL
    summarizer_model_name: str = DEFAULT_GEMINI_SUMMARY_MODEL
    max_messages_per_fetch: int = 300
    api_max_retries: int = 2
    api_rate_limit_retry_after: int = 5
    api_request_timeout: float = 120.0
    zai_request_timeout: float = 300.0  # 5 minutes for Z.AI (for complex coding tasks)
    api_retry_backoff_base: float = 2.0  # Exponential backoff base
    api_retry_backoff_max: float = 32.0  # Max backoff cap (seconds)
    progress_update_interval: float = 0.5
    countdown_update_interval: int = 5
    command_prefix: str = "!"
    service_tier: str = "flex"
    openai_finetuned_model: str | None = None

    # Gemini/LLM model tuning
    # Temperature controls creativity (0.0 = deterministic, higher = more creative)
    temperature: float = 1.0
    # Nucleus sampling (Top-p) controls diversity
    top_p: float = 1.0
    # Gemini Context Caching (2.5/3 Flash: 1024, 2.5/3 Pro: 4096 tokens minimum)
    # Default is 1024 for Flash models; service uses model-specific values
    gemini_cache_min_tokens: int = 1024
    gemini_cache_ttl_minutes: int = 60
    # Gemini Thinking Budget (in tokens)
    thinking_budget: int | None = None
    max_history: int = 50

    # --- Context Summarization ---
    summarization_threshold: int = 40  # Messages before summarization triggers
    summarization_keep_recent: int = 7  # Recent messages to keep unsummarized
    summarization_model: str = "gemini-2.5-flash"  # Cheaper model for summarization
    summarization_max_tokens: int = 500  # Max tokens for summary

    # Channels where every message should be auto-processed by Gemini
    auto_reply_channel_ids: tuple[int, ...] = ()
    log_level: int = logging.INFO
    # --- Session Management ---
    session_cache_limit: int = 200
    session_inactive_minutes: int = 30
    message_buffer_delay: float = 0.1
    break_cut_mode: bool = True
    no_check_permission: bool = False

    # --- Tool Configuration ---
    enable_tools: bool = True
    enable_discord_tools: bool = True
    enable_api_tools: bool = True
    tool_timeout: float = 10.0  # seconds
    weather_api_key: str | None = None
    search_api_key: str | None = None

    # --- Image Generation Rate Limiting ---
    image_rate_limit_per_minute: int = 3
    image_rate_limit_per_hour: int = 15


def _normalize_provider(raw_provider: str | None, default: str) -> str:
    if raw_provider is None or not raw_provider.strip():
        return default
    return raw_provider.strip().lower()


def _validate_provider(provider: str) -> str:
    if provider not in {"gemini", "openai", "zai"}:
        logger.error(
            "에러: LLM 공급자는 'gemini', 'openai' 또는 'zai'여야 합니다. (입력값: %s)",
            provider,
        )
        sys.exit(1)
    return provider


def _first_nonempty_env(*names: str) -> str | None:
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
        logger.warning("%s 설정이 숫자가 아닙니다. 기본값 %s을 사용합니다.", name, default)
        return default


def _parse_int_env(name: str, default: int) -> int:
    """Parse int from environment variable with fallback."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("%s 설정이 숫자가 아닙니다. 기본값 %s을 사용합니다.", name, default)
        return default


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse boolean from environment variable with fallback."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes")


def _parse_thinking_budget() -> int | None:
    """Parse THINKING_BUDGET with special 'off' handling."""
    raw = os.environ.get("THINKING_BUDGET", "off").strip().lower()
    if raw == "off":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("THINKING_BUDGET 설정이 올바르지 않습니다. 기본값 'off'를 사용합니다.")
        return None


def _parse_auto_channel_ids() -> tuple[int, ...]:
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
        logger.warning("AUTO_REPLY_CHANNEL_IDS에 잘못된 값이 있어 무시됨: %s", invalid_entries)

    return tuple(valid_ids)


def _resolve_model_name(provider: str, *, role: str) -> str:
    """Return model name for given provider/role using clear priority.

    Priority order:
    1. Role-specific override (e.g., OPENAI_ASSISTANT_MODEL_NAME)
    2. Sensible provider defaults
    """

    if provider == "openai":
        if role == "assistant":
            return (
                _first_nonempty_env("OPENAI_ASSISTANT_MODEL_NAME") or DEFAULT_OPENAI_ASSISTANT_MODEL
            )
        return _first_nonempty_env("OPENAI_SUMMARY_MODEL_NAME") or DEFAULT_OPENAI_SUMMARY_MODEL

    if provider == "zai":
        if role == "assistant":
            return _first_nonempty_env("ZAI_ASSISTANT_MODEL_NAME") or DEFAULT_ZAI_ASSISTANT_MODEL
        return _first_nonempty_env("ZAI_SUMMARY_MODEL_NAME") or DEFAULT_ZAI_SUMMARY_MODEL

    # Gemini
    if role == "assistant":
        return _first_nonempty_env("GEMINI_ASSISTANT_MODEL_NAME") or DEFAULT_GEMINI_ASSISTANT_MODEL
    return _first_nonempty_env("GEMINI_SUMMARY_MODEL_NAME") or DEFAULT_GEMINI_SUMMARY_MODEL


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    discord_token = os.environ.get("DISCORD_TOKEN")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    zai_api_key = os.environ.get("ZAI_API_KEY")
    zai_coding_plan = _parse_bool_env("ZAI_CODING_PLAN")
    # Use Coding Plan API endpoint if enabled, otherwise use standard API
    default_base_url = (
        "https://api.z.ai/api/coding/paas/v4/"
        if zai_coding_plan
        else "https://api.z.ai/api/paas/v4/"
    )
    zai_base_url = os.environ.get("ZAI_BASE_URL", default_base_url)
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_image_model = os.environ.get(
        "OPENROUTER_IMAGE_MODEL", "black-forest-labs/flux.2-klein-4b"
    )
    service_tier = os.environ.get("SERVICE_TIER", "flex")

    # Provider별 설정 (어시스턴트/요약 분리)
    assistant_llm_provider = _validate_provider(
        _normalize_provider(
            os.environ.get("ASSISTANT_LLM_PROVIDER"),
            "gemini",  # Default from AppConfig dataclass
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

    summarizer_model_name = _resolve_model_name(summarizer_llm_provider, role="summary")

    # 필수 키 검증
    if not discord_token:
        logger.error("에러: DISCORD_TOKEN 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    if "gemini" in {assistant_llm_provider, summarizer_llm_provider} and not gemini_api_key:
        logger.error("에러: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    uses_openai = "openai" in {assistant_llm_provider, summarizer_llm_provider}

    if uses_openai and not openai_api_key:
        logger.error("에러: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    uses_zai = "zai" in {assistant_llm_provider, summarizer_llm_provider}

    if uses_zai and not zai_api_key:
        logger.error("에러: ZAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    # Parse simple environment values using helpers
    auto_reply_channel_ids = _parse_auto_channel_ids()
    message_buffer_delay = _parse_float_env("MESSAGE_BUFFER_DELAY", 2.5)
    temperature = _parse_float_env("TEMPERATURE", 1.0)
    top_p = _parse_float_env("TOP_P", 1.0)
    thinking_budget = _parse_thinking_budget()
    max_history = _parse_int_env("MAX_HISTORY", 50)

    # Parse tool configuration
    enable_tools = _parse_bool_env("ENABLE_TOOLS", default=True)
    enable_discord_tools = _parse_bool_env("ENABLE_DISCORD_TOOLS", default=True)
    enable_api_tools = _parse_bool_env("ENABLE_API_TOOLS", default=True)
    tool_timeout = _parse_float_env("TOOL_TIMEOUT", 10.0)
    weather_api_key = os.environ.get("WEATHER_API_KEY")
    search_api_key = os.environ.get("SEARCH_API_KEY")
    api_request_timeout = _parse_float_env("API_REQUEST_TIMEOUT", 120.0)
    zai_request_timeout = _parse_float_env("ZAI_REQUEST_TIMEOUT", 300.0)

    # Parse image rate limiting configuration
    image_rate_limit_per_minute = _parse_int_env("IMAGE_RATE_LIMIT_PER_MINUTE", 3)
    image_rate_limit_per_hour = _parse_int_env("IMAGE_RATE_LIMIT_PER_HOUR", 15)

    return AppConfig(
        discord_token=discord_token,
        assistant_llm_provider=assistant_llm_provider,
        summarizer_llm_provider=summarizer_llm_provider,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        zai_api_key=zai_api_key,
        zai_base_url=zai_base_url,
        zai_coding_plan=zai_coding_plan,
        openrouter_api_key=openrouter_api_key,
        openrouter_image_model=openrouter_image_model,
        assistant_model_name=assistant_model_name,
        summarizer_model_name=summarizer_model_name,
        auto_reply_channel_ids=auto_reply_channel_ids,
        log_level=_resolve_log_level(os.environ.get("LOG_LEVEL", "INFO")),
        service_tier=service_tier,
        openai_finetuned_model=openai_finetuned_model,
        message_buffer_delay=message_buffer_delay,
        temperature=temperature,
        top_p=top_p,
        gemini_cache_min_tokens=_parse_int_env("GEMINI_CACHE_MIN_TOKENS", 1024),
        gemini_cache_ttl_minutes=_parse_int_env("GEMINI_CACHE_TTL_MINUTES", 60),
        thinking_budget=thinking_budget,
        max_history=max_history,
        no_check_permission=_parse_bool_env("NO_CHECK_PERMISSION"),
        enable_tools=enable_tools,
        enable_discord_tools=enable_discord_tools,
        enable_api_tools=enable_api_tools,
        tool_timeout=tool_timeout,
        weather_api_key=weather_api_key,
        search_api_key=search_api_key,
        api_request_timeout=api_request_timeout,
        zai_request_timeout=zai_request_timeout,
        image_rate_limit_per_minute=image_rate_limit_per_minute,
        image_rate_limit_per_hour=image_rate_limit_per_hour,
    )

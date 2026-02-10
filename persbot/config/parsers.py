"""Environment variable parsers for SoyeBot configuration."""

import logging
import os
import sys
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _resolve_log_level(raw_level: str) -> int:
    """Return a logging level constant from a string, defaulting to INFO."""
    if not raw_level:
        return logging.INFO

    normalized = raw_level.strip().upper()

    # Use getattr to access logging level constants (public API)
    level = getattr(logging, normalized, None)
    if isinstance(level, int) and level > 0:
        return level

    logger.warning("Unknown LOG_LEVEL '%s'; defaulting to INFO", raw_level)
    return logging.INFO


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


def _parse_thinking_budget() -> Optional[int]:
    """Parse THINKING_BUDGET with special 'off' handling."""
    raw = os.environ.get("THINKING_BUDGET", "off").strip().lower()
    if raw == "off":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("THINKING_BUDGET 설정이 올바르지 않습니다. 기본값 'off'를 사용합니다.")
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
        logger.warning("AUTO_REPLY_CHANNEL_IDS에 잘못된 값이 있어 무시됨: %s", invalid_entries)

    return tuple(valid_ids)


def _first_nonempty_env(*names: str) -> Optional[str]:
    """Get first non-empty environment variable from the given names."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _normalize_provider(raw_provider: Optional[str], default: str) -> str:
    """Normalize provider name to lowercase."""
    if raw_provider is None or not raw_provider.strip():
        return default
    return raw_provider.strip().lower()


def _validate_provider(provider: str) -> str:
    """Validate that the provider is supported."""
    if provider not in {"gemini", "openai", "zai"}:
        logger.error(
            "에러: LLM 공급자는 'gemini', 'openai' 또는 'zai'여야 합니다. (입력값: %s)",
            provider,
        )
        sys.exit(1)
    return provider

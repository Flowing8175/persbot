"""Configuration validators for SoyeBot."""

import logging
import sys
from typing import Optional

from .base import (
    DEFAULT_GEMINI_ASSISTANT_MODEL,
    DEFAULT_GEMINI_SUMMARY_MODEL,
    DEFAULT_OPENAI_ASSISTANT_MODEL,
    DEFAULT_OPENAI_SUMMARY_MODEL,
    DEFAULT_ZAI_ASSISTANT_MODEL,
    DEFAULT_ZAI_SUMMARY_MODEL,
)

logger = logging.getLogger(__name__)


def validate_required_keys(
    assistant_provider: str,
    summarizer_provider: str,
    gemini_api_key: Optional[str],
    openai_api_key: Optional[str],
    zai_api_key: Optional[str],
) -> None:
    """
    Validate that required API keys are present for the configured providers.

    Args:
        assistant_provider: The assistant LLM provider.
        summarizer_provider: The summarizer LLM provider.
        gemini_api_key: Gemini API key.
        openai_api_key: OpenAI API key.
        zai_api_key: Z.AI API key.

    Raises:
        SystemExit: If required keys are missing.
    """
    # Validate Discord token (should be checked earlier)
    if "gemini" in {assistant_provider, summarizer_provider} and not gemini_api_key:
        logger.error("에러: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    uses_openai = "openai" in {assistant_provider, summarizer_provider}
    if uses_openai and not openai_api_key:
        logger.error("에러: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    uses_zai = "zai" in {assistant_provider, summarizer_provider}
    if uses_zai and not zai_api_key:
        logger.error("에러: ZAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)


def validate_model_name(model_name: str, provider: str) -> str:
    """
    Validate and return a model name, using defaults if needed.

    Args:
        model_name: The model name to validate.
        provider: The provider name.

    Returns:
        The validated model name.
    """
    if not model_name:
        return get_default_model(provider, "assistant")

    return model_name


def validate_temperature(value: Optional[float]) -> float:
    """
    Validate temperature value.

    Args:
        value: The temperature value to validate.

    Returns:
        The validated temperature value (0.0 to 2.0).
    """
    if value is None:
        return 1.0

    if not (0.0 <= value <= 2.0):
        logger.warning("Temperature 값 %.2f가 범위를 벗어남. 기본값 1.0 사용.", value)
        return 1.0

    return value


def validate_top_p(value: Optional[float]) -> float:
    """
    Validate top_p value.

    Args:
        value: The top_p value to validate.

    Returns:
        The validated top_p value (0.0 to 1.0).
    """
    if value is None:
        return 1.0

    if not (0.0 <= value <= 1.0):
        logger.warning("Top-p 값 %.2f가 범위를 벗어남. 기본값 1.0 사용.", value)
        return 1.0

    return value


def validate_thinking_budget(value: Optional[int]) -> Optional[int]:
    """
    Validate thinking budget value.

    Args:
        value: The thinking budget value to validate.

    Returns:
        The validated thinking budget, or None if disabled.
    """
    if value is None or value == -1:
        return None  # -1 means "auto", None means "off"

    if not (512 <= value <= 32768):
        logger.warning(
            "Thinking Budget 값 %d가 범위를 벗어남(512~32768). 기본값 None 사용.",
            value,
        )
        return None

    return value


def get_default_model(provider: str, role: str) -> str:
    """
    Get the default model name for a provider and role.

    Args:
        provider: The provider name (gemini, openai, zai).
        role: The role (assistant or summary).

    Returns:
        The default model name.
    """
    if provider == "openai":
        return (
            DEFAULT_OPENAI_ASSISTANT_MODEL
            if role == "assistant"
            else DEFAULT_OPENAI_SUMMARY_MODEL
        )
    elif provider == "zai":
        return (
            DEFAULT_ZAI_ASSISTANT_MODEL if role == "assistant" else DEFAULT_ZAI_SUMMARY_MODEL
        )
    else:  # gemini
        return (
            DEFAULT_GEMINI_ASSISTANT_MODEL
            if role == "assistant"
            else DEFAULT_GEMINI_SUMMARY_MODEL
        )


def validate_buffer_delay(value: Optional[float]) -> float:
    """
    Validate message buffer delay value.

    Args:
        value: The buffer delay value to validate.

    Returns:
        The validated buffer delay (0.0 to 60.0).
    """
    if value is None:
        return 2.5

    if not (0.0 <= value <= 60.0):
        logger.warning("Buffer delay 값 %.2f가 범위를 벗어남. 기본값 2.5 사용.", value)
        return 2.5

    return value

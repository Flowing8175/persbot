"""Cache management utilities for Gemini service."""

import datetime
import hashlib
import logging
import re
from typing import Optional, Tuple

import google.genai as genai
from google.genai import types as genai_types

from .constants import (
    CACHE_REFRESH_BUFFER_MAX_MINUTES,
    CACHE_REFRESH_BUFFER_MIN_MINUTES,
    DEFAULT_CACHE_MIN_TOKENS,
    DEFAULT_CACHE_TTL_MINUTES,
)

logger = logging.getLogger(__name__)


def _get_cache_key(model_name: str, content: str, tools: Optional[list] = None) -> str:
    """Generate a consistent cache key/name based on model and content hash."""
    # Clean model name for use in display_name
    safe_model = re.sub(r"[^a-zA-Z0-9-]", "-", model_name)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    tool_suffix = "-tools" if tools else ""
    return f"persbot-{safe_model}-{content_hash[:10]}{tool_suffix}"


def _get_gemini_cache(
    client: genai.Client,
    model_name: str,
    system_instruction: str,
    tools: Optional[list] = None,
    config: Optional[object] = None,
) -> Tuple[Optional[str], Optional[datetime.datetime]]:
    """
    Attempts to find or create a Gemini cache for the given system instruction.
    Returns: (cache_name, local_expiration_datetime)
    """
    if not system_instruction:
        return None, None

    # 1. Check token count
    try:
        # We must count tokens including tools and system instruction to be accurate.
        count_result = client.models.count_tokens(
            model=model_name,
            contents=[system_instruction],
        )
        token_count = count_result.total_tokens
    except Exception as e:
        logger.warning(
            "Failed to count tokens for caching check: %s. Using standard context.",
            e,
        )
        return None, None

    min_tokens = (
        getattr(config, "gemini_cache_min_tokens", DEFAULT_CACHE_MIN_TOKENS)
        if config
        else DEFAULT_CACHE_MIN_TOKENS
    )
    if token_count < min_tokens:
        logger.info(
            "Gemini Context Caching skipped: Token count (%d) < min_tokens (%d). Using standard context.",
            token_count,
            min_tokens,
        )
        return None, None

    logger.info("Token count (%d) meets requirement for Gemini caching.", token_count)

    # 2. Config setup
    cache_display_name = _get_cache_key(model_name, system_instruction, tools)
    ttl_minutes = (
        getattr(config, "gemini_cache_ttl_minutes", DEFAULT_CACHE_TTL_MINUTES)
        if config
        else DEFAULT_CACHE_TTL_MINUTES
    )
    ttl_seconds = ttl_minutes * 60

    now = datetime.datetime.now(datetime.timezone.utc)
    # local_expiration: trigger refresh halfway through TTL window or with a buffer
    refresh_buffer_minutes = min(
        CACHE_REFRESH_BUFFER_MAX_MINUTES,
        max(CACHE_REFRESH_BUFFER_MIN_MINUTES, ttl_minutes // 2),
    )
    local_expiration = now + datetime.timedelta(minutes=ttl_minutes - refresh_buffer_minutes)

    # 3. Search for existing cache
    try:
        # We iterate to find a cache with our unique display name
        for cache in client.caches.list():
            if cache.display_name == cache_display_name:
                logger.info("Found existing Gemini context cache: %s", cache.name)

                # Refresh TTL to prevent expiration
                try:
                    client.caches.update(
                        name=cache.name,
                        config=genai_types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s"),
                    )
                    logger.info(
                        "Successfully refreshed TTL for %s to %ds.",
                        cache.name,
                        ttl_seconds,
                    )
                    return cache.name, local_expiration
                except Exception as update_err:
                    logger.warning(
                        "Failed to refresh TTL for %s: %s. Will attempt re-creation.",
                        cache.name,
                        update_err,
                    )
                    # If update fails, DISCARD this cache and continue searching or create new
                    continue

    except Exception as e:
        logger.error("Error listing Gemini caches: %s", e)

    # 4. Create new cache using the user's requested style
    try:
        logger.info(
            "Creating new Gemini cache '%s' (TTL: %ds)...",
            cache_display_name,
            ttl_seconds,
        )

        cache = client.caches.create(
            model=model_name,
            config=genai_types.CreateCachedContentConfig(
                display_name=cache_display_name,
                system_instruction=system_instruction,
                tools=tools,
                ttl=f"{ttl_seconds}s",
            ),
        )

        logger.info("Successfully created cache: %s", cache.name)
        return cache.name, local_expiration

    except Exception as e:
        logger.error("Failed to create Gemini context cache: %s", e)
        return None, None


def _check_model_cache_validity(
    model_cache: dict,
    cache_key: int,
    now: datetime.datetime,
) -> Optional:
    """Check if cached model exists and is still valid.

    Args:
        model_cache: The model cache dictionary.
        cache_key: The cache key to check.
        now: Current datetime for comparison.

    Returns:
        The cached model if valid, None otherwise.
    """
    if cache_key not in model_cache:
        return None

    model, expires_at = model_cache[cache_key]
    if expires_at and now >= expires_at:
        logger.info("Cached model expired (TTL reached). Refreshing...")
        del model_cache[cache_key]
        return None

    return model

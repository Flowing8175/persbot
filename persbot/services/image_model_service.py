"""Service for managing image model preferences per channel."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default image model
DEFAULT_IMAGE_MODEL = "black-forest-labs/flux.2-klein-4b"

# Channel-level image model preferences
# Format: {channel_id: model_name}
_channel_image_preferences: Dict[int, str] = {}


def set_channel_image_model(channel_id: int, model_name: str) -> None:
    """Set the image model preference for a channel.

    Args:
        channel_id: Discord channel ID.
        model_name: Image model name (e.g., "sourceful/riverflow-v2-pro").
    """
    _channel_image_preferences[channel_id] = model_name
    logger.info("Set image model for channel %d to %s", channel_id, model_name)


def get_channel_image_model(channel_id: int) -> str:
    """Get the image model preference for a channel.

    Args:
        channel_id: Discord channel ID.

    Returns:
        The image model name for the channel, or DEFAULT_IMAGE_MODEL if not set.
    """
    return _channel_image_preferences.get(channel_id, DEFAULT_IMAGE_MODEL)


def clear_channel_image_model(channel_id: int) -> None:
    """Clear the image model preference for a channel.

    Args:
        channel_id: Discord channel ID.
    """
    if channel_id in _channel_image_preferences:
        del _channel_image_preferences[channel_id]
        logger.info("Cleared image model preference for channel %d", channel_id)

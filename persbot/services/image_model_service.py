"""Service for managing image model preferences per channel.

Image model preferences are kept in-memory only and reset on restart.
The default model is determined by the 'default: true' flag in the model definition.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Path to models.json
MODELS_FILE = "data/models.json"


@dataclass
class ImageModelDefinition:
    """Definition of an image generation model."""

    display_name: str
    api_model_name: str
    description: str
    default: bool = False


# In-memory cache for image model definitions and preferences
_image_models_cache: List[ImageModelDefinition] = []
_channel_image_preferences: Dict[int, str] = {}  # In-memory only, reset on restart
_default_image_model: str = "black-forest-labs/flux.2-klein-4b"


def _load_image_models():
    """Load image model definitions from models.json file.

    The default model is determined by the 'default: true' flag in the model definition.
    Channel preferences are in-memory only and not persisted.
    """
    global _image_models_cache, _default_image_model

    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                models_data = data.get("image_models", [])
                _image_models_cache = [
                    ImageModelDefinition(
                        display_name=m["display_name"],
                        api_model_name=m["api_model_name"],
                        description=m["description"],
                        default=m.get("default", False),
                    )
                    for m in models_data
                ]

                # Determine default from model definition with default=True
                _default_image_model = "black-forest-labs/flux.2-klein-4b"  # fallback
                for model in _image_models_cache:
                    if model.default:
                        _default_image_model = model.api_model_name
                        break

        except Exception:
            logger.exception("Failed to load image models")
            # Use fallback defaults
            _image_models_cache = [
                ImageModelDefinition(
                    display_name="Flux 2 Klein",
                    api_model_name="black-forest-labs/flux.2-klein-4b",
                    description="Fast and efficient image generation model",
                    default=True,
                )
            ]
    else:
        logger.warning(f"Models file not found: {MODELS_FILE}")
        _image_models_cache = [
            ImageModelDefinition(
                display_name="Flux 2 Klein",
                api_model_name="black-forest-labs/flux.2-klein-4b",
                description="Fast and efficient image generation model",
                default=True,
            )
        ]


def get_available_image_models() -> List[ImageModelDefinition]:
    """Get all available image model definitions.

    Returns:
        List of ImageModelDefinition objects.
    """
    if not _image_models_cache:
        _load_image_models()
    return _image_models_cache.copy()


def get_image_model_by_name(model_name: str) -> Optional[ImageModelDefinition]:
    """Get an image model definition by API model name.

    Args:
        model_name: The API model name to search for.

    Returns:
        ImageModelDefinition if found, None otherwise.
    """
    if not _image_models_cache:
        _load_image_models()
    for model in _image_models_cache:
        if model.api_model_name == model_name:
            return model
    return None


def set_channel_image_model(channel_id: int, model_name: str) -> bool:
    """Set the image model preference for a channel (in-memory only).

    Args:
        channel_id: Discord channel ID.
        model_name: Image model name (e.g., "sourceful/riverflow-v2-pro").
                    Can be any valid model identifier - no validation required.

    Returns:
        True if successful.
    """
    _channel_image_preferences[channel_id] = model_name
    return True


def get_channel_image_model(channel_id: int) -> str:
    """Get the image model preference for a channel.

    Args:
        channel_id: Discord channel ID.

    Returns:
        The image model name for the channel, or DEFAULT_IMAGE_MODEL if not set.
    """
    if not _image_models_cache:
        _load_image_models()
    return _channel_image_preferences.get(channel_id, _default_image_model)


def clear_channel_image_model(channel_id: int) -> None:
    """Clear the image model preference for a channel (in-memory only).

    Args:
        channel_id: Discord channel ID.
    """
    if channel_id in _channel_image_preferences:
        del _channel_image_preferences[channel_id]


def get_default_image_model() -> str:
    """Get the default image model name.

    Returns:
        The default image model API name.
    """
    if not _image_models_cache:
        _load_image_models()
    return _default_image_model


def set_default_image_model(model_name: str) -> bool:
    """Set the default image model (in-memory only for current session).

    Note: The persistent default is determined by 'default: true' in the model definition.
    This only overrides the default for the current session.

    Args:
        model_name: Image model name to set as default.
                    Can be any valid model identifier - no validation required.

    Returns:
        True if successful.
    """
    _default_image_model = model_name
    return True


# Initialize on module load
_load_image_models()

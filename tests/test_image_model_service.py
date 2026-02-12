"""Tests for services/image_model_service.py module.

This module provides comprehensive test coverage for:
- ImageModelDefinition dataclass
- _load_image_models function
- _save_preferences function
- get_available_image_models function
- get_image_model_by_name function
- set_channel_image_model function
- get_channel_image_model function
- clear_channel_image_model function
- get_default_image_model function
- set_default_image_model function
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from persbot.services.image_model_service import (
    ImageModelDefinition,
    _load_image_models,
    _save_preferences,
    get_available_image_models,
    get_image_model_by_name,
    set_channel_image_model,
    get_channel_image_model,
    clear_channel_image_model,
    get_default_image_model,
    set_default_image_model,
)


# =============================================================================
# ImageModelDefinition Dataclass Tests
# =============================================================================


class TestImageModelDefinition:
    """Tests for ImageModelDefinition dataclass."""

    def test_init_with_all_fields(self):
        """Test ImageModelDefinition initialization with all fields."""
        model = ImageModelDefinition(
            display_name="Test Model",
            api_model_name="test-model",
            description="A test model",
            default=True,
        )

        assert model.display_name == "Test Model"
        assert model.api_model_name == "test-model"
        assert model.description == "A test model"
        assert model.default is True

    def test_init_with_default_false(self):
        """Test ImageModelDefinition with default=False."""
        model = ImageModelDefinition(
            display_name="Test Model",
            api_model_name="test-model",
            description="A test model",
        )

        assert model.default is False


# =============================================================================
# _load_image_models Function Tests
# =============================================================================


class TestLoadImageModels:
    """Tests for _load_image_models function."""

    def test_load_from_valid_json(self, tmp_path):
        """Test loading models from valid JSON file."""
        # Create data directory and models.json
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        models_file = data_dir / "models.json"
        test_data = {
            "image_models": [
                {
                    "display_name": "Flux 2 Klein",
                    "api_model_name": "black-forest-labs/flux.2-klein-4b",
                    "description": "Fast and efficient",
                    "default": True,
                },
                {
                    "display_name": "Test Model",
                    "api_model_name": "test/model",
                    "description": "Test description",
                    "default": False,
                }
            ],
            "image_model_preferences": {
                "default": "black-forest-labs/flux.2-klein-4b",
                "channels": {
                    "123456": "test/model",
                    "789012": "black-forest-labs/flux.2-klein-4b",
                },
            },
        }

        models_file.write_text(json.dumps(test_data), encoding="utf-8")

        # Import the module to reload global state
        import persbot.services.image_model_service as service
        original_cache = service._image_models_cache.copy()
        original_default = service._default_image_model
        original_prefs = service._channel_image_preferences.copy()

        # Reload
        service._load_image_models()

        # Verify models were loaded
        assert len(service._image_models_cache) == 2
        assert service._image_models_cache[0].display_name == "Flux 2 Klein"
        assert service._image_models_cache[1].display_name == "Test Model"

        # Verify preferences were loaded
        assert service._default_image_model == "black-forest-labs/flux.2-klein-4b"
        assert service._channel_image_preferences[123456] == "test/model"
        assert service._channel_image_preferences[789012] == "black-forest-labs/flux.2-klein-4b"

    def test_load_with_invalid_channel_id(self, tmp_path):
        """Test loading models with non-integer channel ID."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        models_file = data_dir / "models.json"

        test_data = {
            "image_model_preferences": {
                "channels": {
                    "abc": "test/model",  # Invalid channel ID
                },
            },
        }

        models_file.write_text(json.dumps(test_data), encoding="utf-8")

        # Reload - should log warning but not crash
        import persbot.services.image_model_service as service
        original_cache = service._image_models_cache.copy()

        with patch("persbot.services.image_model_service.logger") as mock_logger:
            service._load_image_models()

            # Should log warning for invalid channel ID
            mock_logger.warning.assert_called()
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("Invalid channel ID" in w for w in warning_calls)

    def test_load_from_nonexistent_file(self, tmp_path):
        """Test loading models when file doesn't exist."""
        # Don't create the file
        import persbot.services.image_model_service as service

        original_cache = service._image_models_cache.copy()

        with patch("os.path.exists", return_value=False):
            service._load_image_models()

            # Should use fallback defaults
            assert len(service._image_models_cache) == 1
            assert service._image_models_cache[0].display_name == "Flux 2 Klein"

    def test_load_with_missing_image_models_key(self, tmp_path):
        """Test loading models when image_models key is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        models_file = data_dir / "models.json"

        test_data = {
            # Missing "image_models" key
            "image_model_preferences": {
                "default": "test-model",
            },
        }

        models_file.write_text(json.dumps(test_data), encoding="utf-8")

        import persbot.services.image_model_service as service

        with patch("persbot.services.image_model_service.logger") as mock_logger:
            service._load_image_models()

            # Should use fallback defaults
            assert len(service._image_models_cache) == 1

    def test_load_json_decode_error(self, tmp_path):
        """Test loading models with invalid JSON."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        models_file = data_dir / "models.json"

        # Write invalid JSON
        models_file.write_text("{invalid json}", encoding="utf-8")

        import persbot.services.image_model_service as service

        with patch("persbot.services.image_model_service.logger") as mock_logger:
            service._load_image_models()

            # Should log error and use fallback
            mock_logger.error.assert_called()
            assert len(service._image_models_cache) == 1


# =============================================================================
# _save_preferences Function Tests
# =============================================================================


class TestSavePreferences:
    """Tests for _save_preferences function."""

    def test_save_preferences_to_existing_file(self, tmp_path):
        """Test saving preferences to existing JSON file."""
        # Create initial data
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        models_file = data_dir / "models.json"
        initial_data = {
            "image_models": [],
            "image_model_preferences": {
                "default": "initial-model",
                "channels": {},
            },
        }

        models_file.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")

        # Modify preferences and save
        import persbot.services.image_model_service as service
        service._default_image_model = "new-default"
        service._channel_image_preferences[123456] = "test-model"

        service._save_preferences()

        # Verify file was updated
        with open(models_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["image_model_preferences"]["default"] == "new-default"
        assert saved_data["image_model_preferences"]["channels"]["123456"] == "test-model"

    def test_save_creates_missing_keys(self, tmp_path):
        """Test saving preferences creates image_model_preferences key if missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        models_file = data_dir / "models.json"
        # Create file without image_model_preferences
        initial_data = {"image_models": []}
        models_file.write_text(json.dumps(initial_data), encoding="utf-8")

        import persbot.services.image_model_service as service
        service._default_image_model = "test-default"

        service._save_preferences()

        # Verify key was created
        with open(models_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert "image_model_preferences" in saved_data
        assert saved_data["image_model_preferences"]["default"] == "test-default"

    def test_save_to_nonexistent_file(self, tmp_path):
        """Test saving preferences when models file doesn't exist."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        # Don't create models.json
        import persbot.services.image_model_service as service

        with patch("persbot.services.image_model_service.logger") as mock_logger:
            service._save_preferences()

            # Should log warning and not save
            mock_logger.warning.assert_called()
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("not found" in w.lower() for w in warning_calls)


# =============================================================================
# get_available_image_models Function Tests
# =============================================================================


class TestGetAvailableImageModels:
    """Tests for get_available_image_models function."""

    def test_get_available_loads_if_cache_empty(self):
        """Test that get_available_image_models loads if cache is empty."""
        import persbot.services.image_model_service as service

        # Clear cache
        service._image_models_cache.clear()

        with patch("persbot.services.image_model_service._load_image_models") as mock_load:
            models = get_available_image_models()

            # Should load models
            mock_load.assert_called_once()
            assert len(models) > 0

    def test_get_available_returns_cached_models(self):
        """Test that get_available_image_models returns cached models."""
        import persbot.services.image_model_service as service

        # Set up cache
        test_model = ImageModelDefinition(
            display_name="Cached Model",
            api_model_name="cached-model",
            description="Cached",
        )
        service._image_models_cache = [test_model]

        models = get_available_image_models()

        assert len(models) == 1
        assert models[0].display_name == "Cached Model"
        # Verify it's a copy
        models[0].display_name = "Modified"
        cached_models = get_available_image_models()
        assert cached_models[0].display_name == "Cached Model"


# =============================================================================
# get_image_model_by_name Function Tests
# =============================================================================


class TestGetImageModelByName:
    """Tests for get_image_model_by_name function."""

    def test_get_model_by_name_found(self):
        """Test getting model by name when found."""
        import persbot.services.image_model_service as service

        test_model = ImageModelDefinition(
            display_name="Test Model",
            api_model_name="test-model",
            description="Test",
        )
        service._image_models_cache = [test_model]

        result = get_image_model_by_name("test-model")

        assert result is not None
        assert result.api_model_name == "test-model"

    def test_get_model_by_name_not_found(self):
        """Test getting model by name when not found."""
        import persbot.services.image_model_service as service

        service._image_models_cache.clear()

        result = get_image_model_by_name("nonexistent")

        assert result is None

    def test_get_model_by_name_loads_if_cache_empty(self):
        """Test that get_model_by_name loads models if cache is empty."""
        import persbot.services.image_model_service as service

        service._image_models_cache.clear()

        with patch("persbot.services.image_model_service._load_image_models") as mock_load:
            get_image_model_by_name("any-name")

            # Should trigger load
            mock_load.assert_called_once()


# =============================================================================
# set_channel_image_model Function Tests
# =============================================================================


class TestSetChannelImageModel:
    """Tests for set_channel_image_model function."""

    def test_set_channel_model_success(self, tmp_path):
        """Test setting channel image model successfully."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        models_file = data_dir / "models.json"
        initial_data = {
            "image_models": [],
            "image_model_preferences": {
                "default": "default-model",
                "channels": {},
            },
        }
        models_file.write_text(json.dumps(initial_data), encoding="utf-8")

        import persbot.services.image_model_service as service

        # Set channel model
        result = set_channel_image_model(123456, "new-model")

        # Verify preference was set
        assert service._channel_image_preferences[123456] == "new-model"

        # Verify save was called
        with open(models_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data["image_model_preferences"]["channels"]["123456"] == "new-model"

    def test_set_channel_model_returns_true(self):
        """Test that set_channel_image_model returns True."""
        import persbot.services.image_model_service as service

        result = set_channel_image_model(123456, "any-model")

        assert result is True


# =============================================================================
# get_channel_image_model Function Tests
# =============================================================================


class TestGetChannelImageModel:
    """Tests for get_channel_image_model function."""

    def test_get_channel_model_preference_exists(self):
        """Test getting channel model when preference exists."""
        import persbot.services.image_model_service as service

        service._channel_image_preferences[123456] = "channel-model"
        service._default_image_model = "default-model"

        result = get_channel_image_model(123456)

        assert result == "channel-model"

    def test_get_channel_model_returns_default(self):
        """Test getting channel model when no preference exists."""
        import persbot.services.image_model_service as service

        service._channel_image_preferences.clear()
        service._default_image_model = "default-model"

        result = get_channel_image_model(999888)

        assert result == "default-model"


# =============================================================================
# clear_channel_image_model Function Tests
# =============================================================================


class TestClearChannelImageModel:
    """Tests for clear_channel_image_model function."""

    def test_clear_channel_model_exists(self, tmp_path):
        """Test clearing existing channel model preference."""
        import persbot.services.image_model_service as service

        service._channel_image_preferences[123456] = "old-model"

        with patch("persbot.services.image_model_service._save_preferences") as mock_save:
            clear_channel_image_model(123456)

            # Verify preference was removed
            assert 123456 not in service._channel_image_preferences

            # Verify save was called
            mock_save.assert_called_once()

    def test_clear_channel_model_not_exists(self):
        """Test clearing non-existent channel model preference."""
        import persbot.services.image_model_service as service

        service._channel_image_preferences.clear()

        with patch("persbot.services.image_model_service._save_preferences") as mock_save:
            clear_channel_image_model(123456)

            # Save should still be called (even if nothing to clear)
            mock_save.assert_called_once()


# =============================================================================
# get_default_image_model Function Tests
# =============================================================================


class TestGetDefaultImageModel:
    """Tests for get_default_image_model function."""

    def test_get_default_model(self):
        """Test getting default image model."""
        import persbot.services.image_model_service as service

        service._default_image_model = "test-default"

        result = get_default_image_model()

        assert result == "test-default"


# =============================================================================
# set_default_image_model Function Tests
# =============================================================================


class TestSetDefaultImageModel:
    """Tests for set_default_image_model function."""

    def test_set_default_model_success(self):
        """Test setting default image model successfully."""
        import persbot.services.image_model_service as service

        result = set_default_image_model("new-default")

        assert result is True
        assert service._default_image_model == "new-default"

    def test_set_default_model_saves_preferences(self, tmp_path):
        """Test that set_default_image_model saves to file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        models_file = data_dir / "models.json"
        initial_data = {
            "image_models": [],
            "image_model_preferences": {
                "default": "old-default",
                "channels": {},
            },
        }
        models_file.write_text(json.dumps(initial_data), encoding="utf-8")

        import persbot.services.image_model_service as service

        set_default_image_model("new-default")

        with open(models_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["image_model_preferences"]["default"] == "new-default"

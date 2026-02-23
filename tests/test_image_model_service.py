"""Tests for image_model_service module.

Tests focus on:
- ImageModelDefinition dataclass behavior
- Basic module functionality
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from persbot.services.image_model_service import ImageModelDefinition


# =============================================================================
# ImageModelDefinition Tests
# =============================================================================

class TestImageModelDefinition:
    """Tests for ImageModelDefinition dataclass."""

    def test_creates_with_required_fields(self):
        """ImageModelDefinition creates with required fields."""
        model = ImageModelDefinition(
            display_name="Flux 2 Klein",
            api_model_name="black-forest-labs/flux.2-klein-4b",
            description="Fast image generation model"
        )
        assert model.display_name == "Flux 2 Klein"
        assert model.api_model_name == "black-forest-labs/flux.2-klein-4b"
        assert model.description == "Fast image generation model"

    def test_default_is_false(self):
        """default field defaults to False."""
        model = ImageModelDefinition(
            display_name="Test Model",
            api_model_name="test/model",
            description="Test description"
        )
        assert model.default is False

    def test_default_can_be_set_true(self):
        """default field can be set to True."""
        model = ImageModelDefinition(
            display_name="Test Model",
            api_model_name="test/model",
            description="Test description",
            default=True
        )
        assert model.default is True

    def test_is_mutable(self):
        """ImageModelDefinition fields are mutable (not frozen)."""
        model = ImageModelDefinition(
            display_name="Original Name",
            api_model_name="original/model",
            description="Original description"
        )
        model.display_name = "New Name"
        assert model.display_name == "New Name"

    def test_equality(self):
        """ImageModelDefinition compares by value."""
        model1 = ImageModelDefinition(
            display_name="Test",
            api_model_name="test/model",
            description="Desc"
        )
        model2 = ImageModelDefinition(
            display_name="Test",
            api_model_name="test/model",
            description="Desc"
        )
        # Dataclasses compare field values by default
        assert model1 == model2

    def test_different_models_not_equal(self):
        """Different ImageModelDefinition instances are not equal."""
        model1 = ImageModelDefinition(
            display_name="Test1",
            api_model_name="test/model1",
            description="Desc"
        )
        model2 = ImageModelDefinition(
            display_name="Test2",
            api_model_name="test/model2",
            description="Desc"
        )
        assert model1 != model2

    def test_repr_contains_info(self):
        """ImageModelDefinition repr contains useful information."""
        model = ImageModelDefinition(
            display_name="Test Model",
            api_model_name="test/model",
            description="Test description"
        )
        repr_str = repr(model)
        assert "ImageModelDefinition" in repr_str
        assert "Test Model" in repr_str


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_module_exists(self):
        """Module can be imported."""
        from persbot.services import image_model_service
        assert image_model_service is not None

    def test_has_load_function(self):
        """Module has _load_image_models function."""
        from persbot.services.image_model_service import _load_image_models
        assert callable(_load_image_models)

    def test_has_get_available_function(self):
        """Module has get_available_image_models function."""
        from persbot.services.image_model_service import get_available_image_models
        assert callable(get_available_image_models)

    def test_has_get_by_name_function(self):
        """Module has get_image_model_by_name function."""
        from persbot.services.image_model_service import get_image_model_by_name
        assert callable(get_image_model_by_name)

    def test_has_set_channel_function(self):
        """Module has set_channel_image_model function."""
        from persbot.services.image_model_service import set_channel_image_model
        assert callable(set_channel_image_model)

    def test_has_get_channel_function(self):
        """Module has get_channel_image_model function."""
        from persbot.services.image_model_service import get_channel_image_model
        assert callable(get_channel_image_model)

    def test_has_clear_channel_function(self):
        """Module has clear_channel_image_model function."""
        from persbot.services.image_model_service import clear_channel_image_model
        assert callable(clear_channel_image_model)

    def test_has_get_default_function(self):
        """Module has get_default_image_model function."""
        from persbot.services.image_model_service import get_default_image_model
        assert callable(get_default_image_model)

    def test_has_set_default_function(self):
        """Module has set_default_image_model function."""
        from persbot.services.image_model_service import set_default_image_model
        assert callable(set_default_image_model)

"""Tests for Z.AI model wrapper.

Tests cover:
- ZAIChatModel: Wrapper for Z.AI chat models with OpenAI-compatible API
"""

from unittest.mock import Mock

import pytest

from persbot.services.model_wrappers.zai_model import ZAIChatModel
from persbot.services.session_wrappers.zai_session import ZAIChatSession


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client (Z.AI uses OpenAI-compatible API)."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(
        return_value=Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Test response",
                        tool_calls=None,
                    )
                )
            ]
        )
    )
    return client


@pytest.fixture
def zai_model_params():
    """Create basic ZAI model parameters."""
    return {
        "client": Mock(),  # Will be overridden in tests
        "model_name": "zai-model",
        "system_instruction": "You are a helpful assistant.",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_messages": 50,
        "text_extractor": None,
    }


# ============================================================================
# ZAIChatModel.__init__ Tests
# ============================================================================


class TestZAIChatModelInit:
    """Tests for ZAIChatModel.__init__()."""

    def test_initializes_with_all_parameters(self, mock_openai_client):
        """__init__ stores all provided parameters."""
        text_extractor = lambda x: x.get("content", "")
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-large",
            system_instruction="You are helpful.",
            temperature=0.8,
            top_p=0.95,
            max_messages=100,
            text_extractor=text_extractor,
        )

        assert model._client == mock_openai_client
        assert model._model_name == "zai-large"
        assert model._system_instruction == "You are helpful."
        assert model._temperature == 0.8
        assert model._top_p == 0.95
        assert model._max_messages == 100
        assert model._text_extractor == text_extractor

    def test_initializes_with_required_parameters_only(self, mock_openai_client):
        """__init__ uses defaults for optional parameters."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-large",
            system_instruction="System instruction",
            temperature=0.7,
            top_p=0.9,
        )

        assert model._max_messages == 50  # Default
        assert model._text_extractor is None  # Default

    def test_initializes_with_different_parameter_combinations(
        self, mock_openai_client
    ):
        """__init__ handles various parameter combinations."""
        # With max_messages, without text_extractor
        model1 = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model-1",
            system_instruction="Instruction 1",
            temperature=0.5,
            top_p=0.8,
            max_messages=200,
        )
        assert model1._max_messages == 200
        assert model1._text_extractor is None

        # Without max_messages, with text_extractor
        extractor = lambda x: "extracted"
        model2 = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model-2",
            system_instruction="Instruction 2",
            temperature=1.0,
            top_p=1.0,
            text_extractor=extractor,
        )
        assert model2._max_messages == 50
        assert model2._text_extractor == extractor


# ============================================================================
# ZAIChatModel.model_name Property Tests
# ============================================================================


class TestZAIChatModelModelNameProperty:
    """Tests for ZAIChatModel.model_name property."""

    def test_model_name_returns_configured_name(self, mock_openai_client):
        """model_name returns the configured model name."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-pro-model",
            system_instruction="System",
            temperature=0.7,
            top_p=0.9,
        )

        assert model.model_name == "zai-pro-model"

    def test_model_name_different_models(self, mock_openai_client):
        """model_name works with different model names."""
        model_names = ["zai-small", "zai-medium", "zai-large", "zai-coding"]

        for name in model_names:
            model = ZAIChatModel(
                client=mock_openai_client,
                model_name=name,
                system_instruction="System",
                temperature=0.7,
                top_p=0.9,
            )
            assert model.model_name == name


# ============================================================================
# ZAIChatModel.start_chat Tests
# ============================================================================


class TestZAIChatModelStartChat:
    """Tests for ZAIChatModel.start_chat()."""

    def test_start_chat_returns_zai_chat_session(self, mock_openai_client):
        """start_chat returns a ZAIChatSession instance."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="Base instruction",
            temperature=0.7,
            top_p=0.9,
            max_messages=100,
        )

        session = model.start_chat()

        assert isinstance(session, ZAIChatSession)
        assert session._client == mock_openai_client
        assert session._model_name == "zai-model"
        assert session._system_instruction == "Base instruction"
        assert session._temperature == 0.7
        assert session._top_p == 0.9
        assert session._max_messages == 100
        # BaseOpenAISession provides a default lambda when None is passed
        assert session._text_extractor is not None
        assert session._text_extractor("test") == ""

    def test_start_chat_with_system_instruction_override(self, mock_openai_client):
        """start_chat with system_instruction parameter overrides base instruction."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="Base instruction",
            temperature=0.7,
            top_p=0.9,
        )

        custom_instruction = "Custom override instruction"
        session = model.start_chat(system_instruction=custom_instruction)

        assert session._system_instruction == custom_instruction

    def test_start_chat_without_system_instruction_override(self, mock_openai_client):
        """start_chat without system_instruction uses base instruction."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="Base instruction",
            temperature=0.7,
            top_p=0.9,
        )

        session = model.start_chat(system_instruction=None)

        assert session._system_instruction == "Base instruction"

    def test_start_chat_preserves_all_parameters(self, mock_openai_client):
        """start_chat preserves all model parameters in the session."""
        text_extractor = lambda x: x.get("content", "")
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-large",
            system_instruction="You are helpful.",
            temperature=0.85,
            top_p=0.92,
            max_messages=150,
            text_extractor=text_extractor,
        )

        session = model.start_chat()

        assert session._client == mock_openai_client
        assert session._model_name == "zai-large"
        assert session._system_instruction == "You are helpful."
        assert session._temperature == 0.85
        assert session._top_p == 0.92
        assert session._max_messages == 150
        assert session._text_extractor == text_extractor

    def test_start_chat_with_empty_system_instruction(self, mock_openai_client):
        """start_chat with empty string system_instruction falls back to base instruction."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="Base instruction",
            temperature=0.7,
            top_p=0.9,
        )

        session = model.start_chat(system_instruction="")

        # Empty string is falsy, so falls back to base instruction due to "or" logic
        assert session._system_instruction == "Base instruction"

    def test_start_chat_creates_independent_sessions(self, mock_openai_client):
        """start_chat creates independent session instances."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="Base",
            temperature=0.7,
            top_p=0.9,
        )

        session1 = model.start_chat()
        session2 = model.start_chat()

        # Sessions should be different instances
        assert session1 is not session2
        # But have the same configuration
        assert session1._model_name == session2._model_name
        assert session1._temperature == session2._temperature

    def test_start_chat_with_different_overrides(self, mock_openai_client):
        """start_chat with different system_instruction overrides creates varied sessions."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="Base instruction",
            temperature=0.7,
            top_p=0.9,
        )

        session1 = model.start_chat(system_instruction="Custom 1")
        session2 = model.start_chat(system_instruction="Custom 2")
        session3 = model.start_chat()

        assert session1._system_instruction == "Custom 1"
        assert session2._system_instruction == "Custom 2"
        assert session3._system_instruction == "Base instruction"


# ============================================================================
# ZAIChatModel Integration Tests
# ============================================================================


class TestZAIChatModelIntegration:
    """Integration tests for ZAIChatModel behavior."""

    def test_multiple_sessions_from_same_model(self, mock_openai_client):
        """Multiple sessions can be created from the same model instance."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="You are helpful.",
            temperature=0.7,
            top_p=0.9,
            max_messages=50,
        )

        sessions = [model.start_chat() for _ in range(3)]

        # All sessions should be independent
        for i, session in enumerate(sessions):
            assert isinstance(session, ZAIChatSession)
            assert session._model_name == "zai-model"
            # Verify each is a different instance
            for j, other_session in enumerate(sessions):
                if i != j:
                    assert session is not other_session

    def test_model_configuration_immutability(self, mock_openai_client):
        """Model configuration remains unchanged across session creations."""
        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="Original",
            temperature=0.7,
            top_p=0.9,
            max_messages=100,
        )

        # Create multiple sessions with different overrides
        session1 = model.start_chat(system_instruction="Override 1")
        session2 = model.start_chat(system_instruction="Override 2")

        # Original model configuration should be unchanged
        assert model._system_instruction == "Original"
        assert model._temperature == 0.7
        assert model._top_p == 0.9
        assert model._max_messages == 100

        # Sessions should have their overrides
        assert session1._system_instruction == "Override 1"
        assert session2._system_instruction == "Override 2"

    def test_text_extractor_propagation(self, mock_openai_client):
        """text_extractor is correctly propagated to sessions."""
        extractor1 = lambda x: "extracted1"
        extractor2 = lambda x: "extracted2"

        model = ZAIChatModel(
            client=mock_openai_client,
            model_name="zai-model",
            system_instruction="System",
            temperature=0.7,
            top_p=0.9,
            text_extractor=extractor1,
        )

        session = model.start_chat()

        assert session._text_extractor == extractor1
        assert session._text_extractor("any_input") == "extracted1"

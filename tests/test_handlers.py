"""Feature tests for handler modules.

Tests focus on behavior using mocking to avoid Discord dependencies:
- ModelCommandHandler: model management commands
- PersonaCommandHandler: persona management commands
"""

import sys
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest


# Mock external dependencies before any imports
_mock_ddgs = MagicMock()
_mock_ddgs.DDGS = MagicMock
_mock_ddgs.exceptions = MagicMock()
_mock_ddgs.exceptions.RatelimitException = Exception
_mock_ddgs.exceptions.DDGSException = Exception
sys.modules['ddgs'] = _mock_ddgs
sys.modules['ddgs.exceptions'] = _mock_ddgs.exceptions

_mock_bs4 = MagicMock()
sys.modules['bs4'] = _mock_bs4


# Mock discord module before importing
@pytest.fixture(autouse=True)
def mock_discord():
    """Mock discord module to avoid import issues."""
    mock_discord = MagicMock()
    mock_discord.Embed = MagicMock
    mock_discord.Color = MagicMock()
    mock_discord.Color.blue = MagicMock(return_value=0x3498DB)
    mock_discord.Color.green = MagicMock(return_value=0x2ECC71)
    mock_discord.Color.red = MagicMock(return_value=0xE74C3C)
    sys.modules['discord'] = mock_discord
    sys.modules['discord.ext'] = MagicMock()
    sys.modules['discord.ext.commands'] = MagicMock()
    yield mock_discord
    if 'discord' in sys.modules:
        del sys.modules['discord']
    if 'discord.ext' in sys.modules:
        del sys.modules['discord.ext']
    if 'discord.ext.commands' in sys.modules:
        del sys.modules['discord.ext.commands']


class TestModelCommandHandler:
    """Tests for ModelCommandHandler class."""

    @pytest.fixture
    def mock_bot(self):
        """Create a mock bot."""
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.assistant_model_name = "gemini-2.5-flash"
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        service.provider_label = "Gemini"
        service.update_parameters = Mock()
        return service

    @pytest.fixture
    def mock_model_usage_service(self):
        """Create a mock model usage service."""
        service = Mock()
        service.MODEL_DEFINITIONS = {
            "gemini-flash": Mock(
                provider="gemini",
                display_name="Gemini 2.5 Flash",
                api_model="gemini-2.5-flash",
            ),
            "gpt-4": Mock(
                provider="openai",
                display_name="GPT-4",
                api_model="gpt-4",
            ),
        }
        service.get_available_models = Mock(return_value=service.MODEL_DEFINITIONS)
        return service

    def test_model_command_handler_exists(self):
        """ModelCommandHandler class exists."""
        from persbot.bot.handlers.model_handler import ModelCommandHandler
        assert ModelCommandHandler is not None

    def test_creates_with_dependencies(self, mock_bot, mock_config, mock_llm_service, mock_model_usage_service):
        """ModelCommandHandler creates with dependencies."""
        from persbot.bot.handlers.model_handler import ModelCommandHandler

        # Create a concrete implementation for testing
        class ConcreteModelHandler(ModelCommandHandler):
            async def handle(self, ctx):
                pass

        handler = ConcreteModelHandler(
            mock_bot, mock_config, mock_llm_service, mock_model_usage_service
        )
        assert handler.llm_service == mock_llm_service
        assert handler.model_usage_service == mock_model_usage_service


class TestPersonaCommandHandler:
    """Tests for PersonaCommandHandler class."""

    @pytest.fixture
    def mock_bot(self):
        """Create a mock bot."""
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return Mock()

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        return Mock()

    @pytest.fixture
    def mock_prompt_service(self):
        """Create a mock prompt service."""
        service = Mock()
        service.list_personas = Mock(return_value=[])
        service.get_persona = Mock(return_value=None)
        service.get_active_persona = Mock(return_value=None)
        service.set_active_persona = Mock()
        service.save_persona = Mock()
        return service

    def test_persona_command_handler_exists(self):
        """PersonaCommandHandler class exists."""
        from persbot.bot.handlers.persona_handler import PersonaCommandHandler
        assert PersonaCommandHandler is not None

    def test_creates_with_dependencies(self, mock_bot, mock_config, mock_llm_service, mock_prompt_service):
        """PersonaCommandHandler creates with dependencies."""
        from persbot.bot.handlers.persona_handler import PersonaCommandHandler

        # Create a concrete implementation for testing
        class ConcretePersonaHandler(PersonaCommandHandler):
            async def handle(self, ctx):
                pass

        handler = ConcretePersonaHandler(
            mock_bot, mock_config, mock_llm_service, mock_prompt_service
        )
        assert handler.llm_service == mock_llm_service
        assert handler.prompt_service == mock_prompt_service


class TestBaseHandler:
    """Tests for BaseHandler class."""

    def test_base_handler_exists(self):
        """BaseHandler class exists."""
        from persbot.bot.handlers.base_handler import BaseHandler
        assert BaseHandler is not None

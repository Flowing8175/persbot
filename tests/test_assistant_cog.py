"""Feature tests for AssistantCog.

Tests focus on behavior:
- AssistantCog initialization and setup
- Message handling and filtering
"""

import asyncio
import sys
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest
import discord
from discord.ext import commands

# Mock ddgs module before any imports that depend on it
_mock_ddgs = MagicMock()
_mock_ddgs.DDGS = MagicMock
_mock_ddgs.exceptions = MagicMock()
_mock_ddgs.exceptions.RatelimitException = Exception
_mock_ddgs.exceptions.DDGSException = Exception
sys.modules['ddgs'] = _mock_ddgs
_sys_modules = {'ddgs': _mock_ddgs, 'ddgs.exceptions': _mock_ddgs.exceptions}

# Mock bs4 module before any imports that depend on it
_mock_bs4 = MagicMock()
_sys_modules['bs4'] = _mock_bs4
for k, v in _sys_modules.items():
    sys.modules[k] = v


class TestAssistantCogInit:
    """Tests for AssistantCog initialization."""

    @pytest.fixture
    def mock_bot(self):
        """Create a mock bot."""
        bot = MagicMock(spec=commands.Bot)
        bot.user = Mock()
        bot.user.id = 12345
        return bot

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.auto_reply_channel_ids = []
        config.message_buffer_delay = 0.5
        config.break_cut_mode = False
        config.session_inactive_minutes = 0
        config.session_cache_limit = 100
        config.assistant_model_name = "gemini-2.5-flash"
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        service.model_usage_service = Mock()
        service.model_usage_service.MODEL_DEFINITIONS = {}
        service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-flash"
        service.get_assistant_role_name = Mock(return_value="assistant")
        service.get_user_role_name = Mock(return_value="user")
        return service

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        manager = Mock()
        manager.sessions = {}
        manager.session_contexts = {}
        return manager

    @pytest.fixture
    def mock_prompt_service(self):
        """Create a mock prompt service."""
        service = Mock()
        return service

    def test_cog_exists(self):
        """AssistantCog class exists."""
        from persbot.bot.cogs.assistant import AssistantCog
        assert AssistantCog is not None

    def test_creates_with_required_dependencies(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service):
        """AssistantCog creates with required dependencies."""
        from persbot.bot.cogs.assistant import AssistantCog

        cog = AssistantCog(mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service)
        assert cog.bot == mock_bot
        assert cog.config == mock_config
        assert cog.llm_service == mock_llm_service
        assert cog.session_manager == mock_session_manager
        assert cog.prompt_service == mock_prompt_service

    def test_extends_base_chat_cog(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service):
        """AssistantCog extends BaseChatCog."""
        from persbot.bot.cogs.assistant import AssistantCog
        from persbot.bot.cogs.base import BaseChatCog

        cog = AssistantCog(mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service)
        assert isinstance(cog, BaseChatCog)

    def test_initializes_message_buffer(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service):
        """Message buffer is initialized."""
        from persbot.bot.cogs.assistant import AssistantCog

        cog = AssistantCog(mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service)
        assert cog.message_buffer is not None

    def test_initializes_task_tracking_dicts(self, mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service):
        """Task tracking dictionaries are initialized."""
        from persbot.bot.cogs.assistant import AssistantCog

        cog = AssistantCog(mock_bot, mock_config, mock_llm_service, mock_session_manager, mock_prompt_service)
        assert cog.processing_tasks == {}
        assert cog.sending_tasks == {}
        assert cog.active_batches == {}
        assert cog.cancellation_signals == {}
        assert cog.active_api_calls == {}


class TestAssistantCogMessageHandling:
    """Tests for AssistantCog message handling."""

    @pytest.fixture
    def setup_cog(self):
        """Setup a complete cog instance for testing."""
        from persbot.bot.cogs.assistant import AssistantCog

        bot = MagicMock(spec=commands.Bot)
        bot.user = Mock()
        bot.user.id = 12345

        config = Mock()
        config.auto_reply_channel_ids = []
        config.message_buffer_delay = 0.1
        config.break_cut_mode = False
        config.session_inactive_minutes = 0
        config.session_cache_limit = 100
        config.assistant_model_name = "gemini-2.5-flash"

        llm_service = Mock()
        llm_service.model_usage_service = Mock()
        llm_service.model_usage_service.MODEL_DEFINITIONS = {}
        llm_service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-flash"
        llm_service.get_assistant_role_name = Mock(return_value="assistant")
        llm_service.get_user_role_name = Mock(return_value="user")

        session_manager = Mock()
        session_manager.sessions = {}
        session_manager.session_contexts = {}

        prompt_service = Mock()

        cog = AssistantCog(bot, config, llm_service, session_manager, prompt_service)
        return cog, bot, config

    @pytest.mark.asyncio
    async def test_on_message_ignores_bot_messages(self, setup_cog):
        """Bot messages are ignored."""
        cog, bot, config = setup_cog

        message = Mock(spec=discord.Message)
        message.author = Mock()
        message.author.bot = True
        message.author.id = 999

        await cog.on_message(message)
        # Should not process - no task created
        assert len(cog.processing_tasks) == 0

    @pytest.mark.asyncio
    async def test_on_message_ignores_without_mention(self, setup_cog):
        """Messages without bot mention are ignored."""
        cog, bot, config = setup_cog

        message = Mock(spec=discord.Message)
        message.author = Mock()
        message.author.bot = False
        message.author.id = 999
        message.channel = Mock()
        message.channel.id = 123
        message.mention_everyone = False

        bot.user.mentioned_in = Mock(return_value=False)

        await cog.on_message(message)
        # Should not process - no task created
        assert len(cog.processing_tasks) == 0


class TestAssistantCogTyping:
    """Tests for AssistantCog typing event handling."""

    @pytest.fixture
    def setup_cog(self):
        """Setup a complete cog instance for testing."""
        from persbot.bot.cogs.assistant import AssistantCog

        bot = MagicMock(spec=commands.Bot)
        bot.user = Mock()
        bot.user.id = 12345

        config = Mock()
        config.auto_reply_channel_ids = []
        config.message_buffer_delay = 0.1
        config.break_cut_mode = True  # Enable break-cut mode
        config.session_inactive_minutes = 0
        config.session_cache_limit = 100
        config.assistant_model_name = "gemini-2.5-flash"

        llm_service = Mock()
        llm_service.model_usage_service = Mock()
        llm_service.model_usage_service.MODEL_DEFINITIONS = {}
        llm_service.model_usage_service.DEFAULT_MODEL_ALIAS = "gemini-flash"

        session_manager = Mock()
        prompt_service = Mock()

        cog = AssistantCog(bot, config, llm_service, session_manager, prompt_service)
        return cog, bot, config

    @pytest.mark.asyncio
    async def test_on_typing_ignored_for_bot_users(self, setup_cog):
        """Typing events from bot users are ignored."""
        cog, bot, config = setup_cog

        channel = Mock()
        user = Mock()
        user.bot = True
        when = Mock()

        # Should not raise or process
        await cog.on_typing(channel, user, when)

    @pytest.mark.asyncio
    async def test_on_typing_does_not_raise(self, setup_cog):
        """Typing events don't raise exceptions."""
        cog, bot, config = setup_cog

        channel = Mock()
        channel.id = 123
        user = Mock()
        user.bot = False
        when = Mock()

        # Should not raise
        await cog.on_typing(channel, user, when)

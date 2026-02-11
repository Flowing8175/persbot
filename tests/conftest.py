"""Pytest configuration and shared fixtures for SoyeBot testing."""

# Import Mock first before using it
import sys
from unittest.mock import Mock
from unittest.mock import Mock as MockClass

# Mock external dependencies before ANY imports - must be at top level
# Create mock for google.genai.errors
mock_genai_errors = MockClass()
mock_genai_errors.ClientError = Exception
mock_genai_errors.APIError = Exception

# Create mock for google.genai
mock_genai = MockClass()
mock_genai.errors = mock_genai_errors
mock_genai.types = MockClass()

sys.modules["google"] = MockClass()
sys.modules["google.genai"] = mock_genai
sys.modules["google.genai.errors"] = mock_genai_errors

sys.modules["openai"] = Mock()
sys.modules["anthropic"] = Mock()

mock_pil = Mock()
mock_pil.Image = Mock()
mock_pil.Image.Resampling = Mock()
mock_pil.Image.Resampling.LANCZOS = 1
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil.Image

import asyncio
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure pytest-asyncio mode
pytest_asyncio_modes = "auto"


@pytest.fixture
def mock_discord_config():
    """Mock Discord configuration for testing."""
    return {
        "DISCORD_TOKEN": "mock_discord_token_12345",
        "GEMINI_API_KEY": "mock_gemini_key_67890",
        "OPENAI_API_KEY": "mock_openai_key_abcde",
        "ZAI_API_KEY": "mock_zai_key_fghij",
    }


@pytest.fixture
def mock_app_config(mock_discord_config):
    """Create a mock AppConfig for testing."""
    from persbot.config import AppConfig

    with patch.dict(os.environ, mock_discord_config, clear=True):
        config = AppConfig(discord_token=mock_discord_config["DISCORD_TOKEN"])
        # Set test-specific values
        config.discord_token = mock_discord_config["DISCORD_TOKEN"]
        config.gemini_api_key = mock_discord_config["GEMINI_API_KEY"]
        config.openai_api_key = mock_discord_config["OPENAI_API_KEY"]
        config.zai_api_key = mock_discord_config["ZAI_API_KEY"]
        config.command_prefix = "!"
        config.auto_reply_channel_ids = ()
        config.no_check_permission = True  # Disable permission checks in tests
        config.api_request_timeout = 30.0
        config.api_max_retries = 3
        config.api_rate_limit_retry_after = 10.0
        config.api_retry_backoff_base = 2.0
        config.api_retry_backoff_max = 60.0
        config.session_cache_limit = 10
        config.session_ttl_minutes = 30
        config.temperature = 1.0
        config.top_p = 1.0
        config.thinking_budget = None
        config.break_cut_mode = False
        config.max_messages_per_fetch = 100

        yield config


@pytest.fixture
def mock_bot(mock_app_config):
    """Create a mock Discord bot instance."""
    bot = Mock()
    bot.user = Mock(id=123456789, name="TestBot")
    bot.add_cog = Mock()
    bot.tree = Mock()
    bot.tree.sync = AsyncMock(return_value=[])
    bot.get_cog = Mock()

    return bot


@pytest.fixture
def mock_guild():
    """Create a mock Discord guild."""
    guild = Mock()
    guild.id = 987654321
    guild.name = "Test Server"
    guild.permissions_in = Mock()
    return guild


@pytest.fixture
def mock_channel(mock_guild):
    """Create a mock Discord channel."""
    channel = Mock()
    channel.id = 111222333
    channel.name = "test-channel"
    channel.guild = mock_guild
    channel.typing = Mock()
    channel.history = AsyncMock()
    channel.send = AsyncMock(return_value=Mock())
    channel.fetch_message = AsyncMock()
    channel.mention_everyone = False
    return channel


@pytest.fixture
def mock_user():
    """Create a mock Discord user."""
    user = Mock()
    user.id = 123456789
    user.name = "TestUser"
    user.display_name = "Test User"
    user.avatar = Mock()
    user.bot = False
    user.guild_permissions = Mock()
    user.guild_permissions.manage_guild = True  # Admin permissions
    return user


@pytest.fixture
def mock_message(mock_user, mock_channel):
    """Create a mock Discord message."""
    message = Mock()
    message.id = "999888777666555444"
    message.author = mock_user
    message.channel = mock_channel
    message.content = "Test message content"
    message.clean_content = "Test message content"
    message.mentions = []
    message.reference = None
    message.attachments = []
    message.embeds = []
    message.created_at = datetime.now(timezone.utc)
    message.reply = AsyncMock()
    message.delete = AsyncMock()
    message.add_reaction = Mock()
    return message


@pytest.fixture
def mock_mention_message(mock_message, mock_bot):
    """Create a mock message with bot mention."""
    message = Mock()
    message.id = "999888777666555445"
    message.author = mock_message.author
    message.channel = mock_message.channel
    message.content = "<@123456789> Hello bot"
    message.clean_content = "Hello bot"
    message.mentions = [mock_bot.user]
    message.mention_everyone = False
    message.reference = None
    message.attachments = []
    message.embeds = []
    message.created_at = datetime.now(timezone.utc)
    message.reply = AsyncMock()
    message.delete = AsyncMock()
    return message


@pytest.fixture
def mock_attachment():
    """Create a mock Discord attachment."""
    attachment = Mock()
    attachment.id = "555444333222111"
    attachment.filename = "test_image.png"
    attachment.content_type = "image/png"
    attachment.size = 102400  # 100KB
    attachment.read = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n...")
    return attachment


@pytest.fixture
def mock_image_message(mock_user, mock_channel, mock_attachment):
    """Create a mock message with image attachment."""
    message = Mock()
    message.id = "999888777666555446"
    message.author = mock_user
    message.channel = mock_channel
    message.content = ""
    message.clean_content = ""
    message.mentions = []
    message.mention_everyone = False
    message.reference = None
    message.attachments = [mock_attachment]
    message.embeds = []
    message.created_at = datetime.now(timezone.utc)
    message.reply = AsyncMock()
    message.delete = AsyncMock()
    return message


@pytest.fixture
def mock_interaction(mock_user, mock_channel):
    """Create a mock Discord interaction (slash command)."""
    interaction = Mock()
    interaction.id = "111222333444555"
    interaction.user = mock_user
    interaction.channel = mock_channel
    interaction.response = Mock()
    interaction.response.is_done = Mock(return_value=False)
    interaction.response.defer = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.response.send_message = AsyncMock(return_value=Mock())
    interaction.response.followup = AsyncMock()
    interaction.response.defer = AsyncMock()
    interaction.message = Mock()
    return interaction


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = Mock()
    service.generate_chat_response = AsyncMock(return_value=("Mock LLM response", None))
    service.summarize_text = AsyncMock(return_value="Mock summary")
    service.generate_prompt_from_concept = AsyncMock(return_value="Mock generated prompt")
    service.create_assistant_model = Mock(
        return_value=Mock(start_chat=Mock(return_value="Chat response"))
    )
    service.create_chat_session_for_alias = Mock(return_value=Mock())
    service.update_parameters = Mock()
    service.get_user_role_name = Mock(return_value="user")
    service.get_assistant_role_name = Mock(return_value="model")
    service.get_backend_for_model = Mock(return_value=Mock())
    return service


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager."""
    manager = Mock()
    manager.sessions = OrderedDict()
    manager.session_contexts = OrderedDict()
    manager.channel_prompts = {}
    manager.channel_model_preferences = {}
    manager.get_or_create = AsyncMock(return_value=(Mock(), "channel:123"))
    manager.resolve_session = Mock(
        return_value=Mock(session_key="channel:123", cleaned_message="test")
    )
    manager.reset_session_by_channel = Mock(return_value=True)
    manager.undo_last_exchanges = Mock(return_value=[])
    manager.set_session_model = Mock()
    manager.set_channel_prompt = Mock()
    return manager


@pytest.fixture
def mock_prompt_service():
    """Create a mock PromptService."""
    service = Mock()
    service.list_prompts = Mock(
        return_value=[
            {
                "name": "Default",
                "content": "Default prompt",
                "path": "/path/to/default.md",
            },
            {
                "name": "Custom",
                "content": "Custom prompt",
                "path": "/path/to/custom.md",
            },
        ]
    )
    service.get_prompt = Mock(
        return_value={"name": "Test", "content": "Test content", "path": "/test.md"}
    )
    service.add_prompt = AsyncMock(return_value=0)
    service.rename_prompt = AsyncMock(return_value=True)
    service.delete_prompt = AsyncMock(return_value=True)
    service.check_today_limit = AsyncMock(return_value=True)
    service.increment_today_usage = AsyncMock()
    return service


@pytest.fixture
def mock_model_usage_service():
    """Create a mock ModelUsageService."""
    service = Mock()
    service.check_and_increment_usage = AsyncMock(return_value=(True, "gemini-2.5-flash", None))
    service.get_api_model_name = Mock(return_value="gemini-2.5-flash")
    service.DEFAULT_MODEL_ALIAS = "gemini-2.5-flash"
    service.MODEL_DEFINITIONS = {
        "gemini-2.5-flash": Mock(daily_limit=100, provider="gemini"),
        "gemini-2.5-pro": Mock(daily_limit=50, provider="gemini"),
        "glm-4.7-flash": Mock(daily_limit=200, provider="zai"),
    }
    return service


@pytest.fixture
def mock_usage_service():
    """Create a mock ImageUsageService."""
    service = Mock()
    service.check_can_upload = Mock(return_value=True)
    service.record_upload = AsyncMock()
    return service


@pytest.fixture
def mock_chat_session():
    """Create a mock chat session."""
    session = Mock()
    session.history = []
    session.model_alias = "gemini-2.5-flash"
    session.send_message = AsyncMock(return_value="Response text")
    session.start_chat = Mock(return_value=session)
    return session


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file I/O tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_discord_client():
    """Create a mock Discord client."""
    client = Mock()
    client.user = Mock(id=123456789, name="TestBot")
    client.get_guild = AsyncMock()
    client.get_channel = AsyncMock()
    client.get_user = AsyncMock()
    return client


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client."""
    client = AsyncMock()
    client.aio_client = AsyncMock()
    client.cached_content = AsyncMock()
    client.models = Mock()
    client.models.count_tokens = Mock(return_value=Mock(total_tokens=0))
    return client


@pytest.fixture(autouse=True)
def mock_gemini_cache_methods():
    """Automatically mock Gemini cache methods to prevent API calls during tests."""
    # Configure default return values for config attributes that Mock might return
    original_getattr = Mock.__getattr__
    original_setattr = Mock.__setattr__

    def safe_getattr(self, name):
        """Return sensible defaults for common config attributes."""
        if name == "gemini_cache_ttl_minutes":
            return 60
        if name == "gemini_cache_min_tokens":
            return 32768
        if name == "temperature":
            return 1.0
        if name == "top_p":
            return 1.0
        if name == "thinking_budget":
            return None
        if name == "service_tier":
            return "flex"
        # Fall back to default Mock behavior
        return original_getattr(self, name)

    def safe_setattr(self, name, value):
        """Allow setting attributes to None explicitly in tests."""
        if name in ("openai_api_key", "zai_api_key", "gemini_api_key"):
            if value is None:
                self.__dict__[name] = None
                return
        return original_setattr(self, name, value)

    with patch.object(Mock, "__getattr__", safe_getattr):
        with patch.object(Mock, "__setattr__", safe_setattr):
            with (
                patch(
                    "persbot.services.gemini_service.GeminiService._get_gemini_cache",
                    return_value=(None, None),
                ),
                patch("asyncio.create_task", return_value=Mock()),
            ):
                yield


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    client.chat = AsyncMock()
    client.models = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_zai_client():
    """Create a mock Z.AI client."""
    client = AsyncMock()
    client.chat = AsyncMock()
    client.models = AsyncMock(return_value=[])
    return client


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_korean_times():
    """Sample Korean time expressions for testing."""
    return {
        "30분": 30,
        "1시간": 60,
        "2시간30분": 150,
        "24시간": 1440,
        "1일": 1440,
    }


@pytest.fixture
def sample_models_json():
    """Sample models.json content for testing."""
    return {
        "gemini-2.5-flash": {
            "display_name": "Gemini 2.5 Flash",
            "api_model_name": "gemini-2.5-flash",
            "daily_limit": 100,
            "scope": "guild",
            "provider": "gemini",
            "fallback_alias": None,
        },
        "gemini-2.5-pro": {
            "display_name": "Gemini 2.5 Pro",
            "api_model_name": "gemini-2.5-pro",
            "daily_limit": 50,
            "scope": "guild",
            "provider": "gemini",
            "fallback_alias": "gemini-2.5-flash",
        },
        "glm-4.7-flash": {
            "display_name": "GLM 4.7 Flash",
            "api_model_name": "glm-4.7-flash",
            "daily_limit": 200,
            "scope": "guild",
            "provider": "zai",
            "fallback_alias": None,
        },
    }


@pytest.fixture
def sample_prompts():
    """Sample prompt data for testing."""
    return [
        {
            "name": "default_persona",
            "content": "You are a helpful AI assistant.",
            "path": "/default/path.md",
        },
        {
            "name": "developer_persona",
            "content": "You are an expert developer assistant.",
            "path": "/dev/path.md",
        },
        {
            "name": "creative_writer",
            "content": "You are a creative writing assistant.",
            "path": "/writer/path.md",
        },
    ]


@pytest.fixture
def sample_long_text():
    """Generate a long text for testing splitting logic."""
    return "This is line one.\nThis is line two.\nThis is line three with much more content " * 100


@pytest.fixture
def sample_message_history():
    """Sample chat message history."""
    from persbot.services.base import ChatMessage

    return [
        ChatMessage(role="user", content="Hello", author_id=123456789),
        ChatMessage(role="model", content="Hi there! How can I help?", author_id=None),
        ChatMessage(role="user", content="What is 2+2?", author_id=123456789),
        ChatMessage(role="model", content="2+2 equals 4.", author_id=None),
    ]


# Helper functions for tests
def create_mock_message_with_context(user_id, channel_id, content, mentions=None, attachments=None):
    """Helper to create a mock message with specific attributes."""
    msg = Mock()
    msg.id = f"{user_id}_{channel_id}_{content[:10]}"
    msg.author = Mock(id=user_id, name=f"User{user_id}", bot=False)
    msg.channel = Mock(id=channel_id, name=f"Channel{channel_id}")
    msg.content = content
    msg.clean_content = content
    msg.mentions = mentions or []
    msg.attachments = attachments or []
    msg.embeds = []
    msg.reference = None
    msg.created_at = datetime.now(timezone.utc)
    msg.reply = AsyncMock()
    msg.delete = AsyncMock()
    return msg

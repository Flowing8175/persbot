"""Tests for Discord read-only tools."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from soyebot.tools.discord_tools.channel_tools import (
    get_channel_info,
    get_channel_history,
    get_message,
    list_channels,
)
from soyebot.tools.discord_tools.user_tools import (
    get_user_info,
    get_member_info,
    get_member_roles,
)
from soyebot.tools.discord_tools.guild_tools import (
    get_guild_info,
    get_guild_roles,
    get_guild_emojis,
)


class TestChannelTools:
    """Tests for Discord channel tools."""

    @pytest.fixture
    def mock_guild(self):
        """Create a mock guild."""
        guild = Mock()
        guild.id = 987654321
        guild.name = "Test Server"
        return guild

    @pytest.fixture
    def mock_channel(self, mock_guild):
        """Create a mock channel."""
        channel = Mock()
        channel.id = 111222333
        channel.name = "test-channel"
        channel.type = 0  # TEXT
        category_mock = Mock()
        category_mock.name = "General"
        channel.category = category_mock
        channel.position = 1
        channel.nsfw = False
        channel.topic = "A test channel"
        channel.slowmode_delay = 0
        channel.history = AsyncMock()
        channel.fetch_message = AsyncMock()
        return channel

    @pytest.fixture
    def mock_discord_context(self, mock_channel, mock_guild):
        """Create a mock Discord message context."""
        message = Mock()
        message.guild = mock_guild
        message.channel = mock_channel
        message.guild.get_channel = Mock(return_value=mock_channel)
        message.bot = Mock()
        message.bot.get_guild = Mock(return_value=mock_guild)
        return message

    @pytest.mark.asyncio
    async def test_get_channel_info(self, mock_channel, mock_discord_context):
        """Test getting channel information."""
        result = await get_channel_info(mock_channel.id, mock_discord_context)

        assert result.success is True
        data = result.data
        assert data["id"] == str(mock_channel.id)
        assert data["name"] == "test-channel"
        assert data["type"] == "0"
        assert data["category"] == "General"  # category.name should be accessible
        assert data["position"] == 1
        assert data["nsfw"] is False
        assert data["topic"] == "A test channel"

    @pytest.mark.asyncio
    async def test_get_channel_info_not_found(self, mock_discord_context):
        """Test getting info for non-existent channel."""
        mock_discord_context.guild.get_channel = Mock(return_value=None)

        result = await get_channel_info(999999, mock_discord_context)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_channel_info_no_guild(self):
        """Test getting channel info without guild context."""
        result = await get_channel_info(111222333, None)

        assert result.success is False
        assert "not available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_channel_history(self, mock_channel, mock_discord_context):
        """Test getting channel history."""
        # Create mock messages
        mock_messages = []
        for i in range(5):
            msg = Mock()
            msg.id = f"msg_{i}"
            msg.author = Mock(id=123, name=f"User{i}")
            msg.content = f"Message {i}"
            msg.created_at = datetime.now(timezone.utc)
            msg.attachments = []
            mock_messages.append(msg)

        # Set up async iterator
        async def async_history_iter(*args, **kwargs):
            for msg in mock_messages:
                yield msg

        mock_channel.history = Mock(return_value=async_history_iter())

        result = await get_channel_history(
            mock_channel.id, limit=5, discord_context=mock_discord_context
        )

        assert result.success is True
        data = result.data
        assert data["count"] == 5
        assert len(data["messages"]) == 5
        assert data["messages"][0]["content"] == "Message 0"

    @pytest.mark.asyncio
    async def test_get_message(self, mock_channel, mock_discord_context):
        """Test getting a specific message."""
        mock_msg = Mock()
        mock_msg.id = "msg_123"
        mock_msg.author = Mock(id=123, name="TestUser")
        mock_msg.content = "Test message"
        mock_msg.created_at = datetime.now(timezone.utc)
        mock_msg.edited_at = None
        mock_msg.attachments = []
        mock_msg.reference = None

        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)

        result = await get_message("msg_123", mock_channel.id, mock_discord_context)

        assert result.success is True
        data = result.data
        assert data["id"] == "msg_123"
        assert data["content"] == "Test message"
        assert data["author_id"] == "123"

    @pytest.mark.asyncio
    async def test_list_channels(self, mock_guild, mock_discord_context):
        """Test listing all channels in a guild."""
        # Create mock channels
        text_channel = Mock()
        text_channel.id = 111
        text_channel.name = "general"
        text_channel.type = 0  # TEXT
        text_channel.category = Mock(name="Text Channels")
        text_channel.position = 1

        voice_channel = Mock()
        voice_channel.id = 222
        voice_channel.name = "Voice Chat"
        voice_channel.type = 2  # VOICE
        voice_channel.category = None
        voice_channel.position = 2

        mock_guild.channels = [text_channel, voice_channel]

        result = await list_channels(
            mock_guild.id, discord_context=mock_discord_context
        )

        assert result.success is True
        data = result.data
        assert data["count"] == 2
        assert len(data["channels"]) == 2

        # Check that channels are sorted by position
        assert data["channels"][0]["name"] == "general"
        assert data["channels"][1]["name"] == "Voice Chat"

    @pytest.mark.asyncio
    async def test_list_channels_filtered(self, mock_guild, mock_discord_context):
        """Test listing channels with type filter."""
        text_channel = Mock()
        text_channel.id = 111
        text_channel.name = "general"
        text_channel.type = 0  # Keep as int, the tool converts to str
        text_channel.category = None
        text_channel.position = 1

        voice_channel = Mock()
        voice_channel.id = 222
        voice_channel.name = "Voice Chat"
        voice_channel.type = 2  # Keep as int
        voice_channel.category = None
        voice_channel.position = 2

        mock_guild.channels = [text_channel, voice_channel]

        result = await list_channels(
            mock_guild.id,
            channel_type="0",  # Match the string "0" that the int 0 becomes
            discord_context=mock_discord_context,
        )

        assert result.success is True
        data = result.data
        assert data["count"] == 1
        assert data["channels"][0]["name"] == "general"

    @pytest.mark.asyncio
    async def test_get_channel_info_in_dm(self, mock_channel):
        """Test getting channel info in DM (no guild context)."""
        # Create DM context (no guild)
        dm_context = Mock()
        dm_context.guild = None
        dm_context.channel = mock_channel
        dm_context.bot = Mock()

        result = await get_channel_info(None, dm_context)

        # Should handle gracefully with clear error message
        assert result.success is False
        assert (
            "not available" in result.error.lower()
            or "not in a guild" in result.error.lower()
        )


class TestUserTools:
    """Tests for Discord user tools."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        user = Mock()
        user.id = 123456789
        user.name = "TestUser"
        user.display_name = "Test User"
        user.discriminator = "1234"
        user.bot = False
        user.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        user.avatar = Mock()
        user.avatar.url = "https://example.com/avatar.png"
        user.global_name = "Test User Global"
        return user

    @pytest.fixture
    def mock_member(self, mock_user):
        """Create a mock guild member."""
        member = Mock()
        member.id = mock_user.id
        member.name = mock_user.name
        member.display_name = mock_user.display_name
        member.bot = mock_user.bot
        member.joined_at = datetime(2021, 1, 1, tzinfo=timezone.utc)
        member.premium_since = None
        member.pending = False
        member.avatar = Mock()
        member.avatar.url = "https://example.com/avatar.png"
        member.guild_avatar = None
        member.roles = []
        return member

    @pytest.fixture
    def mock_discord_context(self, mock_user, mock_member):
        """Create a mock Discord message context."""
        message = Mock()
        message.guild = Mock(id=987654321)
        message.bot = Mock()
        message.bot.fetch_user = AsyncMock(return_value=mock_user)
        message.guild.get_member = Mock(return_value=mock_member)
        message.guild.fetch_member = AsyncMock(return_value=mock_member)
        message.bot.get_guild = Mock(return_value=message.guild)
        return message

    @pytest.mark.asyncio
    async def test_get_user_info(self, mock_user, mock_discord_context):
        """Test getting user information."""
        result = await get_user_info(mock_user.id, mock_discord_context)

        assert result.success is True
        data = result.data
        assert data["id"] == str(mock_user.id)
        assert data["username"] == "TestUser"
        assert data["display_name"] == "Test User"
        assert data["bot"] is False

    @pytest.mark.asyncio
    async def test_get_member_info(self, mock_member, mock_discord_context):
        """Test getting member information."""
        result = await get_member_info(
            mock_member.id,
            mock_discord_context.guild.id,
            mock_discord_context,
        )

        assert result.success is True
        data = result.data
        assert data["id"] == str(mock_member.id)
        assert data["username"] == "TestUser"
        assert data["joined_at"] is not None

    @pytest.mark.asyncio
    async def test_get_member_roles(self, mock_member, mock_discord_context):
        """Test getting member roles."""
        # Create mock roles
        admin_role = Mock()
        admin_role.id = 1
        admin_role.name = "Admin"
        admin_role.color = Mock(__str__=Mock(return_value="#FF0000"))
        admin_role.position = 10
        admin_role.permissions = Mock()
        admin_role.permissions.value = 8
        admin_role.managed = False
        admin_role.mentionable = True
        admin_role.hoist = True
        admin_role.is_default = Mock(return_value=False)

        mod_role = Mock()
        mod_role.id = 2
        mod_role.name = "Moderator"
        mod_role.color = Mock(__str__=Mock(return_value="#00FF00"))
        mod_role.position = 5
        mod_role.permissions = Mock()
        mod_role.permissions.value = 4
        mod_role.managed = False
        mod_role.mentionable = False
        mod_role.hoist = False
        mod_role.is_default = Mock(return_value=False)

        everyone_role = Mock()
        everyone_role.id = 3
        everyone_role.name = "@everyone"
        everyone_role.is_default = Mock(return_value=True)

        mock_member.roles = [admin_role, mod_role, everyone_role]

        result = await get_member_roles(
            mock_member.id,
            mock_discord_context.guild.id,
            mock_discord_context,
        )

        assert result.success is True
        data = result.data
        # @everyone should be filtered out
        assert data["count"] == 2
        assert len(data["roles"]) == 2

        # Check sorting by position (highest first)
        assert data["roles"][0]["name"] == "Admin"
        assert data["roles"][1]["name"] == "Moderator"

    @pytest.mark.asyncio
    async def test_get_user_info_uses_context(self, mock_user, mock_discord_context):
        """Test that get_user_info uses discord_context when user_id is None."""
        # Call with None user_id - should auto-fill from context
        result = await get_user_info(None, mock_discord_context)

        assert result.success is True
        data = result.data
        assert data["id"] == str(mock_user.id)
        assert data["username"] == "TestUser"
        assert data["display_name"] == "Test User"

    @pytest.mark.asyncio
    async def test_explicit_id_overrides_context(self, mock_discord_context):
        """Test that explicit user_id parameter overrides context."""
        # Create a different user
        other_user = Mock()
        other_user.id = 999888777
        other_user.name = "OtherUser"
        other_user.display_name = "Other User"
        other_user.discriminator = "5678"
        other_user.bot = False
        other_user.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        other_user.avatar = Mock()
        other_user.avatar.url = "https://example.com/other.png"
        other_user.global_name = "Other User Global"

        # Update bot to return the other user when requested
        mock_discord_context.bot.fetch_user = AsyncMock(return_value=other_user)

        # Call with explicit user_id - should use this instead of context
        result = await get_user_info(other_user.id, mock_discord_context)

        assert result.success is True
        data = result.data
        assert data["id"] == str(other_user.id)
        assert data["username"] == "OtherUser"
        assert data["display_name"] == "Other User"


class TestGuildTools:
    """Tests for Discord guild tools."""

    @pytest.fixture
    def mock_guild(self):
        """Create a mock guild."""
        guild = Mock()
        guild.id = 987654321
        guild.name = "Test Server"
        guild.description = "A test server for testing"
        guild.owner_id = 123456789
        guild.owner = Mock(id=123456789, __str__=Mock(return_value="OwnerUser"))
        guild.region = Mock(__str__=Mock(return_value="us-east"))
        guild.verification_level = Mock(__str__=Mock(return_value="HIGH"))
        guild.default_notifications = Mock(__str__=Mock(return_value="ALL"))
        guild.explicit_content_filter = Mock(__str__=Mock(return_value="DISABLED"))
        guild.mfa_level = Mock(__str__=Mock(return_value="NONE"))
        guild.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        guild.member_count = 100
        guild.premium_tier = Mock(__str__=Mock(return_value="TIER_1"))
        guild.premium_subscription_count = 5
        guild.banner = None
        guild.icon = Mock()
        guild.icon.url = "https://example.com/icon.png"
        guild.vanity_url_code = "testserver"
        guild.roles = []  # Empty list instead of Mock to support len()
        guild.channels = []  # Empty list instead of Mock to support len()
        guild.emojis = []  # Empty list instead of Mock to support len()
        return guild

    @pytest.fixture
    def mock_discord_context(self, mock_guild):
        """Create a mock Discord message context."""
        message = Mock()
        message.guild = mock_guild
        message.bot = Mock()
        message.bot.get_guild = Mock(return_value=mock_guild)
        return message

    @pytest.mark.asyncio
    async def test_get_guild_info(self, mock_guild, mock_discord_context):
        """Test getting guild information."""
        result = await get_guild_info(mock_guild.id, mock_discord_context)

        assert result.success is True
        data = result.data
        assert data["id"] == str(mock_guild.id)
        assert data["name"] == "Test Server"
        assert data["description"] == "A test server for testing"
        assert data["member_count"] == 100

    @pytest.mark.asyncio
    async def test_get_guild_roles(self, mock_guild, mock_discord_context):
        """Test getting all guild roles."""
        # Create mock roles
        admin_role = Mock()
        admin_role.id = 1
        admin_role.name = "Admin"
        admin_role.color = Mock(__str__=Mock(return_value="#FF0000"))
        admin_role.position = 10
        admin_role.managed = False
        admin_role.mentionable = True
        admin_role.hoist = True
        admin_role.is_default = Mock(return_value=False)
        admin_role.tags = None

        everyone_role = Mock()
        everyone_role.id = 0
        everyone_role.name = "@everyone"
        everyone_role.color = Mock(__str__=Mock(return_value="#000000"))
        everyone_role.position = 0
        everyone_role.managed = False
        everyone_role.mentionable = False
        everyone_role.hoist = False
        everyone_role.is_default = Mock(return_value=True)
        everyone_role.tags = None

        mock_guild.roles = [admin_role, everyone_role]

        result = await get_guild_roles(mock_guild.id, mock_discord_context)

        assert result.success is True
        data = result.data
        assert data["count"] == 2

        # Check sorting by position (highest first)
        assert data["roles"][0]["name"] == "Admin"
        assert data["roles"][0]["position"] == 10

    @pytest.mark.asyncio
    async def test_get_guild_emojis(self, mock_guild, mock_discord_context):
        """Test getting guild emojis."""
        # Create mock emojis
        emoji1 = Mock()
        emoji1.id = 111
        emoji1.name = "thumbs_up"
        emoji1.animated = False
        emoji1.available = True
        emoji1.managed = False
        emoji1.require_colons = True
        emoji1.roles = []
        emoji1.url = "https://example.com/emojis/111.png"

        emoji2 = Mock()
        emoji2.id = 222
        emoji2.name = "wave"
        emoji2.animated = True
        emoji2.available = True
        emoji2.managed = False
        emoji2.require_colons = False
        emoji2.roles = []
        emoji2.url = "https://example.com/emojis/222.gif"

        mock_guild.emojis = [emoji1, emoji2]

        result = await get_guild_emojis(mock_guild.id, mock_discord_context)

        assert result.success is True
        data = result.data
        assert data["count"] == 2
        assert len(data["emojis"]) == 2

        # Check emoji data
        assert data["emojis"][0]["name"] == "thumbs_up"
        assert data["emojis"][0]["animated"] is False
        assert data["emojis"][1]["name"] == "wave"
        assert data["emojis"][1]["animated"] is True

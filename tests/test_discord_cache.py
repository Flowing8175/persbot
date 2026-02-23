"""Feature tests for discord_cache module.

Tests focus on behavior:
- get_discord_cache(): singleton pattern with lazy initialization
- Member caching: get_cached_member(), cache_member(), invalidate_member_cache()
- Message caching: get_cached_message(), cache_message()
- User caching: get_cached_user(), cache_user()
- Stats and cleanup: get_cache_stats(), cleanup_discord_cache()
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persbot.tools.discord_cache import (
    get_discord_cache,
    get_cached_member,
    cache_member,
    get_cached_message,
    cache_message,
    get_cached_user,
    cache_user,
    invalidate_member_cache,
    get_cache_stats,
    cleanup_discord_cache,
)


class TestGetDiscordCache:
    """Tests for get_discord_cache singleton pattern."""

    @pytest.fixture
    def reset_cache(self):
        """Reset global cache state before and after each test."""
        # Reset before
        import persbot.tools.discord_cache as dc_module
        dc_module._discord_cache = None
        yield
        # Reset after
        dc_module._discord_cache = None

    @pytest.mark.asyncio
    async def test_returns_cache_manager_instance(self, reset_cache):
        """get_discord_cache returns CacheManager instance."""
        with patch("persbot.tools.discord_cache.CacheManager") as mock_cm:
            mock_instance = MagicMock()
            mock_instance.start_cleanup_task = AsyncMock()
            mock_cm.return_value = mock_instance

            cache = await get_discord_cache()

            assert cache == mock_instance
            mock_cm.assert_called_once_with(
                ttl_minutes=5,
                max_size=500,
                cleanup_interval_minutes=5,
            )
            mock_instance.start_cleanup_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_singleton_pattern_returns_same_instance(self, reset_cache):
        """get_discord_cache returns same instance on subsequent calls."""
        with patch("persbot.tools.discord_cache.CacheManager") as mock_cm:
            mock_instance = MagicMock()
            mock_instance.start_cleanup_task = AsyncMock()
            mock_cm.return_value = mock_instance

            cache1 = await get_discord_cache()
            cache2 = await get_discord_cache()

            assert cache1 is cache2
            # CacheManager should only be created once
            mock_cm.assert_called_once()

    @pytest.mark.asyncio
    async def test_thread_safe_initialization(self, reset_cache):
        """get_discord_cache is thread-safe during initialization."""
        with patch("persbot.tools.discord_cache.CacheManager") as mock_cm:
            mock_instance = MagicMock()
            mock_instance.start_cleanup_task = AsyncMock()
            mock_cm.return_value = mock_instance

            # Call concurrently
            results = await asyncio.gather(
                get_discord_cache(),
                get_discord_cache(),
                get_discord_cache(),
            )

            # All should return the same instance
            assert all(r is mock_instance for r in results)
            # Only one instance should be created
            mock_cm.assert_called_once()


class TestMemberCaching:
    """Tests for member caching functions."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache manager."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.delete = AsyncMock(return_value=True)
        return cache

    @pytest.fixture
    def reset_cache(self):
        """Reset global cache state."""
        import persbot.tools.discord_cache as dc_module
        original = dc_module._discord_cache
        dc_module._discord_cache = None
        yield
        dc_module._discord_cache = original

    @pytest.mark.asyncio
    async def test_get_cached_member_delegates_to_cache(self, reset_cache, mock_cache):
        """get_cached_member delegates to cache manager."""
        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            result = await get_cached_member(123, 456)

            mock_cache.get.assert_called_once_with("member", 123, 456)
            assert result is None  # Default mock return

    @pytest.mark.asyncio
    async def test_get_cached_member_returns_member_data(self, reset_cache, mock_cache):
        """get_cached_member returns cached member data."""
        member_data = {"nick": "TestUser", "roles": [1, 2, 3]}
        mock_cache.get.return_value = member_data

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            result = await get_cached_member(123, 456)

            assert result == member_data
            mock_cache.get.assert_called_once_with("member", 123, 456)

    @pytest.mark.asyncio
    async def test_cache_member_stores_data(self, reset_cache, mock_cache):
        """cache_member stores data in cache."""
        member_data = {"nick": "TestUser", "joined_at": "2024-01-01"}

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await cache_member(123, 456, member_data)

            mock_cache.set.assert_called_once_with(member_data, "member", 123, 456)

    @pytest.mark.asyncio
    async def test_cache_member_overwrites_existing(self, reset_cache, mock_cache):
        """cache_member overwrites existing cached data."""
        old_data = {"nick": "OldNick"}
        new_data = {"nick": "NewNick"}

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await cache_member(123, 456, old_data)
            await cache_member(123, 456, new_data)

            assert mock_cache.set.call_count == 2
            # Last call should be with new_data
            mock_cache.set.assert_called_with(new_data, "member", 123, 456)

    @pytest.mark.asyncio
    async def test_invalidate_member_cache_deletes_entry(self, reset_cache, mock_cache):
        """invalidate_member_cache deletes entry from cache."""
        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await invalidate_member_cache(123, 456)

            mock_cache.delete.assert_called_once_with("member", 123, 456)

    @pytest.mark.asyncio
    async def test_cache_then_get_then_invalidate_flow(self, reset_cache, mock_cache):
        """Full cache flow: cache -> get -> invalidate."""
        member_data = {"nick": "TestUser"}

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            # Cache the member
            await cache_member(123, 456, member_data)

            # Retrieve it (simulate)
            mock_cache.get.return_value = member_data
            result = await get_cached_member(123, 456)
            assert result == member_data

            # Invalidate
            await invalidate_member_cache(123, 456)

            # Verify calls
            mock_cache.set.assert_called_once()
            mock_cache.get.assert_called_once()
            mock_cache.delete.assert_called_once()


class TestMessageCaching:
    """Tests for message caching functions."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache manager."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        return cache

    @pytest.fixture
    def reset_cache(self):
        """Reset global cache state."""
        import persbot.tools.discord_cache as dc_module
        original = dc_module._discord_cache
        dc_module._discord_cache = None
        yield
        dc_module._discord_cache = original

    @pytest.mark.asyncio
    async def test_get_cached_message_delegates_to_cache(self, reset_cache, mock_cache):
        """get_cached_message delegates to cache manager."""
        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            result = await get_cached_message(789, 999)

            mock_cache.get.assert_called_once_with("message", 789, 999)
            assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_message_returns_message_data(self, reset_cache, mock_cache):
        """get_cached_message returns cached message data."""
        message_data = {
            "id": 999,
            "content": "Hello, world!",
            "author": {"id": 123, "username": "TestUser"},
        }
        mock_cache.get.return_value = message_data

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            result = await get_cached_message(789, 999)

            assert result == message_data
            assert result["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_cache_message_stores_data(self, reset_cache, mock_cache):
        """cache_message stores data in cache."""
        message_data = {
            "id": 999,
            "content": "Test message",
            "author": {"id": 123},
        }

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await cache_message(789, 999, message_data)

            mock_cache.set.assert_called_once_with(message_data, "message", 789, 999)

    @pytest.mark.asyncio
    async def test_cache_message_handles_complex_data(self, reset_cache, mock_cache):
        """cache_message handles complex message structures."""
        message_data = {
            "id": 999,
            "content": "Complex message",
            "embeds": [{"title": "Embed"}],
            "attachments": [{"id": 1, "filename": "file.png"}],
            "reactions": [{"emoji": "👍", "count": 5}],
        }

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await cache_message(789, 999, message_data)

            mock_cache.set.assert_called_once()
            args = mock_cache.set.call_args[0]
            assert args[0]["embeds"][0]["title"] == "Embed"

    @pytest.mark.asyncio
    async def test_cache_different_messages_in_different_channels(self, reset_cache, mock_cache):
        """Messages in different channels are cached separately."""
        msg1 = {"id": 1, "content": "Channel 1"}
        msg2 = {"id": 2, "content": "Channel 2"}

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await cache_message(111, 1, msg1)
            await cache_message(222, 2, msg2)

            assert mock_cache.set.call_count == 2
            calls = mock_cache.set.call_args_list
            assert calls[0][0] == (msg1, "message", 111, 1)
            assert calls[1][0] == (msg2, "message", 222, 2)


class TestUserCaching:
    """Tests for user caching functions."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache manager."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        return cache

    @pytest.fixture
    def reset_cache(self):
        """Reset global cache state."""
        import persbot.tools.discord_cache as dc_module
        original = dc_module._discord_cache
        dc_module._discord_cache = None
        yield
        dc_module._discord_cache = original

    @pytest.mark.asyncio
    async def test_get_cached_user_delegates_to_cache(self, reset_cache, mock_cache):
        """get_cached_user delegates to cache manager."""
        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            result = await get_cached_user(123)

            mock_cache.get.assert_called_once_with("user", 123)
            assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_user_returns_user_data(self, reset_cache, mock_cache):
        """get_cached_user returns cached user data."""
        user_data = {
            "id": 123,
            "username": "TestUser",
            "discriminator": "1234",
            "avatar": "abc123",
        }
        mock_cache.get.return_value = user_data

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            result = await get_cached_user(123)

            assert result == user_data
            assert result["username"] == "TestUser"

    @pytest.mark.asyncio
    async def test_cache_user_stores_data(self, reset_cache, mock_cache):
        """cache_user stores data in cache."""
        user_data = {
            "id": 456,
            "username": "AnotherUser",
            "global_name": "Another",
        }

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await cache_user(456, user_data)

            mock_cache.set.assert_called_once_with(user_data, "user", 456)

    @pytest.mark.asyncio
    async def test_cache_user_only_uses_user_id(self, reset_cache, mock_cache):
        """cache_user uses only user_id (not guild_id like member)."""
        user_data = {"id": 789, "username": "SoloUser"}

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await cache_user(789, user_data)

            # Should only have 3 args: data, "user", user_id
            mock_cache.set.assert_called_once_with(user_data, "user", 789)

    @pytest.mark.asyncio
    async def test_same_user_cached_once(self, reset_cache, mock_cache):
        """Same user data overwrites in cache."""
        user_v1 = {"id": 123, "username": "OldName"}
        user_v2 = {"id": 123, "username": "NewName"}

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            await cache_user(123, user_v1)
            await cache_user(123, user_v2)

            assert mock_cache.set.call_count == 2
            # Both calls with same user_id
            assert mock_cache.set.call_args_list[0][0] == (user_v1, "user", 123)
            assert mock_cache.set.call_args_list[1][0] == (user_v2, "user", 123)


class TestCacheStats:
    """Tests for get_cache_stats function."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache manager."""
        cache = MagicMock()
        cache.stats = AsyncMock(return_value={
            "total_entries": 10,
            "expired_entries": 2,
            "active_entries": 8,
            "max_size": 500,
            "ttl_minutes": 5,
        })
        return cache

    @pytest.fixture
    def reset_cache(self):
        """Reset global cache state."""
        import persbot.tools.discord_cache as dc_module
        original = dc_module._discord_cache
        dc_module._discord_cache = None
        yield
        dc_module._discord_cache = original

    @pytest.mark.asyncio
    async def test_get_cache_stats_delegates_to_cache(self, reset_cache, mock_cache):
        """get_cache_stats delegates to cache manager."""
        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            stats = await get_cache_stats()

            mock_cache.stats.assert_called_once()
            assert stats["total_entries"] == 10
            assert stats["active_entries"] == 8

    @pytest.mark.asyncio
    async def test_get_cache_stats_returns_all_fields(self, reset_cache, mock_cache):
        """get_cache_stats returns all stat fields."""
        expected_stats = {
            "total_entries": 100,
            "expired_entries": 5,
            "active_entries": 95,
            "max_size": 500,
            "ttl_minutes": 5,
        }
        mock_cache.stats.return_value = expected_stats

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            stats = await get_cache_stats()

            assert stats == expected_stats

    @pytest.mark.asyncio
    async def test_get_cache_stats_handles_empty_cache(self, reset_cache, mock_cache):
        """get_cache_stats handles empty cache."""
        mock_cache.stats.return_value = {
            "total_entries": 0,
            "expired_entries": 0,
            "active_entries": 0,
            "max_size": 500,
            "ttl_minutes": 5,
        }

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            stats = await get_cache_stats()

            assert stats["total_entries"] == 0
            assert stats["active_entries"] == 0


class TestCleanupDiscordCache:
    """Tests for cleanup_discord_cache function."""

    @pytest.fixture
    def reset_cache(self):
        """Reset global cache state."""
        import persbot.tools.discord_cache as dc_module
        original = dc_module._discord_cache
        dc_module._discord_cache = None
        yield
        dc_module._discord_cache = original

    @pytest.mark.asyncio
    async def test_cleanup_stops_cleanup_task(self, reset_cache):
        """cleanup_discord_cache stops the background cleanup task."""
        mock_cache = MagicMock()
        mock_cache.stop_cleanup_task = AsyncMock()
        mock_cache.clear = AsyncMock()

        import persbot.tools.discord_cache as dc_module
        dc_module._discord_cache = mock_cache

        await cleanup_discord_cache()

        mock_cache.stop_cleanup_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_clears_cache(self, reset_cache):
        """cleanup_discord_cache clears all cache entries."""
        mock_cache = MagicMock()
        mock_cache.stop_cleanup_task = AsyncMock()
        mock_cache.clear = AsyncMock()

        import persbot.tools.discord_cache as dc_module
        dc_module._discord_cache = mock_cache

        await cleanup_discord_cache()

        mock_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_sets_global_cache_to_none(self, reset_cache):
        """cleanup_discord_cache sets global cache to None."""
        mock_cache = MagicMock()
        mock_cache.stop_cleanup_task = AsyncMock()
        mock_cache.clear = AsyncMock()

        import persbot.tools.discord_cache as dc_module
        dc_module._discord_cache = mock_cache

        await cleanup_discord_cache()

        assert dc_module._discord_cache is None

    @pytest.mark.asyncio
    async def test_cleanup_when_cache_is_none_is_safe(self, reset_cache):
        """cleanup_discord_cache is safe when cache is already None."""
        import persbot.tools.discord_cache as dc_module
        dc_module._discord_cache = None

        # Should not raise
        await cleanup_discord_cache()

        assert dc_module._discord_cache is None

    @pytest.mark.asyncio
    async def test_cleanup_can_be_called_multiple_times(self, reset_cache):
        """cleanup_discord_cache can be called multiple times safely."""
        mock_cache = MagicMock()
        mock_cache.stop_cleanup_task = AsyncMock()
        mock_cache.clear = AsyncMock()

        import persbot.tools.discord_cache as dc_module
        dc_module._discord_cache = mock_cache

        # First cleanup
        await cleanup_discord_cache()
        assert dc_module._discord_cache is None

        # Second cleanup (should be safe)
        await cleanup_discord_cache()
        assert dc_module._discord_cache is None

    @pytest.mark.asyncio
    async def test_cleanup_executes_operations_in_order(self, reset_cache):
        """cleanup executes operations in correct order."""
        mock_cache = MagicMock()
        call_order = []

        async def mock_stop():
            call_order.append("stop")

        async def mock_clear():
            call_order.append("clear")

        mock_cache.stop_cleanup_task = mock_stop
        mock_cache.clear = mock_clear

        import persbot.tools.discord_cache as dc_module
        dc_module._discord_cache = mock_cache

        await cleanup_discord_cache()

        # Should stop task first, then clear, then set to None
        assert call_order == ["stop", "clear"]


class TestIntegrationScenarios:
    """Integration tests for common usage patterns."""

    @pytest.fixture
    def reset_cache(self):
        """Reset global cache state."""
        import persbot.tools.discord_cache as dc_module
        original = dc_module._discord_cache
        dc_module._discord_cache = None
        yield
        dc_module._discord_cache = original

    @pytest.mark.asyncio
    async def test_full_member_lifecycle(self, reset_cache):
        """Test complete member caching lifecycle."""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock()
        mock_cache.set = AsyncMock()
        mock_cache.delete = AsyncMock(return_value=True)
        mock_cache.stats = AsyncMock(return_value={"total_entries": 0, "active_entries": 0, "expired_entries": 0, "max_size": 500, "ttl_minutes": 5})

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            # 1. Cache a member
            member = {"id": 456, "nick": "Member1", "roles": [1, 2]}
            await cache_member(123, 456, member)

            # 2. Retrieve the member (cache hit)
            mock_cache.get.return_value = member
            result = await get_cached_member(123, 456)
            assert result["nick"] == "Member1"

            # 3. Check stats
            stats = await get_cache_stats()
            assert "total_entries" in stats

            # 4. Invalidate when member leaves
            await invalidate_member_cache(123, 456)

            # Verify all calls
            mock_cache.set.assert_called_once_with(member, "member", 123, 456)
            mock_cache.get.assert_called_once_with("member", 123, 456)
            mock_cache.delete.assert_called_once_with("member", 123, 456)

    @pytest.mark.asyncio
    async def test_message_and_user_caching_together(self, reset_cache):
        """Test caching messages with their authors."""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock()
        mock_cache.set = AsyncMock()

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            # Message data with author
            message = {
                "id": 999,
                "content": "Hello!",
                "author": {"id": 123, "username": "Author"},
            }

            # Cache message
            await cache_message(789, 999, message)

            # Cache user separately
            user = {"id": 123, "username": "Author", "discriminator": "0000"}
            await cache_user(123, user)

            # Verify both were cached with different key prefixes
            calls = mock_cache.set.call_args_list
            assert len(calls) == 2
            assert calls[0][0][1] == "message"  # First call is message
            assert calls[1][0][1] == "user"  # Second call is user

    @pytest.mark.asyncio
    async def test_cache_miss_scenario(self, reset_cache):
        """Test behavior when cache misses."""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)  # Cache miss

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            # Try to get uncached member
            result = await get_cached_member(111, 222)

            # Should return None
            assert result is None
            mock_cache.get.assert_called_once_with("member", 111, 222)

    @pytest.mark.asyncio
    async def test_multiple_members_same_guild(self, reset_cache):
        """Test caching multiple members from same guild."""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock()
        mock_cache.set = AsyncMock()

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            guild_id = 123

            # Cache multiple members
            members = [
                (456, {"id": 456, "nick": "Member1"}),
                (789, {"id": 789, "nick": "Member2"}),
                (101, {"id": 101, "nick": "Member3"}),
            ]

            for user_id, member_data in members:
                await cache_member(guild_id, user_id, member_data)

            # Verify all were cached
            assert mock_cache.set.call_count == 3

            # Each should have unique keys (same guild, different users)
            calls = mock_cache.set.call_args_list
            assert calls[0][0] == (members[0][1], "member", guild_id, 456)
            assert calls[1][0] == (members[1][1], "member", guild_id, 789)
            assert calls[2][0] == (members[2][1], "member", guild_id, 101)


class TestCacheKeySeparation:
    """Tests that different data types use separate cache namespaces."""

    @pytest.fixture
    def reset_cache(self):
        """Reset global cache state."""
        import persbot.tools.discord_cache as dc_module
        original = dc_module._discord_cache
        dc_module._discord_cache = None
        yield
        dc_module._discord_cache = original

    @pytest.mark.asyncio
    async def test_member_message_user_separate_namespaces(self, reset_cache):
        """Member, message, and user use separate cache namespaces."""
        mock_cache = MagicMock()
        mock_cache.set = AsyncMock()

        with patch("persbot.tools.discord_cache.get_discord_cache", return_value=mock_cache):
            # Cache with same IDs but different types
            await cache_member(123, 456, {"type": "member"})
            await cache_message(123, 456, {"type": "message"})
            await cache_user(456, {"type": "user"})

            calls = mock_cache.set.call_args_list

            # Verify namespace separation
            assert calls[0][0][1] == "member"  # (data, "member", guild_id, user_id)
            assert calls[1][0][1] == "message"  # (data, "message", channel_id, message_id)
            assert calls[2][0][1] == "user"  # (data, "user", user_id)

            # Member has 4 args, message has 4 args, user has 3 args
            assert len(calls[0][0]) == 4
            assert len(calls[1][0]) == 4
            assert len(calls[2][0]) == 3

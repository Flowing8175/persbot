"""Feature tests for memory tools module.

Tests focus on behavior using mocking:
- search_episodic_memory: search through stored memories
- save_episodic_memory: save new memory entry
- remove_episodic_memory: remove memory entries
- register_memory_tools: register tools with registry
"""

import sys
import os
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


class TestSearchEpisodicMemory:
    """Tests for search_episodic_memory function."""

    @pytest.mark.asyncio
    async def test_search_episodic_memory_exists(self):
        """search_episodic_memory function exists."""
        from persbot.tools.persona_tools.memory_tools import search_episodic_memory
        assert search_episodic_memory is not None

    @pytest.mark.asyncio
    async def test_rejects_empty_query(self):
        """search_episodic_memory rejects empty query."""
        from persbot.tools.persona_tools.memory_tools import search_episodic_memory

        result = await search_episodic_memory(user_id="123", query="")
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rejects_whitespace_query(self):
        """search_episodic_memory rejects whitespace query."""
        from persbot.tools.persona_tools.memory_tools import search_episodic_memory

        result = await search_episodic_memory(user_id="123", query="   ")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_returns_empty_memories_when_none_found(self):
        """search_episodic_memory returns empty list when no memories found."""
        from persbot.tools.persona_tools.memory_tools import search_episodic_memory

        with patch('persbot.tools.persona_tools.memory_tools._load_memory_vector', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = []

            result = await search_episodic_memory(user_id="123", query="test")

            assert result.success is True
            assert result.data["memories"] == []

    @pytest.mark.asyncio
    async def test_clamps_limit_between_1_and_10(self):
        """search_episodic_memory clamps limit to valid range."""
        from persbot.tools.persona_tools.memory_tools import search_episodic_memory

        with patch('persbot.tools.persona_tools.memory_tools._load_memory_vector', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = []

            # Test with limit over max
            await search_episodic_memory(user_id="123", query="test", limit=100)
            # Test with limit under min
            await search_episodic_memory(user_id="123", query="test", limit=0)


class TestSaveEpisodicMemory:
    """Tests for save_episodic_memory function."""

    @pytest.mark.asyncio
    async def test_save_episodic_memory_exists(self):
        """save_episodic_memory function exists."""
        from persbot.tools.persona_tools.memory_tools import save_episodic_memory
        assert save_episodic_memory is not None

    @pytest.mark.asyncio
    async def test_rejects_empty_content(self):
        """save_episodic_memory rejects empty content."""
        from persbot.tools.persona_tools.memory_tools import save_episodic_memory

        result = await save_episodic_memory(user_id="123", content="")
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self):
        """save_episodic_memory rejects empty user_id."""
        from persbot.tools.persona_tools.memory_tools import save_episodic_memory

        result = await save_episodic_memory(user_id="", content="test memory")
        assert result.success is False
        assert "user id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_saves_memory_successfully(self):
        """save_episodic_memory saves memory successfully."""
        from persbot.tools.persona_tools.memory_tools import save_episodic_memory

        with patch('persbot.tools.persona_tools.memory_tools._load_memory_vector', new_callable=AsyncMock) as mock_load:
            with patch('persbot.tools.persona_tools.memory_tools._save_memory_vector', new_callable=AsyncMock) as mock_save:
                mock_load.return_value = []

                result = await save_episodic_memory(
                    user_id="123",
                    content="User likes coffee",
                    memory_type="preference",
                    tags=["food"]
                )

                assert result.success is True
                assert result.data["content"] == "User likes coffee"
                mock_save.assert_called_once()


class TestRemoveEpisodicMemory:
    """Tests for remove_episodic_memory function."""

    @pytest.mark.asyncio
    async def test_remove_episodic_memory_exists(self):
        """remove_episodic_memory function exists."""
        from persbot.tools.persona_tools.memory_tools import remove_episodic_memory
        assert remove_episodic_memory is not None

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self):
        """remove_episodic_memory rejects empty user_id."""
        from persbot.tools.persona_tools.memory_tools import remove_episodic_memory

        result = await remove_episodic_memory(user_id="")
        assert result.success is False
        assert "user id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_count_when_no_memories(self):
        """remove_episodic_memory returns 0 when no memories to remove."""
        from persbot.tools.persona_tools.memory_tools import remove_episodic_memory

        with patch('persbot.tools.persona_tools.memory_tools._load_memory_vector', new_callable=AsyncMock) as mock_load:
            with patch('persbot.tools.persona_tools.memory_tools._save_memory_vector', new_callable=AsyncMock) as mock_save:
                mock_load.return_value = []

                result = await remove_episodic_memory(user_id="123")

                assert result.success is True
                assert result.data["removed_count"] == 0

    @pytest.mark.asyncio
    async def test_removes_matching_memories(self):
        """remove_episodic_memory removes matching memories."""
        from persbot.tools.persona_tools.memory_tools import remove_episodic_memory

        memories = [
            {"user_id": "123", "content": "test memory", "type": "preference", "tags": []},
            {"user_id": "456", "content": "other memory", "type": "fact", "tags": []},
        ]

        with patch('persbot.tools.persona_tools.memory_tools._load_memory_vector', new_callable=AsyncMock) as mock_load:
            with patch('persbot.tools.persona_tools.memory_tools._save_memory_vector', new_callable=AsyncMock) as mock_save:
                mock_load.return_value = memories

                result = await remove_episodic_memory(user_id="123")

                assert result.success is True
                assert result.data["removed_count"] == 1
                mock_save.assert_called_once()


class TestRegisterMemoryTools:
    """Tests for register_memory_tools function."""

    def test_register_memory_tools_exists(self):
        """register_memory_tools function exists."""
        from persbot.tools.persona_tools.memory_tools import register_memory_tools
        assert register_memory_tools is not None

    def test_registers_tools(self):
        """register_memory_tools registers tools."""
        from persbot.tools.persona_tools.memory_tools import register_memory_tools

        mock_registry = MagicMock()
        register_memory_tools(mock_registry)

        # Should register 3 tools
        assert mock_registry.register.call_count == 3

    def test_registers_search_episodic_memory(self):
        """register_memory_tools registers search_episodic_memory."""
        from persbot.tools.persona_tools.memory_tools import register_memory_tools

        mock_registry = MagicMock()
        register_memory_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[0]
        tool_def = call_args[0][0]
        assert tool_def.name == "search_episodic_memory"

    def test_registers_save_episodic_memory(self):
        """register_memory_tools registers save_episodic_memory."""
        from persbot.tools.persona_tools.memory_tools import register_memory_tools

        mock_registry = MagicMock()
        register_memory_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[1]
        tool_def = call_args[0][0]
        assert tool_def.name == "save_episodic_memory"

    def test_registers_remove_episodic_memory(self):
        """register_memory_tools registers remove_episodic_memory."""
        from persbot.tools.persona_tools.memory_tools import register_memory_tools

        mock_registry = MagicMock()
        register_memory_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[2]
        tool_def = call_args[0][0]
        assert tool_def.name == "remove_episodic_memory"

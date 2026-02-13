"""Feature tests for media tools module.

Tests focus on behavior using mocking:
- generate_situational_snapshot: generate image generation prompt
- register_media_tools: register tools with registry
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


class TestGenerateSituationalSnapshot:
    """Tests for generate_situational_snapshot function."""

    @pytest.mark.asyncio
    async def test_generate_situational_snapshot_exists(self):
        """generate_situational_snapshot function exists."""
        from persbot.tools.persona_tools.media_tools import generate_situational_snapshot
        assert generate_situational_snapshot is not None

    @pytest.mark.asyncio
    async def test_generates_snapshot_with_all_params(self):
        """generate_situational_snapshot generates with all parameters."""
        from persbot.tools.persona_tools.media_tools import generate_situational_snapshot

        result = await generate_situational_snapshot(
            time_of_day="morning",
            location="cafe",
            mood="happy",
            activity="reading"
        )

        assert result.success is True
        assert result.data["time_of_day"] == "morning"
        assert result.data["location"] == "cafe"
        assert result.data["mood"] == "happy"
        assert result.data["activity"] == "reading"

    @pytest.mark.asyncio
    async def test_uses_defaults_for_missing_params(self):
        """generate_situational_snapshot uses defaults for missing params."""
        from persbot.tools.persona_tools.media_tools import generate_situational_snapshot

        result = await generate_situational_snapshot()

        assert result.success is True
        assert result.data is not None
        assert len(result.data) > 0

    @pytest.mark.asyncio
    async def test_includes_image_generation_prompt(self):
        """generate_situational_snapshot includes image generation prompt."""
        from persbot.tools.persona_tools.media_tools import generate_situational_snapshot

        result = await generate_situational_snapshot(time_of_day="evening")

        assert result.success is True
        assert "image_prompt" in result.data
        assert len(result.data["image_prompt"]) > 0

    @pytest.mark.asyncio
    async def test_handles_all_times_of_day(self):
        """generate_situational_snapshot handles all times of day."""
        from persbot.tools.persona_tools.media_tools import generate_situational_snapshot

        times = ["morning", "afternoon", "evening", "night"]

        for time in times:
            result = await generate_situational_snapshot(time_of_day=time)
            assert result.success is True
            assert result.data["time_of_day"] == time

    @pytest.mark.asyncio
    async def test_handles_all_moods(self):
        """generate_situational_snapshot handles all mood types."""
        from persbot.tools.persona_tools.media_tools import generate_situational_snapshot

        moods = ["happy", "tired", "focused", "relaxed", "calm"]

        for mood in moods:
            result = await generate_situational_snapshot(mood=mood)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_handles_all_activities(self):
        """generate_situational_snapshot handles all activity types."""
        from persbot.tools.persona_tools.media_tools import generate_situational_snapshot

        activities = ["working", "reading", "gaming", "resting", "relaxing"]

        for activity in activities:
            result = await generate_situational_snapshot(activity=activity)
            assert result.success is True


class TestRegisterMediaTools:
    """Tests for register_media_tools function."""

    def test_register_media_tools_exists(self):
        """register_media_tools function exists."""
        from persbot.tools.persona_tools.media_tools import register_media_tools
        assert register_media_tools is not None

    def test_registers_tools(self):
        """register_media_tools registers tools."""
        from persbot.tools.persona_tools.media_tools import register_media_tools

        mock_registry = MagicMock()
        register_media_tools(mock_registry)

        # Should register at least 1 tool
        assert mock_registry.register.call_count >= 1

    def test_registers_generate_situational_snapshot(self):
        """register_media_tools registers generate_situational_snapshot."""
        from persbot.tools.persona_tools.media_tools import register_media_tools

        mock_registry = MagicMock()
        register_media_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[0]
        tool_def = call_args[0][0]
        assert tool_def.name == "generate_situational_snapshot"

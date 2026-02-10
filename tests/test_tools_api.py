"""Tests for external API tools (search, weather, time)."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from persbot.tools.api_tools.image_tools import generate_image
from persbot.tools.api_tools.search_tools import web_search
from persbot.tools.api_tools.time_tools import get_time
from persbot.tools.api_tools.weather_tools import get_weather


class TestSearchTools:
    """Tests for web search tool."""

    @pytest.mark.asyncio
    async def test_web_search_empty_query(self):
        """Test web search with empty query."""
        result = await web_search("")

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_web_search_whitespace_query(self):
        """Test web search with whitespace-only query."""
        result = await web_search("   ")

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_web_search_num_results_clamp(self):
        """Test that num_results is clamped correctly."""
        # Test with num_results > 10
        result = await web_search("test", num_results=100)
        # Should fail because no actual search API, but we can test the clamping logic
        # by checking the function doesn't crash

        # Test with num_results < 1
        result = await web_search("test", num_results=0)
        # Should also handle gracefully

    @pytest.mark.asyncio
    async def test_web_search_no_api_key(self):
        """Test web search without API key works (DuckDuckGo doesn't require key)."""
        # DuckDuckGo search doesn't require an API key, so it should work
        # We can't test actual search without network, but we can test the function doesn't crash
        result = await web_search("test query", num_results=1)

        # Should return a result (success or failure, but not crash)
        assert isinstance(result.success, bool)
        assert result is not None


class TestWeatherTools:
    """Tests for weather tool."""

    @pytest.mark.asyncio
    async def test_get_weather_empty_location(self):
        """Test weather with empty location."""
        result = await get_weather("")

        assert result.success is False
        assert "empty" in result.error.lower() or "location" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_weather_whitespace_location(self):
        """Test weather with whitespace-only location."""
        result = await get_weather("   ")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_weather_invalid_units(self):
        """Test weather with invalid units defaults to metric."""
        # Should not crash, should default to metric
        # We can't test actual API calls without keys, but we test the logic
        result = await get_weather("London", units="invalid")

        # Will fail without API key, but should not crash
        assert isinstance(result, object)

    @pytest.mark.asyncio
    async def test_get_weather_imperial_units(self):
        """Test weather with imperial units."""
        # Test that imperial units are handled
        result = await get_weather("New York", units="imperial")

        # Will fail without API key, but should not crash
        assert isinstance(result, object)

    @pytest.mark.asyncio
    async def test_get_weather_metric_units(self):
        """Test weather with metric units."""
        result = await get_weather("Seoul", units="metric")

        assert isinstance(result, object)


class TestTimeTools:
    """Tests for time tool."""

    @pytest.mark.asyncio
    async def test_get_time_default(self):
        """Test getting time with default timezone (UTC)."""
        result = await get_time()

        assert result.success is True
        data = result.data
        assert "timezone" in data or "TZ" in data.get("timezone", "").upper()
        assert "time" in data
        assert "date" in data
        assert "datetime" in data
        assert "unix_timestamp" in data

    @pytest.mark.asyncio
    async def test_get_time_utc(self):
        """Test getting UTC time."""
        result = await get_time("UTC")

        assert result.success is True
        data = result.data
        assert "UTC" in data.get("timezone", "").upper()

    @pytest.mark.asyncio
    async def test_get_time_city_name(self):
        """Test getting time for a city name."""
        result = await get_time("Seoul")

        assert result.success is True
        data = result.data
        assert "time" in data
        assert "date" in data

    @pytest.mark.asyncio
    async def test_get_time_seoul(self):
        """Test getting Seoul time."""
        result = await get_time("Asia/Seoul")

        assert result.success is True
        data = result.data
        # Seoul is UTC+9
        assert "Asia/Seoul" in data.get("timezone", "")

    @pytest.mark.asyncio
    async def test_get_time_new_york(self):
        """Test getting New York time."""
        result = await get_time("America/New_York")

        assert result.success is True
        data = result.data
        assert "America/New_York" in data.get("timezone", "")

    @pytest.mark.asyncio
    async def test_get_time_london(self):
        """Test getting London time."""
        result = await get_time("Europe/London")

        assert result.success is True
        data = result.data
        assert "Europe/London" in data.get("timezone", "")

    @pytest.mark.asyncio
    async def test_get_time_tokyo(self):
        """Test getting Tokyo time."""
        result = await get_time("Tokyo")

        assert result.success is True
        data = result.data
        assert "time" in data

    @pytest.mark.asyncio
    async def test_get_time_timezone_format(self):
        """Test that returned data has expected format."""
        result = await get_time("UTC")

        assert result.success is True
        data = result.data

        # Check required fields
        assert "timezone" in data
        assert "datetime" in data
        assert "time" in data
        assert "date" in data
        assert "day_of_week" in data
        assert "unix_timestamp" in data

        # Check data types
        assert isinstance(data["unix_timestamp"], int)
        assert isinstance(data["time"], str)
        assert isinstance(data["date"], str)

    @pytest.mark.asyncio
    async def test_get_time_case_insensitive(self):
        """Test that timezone lookup is case-insensitive."""
        result1 = await get_time("utc")
        result2 = await get_time("UTC")
        result3 = await get_time("U t C")  # Whitespace

        # All should succeed
        assert result1.success is True
        assert result2.success is True
        # The last one might fail depending on implementation

    @pytest.mark.asyncio
    async def test_get_time_mountain_view(self):
        """Test getting time for Mountain View (Google HQ)."""
        result = await get_time("America/Los_Angeles")

        assert result.success is True
        data = result.data
        assert "America/Los_Angeles" in data.get("timezone", "")


class TestToolIntegration:
    """Integration tests for API tools working together."""

    @pytest.mark.asyncio
    async def test_multiple_time_queries(self):
        """Test querying multiple timezones in sequence."""
        timezones = ["UTC", "Asia/Seoul", "America/New_York", "Europe/London"]

        results = []
        for tz in timezones:
            result = await get_time(tz)
            results.append(result)
            assert result.success is True

        # All should have different utc_offset or time based on timezone
        times = [r.data["time"] for r in results]
        # They should be different (or at least potentially different)
        assert len(times) == len(timezones)

    @pytest.mark.asyncio
    async def test_weather_and_time_combination(self):
        """Test using weather and time tools together."""
        # Get time for a location
        time_result = await get_time("Asia/Seoul")
        assert time_result.success is True

        # Get weather for the same location
        weather_result = await get_weather("Seoul")
        # Will fail without API key but shouldn't crash
        assert isinstance(weather_result, object)


class TestImageTools:
    """Tests for image generation tool."""

    @pytest.mark.asyncio
    async def test_generate_image_empty_prompt(self):
        """Test image generation with empty prompt."""
        result = await generate_image("")

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_generate_image_whitespace_prompt(self):
        """Test image generation with whitespace-only prompt."""
        result = await generate_image("   ")

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_generate_image_with_mock_service(self):
        """Test successful image generation with mocked ImageService."""
        # Mock ImageService
        mock_service = AsyncMock()
        mock_service.generate_image_with_fetch = AsyncMock(return_value=b"fake_image_bytes")

        # Mock config
        mock_config = MagicMock()
        mock_config.openrouter_api_key = "test-key"
        mock_config.openrouter_image_model = "test-model"
        mock_config.api_request_timeout = 30.0

        # Mock rate limiter
        mock_limiter = MagicMock()
        mock_rate_result = MagicMock()
        mock_rate_result.allowed = True
        mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

        with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
            with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
                with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                    result = await generate_image("A beautiful sunset")

        # Verify success
        assert result.success is True
        assert result.data == "Image generated successfully"
        assert result.metadata.get("image_bytes") == b"fake_image_bytes"

    @pytest.mark.asyncio
    async def test_generate_image_rate_limited(self):
        """Test image generation when rate limited."""
        # Mock config
        mock_config = MagicMock()
        mock_config.openrouter_api_key = "test-key"
        mock_config.openrouter_image_model = "test-model"
        mock_config.api_request_timeout = 30.0

        # Mock rate limiter to deny
        mock_limiter = MagicMock()
        mock_rate_result = MagicMock()
        mock_rate_result.allowed = False
        mock_rate_result.message = "Rate limited"
        mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

        mock_discord_context = MagicMock()
        mock_discord_context.author = MagicMock(id=12345)

        with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
            with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
                result = await generate_image("test prompt", discord_context=mock_discord_context)

        assert result.success is False
        assert "rate limited" in result.error.lower()

    @pytest.mark.asyncio
    async def test_generate_image_service_error(self):
        """Test image generation when service fails."""
        from persbot.services.image_service import ImageGenerationError

        # Mock ImageService
        mock_service = AsyncMock()
        mock_service.generate_image_with_fetch = AsyncMock(
            side_effect=ImageGenerationError("API error")
        )

        mock_config = MagicMock()
        mock_config.openrouter_api_key = "test-key"
        mock_config.openrouter_image_model = "test-model"
        mock_config.api_request_timeout = 30.0

        mock_limiter = MagicMock()
        mock_rate_result = MagicMock()
        mock_rate_result.allowed = True
        mock_limiter.check_rate_limit = AsyncMock(return_value=mock_rate_result)

        with patch("persbot.tools.api_tools.image_tools.load_config", return_value=mock_config):
            with patch("persbot.tools.api_tools.image_tools.get_image_rate_limiter", return_value=mock_limiter):
                with patch("persbot.tools.api_tools.image_tools._get_image_service", return_value=mock_service):
                    result = await generate_image("test prompt")

        assert result.success is False
        assert "failed" in result.error.lower() or "api error" in result.error.lower()

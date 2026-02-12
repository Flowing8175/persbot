"""Tests for services/search_service.py module.

This module provides comprehensive test coverage for:
- SearchError exception
- SearchRateLimitError exception
- SearchService class
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from persbot.services.search_service import (
    SearchService,
    SearchError,
    SearchRateLimitError,
)


# =============================================================================
# Exception Tests
# =============================================================================


class TestSearchExceptions:
    """Tests for SearchService exceptions."""

    def test_search_error_message(self):
        """Test SearchError can be raised with message."""
        with pytest.raises(SearchError) as exc_info:
            raise SearchError("Test error")

        assert str(exc_info.value) == "Test error"

    def test_search_rate_limit_error_message(self):
        """Test SearchRateLimitError has correct type."""
        error = SearchRateLimitError("Rate limited")

        assert isinstance(error, SearchError)
        assert isinstance(error, SearchRateLimitError)


# =============================================================================
# SearchService Class Tests
# =============================================================================


class TestSearchService:
    """Tests for SearchService class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        from types import SimpleNamespace
        return SimpleNamespace(
            no_check_permission=True,
        )

    def test_init_default_values(self):
        """Test SearchService initialization with defaults."""
        config = Mock()
        service = SearchService(config)

        assert service._max_retries == 2
        assert service._base_delay == 2.0
        assert service._max_delay == 10.0
        assert service._request_timeout == 30.0

    @pytest.mark.asyncio
    async def test_web_search_success(self, tmp_path):
        """Test successful web search."""
        # Create models.json with search results
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        models_file = data_dir / "models.json"
        test_data = {
            "image_models": [],
            "image_model_preferences": {},
        }
        models_file.write_text('{"results": []}', encoding="utf-8")

        # Mock DDGS module
        with patch("persbot.services.search_service.DDGS") as mock_ddgs:
            mock_ddgs_class = Mock()
            mock_ddgs = Mock()
            mock_ddgs.text = Mock(return_value=mock_ddgs)
            mock_ddgs_instance = mock_ddgs_class.return_value

            # Mock search results
            mock_result1 = {"title": "Result 1", "href": "http://example.com/1", "body": "Snippet 1"}
            mock_result2 = {"title": "Result 2", "href": "http://example.com/2", "body": "Snippet 2"}

            mock_ddgs_instance.text = Mock(return_value=[mock_result1, mock_result2])

            service = SearchService()
            service._config = self.mock_config()

            results = await service.web_search(
                query="test query",
                num_results=5,
            )

            # Verify results
            assert results is not None
            assert len(results) == 2
            assert results[0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_web_search_empty_query(self):
        """Test web search with empty query raises error."""
        service = SearchService()
        service._config = Mock()

        with pytest.raises(SearchError) as exc_info:
            await service.web_search(query="   ")

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_web_search_rate_limit(self):
        """Test web search handles rate limit errors."""
        service = SearchService()
        service._config = Mock()

        with patch("persbot.services.search_service.DDGS") as mock_ddgs:
            mock_ddgs_class = Mock()
            mock_ddgs = Mock()
            mock_rate_limit_exc = Exception("429 Too Many Requests")
            mock_ddgs_instance = mock_ddgs_class.return_value

            # First call raises rate limit, second also raises
            mock_ddgs_instance.text = Mock(
                side_effect=[
                    mock_rate_limit_exc,
                    mock_rate_limit_exc,
                ]
            )

            with pytest.raises(SearchRateLimitError):
                await service.web_search(query="test query")

    @pytest.mark.asyncio
    async def test_web_search_timeout(self):
        """Test web search handles timeout."""
        service = SearchService()
        service._config = Mock()

        with patch("persbot.services.search_service.DDGS") as mock_ddgs:
            mock_ddgs_class = Mock()
            mock_ddgs = Mock()
            mock_timeout_exc = asyncio.TimeoutError()
            mock_ddgs_instance = mock_ddgs_class.return_value

            mock_ddgs_instance.text = Mock(side_effect=mock_timeout_exc)

            with pytest.raises(SearchError):
                await service.web_search(query="test query")

    @pytest.mark.asyncio
    async def test_web_search_cancel_before_execution(self):
        """Test cancellation before search execution."""
        service = SearchService()
        cancel_event = asyncio.Event()
        cancel_event.set()

        result = await service.web_search(
            query="test",
            cancel_event=cancel_event,
        )

        # Should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            # Get result to trigger the error
            await result if result else None

    @pytest.mark.asyncio
    async def test_web_search_num_results_clamping(self):
        """Test that num_results is clamped between 1-10."""
        service = SearchService()

        with patch("persbot.services.search_service.DDGS") as mock_ddgs:
            mock_ddgs_class = Mock()
            mock_ddgs = Mock()
            mock_ddgs_instance = mock_ddgs_class.return_value

            mock_ddgs_instance.text = Mock(return_value=[])

            # Request 15 results - should be clamped to 10
            await service.web_search(query="test", num_results=15)

            # Verify clamping to 10
            mock_ddgs_instance.text.assert_called_once()
            call_args = mock_ddgs_instance.text.call_args
            assert call_args[1]["max_results"] == 10

    @pytest.mark.asyncio
    async def test_calculate_backoff(self):
        """Test exponential backoff calculation."""
        service = SearchService()

        # Test various attempts
        assert service._calculate_backoff(1) == 2.0
        assert service._calculate_backoff(2) == 4.0
        assert service._calculate_backoff(3) == 8.0

        # Verify max delay cap
        assert service._calculate_backoff(10) == 10.0

    def test_perform_search(self):
        """Test _perform_search method."""
        service = SearchService()

        with patch("persbot.services.search_service.DDGS") as mock_ddgs:
            mock_ddgs_class = Mock()
            mock_ddgs = Mock()
            mock_ddgs_instance = mock_ddgs_class.return_value

            # Mock results
            mock_results = [
                {"title": "Test", "href": "http://test.com", "body": "Content"},
            ]

            mock_ddgs_instance.text = Mock(return_value=mock_results)

            # Mock asyncio.to_thread
            with patch("asyncio.to_thread") as mock_to_thread:
                mock_to_thread.return_value = mock_ddgs_instance.text

            result = service._perform_search(
                lambda: service._perform_search(mock_ddgs_instance.text, "test", 5)
            )

            # Verify DDGS.text was called correctly
            mock_ddgs_instance.text.assert_called_once()
            call_args = mock_ddgs_instance.text.call_args
            assert call_args[0][0] == "test"
            assert call_args[0][1] == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestSearchServiceIntegration:
    """Integration tests for SearchService."""

    @pytest.mark.asyncio
    async def test_full_search_flow(self, tmp_path):
        """Test complete search flow."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        models_file = data_dir / "models.json"
        test_data = {
            "image_models": [],
            "image_model_preferences": {},
        }
        models_file.write_text('{}', encoding="utf-8")

        with patch("persbot.services.search_service.DDGS") as mock_ddgs:
            mock_ddgs_class = Mock()
            mock_ddgs = Mock()
            mock_ddgs_instance = mock_ddgs_class.return_value

            # Set up results
            mock_ddgs_instance.text = Mock(return_value=[
                {"title": "Test Result", "href": "http://example.com", "body": "Test content"},
            ])

            service = SearchService()

            result = await service.web_search(query="test query")

            # Verify complete flow
            assert result is not None
            assert len(result) == 1
            assert result[0]["title"] == "Test Result"

"""Feature tests for search service module.

Tests focus on behavior using mocking to avoid external API dependencies:
- SearchError: base exception for search errors
- SearchRateLimitError: rate limit exception
- SearchService: web search via DuckDuckGo
"""

import asyncio
import sys
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest


# Mock ddgs module before importing anything else
@pytest.fixture(autouse=True)
def mock_ddgs():
    """Mock ddgs module to avoid import issues."""
    mock_ddgs_module = MagicMock()
    mock_ddgs_module.DDGS = MagicMock
    mock_ddgs_module.exceptions = MagicMock()
    mock_ddgs_module.exceptions.RatelimitException = Exception
    mock_ddgs_module.exceptions.DDGSException = Exception
    sys.modules['ddgs'] = mock_ddgs_module
    sys.modules['ddgs.exceptions'] = mock_ddgs_module.exceptions
    yield mock_ddgs_module
    if 'ddgs' in sys.modules:
        del sys.modules['ddgs']
    if 'ddgs.exceptions' in sys.modules:
        del sys.modules['ddgs.exceptions']


class TestSearchError:
    """Tests for SearchError exception."""

    def test_search_error_exists(self):
        """SearchError class exists."""
        from persbot.services.search_service import SearchError
        assert SearchError is not None

    def test_search_error_is_exception(self):
        """SearchError is an Exception."""
        from persbot.services.search_service import SearchError
        assert issubclass(SearchError, Exception)

    def test_search_error_can_be_raised(self):
        """SearchError can be raised with message."""
        from persbot.services.search_service import SearchError

        with pytest.raises(SearchError, match="test error"):
            raise SearchError("test error")


class TestSearchRateLimitError:
    """Tests for SearchRateLimitError exception."""

    def test_search_rate_limit_error_exists(self):
        """SearchRateLimitError class exists."""
        from persbot.services.search_service import SearchRateLimitError
        assert SearchRateLimitError is not None

    def test_search_rate_limit_error_is_search_error(self):
        """SearchRateLimitError is a SearchError."""
        from persbot.services.search_service import SearchRateLimitError, SearchError
        assert issubclass(SearchRateLimitError, SearchError)

    def test_search_rate_limit_error_can_be_raised(self):
        """SearchRateLimitError can be raised with message."""
        from persbot.services.search_service import SearchRateLimitError

        with pytest.raises(SearchRateLimitError, match="rate limited"):
            raise SearchRateLimitError("rate limited")


class TestSearchService:
    """Tests for SearchService class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return Mock()

    def test_search_service_exists(self):
        """SearchService class exists."""
        from persbot.services.search_service import SearchService
        assert SearchService is not None

    def test_creates_with_optional_config(self, mock_config):
        """SearchService creates with optional config."""
        from persbot.services.search_service import SearchService

        service = SearchService(config=mock_config)
        assert service._config == mock_config

    def test_creates_without_config(self):
        """SearchService creates without config."""
        from persbot.services.search_service import SearchService

        service = SearchService()
        assert service._config is None

    def test_default_max_retries_is_two(self, mock_config):
        """Default max_retries is 2."""
        from persbot.services.search_service import SearchService

        service = SearchService(config=mock_config)
        assert service._max_retries == 2

    def test_default_base_delay_is_two(self, mock_config):
        """Default base_delay is 2.0."""
        from persbot.services.search_service import SearchService

        service = SearchService(config=mock_config)
        assert service._base_delay == 2.0

    def test_default_max_delay_is_ten(self, mock_config):
        """Default max_delay is 10.0."""
        from persbot.services.search_service import SearchService

        service = SearchService(config=mock_config)
        assert service._max_delay == 10.0

    def test_default_request_timeout_is_thirty(self, mock_config):
        """Default request_timeout is 30.0."""
        from persbot.services.search_service import SearchService

        service = SearchService(config=mock_config)
        assert service._request_timeout == 30.0


class TestSearchServiceWebSearch:
    """Tests for SearchService.web_search method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return Mock()

    @pytest.fixture
    def service(self, mock_config):
        """Create a SearchService."""
        from persbot.services.search_service import SearchService
        return SearchService(config=mock_config)

    @pytest.mark.asyncio
    async def test_raises_error_for_empty_query(self, service):
        """web_search raises SearchError for empty query."""
        from persbot.services.search_service import SearchError

        with pytest.raises(SearchError, match="cannot be empty"):
            await service.web_search("")

    @pytest.mark.asyncio
    async def test_raises_error_for_whitespace_query(self, service):
        """web_search raises SearchError for whitespace query."""
        from persbot.services.search_service import SearchError

        with pytest.raises(SearchError, match="cannot be empty"):
            await service.web_search("   ")

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_when_cancel_event_set(self, service):
        """web_search raises CancelledError when cancel_event is set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service.web_search("test query", cancel_event=cancel_event)


class TestSearchServiceCalculateBackoff:
    """Tests for SearchService._calculate_backoff method."""

    @pytest.fixture
    def service(self):
        """Create a SearchService."""
        from persbot.services.search_service import SearchService
        return SearchService()

    def test_first_attempt_returns_base_delay(self, service):
        """_calculate_backoff returns base_delay for first attempt."""
        # base_delay^1 = 2.0
        result = service._calculate_backoff(1)
        assert result == 2.0

    def test_second_attempt_returns_squared_delay(self, service):
        """_calculate_backoff returns squared delay for second attempt."""
        # base_delay^2 = 4.0
        result = service._calculate_backoff(2)
        assert result == 4.0

    def test_caps_at_max_delay(self, service):
        """_calculate_backoff caps at max_delay."""
        # base_delay^10 would be 1024, but capped at 10.0
        result = service._calculate_backoff(10)
        assert result == 10.0


class TestSearchServicePerformSearch:
    """Tests for SearchService._perform_search method."""

    @pytest.fixture
    def service(self):
        """Create a SearchService."""
        from persbot.services.search_service import SearchService
        return SearchService()

    def test_perform_search_exists(self, service):
        """_perform_search method exists."""
        assert hasattr(service, '_perform_search')

    def test_perform_search_returns_list(self, service):
        """_perform_search returns a list."""
        # This would need DDGS to work, so we just check the method signature
        import inspect
        sig = inspect.signature(service._perform_search)
        assert 'query' in sig.parameters
        assert 'num_results' in sig.parameters


class TestSearchServiceExecuteWithRetry:
    """Tests for SearchService._execute_with_retry method."""

    @pytest.fixture
    def service(self):
        """Create a SearchService."""
        from persbot.services.search_service import SearchService
        return SearchService()

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self, service):
        """_execute_with_retry returns result on success."""
        def success_func():
            return ["result1", "result2"]

        result = await service._execute_with_retry(success_func)
        assert result == ["result1", "result2"]

    @pytest.mark.asyncio
    async def test_raises_cancelled_error_when_event_set(self, service):
        """_execute_with_retry raises CancelledError when cancel_event set."""
        cancel_event = asyncio.Event()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError):
            await service._execute_with_retry(lambda: [], cancel_event=cancel_event)

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self, service):
        """_execute_with_retry retries on timeout."""
        call_count = 0

        def timeout_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return ["success"]

        # Reduce timeout for faster test
        service._request_timeout = 0.1
        service._base_delay = 0.01

        result = await service._execute_with_retry(timeout_then_success)
        assert call_count == 2
        assert result == ["success"]

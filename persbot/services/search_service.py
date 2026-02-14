"""Web search service for SoyeBot AI.

This service handles web search through DuckDuckGo with retry logic
and cancellation support.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ddgs import DDGS
from ddgs.exceptions import RatelimitException, DDGSException

from persbot.config import AppConfig

logger = logging.getLogger(__name__)


class SearchError(Exception):
    """Base exception for search errors."""

    pass


class SearchRateLimitError(SearchError):
    """Exception raised when search rate limit is exceeded."""

    pass


class SearchService:
    """Service for web search via DuckDuckGo.

    This service handles:
    - Web search using DuckDuckGo
    - Retry logic for transient failures
    - Cancellation support via cancel_event
    - Proper timeout handling
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the search service.

        Args:
            config: Application configuration (optional, for future settings).
        """
        self._config = config
        self._max_retries = 2
        self._base_delay = 2.0
        self._max_delay = 10.0
        self._request_timeout = 30.0

    async def web_search(
        self,
        query: str,
        num_results: int = 5,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[List[Dict[str, str]]]:
        """
        Search the web for information using DuckDuckGo.

        Args:
            query: The search query string.
            num_results: Number of results to return (default: 5, max: 10).
            cancel_event: AsyncIO event to check for cancellation before API calls.

        Returns:
            List of search result dictionaries with keys: title, url, snippet, source.

        Raises:
            asyncio.CancelledError: If cancel_event is set before/during search.
            SearchError: If search fails after retries.
            SearchRateLimitError: If rate limit is exceeded.
        """
        # Check cancellation event before starting
        if cancel_event and cancel_event.is_set():
            logger.debug("Web search aborted before API call")
            raise asyncio.CancelledError("Web search aborted by user")

        if not query or not query.strip():
            raise SearchError("Search query cannot be empty")

        num_results = min(max(1, num_results), 10)  # Clamp between 1-10

        # Check for cancellation before API call
        if cancel_event and cancel_event.is_set():
            logger.debug("Web search aborted before API call")
            raise asyncio.CancelledError("Web search aborted by user")

        logger.debug(
            "Performing web search: query='%s' (num_results=%d)",
            query[:100] if len(query) > 100 else query,
            num_results,
        )

        # Execute with retry logic
        result = await self._execute_with_retry(
            lambda: self._perform_search(query, num_results),
            cancel_event=cancel_event,
        )

        if result is None:
            return None

        logger.info(
            "Web search completed successfully: %d results for query='%s'",
            len(result),
            query[:50] if len(query) > 50 else query,
        )

        return result

    async def _execute_with_retry(
        self,
        search_func: callable,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Optional[Any]:
        """Execute search with retry logic.

        Args:
            search_func: Function to execute that returns search results.
            cancel_event: AsyncIO event to check for cancellation.

        Returns:
            Search results on success, None on failure.

        Raises:
            asyncio.CancelledError: If cancel_event is set.
            SearchError: If all retries fail.
        """
        last_error = None

        for attempt in range(1, self._max_retries + 1):
            # Check for cancellation at the start of each retry iteration
            if cancel_event and cancel_event.is_set():
                logger.debug("Search retry loop aborted")
                raise asyncio.CancelledError("Search aborted by user")

            try:
                # Execute the search with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(search_func),
                    timeout=self._request_timeout,
                )
                return result

            except asyncio.TimeoutError:
                last_error = SearchError("Search request timed out")
                logger.warning(
                    "Search timeout (attempt %d/%d)",
                    attempt,
                    self._max_retries,
                )

                if attempt < self._max_retries:
                    delay = self._calculate_backoff(attempt)
                    # Check for cancellation before backoff sleep
                    if cancel_event and cancel_event.is_set():
                        logger.debug("Search retry aborted during backoff")
                        raise asyncio.CancelledError("Search aborted by user")
                    await asyncio.sleep(delay)
                    continue

            except RatelimitException as e:
                last_error = SearchRateLimitError(str(e))
                logger.warning(
                    "Search rate limit exceeded (attempt %d/%d)",
                    attempt,
                    self._max_retries,
                )

                if attempt < self._max_retries:
                    delay = 5.0  # Fixed delay for rate limit
                    # Check for cancellation before backoff sleep
                    if cancel_event and cancel_event.is_set():
                        logger.debug("Search retry aborted during backoff")
                        raise asyncio.CancelledError("Search aborted by user")
                    await asyncio.sleep(delay)
                    continue

                # Rate limit error, don't retry further
                break

            except DDGSException as e:
                last_error = SearchError(f"Search service error: {str(e)}")
                logger.error(
                    "Search service error (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                )

                if attempt < self._max_retries:
                    delay = self._calculate_backoff(attempt)
                    # Check for cancellation before backoff sleep
                    if cancel_event and cancel_event.is_set():
                        logger.debug("Search retry aborted during backoff")
                        raise asyncio.CancelledError("Search aborted by user")
                    await asyncio.sleep(delay)
                    continue

            except asyncio.CancelledError:
                logger.debug("Search cancelled during execution")
                raise

            except Exception as e:
                last_error = SearchError(f"Unexpected search error: {str(e)}")
                logger.error(
                    "Unexpected search error (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                    exc_info=True,
                )

                if attempt < self._max_retries:
                    delay = self._calculate_backoff(attempt)
                    # Check for cancellation before backoff sleep
                    if cancel_event and cancel_event.is_set():
                        logger.debug("Search retry aborted during backoff")
                        raise asyncio.CancelledError("Search aborted by user")
                    await asyncio.sleep(delay)
                    continue

        # All retries failed
        if isinstance(last_error, SearchRateLimitError):
            raise last_error
        elif last_error:
            raise last_error
        else:
            raise SearchError("Search failed after retries")

    def _perform_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Perform synchronous DuckDuckGo search.

        Args:
            query: The search query string.
            num_results: Number of results to return.

        Returns:
            List of search result dictionaries.

        Raises:
            RatelimitException: When DuckDuckGo rate limits the request.
            DDGSException: For other search errors.
        """
        ddgs = DDGS()
        search_results = ddgs.text(
            query,
            max_results=num_results,
            region="ko-kr",
            safesearch="moderate",
            backend="auto",
        )

        # Transform DDGS results to match expected format
        results = []
        for result in search_results or []:
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "DuckDuckGo",
                }
            )

        return results

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (1-indexed).

        Returns:
            Delay in seconds.
        """
        return min(self._base_delay**attempt, self._max_delay)

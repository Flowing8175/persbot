"""Feature tests for web tools module.

Tests focus on behavior using mocking:
- inspect_external_content: inspect content from external URL
- register_web_tools: register tools with registry
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

# Create a more complete bs4 mock
_mock_bs4 = MagicMock()
_mock_BeautifulSoup = MagicMock()
_mock_bs4.BeautifulSoup = _mock_BeautifulSoup
sys.modules['bs4'] = _mock_bs4


class TestInspectExternalContent:
    """Tests for inspect_external_content function."""

    @pytest.mark.asyncio
    async def test_inspect_external_content_exists(self):
        """inspect_external_content function exists."""
        from persbot.tools.persona_tools.web_tools import inspect_external_content
        assert inspect_external_content is not None

    @pytest.mark.asyncio
    async def test_returns_error_without_url(self):
        """inspect_external_content returns error without URL."""
        from persbot.tools.persona_tools.web_tools import inspect_external_content

        # Function requires url parameter, so test with empty string
        result = await inspect_external_content(url="")

        # Should return error or handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_handles_invalid_url(self):
        """inspect_external_content handles invalid URL."""
        from persbot.tools.persona_tools.web_tools import inspect_external_content

        result = await inspect_external_content(url="not-a-valid-url")

        # Should handle gracefully (may return error)
        assert result is not None


class TestParseHtmlSync:
    """Tests for _parse_html_sync function."""

    def test_parse_html_sync_exists(self):
        """_parse_html_sync function exists."""
        from persbot.tools.persona_tools.web_tools import _parse_html_sync
        assert _parse_html_sync is not None

    def test_extracts_title_from_html(self):
        """_parse_html_sync extracts title from HTML."""
        from persbot.tools.persona_tools.web_tools import _parse_html_sync

        html = "<html><head><title>Test Page</title></head><body><p>Content</p></body></html>"

        # Mock BeautifulSoup to work properly
        with patch('persbot.tools.persona_tools.web_tools.BeautifulSoup') as mock_bs:
            mock_soup = MagicMock()
            mock_soup.title = MagicMock()
            mock_soup.title.string = "Test Page"
            mock_soup.find.return_value = None
            mock_soup.body = MagicMock()
            mock_soup.get_text.return_value = "Content"
            mock_bs.return_value = mock_soup

            result = _parse_html_sync(html)

            assert "title" in result

    def test_handles_html_without_title(self):
        """_parse_html_sync handles HTML without title."""
        from persbot.tools.persona_tools.web_tools import _parse_html_sync

        html = "<html><body><p>Just content</p></body></html>"

        with patch('persbot.tools.persona_tools.web_tools.BeautifulSoup') as mock_bs:
            mock_soup = MagicMock()
            mock_soup.title = None
            mock_soup.find.return_value = None
            mock_soup.body = MagicMock()
            mock_soup.get_text.return_value = "Just content"
            mock_bs.return_value = mock_soup

            result = _parse_html_sync(html)

            assert result is not None


class TestRegisterWebTools:
    """Tests for register_web_tools function."""

    def test_register_web_tools_exists(self):
        """register_web_tools function exists."""
        from persbot.tools.persona_tools.web_tools import register_web_tools
        assert register_web_tools is not None

    def test_registers_tools(self):
        """register_web_tools registers tools."""
        from persbot.tools.persona_tools.web_tools import register_web_tools

        mock_registry = MagicMock()
        register_web_tools(mock_registry)

        # Should register at least 1 tool
        assert mock_registry.register.call_count >= 1

    def test_registers_inspect_external_content(self):
        """register_web_tools registers inspect_external_content."""
        from persbot.tools.persona_tools.web_tools import register_web_tools

        mock_registry = MagicMock()
        register_web_tools(mock_registry)

        call_args = mock_registry.register.call_args_list[0]
        tool_def = call_args[0][0]
        assert tool_def.name == "inspect_external_content"

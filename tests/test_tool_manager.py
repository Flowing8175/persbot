"""Feature tests for tool manager module.

Tests focus on behavior using mocking:
- ToolManager: manages tool registration and execution
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


class TestToolManager:
    """Tests for ToolManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.enable_tools = True
        config.enable_discord_tools = True
        config.enable_api_tools = True
        config.enable_persona_tools = True
        config.search_api_key = None
        config.weather_api_key = None
        return config

    def test_tool_manager_exists(self):
        """ToolManager class exists."""
        from persbot.tools.manager import ToolManager
        assert ToolManager is not None

    def test_creates_with_config(self, mock_config):
        """ToolManager creates with config."""
        from persbot.tools.manager import ToolManager

        with patch('persbot.tools.manager.register_all_discord_tools') as mock_discord:
            with patch('persbot.tools.manager.register_all_api_tools') as mock_api:
                with patch('persbot.tools.manager.register_all_persona_tools') as mock_persona:
                    manager = ToolManager(mock_config)

                    assert manager.config == mock_config
                    assert manager.registry is not None
                    assert manager.executor is not None

    def test_registers_tools_when_enabled(self, mock_config):
        """ToolManager registers tools when enabled."""
        from persbot.tools.manager import ToolManager

        with patch('persbot.tools.manager.register_all_discord_tools') as mock_discord:
            with patch('persbot.tools.manager.register_all_api_tools') as mock_api:
                with patch('persbot.tools.manager.register_all_persona_tools') as mock_persona:
                    manager = ToolManager(mock_config)

                    mock_discord.assert_called_once()
                    mock_api.assert_called_once()
                    mock_persona.assert_called_once()

    def test_skips_registration_when_disabled(self):
        """ToolManager skips registration when disabled."""
        from persbot.tools.manager import ToolManager

        mock_config = Mock()
        mock_config.enable_tools = False

        with patch('persbot.tools.manager.register_all_discord_tools') as mock_discord:
            with patch('persbot.tools.manager.register_all_api_tools') as mock_api:
                with patch('persbot.tools.manager.register_all_persona_tools') as mock_persona:
                    manager = ToolManager(mock_config)

                    mock_discord.assert_not_called()
                    mock_api.assert_not_called()
                    mock_persona.assert_not_called()


class TestToolManagerGetEnabledTools:
    """Tests for ToolManager.get_enabled_tools method."""

    @pytest.fixture
    def manager(self):
        """Create a ToolManager instance."""
        from persbot.tools.manager import ToolManager

        mock_config = Mock()
        mock_config.enable_tools = True
        mock_config.enable_discord_tools = False
        mock_config.enable_api_tools = False
        mock_config.enable_persona_tools = False

        with patch('persbot.tools.manager.register_all_discord_tools'):
            with patch('persbot.tools.manager.register_all_api_tools'):
                with patch('persbot.tools.manager.register_all_persona_tools'):
                    return ToolManager(mock_config)

    def test_get_enabled_tools_exists(self, manager):
        """get_enabled_tools method exists."""
        assert manager.get_enabled_tools is not None

    def test_returns_dict(self, manager):
        """get_enabled_tools returns dict."""
        result = manager.get_enabled_tools()
        assert isinstance(result, dict)


class TestToolManagerGetToolsByCategory:
    """Tests for ToolManager.get_tools_by_category method."""

    @pytest.fixture
    def manager(self):
        """Create a ToolManager instance."""
        from persbot.tools.manager import ToolManager

        mock_config = Mock()
        mock_config.enable_tools = True
        mock_config.enable_discord_tools = False
        mock_config.enable_api_tools = False
        mock_config.enable_persona_tools = False

        with patch('persbot.tools.manager.register_all_discord_tools'):
            with patch('persbot.tools.manager.register_all_api_tools'):
                with patch('persbot.tools.manager.register_all_persona_tools'):
                    return ToolManager(mock_config)

    def test_get_tools_by_category_exists(self, manager):
        """get_tools_by_category method exists."""
        assert manager.get_tools_by_category is not None

    def test_returns_list(self, manager):
        """get_tools_by_category returns list."""
        from persbot.tools.base import ToolCategory

        result = manager.get_tools_by_category(ToolCategory.DISCORD_GUILD)
        assert isinstance(result, list)


class TestToolManagerExecuteTool:
    """Tests for ToolManager.execute_tool method."""

    @pytest.fixture
    def manager(self):
        """Create a ToolManager instance."""
        from persbot.tools.manager import ToolManager

        mock_config = Mock()
        mock_config.enable_tools = True
        mock_config.enable_discord_tools = False
        mock_config.enable_api_tools = False
        mock_config.enable_persona_tools = False

        with patch('persbot.tools.manager.register_all_discord_tools'):
            with patch('persbot.tools.manager.register_all_api_tools'):
                with patch('persbot.tools.manager.register_all_persona_tools'):
                    return ToolManager(mock_config)

    @pytest.mark.asyncio
    async def test_execute_tool_exists(self, manager):
        """execute_tool method exists."""
        assert manager.execute_tool is not None

    @pytest.mark.asyncio
    async def test_execute_tool_calls_executor(self, manager):
        """execute_tool calls executor."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = "test"
        mock_result.error = None

        with patch.object(manager.executor, 'execute_tool', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result

            result = await manager.execute_tool("test_tool", {})

            mock_exec.assert_called_once()


class TestToolManagerSetToolEnabled:
    """Tests for ToolManager.set_tool_enabled method."""

    @pytest.fixture
    def manager(self):
        """Create a ToolManager instance."""
        from persbot.tools.manager import ToolManager

        mock_config = Mock()
        mock_config.enable_tools = True
        mock_config.enable_discord_tools = False
        mock_config.enable_api_tools = False
        mock_config.enable_persona_tools = False

        with patch('persbot.tools.manager.register_all_discord_tools'):
            with patch('persbot.tools.manager.register_all_api_tools'):
                with patch('persbot.tools.manager.register_all_persona_tools'):
                    return ToolManager(mock_config)

    def test_set_tool_enabled_exists(self, manager):
        """set_tool_enabled method exists."""
        assert manager.set_tool_enabled is not None

    def test_set_tool_enabled_calls_registry(self, manager):
        """set_tool_enabled calls registry."""
        with patch.object(manager.registry, 'set_tool_enabled', return_value=True) as mock_set:
            result = manager.set_tool_enabled("test_tool", True)

            mock_set.assert_called_once_with("test_tool", True)
            assert result is True


class TestToolManagerIsEnabled:
    """Tests for ToolManager.is_enabled method."""

    def test_is_enabled_returns_true_when_enabled(self):
        """is_enabled returns True when tools enabled."""
        from persbot.tools.manager import ToolManager

        mock_config = Mock()
        mock_config.enable_tools = True
        mock_config.enable_discord_tools = False
        mock_config.enable_api_tools = False
        mock_config.enable_persona_tools = False

        with patch('persbot.tools.manager.register_all_discord_tools'):
            with patch('persbot.tools.manager.register_all_api_tools'):
                with patch('persbot.tools.manager.register_all_persona_tools'):
                    manager = ToolManager(mock_config)

                    assert manager.is_enabled() is True

    def test_is_enabled_returns_false_when_disabled(self):
        """is_enabled returns False when tools disabled."""
        from persbot.tools.manager import ToolManager

        mock_config = Mock()
        mock_config.enable_tools = False

        with patch('persbot.tools.manager.register_all_discord_tools'):
            with patch('persbot.tools.manager.register_all_api_tools'):
                with patch('persbot.tools.manager.register_all_persona_tools'):
                    manager = ToolManager(mock_config)

                    assert manager.is_enabled() is False


class TestToolManagerGetMetrics:
    """Tests for ToolManager.get_metrics method."""

    @pytest.fixture
    def manager(self):
        """Create a ToolManager instance."""
        from persbot.tools.manager import ToolManager

        mock_config = Mock()
        mock_config.enable_tools = True
        mock_config.enable_discord_tools = False
        mock_config.enable_api_tools = False
        mock_config.enable_persona_tools = False

        with patch('persbot.tools.manager.register_all_discord_tools'):
            with patch('persbot.tools.manager.register_all_api_tools'):
                with patch('persbot.tools.manager.register_all_persona_tools'):
                    return ToolManager(mock_config)

    def test_get_metrics_exists(self, manager):
        """get_metrics method exists."""
        assert manager.get_metrics is not None

    def test_get_metrics_calls_executor(self, manager):
        """get_metrics calls executor."""
        with patch.object(manager.executor, 'get_metrics', return_value={}) as mock_metrics:
            result = manager.get_metrics()

            mock_metrics.assert_called_once()
            assert result == {}

"""Feature tests for tool base definitions.

Tests focus on behavior rather than implementation details:
- ToolDefinition: tool definition and execution
- ToolResult: execution result dataclass
- ToolParameter: parameter definition dataclass
- ToolCategory: tool category enumeration
"""

import asyncio

import pytest

from persbot.tools.base import (
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


# ==============================================================================
# ToolCategory Feature Tests
# ==============================================================================

class TestToolCategory:
    """Tests for ToolCategory enumeration."""

    def test_has_discord_categories(self):
        """ToolCategory includes Discord-related categories."""
        assert ToolCategory.DISCORD_CHANNEL.value == "discord_channel"
        assert ToolCategory.DISCORD_USER.value == "discord_user"
        assert ToolCategory.DISCORD_GUILD.value == "discord_guild"

    def test_has_api_categories(self):
        """ToolCategory includes API-related categories."""
        assert ToolCategory.API_SEARCH.value == "api_search"
        assert ToolCategory.API_WEATHER.value == "api_weather"
        assert ToolCategory.API_TIME.value == "api_time"

    def test_has_persona_categories(self):
        """ToolCategory includes persona-related categories."""
        assert ToolCategory.PERSONA_MEMORY.value == "persona_memory"
        assert ToolCategory.PERSONA_MEDIA.value == "persona_media"
        assert ToolCategory.PERSONA_ROUTINE.value == "persona_routine"
        assert ToolCategory.PERSONA_WEB.value == "persona_web"

    def test_can_iterate_all_categories(self):
        """ToolCategory can be iterated to get all categories."""
        categories = list(ToolCategory)
        assert len(categories) == 10


# ==============================================================================
# ToolParameter Feature Tests
# ==============================================================================

class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_creates_with_required_fields(self):
        """ToolParameter creates with required fields."""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
        )
        assert param.name == "query"
        assert param.type == "string"
        assert param.description == "Search query"
        assert param.required is False
        assert param.default is None
        assert param.enum is None
        assert param.items_type is None

    def test_creates_with_all_fields(self):
        """ToolParameter creates with all fields specified."""
        param = ToolParameter(
            name="count",
            type="integer",
            description="Number of results",
            required=True,
            default=10,
            enum=None,
            items_type=None,
        )
        assert param.name == "count"
        assert param.type == "integer"
        assert param.description == "Number of results"
        assert param.required is True
        assert param.default == 10

    def test_creates_enum_parameter(self):
        """ToolParameter supports enum parameters."""
        param = ToolParameter(
            name="sort_order",
            type="string",
            description="Sort order",
            enum=["asc", "desc"],
        )
        assert param.enum == ["asc", "desc"]

    def test_creates_array_parameter(self):
        """ToolParameter supports array parameters with items_type."""
        param = ToolParameter(
            name="ids",
            type="array",
            description="List of IDs",
            items_type="integer",
        )
        assert param.type == "array"
        assert param.items_type == "integer"

    def test_is_mutable(self):
        """ToolParameter is mutable (not frozen)."""
        param = ToolParameter(name="test", type="string", description="Test")
        param.required = True
        param.default = "default_value"
        assert param.required is True
        assert param.default == "default_value"


# ==============================================================================
# ToolResult Feature Tests
# ==============================================================================

class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_creates_successful_result(self):
        """ToolResult creates a successful result."""
        result = ToolResult(success=True, data="test data")
        assert result.success is True
        assert result.data == "test data"
        assert result.error is None
        assert result.metadata == {}

    def test_creates_failed_result(self):
        """ToolResult creates a failed result."""
        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None

    def test_creates_result_with_metadata(self):
        """ToolResult can include metadata."""
        result = ToolResult(
            success=True,
            data={"key": "value"},
            metadata={"duration_ms": 150, "cache_hit": True},
        )
        assert result.metadata["duration_ms"] == 150
        assert result.metadata["cache_hit"] is True

    def test_default_metadata_is_empty_dict(self):
        """ToolResult metadata defaults to empty dict."""
        result = ToolResult(success=True)
        assert result.metadata == {}
        # Ensure it's a mutable dict, not shared
        result.metadata["key"] = "value"
        assert result.metadata["key"] == "value"

    def test_is_mutable(self):
        """ToolResult is mutable (not frozen)."""
        result = ToolResult(success=True)
        result.success = False
        result.error = "New error"
        result.metadata["key"] = "value"
        assert result.success is False
        assert result.error == "New error"
        assert result.metadata["key"] == "value"


# ==============================================================================
# ToolDefinition Feature Tests
# ==============================================================================

class TestToolDefinitionCreation:
    """Tests for ToolDefinition instantiation."""

    def test_creates_with_required_fields(self):
        """ToolDefinition creates with required fields."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.category == ToolCategory.API_SEARCH
        assert tool.parameters == []
        assert tool.handler is handler
        assert tool.enabled is True
        assert tool.requires_permission is None
        assert tool.timeout is None

    def test_creates_with_all_fields(self):
        """ToolDefinition creates with all fields specified."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        params = [ToolParameter(name="q", type="string", description="Query")]
        tool = ToolDefinition(
            name="search",
            description="Search tool",
            category=ToolCategory.API_SEARCH,
            parameters=params,
            handler=handler,
            requires_permission="read_messages",
            enabled=False,
            timeout=30.0,
        )
        assert tool.name == "search"
        assert tool.requires_permission == "read_messages"
        assert tool.enabled is False
        assert tool.timeout == 30.0
        assert len(tool.parameters) == 1


class TestToolDefinitionExecute:
    """Tests for ToolDefinition.execute behavior."""

    @pytest.mark.asyncio
    async def test_execute_returns_result_from_handler(self):
        """execute returns ToolResult from handler."""
        async def handler(**kwargs):
            return ToolResult(success=True, data="result")

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        result = await tool.execute()

        assert result.success is True
        assert result.data == "result"

    @pytest.mark.asyncio
    async def test_execute_wraps_non_tool_result(self):
        """execute wraps non-ToolResult return values."""
        async def handler(**kwargs):
            return {"key": "value"}

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        result = await tool.execute()

        assert result.success is True
        assert result.data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_execute_passes_kwargs_to_handler(self):
        """execute passes kwargs to the handler."""
        received = {}

        async def handler(**kwargs):
            received.update(kwargs)
            return ToolResult(success=True)

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        await tool.execute(query="test", count=5)

        assert received["query"] == "test"
        assert received["count"] == 5

    @pytest.mark.asyncio
    async def test_execute_returns_failed_result_on_exception(self):
        """execute returns failed result when handler raises exception."""
        async def handler(**kwargs):
            raise ValueError("Handler error")

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        result = await tool.execute()

        assert result.success is False
        assert "Handler error" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_cancel_event_set_returns_aborted(self):
        """execute returns aborted result when cancel_event is set."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        cancel_event = asyncio.Event()
        cancel_event.set()

        result = await tool.execute(cancel_event=cancel_event)

        assert result.success is False
        assert "aborted" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_cancel_event_not_set_proceeds(self):
        """execute proceeds normally when cancel_event is not set."""
        async def handler(**kwargs):
            return ToolResult(success=True, data="ok")

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        cancel_event = asyncio.Event()

        result = await tool.execute(cancel_event=cancel_event)

        assert result.success is True
        assert result.data == "ok"

    @pytest.mark.asyncio
    async def test_execute_passes_cancel_event_to_handler_if_accepted(self):
        """execute passes cancel_event to handler if it accepts it."""
        received = {}

        async def handler(cancel_event=None, **kwargs):
            received["cancel_event"] = cancel_event
            return ToolResult(success=True)

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        cancel_event = asyncio.Event()
        await tool.execute(cancel_event=cancel_event, query="test")

        assert received["cancel_event"] is cancel_event

    @pytest.mark.asyncio
    async def test_execute_reraises_cancelled_error(self):
        """execute re-raises asyncio.CancelledError."""
        async def handler(**kwargs):
            raise asyncio.CancelledError()

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )

        with pytest.raises(asyncio.CancelledError):
            await tool.execute()


class TestToolDefinitionOpenAIFormat:
    """Tests for ToolDefinition.to_openai_format behavior."""

    def test_converts_to_openai_format(self):
        """to_openai_format converts to OpenAI function format."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        tool = ToolDefinition(
            name="search",
            description="Search for items",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )
        result = tool.to_openai_format()

        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Search for items"
        assert result["function"]["parameters"]["type"] == "object"

    def test_includes_parameters_in_openai_format(self):
        """to_openai_format includes parameter definitions."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        params = [
            ToolParameter(name="query", type="string", description="Search query", required=True),
            ToolParameter(name="count", type="integer", description="Max results", required=False, default=10),
        ]
        tool = ToolDefinition(
            name="search",
            description="Search",
            category=ToolCategory.API_SEARCH,
            parameters=params,
            handler=handler,
        )
        result = tool.to_openai_format()

        props = result["function"]["parameters"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"
        assert props["query"]["description"] == "Search query"
        assert "count" in props
        assert props["count"]["default"] == 10
        assert result["function"]["parameters"]["required"] == ["query"]

    def test_includes_enum_in_openai_format(self):
        """to_openai_format includes enum values."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        params = [
            ToolParameter(
                name="sort",
                type="string",
                description="Sort order",
                enum=["asc", "desc"],
            ),
        ]
        tool = ToolDefinition(
            name="search",
            description="Search",
            category=ToolCategory.API_SEARCH,
            parameters=params,
            handler=handler,
        )
        result = tool.to_openai_format()

        props = result["function"]["parameters"]["properties"]
        assert props["sort"]["enum"] == ["asc", "desc"]

    def test_includes_array_items_in_openai_format(self):
        """to_openai_format includes array items type."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        params = [
            ToolParameter(
                name="ids",
                type="array",
                description="List of IDs",
                items_type="integer",
            ),
        ]
        tool = ToolDefinition(
            name="search",
            description="Search",
            category=ToolCategory.API_SEARCH,
            parameters=params,
            handler=handler,
        )
        result = tool.to_openai_format()

        props = result["function"]["parameters"]["properties"]
        assert props["ids"]["type"] == "array"
        assert props["ids"]["items"]["type"] == "integer"

    def test_empty_required_list_when_no_required_params(self):
        """to_openai_format has empty required list when no required params."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        params = [
            ToolParameter(name="optional", type="string", description="Optional", required=False),
        ]
        tool = ToolDefinition(
            name="tool",
            description="Tool",
            category=ToolCategory.API_SEARCH,
            parameters=params,
            handler=handler,
        )
        result = tool.to_openai_format()

        assert result["function"]["parameters"]["required"] == []


class TestToolDefinitionGeminiFormat:
    """Tests for ToolDefinition.to_gemini_format behavior."""

    def test_raises_import_error_without_genai(self, monkeypatch):
        """to_gemini_format raises ImportError when google.genai is not available."""
        async def handler(**kwargs):
            return ToolResult(success=True)

        tool = ToolDefinition(
            name="test",
            description="Test",
            category=ToolCategory.API_SEARCH,
            parameters=[],
            handler=handler,
        )

        # Mock HAS_GENAI to False
        import persbot.tools.base as base_module
        monkeypatch.setattr(base_module, "HAS_GENAI", False)

        with pytest.raises(ImportError) as exc_info:
            tool.to_gemini_format()

        assert "google.genai" in str(exc_info.value)

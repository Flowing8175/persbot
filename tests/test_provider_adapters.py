"""Feature tests for provider adapters.

Tests cover:
- GeminiToolAdapter: convert_tools, extract_function_calls, format_results
- OpenAIToolAdapter: convert_tools, extract_function_calls, format_results
- ZAIToolAdapter: inherits from OpenAIToolAdapter, format_results with binary support
- ToolAdapterRegistry: register, get, get_or_create
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock google.genai.types before importing adapters
@pytest.fixture(autouse=True)
def mock_genai_types():
    """Mock google.genai.types module to avoid import issues."""
    # Create mock types module
    mock_types = MagicMock()

    # Create mock classes for genai types
    class MockTool:
        def __init__(self, function_declarations=None):
            self.function_declarations = function_declarations or []

    class MockFunctionDeclaration:
        def __init__(self, name=None, description=None, parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters

    class MockSchema:
        def __init__(self, type=None, description=None, properties=None, required=None):
            self.type = type
            self.description = description
            self.properties = properties
            self.required = required

    class MockPart:
        def __init__(self, function_response=None):
            self.function_response = function_response

    class MockFunctionResponse:
        def __init__(self, name=None, response=None):
            self.name = name
            self.response = response

    mock_types.Tool = MockTool
    mock_types.FunctionDeclaration = MockFunctionDeclaration
    mock_types.Schema = MockSchema
    mock_types.Part = MockPart
    mock_types.FunctionResponse = MockFunctionResponse

    # Mock the module
    mock_genai = MagicMock()
    mock_genai.types = mock_types
    sys.modules['google'] = mock_genai
    sys.modules['google.genai'] = mock_genai
    sys.modules['google.genai.types'] = mock_types

    yield mock_types

    # Cleanup
    for mod in ['google.genai.types', 'google.genai', 'google']:
        if mod in sys.modules:
            del sys.modules[mod]


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockToolParameter:
    """Mock ToolParameter for testing."""

    def __init__(self, name, type="string", description="", required=False, enum=None, default=None, items_type=None):
        self.name = name
        self.type = type
        self.description = description
        self.required = required
        self.enum = enum
        self.default = default
        self.items_type = items_type


class MockToolDefinition:
    """Mock ToolDefinition for testing."""

    def __init__(self, name, description, parameters=None, enabled=True):
        self.name = name
        self.description = description
        self.parameters = parameters or []
        self.enabled = enabled

    def to_openai_format(self):
        """Convert to OpenAI format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop_def = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop_def["enum"] = param.enum
            if param.default is not None:
                prop_def["default"] = param.default
            if param.type == "array" and param.items_type:
                prop_def["items"] = {"type": param.items_type}

            properties[param.name] = prop_def
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_gemini_format(self):
        """Convert to Gemini format."""
        from google.genai import types as genai_types

        properties = {}
        required = []

        for param in self.parameters:
            prop_def = genai_types.Schema(
                type=param.type,
                description=param.description,
            )
            if param.required:
                required.append(param.name)

            properties[param.name] = prop_def

        # Return a simple object that can be wrapped in a list
        # The actual genai_types.Tool expects function_declarations as a list
        class _FunctionDecl:
            def __init__(self, name, description, parameters):
                self.name = name
                self.description = description
                self.parameters = parameters

        return _FunctionDecl(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type="object",
                properties=properties,
                required=required,
            ),
        )


class MockGeminiFunctionCall:
    """Mock Gemini function call."""

    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


class MockGeminiPart:
    """Mock Gemini response part."""

    def __init__(self, function_call=None):
        self.function_call = function_call


class MockGeminiContent:
    """Mock Gemini content."""

    def __init__(self, parts=None):
        self.parts = parts or []


class MockGeminiCandidate:
    """Mock Gemini candidate."""

    def __init__(self, content=None):
        self.content = content


class MockGeminiResponse:
    """Mock Gemini response."""

    def __init__(self, candidates=None):
        self.candidates = candidates or []


class MockOpenAIFunction:
    """Mock OpenAI function call."""

    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class MockOpenAIToolCall:
    """Mock OpenAI tool call."""

    def __init__(self, id, function):
        self.id = id
        self.function = function


class MockOpenAIMessage:
    """Mock OpenAI message."""

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls


class MockOpenAIChoice:
    """Mock OpenAI choice."""

    def __init__(self, message=None):
        self.message = message


class MockOpenAIResponse:
    """Mock OpenAI response."""

    def __init__(self, choices=None):
        self.choices = choices or []


# ============================================================================
# GeminiToolAdapter Tests
# ============================================================================

class TestGeminiToolAdapterConvertTools:
    """Tests for GeminiToolAdapter.convert_tools."""

    def test_convert_tools_returns_list(self):
        """convert_tools returns a list."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter
        from unittest.mock import MagicMock

        # Create a mock tool with to_gemini_format method
        mock_tool = MagicMock()
        mock_tool.enabled = True
        mock_tool.to_gemini_format = MagicMock(return_value=MagicMock())

        tools = [mock_tool]
        result = GeminiToolAdapter.convert_tools(tools)

        assert isinstance(result, list)

    def test_convert_tools_converts_single_tool(self):
        """convert_tools converts a single tool."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter
        from unittest.mock import MagicMock

        mock_tool = MagicMock()
        mock_tool.enabled = True
        mock_tool.to_gemini_format = MagicMock(return_value=MagicMock())

        tools = [mock_tool]
        result = GeminiToolAdapter.convert_tools(tools)

        # Returns list containing one Tool with function_declarations
        assert len(result) == 1

    def test_convert_tools_converts_multiple_tools(self):
        """convert_tools converts multiple tools."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter
        from unittest.mock import MagicMock

        tools = []
        for i in range(3):
            mock_tool = MagicMock()
            mock_tool.enabled = True
            mock_tool.to_gemini_format = MagicMock(return_value=MagicMock())
            tools.append(mock_tool)

        result = GeminiToolAdapter.convert_tools(tools)

        # All tools go into a single Tool object with multiple function_declarations
        assert len(result) == 1
        assert len(result[0].function_declarations) == 3

    def test_convert_tools_empty_list(self):
        """convert_tools handles empty list."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        result = GeminiToolAdapter.convert_tools([])

        assert result == []

    def test_convert_tools_with_parameters(self):
        """convert_tools includes parameters in converted tools."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter
        from unittest.mock import MagicMock

        mock_tool = MagicMock()
        mock_tool.enabled = True
        mock_tool.to_gemini_format = MagicMock(return_value=MagicMock())

        tools = [mock_tool]
        result = GeminiToolAdapter.convert_tools(tools)

        assert len(result) == 1


class TestGeminiToolAdapterExtractFunctionCalls:
    """Tests for GeminiToolAdapter.extract_function_calls."""

    def test_extract_function_calls_returns_list(self):
        """extract_function_calls returns a list."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        response = MockGeminiResponse(candidates=[])
        result = GeminiToolAdapter.extract_function_calls(response)

        assert isinstance(result, list)

    def test_extract_function_calls_extracts_single_call(self):
        """extract_function_calls extracts a single function call."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        response = MockGeminiResponse(
            candidates=[
                MockGeminiCandidate(
                    content=MockGeminiContent(
                        parts=[
                            MockGeminiPart(
                                function_call=MockGeminiFunctionCall(
                                    name="get_weather",
                                    args={"location": "Seoul"},
                                )
                            )
                        ]
                    )
                )
            ]
        )

        result = GeminiToolAdapter.extract_function_calls(response)

        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["parameters"] == {"location": "Seoul"}

    def test_extract_function_calls_extracts_multiple_calls(self):
        """extract_function_calls extracts multiple function calls."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        response = MockGeminiResponse(
            candidates=[
                MockGeminiCandidate(
                    content=MockGeminiContent(
                        parts=[
                            MockGeminiPart(function_call=MockGeminiFunctionCall("func1", {"a": 1})),
                            MockGeminiPart(function_call=MockGeminiFunctionCall("func2", {"b": 2})),
                        ]
                    )
                )
            ]
        )

        result = GeminiToolAdapter.extract_function_calls(response)

        assert len(result) == 2
        assert result[0]["name"] == "func1"
        assert result[1]["name"] == "func2"

    def test_extract_function_calls_empty_response(self):
        """extract_function_calls handles empty response."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        response = MockGeminiResponse(candidates=[])
        result = GeminiToolAdapter.extract_function_calls(response)

        assert result == []

    def test_extract_function_calls_no_function_call(self):
        """extract_function_calls handles parts without function calls."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        response = MockGeminiResponse(
            candidates=[
                MockGeminiCandidate(
                    content=MockGeminiContent(
                        parts=[MockGeminiPart(function_call=None)]
                    )
                )
            ]
        )

        result = GeminiToolAdapter.extract_function_calls(response)

        assert result == []

    def test_extract_function_calls_multiple_candidates(self):
        """extract_function_calls handles multiple candidates."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        response = MockGeminiResponse(
            candidates=[
                MockGeminiCandidate(
                    content=MockGeminiContent(
                        parts=[MockGeminiPart(function_call=MockGeminiFunctionCall("func_a", {}))]
                    )
                ),
                MockGeminiCandidate(
                    content=MockGeminiContent(
                        parts=[MockGeminiPart(function_call=MockGeminiFunctionCall("func_b", {}))]
                    )
                ),
            ]
        )

        result = GeminiToolAdapter.extract_function_calls(response)

        assert len(result) == 2


class TestGeminiToolAdapterFormatResults:
    """Tests for GeminiToolAdapter.format_results."""

    def test_format_results_returns_list(self):
        """format_results returns a list."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        results = [{"name": "test", "result": "ok"}]
        formatted = GeminiToolAdapter.format_results(results)

        assert isinstance(formatted, list)

    def test_format_results_single_result(self):
        """format_results formats a single result."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        results = [{"name": "get_weather", "result": "Sunny"}]
        formatted = GeminiToolAdapter.format_results(results)

        assert len(formatted) == 1

    def test_format_results_multiple_results(self):
        """format_results formats multiple results."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        results = [
            {"name": "func1", "result": "result1"},
            {"name": "func2", "result": "result2"},
        ]
        formatted = GeminiToolAdapter.format_results(results)

        assert len(formatted) == 2

    def test_format_results_with_dict_result(self):
        """format_results formats dict result as JSON."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        results = [{"name": "search", "result": {"items": [1, 2, 3]}}]
        formatted = GeminiToolAdapter.format_results(results)

        assert len(formatted) == 1

    def test_format_results_with_error(self):
        """format_results includes error message."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        results = [{"name": "failing_tool", "error": "Something went wrong"}]
        formatted = GeminiToolAdapter.format_results(results)

        assert len(formatted) == 1

    def test_format_results_with_none_result(self):
        """format_results handles None result."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        results = [{"name": "tool", "result": None}]
        formatted = GeminiToolAdapter.format_results(results)

        assert len(formatted) == 1


# ============================================================================
# OpenAIToolAdapter Tests
# ============================================================================

class TestOpenAIToolAdapterConvertTools:
    """Tests for OpenAIToolAdapter.convert_tools."""

    def test_convert_tools_returns_list(self):
        """convert_tools returns a list."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        tools = [
            MockToolDefinition(name="test_tool", description="A test tool", parameters=[])
        ]

        result = OpenAIToolAdapter.convert_tools(tools)

        assert isinstance(result, list)

    def test_convert_tools_converts_single_tool(self):
        """convert_tools converts a single tool to OpenAI format."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        tools = [
            MockToolDefinition(
                name="get_weather",
                description="Get weather info",
                parameters=[
                    MockToolParameter(name="location", type="string", description="City", required=True),
                ],
            )
        ]

        result = OpenAIToolAdapter.convert_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"

    def test_convert_tools_converts_multiple_tools(self):
        """convert_tools converts multiple tools."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        tools = [
            MockToolDefinition(name="tool1", description="First", parameters=[]),
            MockToolDefinition(name="tool2", description="Second", parameters=[]),
        ]

        result = OpenAIToolAdapter.convert_tools(tools)

        assert len(result) == 2

    def test_convert_tools_empty_list(self):
        """convert_tools handles empty list."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        result = OpenAIToolAdapter.convert_tools([])

        assert result == []

    def test_convert_tools_includes_parameters(self):
        """convert_tools includes parameter schema."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        tools = [
            MockToolDefinition(
                name="search",
                description="Search",
                parameters=[
                    MockToolParameter(name="query", type="string", description="Query", required=True),
                ],
            )
        ]

        result = OpenAIToolAdapter.convert_tools(tools)

        params = result[0]["function"]["parameters"]
        assert "properties" in params
        assert "query" in params["properties"]
        assert "required" in params
        assert "query" in params["required"]


class TestOpenAIToolAdapterExtractFunctionCalls:
    """Tests for OpenAIToolAdapter.extract_function_calls."""

    def test_extract_function_calls_returns_list(self):
        """extract_function_calls returns a list."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        response = MockOpenAIResponse(choices=[])
        result = OpenAIToolAdapter.extract_function_calls(response)

        assert isinstance(result, list)

    def test_extract_function_calls_extracts_single_call(self):
        """extract_function_calls extracts a single function call."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        response = MockOpenAIResponse(
            choices=[
                MockOpenAIChoice(
                    message=MockOpenAIMessage(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id="call_123",
                                function=MockOpenAIFunction(
                                    name="get_weather",
                                    arguments='{"location": "Seoul"}',
                                )
                            )
                        ]
                    )
                )
            ]
        )

        result = OpenAIToolAdapter.extract_function_calls(response)

        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["name"] == "get_weather"
        assert result[0]["parameters"] == {"location": "Seoul"}

    def test_extract_function_calls_extracts_multiple_calls(self):
        """extract_function_calls extracts multiple function calls."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        response = MockOpenAIResponse(
            choices=[
                MockOpenAIChoice(
                    message=MockOpenAIMessage(
                        tool_calls=[
                            MockOpenAIToolCall("call_1", MockOpenAIFunction("func1", '{"a": 1}')),
                            MockOpenAIToolCall("call_2", MockOpenAIFunction("func2", '{"b": 2}')),
                        ]
                    )
                )
            ]
        )

        result = OpenAIToolAdapter.extract_function_calls(response)

        assert len(result) == 2
        assert result[0]["name"] == "func1"
        assert result[1]["name"] == "func2"

    def test_extract_function_calls_empty_response(self):
        """extract_function_calls handles empty response."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        response = MockOpenAIResponse(choices=[])
        result = OpenAIToolAdapter.extract_function_calls(response)

        assert result == []

    def test_extract_function_calls_no_tool_calls(self):
        """extract_function_calls handles message without tool calls."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        response = MockOpenAIResponse(
            choices=[
                MockOpenAIChoice(message=MockOpenAIMessage(tool_calls=None))
            ]
        )

        result = OpenAIToolAdapter.extract_function_calls(response)

        assert result == []

    def test_extract_function_calls_empty_tool_calls(self):
        """extract_function_calls handles empty tool calls list."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        response = MockOpenAIResponse(
            choices=[
                MockOpenAIChoice(message=MockOpenAIMessage(tool_calls=[]))
            ]
        )

        result = OpenAIToolAdapter.extract_function_calls(response)

        assert result == []


class TestOpenAIToolAdapterFormatResults:
    """Tests for OpenAIToolAdapter.format_results."""

    def test_format_results_returns_list(self):
        """format_results returns a list."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        results = [{"id": "call_1", "name": "test", "result": "ok"}]
        formatted = OpenAIToolAdapter.format_results(results)

        assert isinstance(formatted, list)

    def test_format_results_single_result(self):
        """format_results formats a single result."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        results = [{"id": "call_123", "name": "get_weather", "result": "Sunny"}]
        formatted = OpenAIToolAdapter.format_results(results)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "tool"
        assert formatted[0]["tool_call_id"] == "call_123"
        assert formatted[0]["name"] == "get_weather"

    def test_format_results_multiple_results(self):
        """format_results formats multiple results."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        results = [
            {"id": "call_1", "name": "func1", "result": "result1"},
            {"id": "call_2", "name": "func2", "result": "result2"},
        ]
        formatted = OpenAIToolAdapter.format_results(results)

        assert len(formatted) == 2

    def test_format_results_with_dict_result(self):
        """format_results formats dict result as string."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        results = [{"id": "call_1", "name": "search", "result": {"items": [1, 2, 3]}}]
        formatted = OpenAIToolAdapter.format_results(results)

        # Dict is converted using str() which gives Python repr
        assert "items" in formatted[0]["content"]

    def test_format_results_with_error(self):
        """format_results includes error message."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "error": "Failed"}]
        formatted = OpenAIToolAdapter.format_results(results)

        assert formatted[0]["content"] == "Error: Failed"

    def test_format_results_with_none_result(self):
        """format_results handles None result."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "result": None}]
        formatted = OpenAIToolAdapter.format_results(results)

        assert formatted[0]["content"] == ""

    def test_format_results_with_string_result(self):
        """format_results formats string result."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "result": "Hello world"}]
        formatted = OpenAIToolAdapter.format_results(results)

        assert formatted[0]["content"] == "Hello world"

    def test_format_results_with_bytes_result(self):
        """format_results handles bytes as string representation."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "result": b"binary data"}]
        formatted = OpenAIToolAdapter.format_results(results)

        # Bytes are converted using str() which gives "b'binary data'"
        assert "binary" in formatted[0]["content"]


# ============================================================================
# ZAIToolAdapter Tests
# ============================================================================

class TestZAIToolAdapterInheritance:
    """Tests for ZAIToolAdapter inheritance from OpenAIStyleAdapter."""

    def test_inherits_from_openai_style_adapter(self):
        """ZAIToolAdapter inherits from OpenAIStyleAdapter."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter
        from persbot.tools.adapters.base_adapter import OpenAIStyleAdapter

        assert issubclass(ZAIToolAdapter, OpenAIStyleAdapter)

    def test_inherits_convert_tools(self):
        """ZAIToolAdapter inherits convert_tools from OpenAIStyleAdapter."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        tools = [MockToolDefinition(name="test", description="Test", parameters=[])]
        result = ZAIToolAdapter.convert_tools(tools)

        assert isinstance(result, list)
        assert len(result) == 1

    def test_inherits_extract_function_calls(self):
        """ZAIToolAdapter inherits extract_function_calls from OpenAIToolAdapter."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        response = MockOpenAIResponse(
            choices=[
                MockOpenAIChoice(
                    message=MockOpenAIMessage(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id="call_zai",
                                function=MockOpenAIFunction("test_func", '{"arg": "value"}')
                            )
                        ]
                    )
                )
            ]
        )

        result = ZAIToolAdapter.extract_function_calls(response)

        assert len(result) == 1
        assert result[0]["name"] == "test_func"


class TestZAIToolAdapterFormatResults:
    """Tests for ZAIToolAdapter.format_results (overridden with binary support)."""

    def test_format_results_returns_list(self):
        """format_results returns a list."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        results = [{"id": "call_1", "name": "test", "result": "ok"}]
        formatted = ZAIToolAdapter.format_results(results)

        assert isinstance(formatted, list)

    def test_format_results_with_dict_result(self):
        """format_results formats dict result as string."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "result": {"key": "value"}}]
        formatted = ZAIToolAdapter.format_results(results)

        # Dict is converted using str() which gives Python repr
        assert "key" in formatted[0]["content"]

    def test_format_results_with_bytes_result_keeps_binary(self):
        """format_results describes binary data with size (Z.AI specific)."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        binary_data = b"\x89PNG\r\n\x1a\n"  # PNG header
        results = [{"id": "call_1", "name": "generate_image", "result": binary_data}]
        formatted = ZAIToolAdapter.format_results(results)

        # Z.AI describes binary data as "Binary data (N bytes)"
        assert "Binary data" in formatted[0]["content"]
        assert "bytes" in formatted[0]["content"]

    def test_format_results_with_error(self):
        """format_results includes error message."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "error": "Z.AI error"}]
        formatted = ZAIToolAdapter.format_results(results)

        assert formatted[0]["content"] == "Error: Z.AI error"

    def test_format_results_with_none_result(self):
        """format_results handles None result."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "result": None}]
        formatted = ZAIToolAdapter.format_results(results)

        assert formatted[0]["content"] == ""

    def test_format_results_with_string_result(self):
        """format_results formats string result."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "result": "Z.AI response"}]
        formatted = ZAIToolAdapter.format_results(results)

        assert formatted[0]["content"] == "Z.AI response"

    def test_format_results_tool_role(self):
        """format_results sets role to 'tool'."""
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        results = [{"id": "call_1", "name": "tool", "result": "data"}]
        formatted = ZAIToolAdapter.format_results(results)

        assert formatted[0]["role"] == "tool"


# ============================================================================
# ToolAdapterRegistry Tests
# ============================================================================

class TestToolAdapterRegistry:
    """Tests for ToolAdapterRegistry."""

    def test_register_and_get_adapter(self):
        """ToolAdapterRegistry can register and retrieve adapters."""
        from persbot.tools.adapters.base_adapter import ToolAdapterRegistry
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        # Clear any existing registration
        ToolAdapterRegistry._adapters.pop("test_provider", None)

        adapter = OpenAIToolAdapter()
        ToolAdapterRegistry.register("test_provider", adapter)

        retrieved = ToolAdapterRegistry.get("test_provider")
        assert retrieved is adapter

        # Cleanup
        ToolAdapterRegistry._adapters.pop("test_provider", None)

    def test_get_returns_none_for_unknown_provider(self):
        """ToolAdapterRegistry.get returns None for unknown provider."""
        from persbot.tools.adapters.base_adapter import ToolAdapterRegistry

        result = ToolAdapterRegistry.get("unknown_provider_xyz")
        assert result is None

    def test_get_is_case_insensitive(self):
        """ToolAdapterRegistry.get is case-insensitive."""
        from persbot.tools.adapters.base_adapter import ToolAdapterRegistry
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        # Clear any existing registration
        ToolAdapterRegistry._adapters.pop("casetest", None)

        adapter = OpenAIToolAdapter()
        ToolAdapterRegistry.register("CaseTest", adapter)

        assert ToolAdapterRegistry.get("casetest") is adapter
        assert ToolAdapterRegistry.get("CASETEST") is adapter
        assert ToolAdapterRegistry.get("CaSeTeSt") is adapter

        # Cleanup
        ToolAdapterRegistry._adapters.pop("casetest", None)

    def test_get_or_create_returns_existing(self):
        """ToolAdapterRegistry.get_or_create returns existing adapter."""
        from persbot.tools.adapters.base_adapter import ToolAdapterRegistry
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        # Clear any existing registration
        ToolAdapterRegistry._adapters.pop("openai", None)

        adapter = OpenAIToolAdapter()
        ToolAdapterRegistry.register("openai", adapter)

        retrieved = ToolAdapterRegistry.get_or_create("openai")
        assert retrieved is adapter

    def test_get_or_create_creates_gemini_adapter(self):
        """ToolAdapterRegistry.get_or_create creates GeminiToolAdapter for 'gemini'."""
        from persbot.tools.adapters.base_adapter import ToolAdapterRegistry
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

        # Clear any existing registration
        ToolAdapterRegistry._adapters.pop("gemini", None)

        adapter = ToolAdapterRegistry.get_or_create("gemini")

        assert isinstance(adapter, GeminiToolAdapter)

        # Cleanup
        ToolAdapterRegistry._adapters.pop("gemini", None)

    def test_get_or_create_creates_openai_adapter(self):
        """ToolAdapterRegistry.get_or_create creates OpenAIToolAdapter for 'openai'."""
        from persbot.tools.adapters.base_adapter import ToolAdapterRegistry
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter

        # Clear any existing registration
        ToolAdapterRegistry._adapters.pop("openai", None)

        adapter = ToolAdapterRegistry.get_or_create("openai")

        assert isinstance(adapter, OpenAIToolAdapter)

        # Cleanup
        ToolAdapterRegistry._adapters.pop("openai", None)

    def test_get_or_create_creates_zai_adapter(self):
        """ToolAdapterRegistry.get_or_create creates ZAIToolAdapter for 'zai'."""
        from persbot.tools.adapters.base_adapter import ToolAdapterRegistry
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        # Clear any existing registration
        ToolAdapterRegistry._adapters.pop("zai", None)

        adapter = ToolAdapterRegistry.get_or_create("zai")

        assert isinstance(adapter, ZAIToolAdapter)

        # Cleanup
        ToolAdapterRegistry._adapters.pop("zai", None)

    def test_get_or_create_raises_for_unknown_provider(self):
        """ToolAdapterRegistry.get_or_create raises ValueError for unknown provider."""
        from persbot.tools.adapters.base_adapter import ToolAdapterRegistry

        with pytest.raises(ValueError) as exc_info:
            ToolAdapterRegistry.get_or_create("unknown_provider")

        assert "Unknown provider" in str(exc_info.value)


class TestGetToolAdapter:
    """Tests for get_tool_adapter convenience function."""

    def test_get_tool_adapter_returns_adapter(self):
        """get_tool_adapter returns an adapter."""
        from persbot.tools.adapters.base_adapter import get_tool_adapter

        adapter = get_tool_adapter("openai")

        assert adapter is not None

    def test_get_tool_adapter_caches_adapter(self):
        """get_tool_adapter caches the adapter."""
        from persbot.tools.adapters.base_adapter import get_tool_adapter, ToolAdapterRegistry

        # Clear cache
        ToolAdapterRegistry._adapters.pop("openai", None)

        adapter1 = get_tool_adapter("openai")
        adapter2 = get_tool_adapter("openai")

        assert adapter1 is adapter2


# ============================================================================
# Comparison Tests
# ============================================================================

class TestAdapterComparison:
    """Tests comparing different adapters."""

    def test_openai_vs_zai_binary_handling(self):
        """OpenAI and ZAI adapters handle binary data differently."""
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        binary_data = b"\x00\x01\x02\x03"
        results = [{"id": "call_1", "name": "tool", "result": binary_data}]

        openai_formatted = OpenAIToolAdapter.format_results(results)
        zai_formatted = ZAIToolAdapter.format_results(results)

        # OpenAI converts to string representation (Python bytes repr)
        assert openai_formatted[0]["content"].startswith("b'")

        # Z.AI describes binary with size info
        assert "Binary data" in zai_formatted[0]["content"]
        assert "4 bytes" in zai_formatted[0]["content"]

    def test_all_adapters_format_results_with_error(self):
        """All adapters handle error results."""
        from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter
        from persbot.tools.adapters.openai_adapter import OpenAIToolAdapter
        from persbot.tools.adapters.zai_adapter import ZAIToolAdapter

        error_results = [{"id": "call_1", "name": "tool", "error": "Test error"}]

        # All should handle errors without exception
        gemini_result = GeminiToolAdapter.format_results(error_results)
        openai_result = OpenAIToolAdapter.format_results(error_results)
        zai_result = ZAIToolAdapter.format_results(error_results)

        assert len(gemini_result) == 1
        assert len(openai_result) == 1
        assert len(zai_result) == 1

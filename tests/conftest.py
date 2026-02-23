"""Pytest configuration and fixtures for persbot tests."""

import sys
from unittest.mock import MagicMock

# Mock google.genai.types and openai modules BEFORE any imports that might use them
# This needs to be done at module import time
def setup_mocks():
    """Setup external dependency mocks for all tests."""

    # Only setup once
    if 'google.genai' in sys.modules and 'openai' in sys.modules:
        return

    # ========== Mock google.genai.types ==========
    if 'google.genai' not in sys.modules:
        # Create mock types module
        mock_types = MagicMock()

        # Create mock classes for genai types
        class MockGenerateContentConfig:
            def __init__(self, **kwargs):
                self.temperature = kwargs.get('temperature', 1.0)
                self.top_p = kwargs.get('top_p', 1.0)
                self.system_instruction = kwargs.get('system_instruction')
                self.cached_content = kwargs.get('cached_content')
                self.thinking_config = kwargs.get('thinking_config')
                self.tools = kwargs.get('tools')

        class MockThinkingConfig:
            def __init__(self, thinking_budget=0):
                self.thinking_budget = thinking_budget

        class MockContent:
            def __init__(self, role=None, parts=None):
                self.role = role
                self.parts = parts or []

            @classmethod
            def model_validate(cls, data):
                """Create a Content from a dict (mimics pydantic model_validate)."""
                parts = []
                for part_data in data.get('parts', []):
                    part = MockPart()
                    if 'text' in part_data:
                        part.text = part_data['text']
                    if 'function_call' in part_data:
                        fc = part_data['function_call']
                        part.function_call = MockFunctionCall(name=fc.get('name'), args=fc.get('args'))
                    if 'function_response' in part_data:
                        fr = part_data['function_response']
                        part.function_response = MockFunctionResponse(name=fr.get('name'), response=fr.get('response'))
                    if 'inline_data' in part_data:
                        id_data = part_data['inline_data']
                        part.inline_data = MockBlob(mime_type=id_data.get('mime_type'), data=id_data.get('data'))
                    if 'thought_signature' in part_data:
                        part.thought_signature = part_data['thought_signature']
                    parts.append(part)
                return cls(role=data.get('role'), parts=parts)

        class MockFunctionCall:
            def __init__(self, name=None, args=None):
                self.name = name
                self.args = args or {}

        class MockFunctionResponse:
            def __init__(self, name=None, response=None):
                self.name = name
                self.response = response or {}

        class MockPart:
            def __init__(self, text=None, function_call=None, function_response=None, inline_data=None, thought=None, thought_signature=None):
                self.text = text
                self.function_call = function_call
                self.function_response = function_response
                self.inline_data = inline_data
                self.thought = thought
                self.thought_signature = thought_signature

            @classmethod
            def from_function_call(cls, name, args):
                """Create a Part with function_call."""
                return cls(function_call=MockFunctionCall(name=name, args=args))

            @classmethod
            def from_function_response(cls, name, response):
                """Create a Part with function_response."""
                return cls(function_response=MockFunctionResponse(name=name, response=response))

        class MockBlob:
            def __init__(self, mime_type=None, data=None):
                self.mime_type = mime_type
                self.data = data

        mock_types.GenerateContentConfig = MockGenerateContentConfig
        mock_types.ThinkingConfig = MockThinkingConfig
        mock_types.Content = MockContent
        mock_types.Part = MockPart
        mock_types.Blob = MockBlob

        # Mock the module
        mock_genai = MagicMock()
        mock_genai.types = mock_types

        # Mock errors module
        mock_errors = MagicMock()

        class MockClientError(Exception):
            def __init__(self, code=None, response_json=None):
                self.code = code
                self.response_json = response_json or {}
                # Extract message from response_json if present
                self.message = ""
                if self.response_json and "error" in self.response_json:
                    self.message = self.response_json["error"].get("message", "")
                super().__init__(f"ClientError: {code}")

        mock_errors.ClientError = MockClientError

        sys.modules['google'] = mock_genai
        sys.modules['google.genai'] = mock_genai
        sys.modules['google.genai.types'] = mock_types
        sys.modules['google.genai.errors'] = mock_errors

    # ========== Mock openai ==========
    if 'openai' not in sys.modules:
        mock_openai = MagicMock()

        # Create OpenAI class mock
        mock_openai_class = MagicMock()
        mock_openai.OpenAI = mock_openai_class

        # Mock errors if needed
        mock_openai.APIError = type('APIError', (Exception,), {})
        mock_openai.APIStatusError = type('APIStatusError', (Exception,), {})

        class MockRateLimitError(Exception):
            def __init__(self, message=None, response=None, body=None):
                self.message = message
                self.response = response
                self.body = body
                super().__init__(message or "Rate limit error")

        mock_openai.RateLimitError = MockRateLimitError
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})

        sys.modules['openai'] = mock_openai

    # ========== Mock ddgs (duckduckgo-search) ==========
    if 'ddgs' not in sys.modules:
        # Create mock module structure for ddgs package
        mock_ddgs = MagicMock()

        # Create mock exceptions sub-module
        mock_exceptions = MagicMock()
        mock_exceptions.RatelimitException = type('RatelimitException', (Exception,), {})
        mock_exceptions.DDGSException = type('DDGSException', (Exception,), {})
        mock_ddgs.exceptions = mock_exceptions

        # Mock DDGS class
        mock_ddgs.DDGS = MagicMock

        sys.modules['ddgs'] = mock_ddgs
        sys.modules['ddgs.exceptions'] = mock_exceptions

    # ========== Mock bs4 (beautifulsoup4) ==========
    if 'bs4' not in sys.modules:
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup = MagicMock
        sys.modules['bs4'] = mock_bs4


# Setup the mocks immediately when conftest is loaded
setup_mocks()

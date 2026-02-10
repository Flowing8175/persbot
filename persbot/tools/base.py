"""Base tool definitions for SoyeBot AI tool system."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import inspect

try:
    from google.genai import types as genai_types

    HAS_GENAI = True
except ImportError:
    genai_types = None
    HAS_GENAI = False


class ToolCategory(Enum):
    """Categories of tools for organization and filtering."""

    DISCORD_CHANNEL = "discord_channel"
    DISCORD_USER = "discord_user"
    DISCORD_GUILD = "discord_guild"
    API_SEARCH = "api_search"
    API_WEATHER = "api_weather"
    API_TIME = "api_time"
    PERSONA_MEMORY = "persona_memory"
    PERSONA_MEDIA = "persona_media"
    PERSONA_ROUTINE = "persona_routine"
    PERSONA_WEB = "persona_web"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    """Parameter name."""

    type: str
    """Parameter type (string, integer, boolean, etc.)"""

    description: str
    """Parameter description."""

    required: bool = False
    """Whether this parameter is required."""

    default: Optional[Any] = None
    """Default value for optional parameters."""

    enum: Optional[List[Any]] = None
    """List of allowed values for enum parameters."""

    items_type: Optional[str] = None
    """Type of items for array parameters (e.g., 'string', 'integer')."""


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    """Whether the tool execution was successful."""

    data: Optional[Any] = None
    """The result data from the tool execution."""

    error: Optional[str] = None
    """Error message if execution failed."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the execution."""


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the AI."""

    name: str
    """Unique name for the tool."""

    description: str
    """Description of what the tool does."""

    category: ToolCategory
    """Category the tool belongs to."""

    parameters: List[ToolParameter]
    """List of parameters the tool accepts."""

    handler: Callable
    """Async function that executes the tool."""

    requires_permission: Optional[str] = None
    """Discord permission required to use this tool (e.g., 'read_messages')."""

    rate_limit: Optional[int] = None
    """Rate limit in seconds between uses (None = no limit)."""

    enabled: bool = True
    """Whether the tool is currently enabled."""

    timeout: Optional[float] = None
    """Timeout in seconds for tool execution. None means use global config timeout."""

    async def execute(self, cancel_event: Optional[asyncio.Event] = None, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            cancel_event: AsyncIO event to check for cancellation before execution.
            **kwargs: Parameter values to pass to the tool handler.

        Returns:
            ToolResult containing the execution result.
        """
        try:
            # Check for cancellation before handler execution
            if cancel_event and cancel_event.is_set():
                return ToolResult(
                    success=False,
                    error=f"Tool '{self.name}' execution aborted by user",
                )

            # Pass cancel_event to handler if it accepts it
            sig = inspect.signature(self.handler)
            if "cancel_event" in sig.parameters:
                result = await self.handler(cancel_event=cancel_event, **kwargs)
            else:
                result = await self.handler(**kwargs)

            if isinstance(result, ToolResult):
                return result
            return ToolResult(success=True, data=result)
        except asyncio.CancelledError:
            # Re-raise cancellation errors to propagate abort signal
            raise
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool definition to OpenAI function calling format.

        Returns:
            Dictionary in OpenAI function format.
        """
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

    def to_gemini_format(self) -> Any:
        """Convert tool definition to Gemini function declaration format.

        Returns:
            genai_types.FunctionDeclaration object.
        """
        if not HAS_GENAI or genai_types is None:
            raise ImportError(
                "google.genai is not installed. Please install it to use Gemini format."
            )

        properties = {}
        required = []

        for param in self.parameters:
            prop_def = genai_types.Schema(
                type=param.type,
                description=param.description,
            )
            if param.enum:
                prop_def.enum = param.enum
            if param.default is not None:
                prop_def.default = param.default
            if param.type == "array" and param.items_type:
                prop_def.items = genai_types.Schema(type=param.items_type)

            properties[param.name] = prop_def
            if param.required:
                required.append(param.name)

        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type="object",
                properties=properties,
                required=required,
            ),
        )

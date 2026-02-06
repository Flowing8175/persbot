"""Base tool definitions for SoyeBot AI tool system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, List, Dict


class ToolCategory(Enum):
    """Categories of tools for organization and filtering."""

    DISCORD_CHANNEL = "discord_channel"
    DISCORD_USER = "discord_user"
    DISCORD_GUILD = "discord_guild"
    API_SEARCH = "api_search"
    API_WEATHER = "api_weather"
    API_TIME = "api_time"


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

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Parameter values to pass to the tool handler.

        Returns:
            ToolResult containing the execution result.
        """
        try:
            result = await self.handler(**kwargs)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(success=True, data=result)
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
        from google.genai import types as genai_types

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

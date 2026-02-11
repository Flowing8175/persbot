"""Model-related value objects."""

from dataclasses import dataclass
from enum import Enum
from typing import Union


class Provider(str, Enum):
    """LLM provider identifiers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ZAI = "zai"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "Provider":
        """Create Provider from string, case-insensitive."""
        normalized = value.lower().strip()
        for provider in cls:
            if provider.value == normalized:
                return provider
        raise ValueError(f"Unknown provider: {value}")


@dataclass(frozen=True, slots=True)
class ModelAlias:
    """A strongly-typed model alias identifier.

    Model aliases are user-friendly names for LLM models.
    Examples: "Gemini 2.5 Flash", "GPT-4o", "GLM 4.7"
    """

    value: str

    def __init__(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"ModelAlias must be a string, got {type(value).__name__}")
        str_value = value.strip()
        if not str_value:
            raise ValueError("ModelAlias cannot be empty")
        object.__setattr__(self, "value", str_value)

    def __str__(self) -> str:
        return self.value

    @property
    def provider(self) -> Provider:
        """Determine the provider for this model alias."""
        value_lower = self.value.lower()
        if "gemini" in value_lower:
            return Provider.GEMINI
        if "gpt" in value_lower or "openai" in value_lower:
            return Provider.OPENAI
        if "glm" in value_lower or "zai" in value_lower:
            return Provider.ZAI
        # Default to Gemini for unknown models
        return Provider.GEMINI

    @classmethod
    def from_raw(cls, value: Union[str, "ModelAlias"]) -> "ModelAlias":
        """Create ModelAlias from various input types."""
        if isinstance(value, ModelAlias):
            return value
        return cls(value)


# Standard model aliases
class StandardModels:
    """Standard model alias constants."""

    # Gemini models
    GEMINI_FLASH = ModelAlias("Gemini 2.5 Flash")
    GEMINI_PRO = ModelAlias("Gemini 2.5 Pro")

    # OpenAI models
    GPT_4O = ModelAlias("GPT-4o")
    GPT_4O_MINI = ModelAlias("GPT-4o Mini")
    GPT_5_MINI = ModelAlias("GPT-5 Mini")

    # ZAI models
    GLM_4_7 = ModelAlias("GLM 4.7")
    GLM_4_FLASH = ModelAlias("GLM 4 Flash")
    GLM_4_6V = ModelAlias("GLM 4.6V")

    DEFAULT = GEMINI_FLASH

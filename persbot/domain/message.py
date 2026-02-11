"""Message value objects."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, slots=True)
class MessageId:
    """A strongly-typed Discord message identifier.

    Discord message IDs are snowflakes, which are large integers.
    This value object provides type safety and validation.
    """

    value: str

    def __init__(self, value: Union[int, str]):
        # Convert to string for consistency (Discord.py returns strings)
        str_value = str(value)

        if not str_value.isdigit():
            raise ValueError(f"MessageId must be numeric, got {str_value}")

        object.__setattr__(self, "value", str_value)

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_raw(cls, value: Union[int, str, "MessageId"]) -> "MessageId":
        """Create MessageId from various input types."""
        if isinstance(value, MessageId):
            return value
        return cls(value)

    def to_int(self) -> int:
        """Convert message ID to integer."""
        return int(self.value)

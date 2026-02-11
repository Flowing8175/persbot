"""Channel value objects."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, slots=True)
class ChannelId:
    """A strongly-typed Discord channel identifier.

    This value object prevents mixing up channel IDs with other integer IDs
    and provides validation at creation time.
    """

    value: int

    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"ChannelId must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"ChannelId must be positive, got {value}")
        object.__setattr__(self, "value", value)

    def __str__(self) -> str:
        return str(self.value)

    def __int__(self) -> int:
        return self.value

    @classmethod
    def from_raw(cls, value: Union[int, str, "ChannelId"]) -> "ChannelId":
        """Create ChannelId from various input types."""
        if isinstance(value, ChannelId):
            return value
        if isinstance(value, str):
            return cls(int(value))
        return cls(value)

    def to_session_key(self) -> str:
        """Convert to session key format."""
        return f"channel:{self.value}"


@dataclass(frozen=True, slots=True)
class ThreadId(ChannelId):
    """A strongly-typed Discord thread identifier.

    Inherits from ChannelId as threads are a type of channel in Discord.
    """

    def to_session_key(self) -> str:
        """Convert to session key format for threads."""
        return f"thread:{self.value}"

"""Session-related value objects."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, slots=True)
class SessionKey:
    """A strongly-typed session key identifier.

    Session keys are used to identify unique chat sessions.
    Format: "{type}:{id}" e.g., "channel:123456789"
    """

    value: str

    def __init__(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"SessionKey must be a string, got {type(value).__name__}")
        str_value = value.strip()
        if not str_value:
            raise ValueError("SessionKey cannot be empty")
        if ":" not in str_value:
            raise ValueError(f"SessionKey must contain ':', got {str_value}")
        object.__setattr__(self, "value", str_value)

    def __str__(self) -> str:
        return self.value

    @property
    def type(self) -> str:
        """Get the session type (prefix before colon)."""
        return self.value.split(":", 1)[0]

    @property
    def id(self) -> str:
        """Get the session ID (suffix after colon)."""
        return self.value.split(":", 1)[1]

    @classmethod
    def from_channel(cls, channel_id: int) -> "SessionKey":
        """Create a channel-based session key."""
        return cls(f"channel:{channel_id}")

    @classmethod
    def from_user(cls, user_id: int) -> "SessionKey":
        """Create a user-based session key."""
        return cls(f"user:{user_id}")

    @classmethod
    def from_thread(cls, thread_id: int) -> "SessionKey":
        """Create a thread-based session key."""
        return cls(f"thread:{thread_id}")

    @classmethod
    def from_raw(cls, value: Union[str, "SessionKey"]) -> "SessionKey":
        """Create SessionKey from various input types."""
        if isinstance(value, SessionKey):
            return value
        return cls(value)

    def is_channel_session(self) -> bool:
        """Check if this is a channel-based session."""
        return self.type == "channel"

    def is_user_session(self) -> bool:
        """Check if this is a user-based session."""
        return self.type == "user"

    def is_thread_session(self) -> bool:
        """Check if this is a thread-based session."""
        return self.type == "thread"

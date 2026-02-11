"""User-related value objects."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, slots=True)
class UserId:
    """A strongly-typed Discord user identifier.

    This value object prevents mixing up user IDs with other integer IDs
    and provides validation at creation time.
    """

    value: int

    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"UserId must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"UserId must be positive, got {value}")
        object.__setattr__(self, "value", value)

    def __str__(self) -> str:
        return str(self.value)

    def __int__(self) -> int:
        return self.value

    @classmethod
    def from_raw(cls, value: Union[int, str, "UserId"]) -> "UserId":
        """Create UserId from various input types."""
        if isinstance(value, UserId):
            return value
        if isinstance(value, str):
            return cls(int(value))
        return cls(value)

    def to_session_key(self) -> str:
        """Convert to session key format."""
        return f"user:{self.value}"


@dataclass(frozen=True, slots=True)
class GuildId:
    """A strongly-typed Discord guild (server) identifier.

    This value object prevents mixing up guild IDs with other integer IDs.
    In DM contexts, the guild ID may be the same as the user ID.
    """

    value: int

    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"GuildId must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"GuildId must be positive, got {value}")
        object.__setattr__(self, "value", value)

    def __str__(self) -> str:
        return str(self.value)

    def __int__(self) -> int:
        return self.value

    @classmethod
    def from_raw(cls, value: Union[int, str, "GuildId"]) -> "GuildId":
        """Create GuildId from various input types."""
        if isinstance(value, GuildId):
            return value
        if isinstance(value, str):
            return cls(int(value))
        return cls(value)

    @classmethod
    def from_user_id(cls, user_id: UserId) -> "GuildId":
        """Create GuildId from UserId (for DM contexts)."""
        return cls(user_id.value)

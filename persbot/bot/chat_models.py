"""Chat models for Discord bot interactions.

This module contains data models for chat-related operations.
"""

from dataclasses import dataclass, field
from typing import Any, List

from persbot.constants import ToolLabels


@dataclass(frozen=True)
class ChatReply:
    """Container for an LLM response tied to a session.

    Attributes:
        text: The response text from the LLM.
        session_key: The session key for this response.
        response: The raw response object from the LLM.
        images: List of generated image bytes.
        notification: Optional notification message to prepend.
        tool_rounds: Number of tool execution rounds.
    """

    text: str
    session_key: str
    response: Any
    images: List[bytes] = field(default_factory=list)
    notification: str = ""
    tool_rounds: int = 0

    @property
    def display_text(self) -> str:
        """Get the text with notification prepended if exists."""
        if self.notification:
            return f"📢 {self.notification}\n\n{self.text}"
        return self.text

    @property
    def has_tools(self) -> bool:
        """Check if tools were executed."""
        return self.tool_rounds > 0

    @property
    def has_images(self) -> bool:
        """Check if images were generated."""
        return len(self.images) > 0


@dataclass
class ToolProgress:
    """Progress notification for tool execution.

    Attributes:
        tool_names: List of tool names being executed.
        korean_names: Korean display names for the tools.
    """

    tool_names: List[str] = field(default_factory=list)

    @property
    def korean_names(self) -> List[str]:
        """Get Korean names for the tools."""
        return [ToolLabels.__dict__.get(name.upper(), name) for name in self.tool_names]

    @property
    def notification_text(self) -> str:
        """Get the notification text."""
        if not self.korean_names:
            return "🔧 도구 실행 중..."
        return f"🔧 {', '.join(self.korean_names)} 사용 중..."


@dataclass
class SessionContext:
    """Context for a chat session.

    Attributes:
        channel_id: The Discord channel ID.
        user_id: The user's Discord ID.
        username: The user's username.
        message_id: The Discord message ID.
        created_at: When the message was created.
        is_reply_to_summary: Whether this is a reply to a summary.
    """

    channel_id: int
    user_id: int
    username: str
    message_id: str
    created_at: Any
    is_reply_to_summary: bool = False
    reference_message_id: str = None

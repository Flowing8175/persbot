"""Context summarization service for reducing token costs in long conversations."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from persbot.services.base import ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for context summarization."""

    # Number of messages before summarization is triggered
    threshold: int = 40
    # Number of recent messages to keep unsummarized
    keep_recent_messages: int = 7
    # Model to use for summarization (cheaper model)
    summarization_model: str = "gemini-2.5-flash"
    # Maximum tokens for the summary
    max_summary_length: int = 500


class SummarizationService:
    """Service for summarizing conversation history to reduce context costs.

    When a conversation exceeds the message threshold, older messages are
    summarized into a condensed form while recent messages are kept intact
    for contextual relevance.
    """

    def __init__(self, config: Optional[SummarizationConfig] = None):
        self.config = config or SummarizationConfig()
        self._summarization_count: int = 0
        self._tokens_saved: int = 0

    def should_summarize(self, history: List[ChatMessage]) -> bool:
        """Check if history exceeds the summarization threshold.

        Args:
            history: List of chat messages.

        Returns:
            True if summarization should be triggered.
        """
        return len(history) >= self.config.threshold

    def get_messages_to_summarize(
        self, history: List[ChatMessage]
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        """Split history into messages to summarize and messages to keep.

        Args:
            history: Full conversation history (may be a deque or list).

        Returns:
            Tuple of (messages_to_summarize, messages_to_keep).
        """
        if len(history) <= self.config.keep_recent_messages:
            return [], list(history)

        # Convert to list to support slicing (history may be a deque)
        history_list = list(history)

        # Keep the most recent messages intact
        keep_count = self.config.keep_recent_messages
        messages_to_summarize = history_list[:-keep_count]
        messages_to_keep = history_list[-keep_count:]

        return messages_to_summarize, messages_to_keep

    def format_history_for_summary(self, messages: List[ChatMessage]) -> str:
        """Format messages into a string suitable for summarization.

        Args:
            messages: Messages to format.

        Returns:
            Formatted string of conversation.
        """
        lines = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content or ""
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def create_summary_prompt(self, formatted_history: str) -> str:
        """Create the prompt for generating a summary.

        Args:
            formatted_history: Formatted conversation history.

        Returns:
            Prompt string for the LLM.
        """
        return f"""Summarize the following Discord conversation concisely.
Keep key topics, decisions, and important context.
The summary should be {self.config.max_summary_length} tokens or less.

Conversation:
{formatted_history}

Summary:"""

    def create_summary_message(self, summary_text: str) -> ChatMessage:
        """Create a ChatMessage containing the summary.

        Args:
            summary_text: The generated summary text.

        Returns:
            ChatMessage with the summary.
        """
        return ChatMessage(
            role="user",
            content=f"[Previous conversation summary: {summary_text}]",
            parts=[{"text": f"[Previous conversation summary: {summary_text}]"}],
        )

    def record_summarization(self, messages_summarized: int, estimated_tokens_saved: int) -> None:
        """Record metrics from a summarization operation.

        Args:
            messages_summarized: Number of messages that were summarized.
            estimated_tokens_saved: Estimated tokens saved by summarization.
        """
        self._summarization_count += 1
        self._tokens_saved += estimated_tokens_saved
        logger.info(
            "Summarization #%d: %d messages summarized, ~%d tokens saved",
            self._summarization_count,
            messages_summarized,
            estimated_tokens_saved,
        )

    def get_stats(self) -> dict:
        """Get summarization statistics.

        Returns:
            Dict with summarization metrics.
        """
        return {
            "summarization_count": self._summarization_count,
            "tokens_saved": self._tokens_saved,
            "threshold": self.config.threshold,
            "keep_recent": self.config.keep_recent_messages,
        }

    def estimate_tokens(self, messages: List[ChatMessage]) -> int:
        """Estimate token count for messages (rough approximation).

        Uses a simple heuristic: ~4 characters per token.

        Args:
            messages: Messages to estimate.

        Returns:
            Estimated token count.
        """
        total_chars = sum(len(msg.content or "") for msg in messages)
        return total_chars // 4

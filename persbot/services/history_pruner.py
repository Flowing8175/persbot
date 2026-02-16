"""Semantic history pruning for managing LLM context window usage.

This module provides intelligent history pruning to prevent context window overflow
while preserving important conversation context.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Standard message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


@dataclass
class PruningConfig:
    """Configuration for history pruning."""

    max_tokens: int = 100000  # Target max tokens after pruning
    keep_first_n: int = 3  # Always keep first N messages (system)
    keep_last_n: int = 6  # Always keep last N messages (recent context)
    recency_bonus_weight: float = 0.3  # Weight for recency in scoring
    tool_call_bonus: float = 0.5  # Bonus for messages with tool calls
    min_tokens_per_message: int = 10  # Minimum estimated tokens per message


@dataclass
class ScoredMessage:
    """A message with its pruning score."""

    index: int
    message: Any
    score: float
    estimated_tokens: int
    has_tool_calls: bool = False
    role: str = ""


class HistoryPruner:
    """Intelligent history pruning with semantic importance scoring.

    Features:
    - Role-based importance scoring (system > tool > user > assistant)
    - Recency bonus for newer messages
    - Always keeps first N (system) and last N (recent) messages
    - Tool call detection (higher importance)
    - Token estimation for proactive pruning

    Usage:
        pruner = HistoryPruner(config)

        # Estimate token count
        total_tokens = pruner.estimate_tokens(history)

        # Prune if needed
        if total_tokens > config.max_tokens:
            pruned_history = pruner.prune(history, target_tokens=80000)
    """

    # Role importance weights (higher = more important to keep)
    ROLE_WEIGHTS: Dict[str, float] = {
        MessageRole.SYSTEM: 1.0,
        MessageRole.TOOL: 0.8,
        MessageRole.FUNCTION: 0.8,
        MessageRole.USER: 0.6,
        MessageRole.ASSISTANT: 0.4,
    }

    # Default weight for unknown roles
    DEFAULT_ROLE_WEIGHT = 0.5

    def __init__(self, config: Optional[PruningConfig] = None):
        """Initialize the history pruner.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or PruningConfig()

    def estimate_tokens(self, messages: List[Any]) -> int:
        """Estimate total tokens in the message history.

        Uses rough estimation: 1 token ~= 4 characters.

        Args:
            messages: List of message objects.

        Returns:
            Estimated total token count.
        """
        total = 0
        for msg in messages:
            total += self._estimate_message_tokens(msg)
        return total

    def _estimate_message_tokens(self, message: Any) -> int:
        """Estimate tokens for a single message.

        Args:
            message: A message object.

        Returns:
            Estimated token count.
        """
        text = self._extract_message_text(message)
        # Rough estimation: 1 token ~= 4 characters
        char_count = len(text)
        tokens = max(char_count // 4, self.config.min_tokens_per_message)

        # Add overhead for role and metadata
        tokens += 4

        return tokens

    def _extract_message_text(self, message: Any) -> str:
        """Extract text content from a message.

        Args:
            message: A message object (various formats).

        Returns:
            Extracted text content.
        """
        # Handle dict-style messages
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Multi-part content
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        if "text" in part:
                            parts.append(part["text"])
                return " ".join(parts)
            return str(content) if content else ""

        # Handle object-style messages
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
                return " ".join(parts)
            return str(content) if content else ""

        # Handle parts attribute (Gemini-style)
        if hasattr(message, "parts"):
            parts = message.parts
            if isinstance(parts, list):
                text_parts = []
                for part in parts:
                    if hasattr(part, "text"):
                        text_parts.append(part.text)
                    elif isinstance(part, str):
                        text_parts.append(part)
                return " ".join(text_parts)

        return str(message) if message else ""

    def _get_message_role(self, message: Any) -> str:
        """Get the role of a message.

        Args:
            message: A message object.

        Returns:
            The message role string.
        """
        if isinstance(message, dict):
            return message.get("role", "").lower()
        if hasattr(message, "role"):
            return str(message.role).lower()
        return ""

    def _has_tool_calls(self, message: Any) -> bool:
        """Check if a message contains tool/function calls.

        Args:
            message: A message object.

        Returns:
            True if message has tool calls.
        """
        # Check dict-style
        if isinstance(message, dict):
            if "tool_calls" in message:
                return bool(message["tool_calls"])
            if "function_call" in message:
                return bool(message["function_call"])

        # Check object-style
        if hasattr(message, "tool_calls") and message.tool_calls:
            return True
        if hasattr(message, "function_call") and message.function_call:
            return True

        # Check for tool call response
        role = self._get_message_role(message)
        if role in (MessageRole.TOOL, MessageRole.FUNCTION):
            return True

        return False

    def _calculate_importance_score(
        self,
        message: Any,
        index: int,
        total_messages: int,
    ) -> float:
        """Calculate importance score for a message.

        Higher scores = more important to keep.

        Args:
            message: The message to score.
            index: Index of the message in the history.
            total_messages: Total number of messages.

        Returns:
            Importance score (0.0 to 1.0+).
        """
        # Base score from role weight
        role = self._get_message_role(message)
        role_weight = self.ROLE_WEIGHTS.get(role, self.DEFAULT_ROLE_WEIGHT)

        # Recency bonus (newer messages are more important)
        if total_messages > 1:
            recency_factor = index / (total_messages - 1)  # 0.0 to 1.0
            recency_bonus = recency_factor * self.config.recency_bonus_weight
        else:
            recency_bonus = 0

        # Tool call bonus
        tool_bonus = self.config.tool_call_bonus if self._has_tool_calls(message) else 0

        return role_weight + recency_bonus + tool_bonus

    def prune(
        self,
        messages: List[Any],
        target_tokens: Optional[int] = None,
    ) -> Tuple[List[Any], int]:
        """Prune message history to fit within target token limit.

        Args:
            messages: List of message objects.
            target_tokens: Target maximum tokens. Uses config default if not provided.

        Returns:
            Tuple of (pruned_messages, removed_count).
        """
        if not messages:
            return messages, 0

        target = target_tokens or self.config.max_tokens
        current_tokens = self.estimate_tokens(messages)

        if current_tokens <= target:
                return messages, 0

        total_messages = len(messages)
        keep_first = min(self.config.keep_first_n, total_messages)
        keep_last = min(self.config.keep_last_n, total_messages)

        # Ensure we don't try to keep more than we have
        if keep_first + keep_last >= total_messages:
            logger.warning("History too short to prune meaningfully")
            return messages, 0

        # Score all removable messages (between first N and last N)
        scored_messages: List[ScoredMessage] = []
        for i, msg in enumerate(messages):
            # Skip messages in protected zones
            if i < keep_first or i >= total_messages - keep_last:
                continue

            score = self._calculate_importance_score(msg, i, total_messages)
            tokens = self._estimate_message_tokens(msg)
            role = self._get_message_role(msg)
            has_tools = self._has_tool_calls(msg)

            scored_messages.append(ScoredMessage(
                index=i,
                message=msg,
                score=score,
                estimated_tokens=tokens,
                has_tool_calls=has_tools,
                role=role,
            ))

        # Sort by score (ascending) - lowest scores removed first
        scored_messages.sort(key=lambda x: x.score)

        # Calculate tokens to remove
        tokens_to_remove = current_tokens - target
        removed_indices = set()
        removed_tokens = 0

        # Remove lowest-scoring messages until we hit target
        for scored in scored_messages:
            if removed_tokens >= tokens_to_remove:
                break
            removed_indices.add(scored.index)
            removed_tokens += scored.estimated_tokens

        # Build pruned history
        pruned = [
            msg for i, msg in enumerate(messages)
            if i not in removed_indices
        ]

        removed_count = len(removed_indices)
        new_tokens = self.estimate_tokens(pruned)


        return pruned, removed_count

    def prune_if_needed(
        self,
        messages: List[Any],
        provider: str = "generic",
    ) -> Tuple[List[Any], bool]:
        """Prune message history if it exceeds provider limits.

        Args:
            messages: List of message objects.
            provider: Provider name for context limit lookup.

        Returns:
            Tuple of (messages, was_pruned).
        """
        # Get provider-specific context limit
        max_tokens = self._get_provider_context_limit(provider)

        # Check if we need to prune
        if self.estimate_tokens(messages) <= max_tokens:
            return messages, False

        # Prune to 80% of limit to leave room for response
        target = int(max_tokens * 0.8)
        pruned, _ = self.prune(messages, target_tokens=target)
        return pruned, True

    def _get_provider_context_limit(self, provider: str) -> int:
        """Get context window limit for a provider.

        Args:
            provider: Provider name.

        Returns:
            Maximum context tokens for the provider.
        """
        # Provider context limits (as of 2025)
        limits = {
            "gemini": 1_000_000,  # 1M tokens
            "openai": 128_000,    # 128K tokens
            "zai": 128_000,       # 128K tokens
        }
        return limits.get(provider.lower(), self.config.max_tokens)

    def get_pruning_stats(self, messages: List[Any]) -> Dict[str, Any]:
        """Get statistics about the message history for pruning decisions.

        Args:
            messages: List of message objects.

        Returns:
            Dictionary with pruning statistics.
        """
        total_tokens = self.estimate_tokens(messages)
        role_counts: Dict[str, int] = {}
        tool_call_count = 0

        for msg in messages:
            role = self._get_message_role(msg)
            role_counts[role] = role_counts.get(role, 0) + 1
            if self._has_tool_calls(msg):
                tool_call_count += 1

        return {
            "total_messages": len(messages),
            "estimated_tokens": total_tokens,
            "role_distribution": role_counts,
            "tool_call_count": tool_call_count,
            "would_prune": total_tokens > self.config.max_tokens,
            "config_max_tokens": self.config.max_tokens,
        }

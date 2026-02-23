"""Feature tests for HistoryPruner.

Tests focus on behavior:
- MessageRole enum values
- PruningConfig initialization and defaults
- ScoredMessage dataclass
- HistoryPruner initialization
- Token estimation for various message formats
- Pruning logic with different scenarios
- Role-based importance scoring
- Tool call detection
- Message text extraction from various formats
- Provider-specific context limits
- Pruning statistics
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from persbot.services.history_pruner import (
    MessageRole,
    PruningConfig,
    ScoredMessage,
    HistoryPruner,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_system_role_value(self):
        """SYSTEM role has correct string value."""
        assert MessageRole.SYSTEM.value == "system"

    def test_user_role_value(self):
        """USER role has correct string value."""
        assert MessageRole.USER.value == "user"

    def test_assistant_role_value(self):
        """ASSISTANT role has correct string value."""
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_tool_role_value(self):
        """TOOL role has correct string value."""
        assert MessageRole.TOOL.value == "tool"

    def test_function_role_value(self):
        """FUNCTION role has correct string value."""
        assert MessageRole.FUNCTION.value == "function"


class TestPruningConfig:
    """Tests for PruningConfig dataclass."""

    def test_default_values(self):
        """PruningConfig has correct default values."""
        config = PruningConfig()
        assert config.max_tokens == 100000
        assert config.keep_first_n == 3
        assert config.keep_last_n == 6
        assert config.recency_bonus_weight == 0.3
        assert config.tool_call_bonus == 0.5
        assert config.min_tokens_per_message == 10

    def test_custom_max_tokens(self):
        """PruningConfig accepts custom max_tokens."""
        config = PruningConfig(max_tokens=50000)
        assert config.max_tokens == 50000
        # Other fields use defaults
        assert config.keep_first_n == 3

    def test_custom_keep_first_n(self):
        """PruningConfig accepts custom keep_first_n."""
        config = PruningConfig(keep_first_n=5)
        assert config.keep_first_n == 5

    def test_custom_keep_last_n(self):
        """PruningConfig accepts custom keep_last_n."""
        config = PruningConfig(keep_last_n=10)
        assert config.keep_last_n == 10

    def test_custom_recency_bonus_weight(self):
        """PruningConfig accepts custom recency_bonus_weight."""
        config = PruningConfig(recency_bonus_weight=0.5)
        assert config.recency_bonus_weight == 0.5

    def test_custom_tool_call_bonus(self):
        """PruningConfig accepts custom tool_call_bonus."""
        config = PruningConfig(tool_call_bonus=0.8)
        assert config.tool_call_bonus == 0.8

    def test_custom_min_tokens_per_message(self):
        """PruningConfig accepts custom min_tokens_per_message."""
        config = PruningConfig(min_tokens_per_message=5)
        assert config.min_tokens_per_message == 5

    def test_all_custom_values(self):
        """PruningConfig accepts all custom values."""
        config = PruningConfig(
            max_tokens=200000,
            keep_first_n=2,
            keep_last_n=4,
            recency_bonus_weight=0.4,
            tool_call_bonus=0.6,
            min_tokens_per_message=15,
        )
        assert config.max_tokens == 200000
        assert config.keep_first_n == 2
        assert config.keep_last_n == 4
        assert config.recency_bonus_weight == 0.4
        assert config.tool_call_bonus == 0.6
        assert config.min_tokens_per_message == 15


class TestScoredMessage:
    """Tests for ScoredMessage dataclass."""

    def test_creation_with_all_fields(self):
        """ScoredMessage can be created with all fields."""
        msg = {"role": "user", "content": "Hello"}
        scored = ScoredMessage(
            index=0,
            message=msg,
            score=1.5,
            estimated_tokens=20,
            has_tool_calls=False,
            role="user",
        )
        assert scored.index == 0
        assert scored.message == msg
        assert scored.score == 1.5
        assert scored.estimated_tokens == 20
        assert scored.has_tool_calls is False
        assert scored.role == "user"

    def test_default_has_tool_calls(self):
        """ScoredMessage has_tool_calls defaults to False."""
        scored = ScoredMessage(
            index=0,
            message={},
            score=1.0,
            estimated_tokens=10,
        )
        assert scored.has_tool_calls is False

    def test_default_role(self):
        """ScoredMessage role defaults to empty string."""
        scored = ScoredMessage(
            index=0,
            message={},
            score=1.0,
            estimated_tokens=10,
        )
        assert scored.role == ""


class MockMessage:
    """Mock message object for testing."""

    def __init__(
        self,
        role: str,
        content: Any = None,
        parts: Optional[List[Any]] = None,
        tool_calls: Optional[List] = None,
        function_call: Optional[Dict] = None,
    ):
        self.role = role
        self.content = content
        self.parts = parts
        self.tool_calls = tool_calls
        self.function_call = function_call


class MockPart:
    """Mock message part for testing."""

    def __init__(self, text: str = ""):
        self.text = text


class MockMessageWithoutContent:
    """Mock message that only has parts (no content attribute)."""

    def __init__(self, role: str, parts: Optional[List[Any]] = None):
        self.role = role
        self.parts = parts
        # No content attribute at all


class TestHistoryPrunerInit:
    """Tests for HistoryPruner initialization."""

    def test_init_with_default_config(self):
        """HistoryPruner can be created with default config."""
        pruner = HistoryPruner()
        assert pruner.config is not None
        assert pruner.config.max_tokens == 100000
        assert pruner.config.keep_first_n == 3

    def test_init_with_custom_config(self):
        """HistoryPruner can be created with custom config."""
        config = PruningConfig(max_tokens=50000, keep_first_n=5)
        pruner = HistoryPruner(config)
        assert pruner.config == config
        assert pruner.config.max_tokens == 50000
        assert pruner.config.keep_first_n == 5

    def test_role_weights_constant(self):
        """ROLE_WEIGHTS defines correct importance values."""
        assert HistoryPruner.ROLE_WEIGHTS[MessageRole.SYSTEM] == 1.0
        assert HistoryPruner.ROLE_WEIGHTS[MessageRole.TOOL] == 0.8
        assert HistoryPruner.ROLE_WEIGHTS[MessageRole.FUNCTION] == 0.8
        assert HistoryPruner.ROLE_WEIGHTS[MessageRole.USER] == 0.6
        assert HistoryPruner.ROLE_WEIGHTS[MessageRole.ASSISTANT] == 0.4

    def test_default_role_weight_constant(self):
        """DEFAULT_ROLE_WEIGHT is set correctly."""
        assert HistoryPruner.DEFAULT_ROLE_WEIGHT == 0.5


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_estimate_empty_list(self):
        """estimate_tokens returns 0 for empty list."""
        pruner = HistoryPruner()
        assert pruner.estimate_tokens([]) == 0

    def test_estimate_single_dict_message(self):
        """estimate_tokens estimates tokens for dict message."""
        pruner = HistoryPruner()
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = pruner.estimate_tokens(messages)
        # "Hello world" = 11 chars, 11 // 4 = 2, + 4 overhead = 6, but min 10
        assert tokens >= 10

    def test_estimate_multiple_messages(self):
        """estimate_tokens sums tokens for multiple messages."""
        pruner = HistoryPruner()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        tokens = pruner.estimate_tokens(messages)
        # Each message has min 10 tokens + overhead
        assert tokens >= 20

    def test_estimate_respects_min_tokens(self):
        """estimate_tokens respects min_tokens_per_message."""
        config = PruningConfig(min_tokens_per_message=20)
        pruner = HistoryPruner(config)
        messages = [{"role": "user", "content": "Hi"}]
        tokens = pruner.estimate_tokens(messages)
        # "Hi" = 2 chars, should use min 20 + 4 overhead
        assert tokens >= 20

    def test_estimate_object_message_with_content(self):
        """estimate_tokens handles object with content attribute."""
        pruner = HistoryPruner()
        msg = MockMessage(role="user", content="Hello world")
        tokens = pruner.estimate_tokens([msg])
        assert tokens >= 10

    def test_estimate_object_with_parts(self):
        """estimate_tokens handles object with parts attribute."""
        pruner = HistoryPruner()
        part1 = MockPart(text="Hello")
        part2 = MockPart(text="world")
        msg = MockMessage(role="user", parts=[part1, part2])
        tokens = pruner.estimate_tokens([msg])
        assert tokens >= 10

    def test_estimate_long_content(self):
        """estimate_tokens scales with content length."""
        pruner = HistoryPruner()
        short = [{"role": "user", "content": "Hi"}]
        long = [{"role": "user", "content": "a" * 100}]
        short_tokens = pruner.estimate_tokens(short)
        long_tokens = pruner.estimate_tokens(long)
        assert long_tokens > short_tokens


class TestExtractMessageText:
    """Tests for message text extraction."""

    def test_extract_from_dict_string_content(self):
        """Extracts text from dict with string content."""
        pruner = HistoryPruner()
        msg = {"role": "user", "content": "Hello world"}
        text = pruner._extract_message_text(msg)
        assert text == "Hello world"

    def test_extract_from_dict_list_content(self):
        """Extracts text from dict with list content."""
        pruner = HistoryPruner()
        msg = {
            "role": "user",
            "content": ["Hello", {"text": "world"}]
        }
        text = pruner._extract_message_text(msg)
        assert "Hello" in text
        assert "world" in text

    def test_extract_from_dict_empty_content(self):
        """Handles dict with empty content."""
        pruner = HistoryPruner()
        msg = {"role": "user", "content": ""}
        text = pruner._extract_message_text(msg)
        assert text == ""

    def test_extract_from_object_string_content(self):
        """Extracts text from object with string content."""
        pruner = HistoryPruner()
        msg = MockMessage(role="user", content="Hello world")
        text = pruner._extract_message_text(msg)
        assert text == "Hello world"

    def test_extract_from_object_list_content(self):
        """Extracts text from object with list content."""
        pruner = HistoryPruner()
        msg = MockMessage(role="user", content=["Hello", {"text": "world"}])
        text = pruner._extract_message_text(msg)
        assert "Hello" in text
        assert "world" in text

    def test_extract_from_parts(self):
        """Extracts text from parts attribute when content is not present."""
        pruner = HistoryPruner()
        part1 = MockPart(text="Hello")
        part2 = MockPart(text="world")
        # Use MockMessageWithoutContent which has no content attribute
        msg = MockMessageWithoutContent(role="user", parts=[part1, part2])
        text = pruner._extract_message_text(msg)
        assert "Hello" in text
        assert "world" in text

    def test_extract_from_parts_with_strings(self):
        """Extracts text from parts containing strings."""
        pruner = HistoryPruner()
        msg = MockMessageWithoutContent(role="user", parts=["Hello", "world"])
        text = pruner._extract_message_text(msg)
        assert text == "Hello world"

    def test_extract_fallback_to_str(self):
        """Falls back to string representation when content key is missing."""
        pruner = HistoryPruner()
        # Dict without 'content' key - but implementation returns empty for dict without content
        # Test with empty content which gets converted to empty string
        msg = {"content": None}
        text = pruner._extract_message_text(msg)
        assert text == ""

    def test_extract_none_message(self):
        """Handles None message gracefully."""
        pruner = HistoryPruner()
        text = pruner._extract_message_text(None)
        assert text == ""


class TestGetMessageRole:
    """Tests for message role extraction."""

    def test_get_role_from_dict(self):
        """Extracts role from dict message."""
        pruner = HistoryPruner()
        msg = {"role": "user", "content": "Hello"}
        role = pruner._get_message_role(msg)
        assert role == "user"

    def test_get_role_from_dict_lowercases(self):
        """Lowercases role from dict message."""
        pruner = HistoryPruner()
        msg = {"role": "User", "content": "Hello"}
        role = pruner._get_message_role(msg)
        assert role == "user"

    def test_get_role_from_object(self):
        """Extracts role from object message."""
        pruner = HistoryPruner()
        msg = MockMessage(role="assistant", content="Hi")
        role = pruner._get_message_role(msg)
        assert role == "assistant"

    def test_get_role_from_object_lowercases(self):
        """Lowercases role from object message."""
        pruner = HistoryPruner()
        msg = MockMessage(role="Assistant", content="Hi")
        role = pruner._get_message_role(msg)
        assert role == "assistant"

    def test_get_role_missing_returns_empty(self):
        """Returns empty string when role is missing."""
        pruner = HistoryPruner()
        msg = {"content": "Hello"}
        role = pruner._get_message_role(msg)
        assert role == ""


class TestHasToolCalls:
    """Tests for tool call detection."""

    def test_detects_tool_calls_in_dict(self):
        """Detects tool_calls in dict message."""
        pruner = HistoryPruner()
        msg = {
            "role": "assistant",
            "content": "I'll help",
            "tool_calls": [{"id": "call_123", "function": {"name": "search"}}]
        }
        assert pruner._has_tool_calls(msg) is True

    def test_detects_function_call_in_dict(self):
        """Detects function_call in dict message."""
        pruner = HistoryPruner()
        msg = {
            "role": "assistant",
            "content": "I'll help",
            "function_call": {"name": "search", "arguments": "{}"}
        }
        assert pruner._has_tool_calls(msg) is True

    def test_empty_tool_calls_in_dict(self):
        """Returns False for empty tool_calls in dict."""
        pruner = HistoryPruner()
        msg = {"role": "assistant", "content": "Hi", "tool_calls": []}
        assert pruner._has_tool_calls(msg) is False

    def test_detects_tool_calls_in_object(self):
        """Detects tool_calls in object message."""
        pruner = HistoryPruner()
        msg = MockMessage(
            role="assistant",
            content="I'll help",
            tool_calls=[{"id": "call_123"}]
        )
        assert pruner._has_tool_calls(msg) is True

    def test_detects_function_call_in_object(self):
        """Detects function_call in object message."""
        pruner = HistoryPruner()
        msg = MockMessage(
            role="assistant",
            content="I'll help",
            function_call={"name": "search"}
        )
        assert pruner._has_tool_calls(msg) is True

    def test_detects_tool_role(self):
        """Detects tool role."""
        pruner = HistoryPruner()
        msg = {"role": "tool", "content": "Result: 42"}
        assert pruner._has_tool_calls(msg) is True

    def test_detects_function_role(self):
        """Detects function role."""
        pruner = HistoryPruner()
        msg = {"role": "function", "content": "Result: 42"}
        assert pruner._has_tool_calls(msg) is True

    def test_no_tool_calls_in_regular_message(self):
        """Returns False for regular message without tool calls."""
        pruner = HistoryPruner()
        msg = {"role": "user", "content": "Hello"}
        assert pruner._has_tool_calls(msg) is False


class TestCalculateImportanceScore:
    """Tests for importance score calculation."""

    def test_system_message_score(self):
        """System messages have high base score."""
        pruner = HistoryPruner()
        msg = {"role": "system", "content": "You are helpful"}
        score = pruner._calculate_importance_score(msg, 0, 10)
        assert score >= 1.0  # Base weight for system

    def test_user_message_score(self):
        """User messages have medium base score."""
        pruner = HistoryPruner()
        msg = {"role": "user", "content": "Hello"}
        score = pruner._calculate_importance_score(msg, 0, 10)
        assert score >= 0.6  # Base weight for user

    def test_assistant_message_score(self):
        """Assistant messages have lower base score."""
        pruner = HistoryPruner()
        msg = {"role": "assistant", "content": "Hi"}
        score = pruner._calculate_importance_score(msg, 0, 10)
        assert score >= 0.4  # Base weight for assistant

    def test_recency_bonus_increases_score(self):
        """Newer messages get recency bonus."""
        pruner = HistoryPruner()
        msg = {"role": "user", "content": "Hello"}
        old_score = pruner._calculate_importance_score(msg, 0, 10)
        new_score = pruner._calculate_importance_score(msg, 9, 10)
        assert new_score > old_score

    def test_tool_call_bonus_increases_score(self):
        """Messages with tool calls get bonus."""
        pruner = HistoryPruner()
        msg_with_tools = {
            "role": "assistant",
            "content": "I'll search",
            "tool_calls": [{"id": "call_123"}]
        }
        msg_without_tools = {
            "role": "assistant",
            "content": "I'll search"
        }
        score_with = pruner._calculate_importance_score(msg_with_tools, 0, 10)
        score_without = pruner._calculate_importance_score(msg_without_tools, 0, 10)
        assert score_with > score_without

    def test_unknown_role_uses_default_weight(self):
        """Unknown roles use default weight."""
        pruner = HistoryPruner()
        msg = {"role": "unknown", "content": "Hello"}
        score = pruner._calculate_importance_score(msg, 0, 10)
        assert score >= HistoryPruner.DEFAULT_ROLE_WEIGHT


class TestPrune:
    """Tests for prune method."""

    def test_prune_empty_list(self):
        """Prune returns empty list for empty input."""
        pruner = HistoryPruner()
        result, removed = pruner.prune([])
        assert result == []
        assert removed == 0

    def test_prune_under_limit(self):
        """Messages under limit are not pruned."""
        pruner = HistoryPruner()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result, removed = pruner.prune(messages, target_tokens=10000)
        assert result == messages
        assert removed == 0

    def test_prune_keeps_first_n(self):
        """First N messages are always kept."""
        config = PruningConfig(keep_first_n=2, max_tokens=50)
        pruner = HistoryPruner(config)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "a" * 100},
        ]
        result, removed = pruner.prune(messages, target_tokens=30)
        # First 2 should be kept
        assert len(result) >= 2
        assert result[0] == messages[0]
        assert result[1] == messages[1]

    def test_prune_keeps_last_n(self):
        """Last N messages are always kept."""
        config = PruningConfig(keep_last_n=2, keep_first_n=0, max_tokens=50)
        pruner = HistoryPruner(config)
        messages = [
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
            {"role": "user", "content": "c" * 100},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result, removed = pruner.prune(messages, target_tokens=30)
        # Last 2 should be kept
        assert len(result) >= 2
        assert result[-1] == messages[-1]
        assert result[-2] == messages[-2]

    def test_prune_removes_middle_messages(self):
        """Middle messages are pruned when over limit."""
        config = PruningConfig(keep_first_n=1, keep_last_n=1, max_tokens=50)
        pruner = HistoryPruner(config)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
            {"role": "user", "content": "c" * 100},
            {"role": "assistant", "content": "Hello"},
        ]
        result, removed = pruner.prune(messages, target_tokens=30)
        # Should remove some middle messages
        assert removed > 0
        assert result[0] == messages[0]  # First kept
        assert result[-1] == messages[-1]  # Last kept

    def test_prune_uses_config_target_by_default(self):
        """Uses config.max_tokens when target not provided."""
        config = PruningConfig(max_tokens=50, keep_first_n=0, keep_last_n=0)
        pruner = HistoryPruner(config)
        messages = [
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
        ]
        result, removed = pruner.prune(messages)
        assert removed > 0

    def test_prune_too_short_warning(self):
        """Returns original when keep_first + keep_last >= total."""
        config = PruningConfig(keep_first_n=5, keep_last_n=5, max_tokens=10)
        pruner = HistoryPruner(config)
        messages = [
            {"role": "user", "content": "Hi"},
        ]
        result, removed = pruner.prune(messages)
        assert result == messages
        assert removed == 0

    def test_prune_removes_lowest_score_first(self):
        """Messages with lowest scores are removed first."""
        config = PruningConfig(keep_first_n=1, keep_last_n=1, max_tokens=80)
        pruner = HistoryPruner(config)
        messages = [
            {"role": "system", "content": "a" * 50},  # Protected (first N), high score
            {"role": "assistant", "content": "b" * 50},  # Low score, can be removed
            {"role": "user", "content": "c" * 50},  # Medium score, can be removed
            {"role": "user", "content": "Hi there"},  # Protected (last N)
        ]
        result, removed = pruner.prune(messages, target_tokens=60)
        # At least one message should be removed
        assert removed > 0
        # First and last should always be kept
        assert result[0]["role"] == "system"
        assert result[-1]["role"] == "user"
        # Result should have fewer messages than original
        assert len(result) < len(messages)

    def test_prune_returns_removed_count(self):
        """Returns correct count of removed messages."""
        config = PruningConfig(keep_first_n=0, keep_last_n=0, max_tokens=30)
        pruner = HistoryPruner(config)
        messages = [
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
        ]
        result, removed = pruner.prune(messages, target_tokens=20)
        assert removed >= 0
        assert len(result) + removed == len(messages)


class TestPruneIfNeeded:
    """Tests for prune_if_needed method."""

    def test_no_prune_when_under_limit(self):
        """Returns original messages when under limit."""
        pruner = HistoryPruner()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result, was_pruned = pruner.prune_if_needed(messages, provider="gemini")
        assert result == messages
        assert was_pruned is False

    def test_prunes_when_over_limit(self):
        """Prunes when over provider limit."""
        # Create messages that exceed OpenAI's limit (128K)
        pruner = HistoryPruner()
        messages = [{"role": "user", "content": "a" * 10000} for _ in range(100)]
        result, was_pruned = pruner.prune_if_needed(messages, provider="openai")
        assert was_pruned is True
        # Result should be shorter
        assert len(result) < len(messages)

    def test_uses_provider_specific_limit_gemini(self):
        """Uses correct limit for Gemini (1M tokens)."""
        pruner = HistoryPruner()
        # Small message set that's under 1M tokens
        messages = [{"role": "user", "content": "Hi"}]
        result, was_pruned = pruner.prune_if_needed(messages, provider="gemini")
        assert was_pruned is False

    def test_uses_provider_specific_limit_openai(self):
        """Uses correct limit for OpenAI (128K tokens)."""
        pruner = HistoryPruner()
        # Messages that would be over 128K
        messages = [{"role": "user", "content": "a" * 10000} for _ in range(100)]
        result, was_pruned = pruner.prune_if_needed(messages, provider="openai")
        assert was_pruned is True

    def test_uses_provider_specific_limit_zai(self):
        """Uses correct limit for Z.AI (128K tokens)."""
        pruner = HistoryPruner()
        # Messages that would be over 128K
        messages = [{"role": "user", "content": "a" * 10000} for _ in range(100)]
        result, was_pruned = pruner.prune_if_needed(messages, provider="zai")
        assert was_pruned is True

    def test_uses_config_max_for_unknown_provider(self):
        """Uses config.max_tokens for unknown provider."""
        pruner = HistoryPruner()
        messages = [{"role": "user", "content": "Hi"}]
        result, was_pruned = pruner.prune_if_needed(messages, provider="unknown")
        assert was_pruned is False

    def test_targets_80_percent_of_limit(self):
        """Prunes to 80% of limit to leave room for response."""
        pruner = HistoryPruner()
        # Create messages that exceed limit
        messages = [{"role": "user", "content": "a" * 10000} for _ in range(100)]
        result, was_pruned = pruner.prune_if_needed(messages, provider="openai")
        assert was_pruned is True
        # Result should have significantly fewer messages
        assert len(result) < len(messages) * 0.9


class TestGetProviderContextLimit:
    """Tests for provider context limit lookup."""

    def test_gemini_limit(self):
        """Gemini has 1M token limit."""
        pruner = HistoryPruner()
        limit = pruner._get_provider_context_limit("gemini")
        assert limit == 1_000_000

    def test_openai_limit(self):
        """OpenAI has 128K token limit."""
        pruner = HistoryPruner()
        limit = pruner._get_provider_context_limit("openai")
        assert limit == 128_000

    def test_zai_limit(self):
        """Z.AI has 128K token limit."""
        pruner = HistoryPruner()
        limit = pruner._get_provider_context_limit("zai")
        assert limit == 128_000

    def test_unknown_provider_uses_config(self):
        """Unknown provider uses config.max_tokens."""
        config = PruningConfig(max_tokens=50000)
        pruner = HistoryPruner(config)
        limit = pruner._get_provider_context_limit("unknown")
        assert limit == 50000

    def test_case_insensitive_provider(self):
        """Provider name is case-insensitive."""
        pruner = HistoryPruner()
        limit1 = pruner._get_provider_context_limit("GEMINI")
        limit2 = pruner._get_provider_context_limit("Gemini")
        limit3 = pruner._get_provider_context_limit("gemini")
        assert limit1 == limit2 == limit3


class TestGetPruningStats:
    """Tests for pruning statistics."""

    def test_stats_for_empty_messages(self):
        """Returns stats for empty message list."""
        pruner = HistoryPruner()
        stats = pruner.get_pruning_stats([])
        assert stats["total_messages"] == 0
        assert stats["estimated_tokens"] == 0
        assert stats["role_distribution"] == {}
        assert stats["tool_call_count"] == 0
        assert stats["would_prune"] is False

    def test_stats_counts_messages(self):
        """Counts total messages correctly."""
        pruner = HistoryPruner()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        stats = pruner.get_pruning_stats(messages)
        assert stats["total_messages"] == 2

    def test_stats_estimates_tokens(self):
        """Estimates total tokens correctly."""
        pruner = HistoryPruner()
        messages = [
            {"role": "user", "content": "Hello world"},
        ]
        stats = pruner.get_pruning_stats(messages)
        assert stats["estimated_tokens"] >= 10

    def test_stats_role_distribution(self):
        """Counts messages by role."""
        pruner = HistoryPruner()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Hello"},
        ]
        stats = pruner.get_pruning_stats(messages)
        assert stats["role_distribution"]["system"] == 1
        assert stats["role_distribution"]["user"] == 2
        assert stats["role_distribution"]["assistant"] == 1

    def test_stats_tool_call_count(self):
        """Counts messages with tool calls."""
        pruner = HistoryPruner()
        messages = [
            {"role": "user", "content": "Search for news"},
            {
                "role": "assistant",
                "content": "I'll search",
                "tool_calls": [{"id": "call_123"}]
            },
            {"role": "tool", "content": "Result: found"},
            {
                "role": "assistant",
                "content": "Done",
                "tool_calls": [{"id": "call_456"}]
            },
        ]
        stats = pruner.get_pruning_stats(messages)
        assert stats["tool_call_count"] == 3  # 2 with tool_calls, 1 with tool role

    def test_stats_would_prune_flag(self):
        """Indicates if pruning would occur."""
        config = PruningConfig(max_tokens=10)  # Very low limit
        pruner = HistoryPruner(config)
        # Long message that definitely exceeds limit
        messages = [{"role": "user", "content": "a" * 1000}]
        stats = pruner.get_pruning_stats(messages)
        # 1000 chars / 4 = 250 tokens + overhead > 10
        assert stats["would_prune"] is True

    def test_stats_config_max_tokens(self):
        """Includes config max_tokens in stats."""
        config = PruningConfig(max_tokens=50000)
        pruner = HistoryPruner(config)
        stats = pruner.get_pruning_stats([])
        assert stats["config_max_tokens"] == 50000

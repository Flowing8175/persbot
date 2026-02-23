"""Feature tests for summarization_service module.

Tests focus on behavior:
- SummarizationConfig: configuration for summarization
- SummarizationService: context summarization to reduce token costs
"""

import pytest
from unittest.mock import Mock

from persbot.services.summarization_service import (
    SummarizationConfig,
    SummarizationService,
)
from persbot.services.base import ChatMessage


class TestSummarizationConfig:
    """Tests for SummarizationConfig dataclass."""

    def test_config_exists(self):
        """SummarizationConfig class exists."""
        assert SummarizationConfig is not None

    def test_creates_with_defaults(self):
        """SummarizationConfig creates with default values."""
        config = SummarizationConfig()

        assert config.threshold == 40
        assert config.keep_recent_messages == 7
        assert config.summarization_model == "gemini-2.5-flash"
        assert config.max_summary_length == 500

    def test_creates_with_custom_threshold(self):
        """SummarizationConfig can be created with custom threshold."""
        config = SummarizationConfig(threshold=20)

        assert config.threshold == 20
        assert config.keep_recent_messages == 7  # Default preserved

    def test_creates_with_custom_keep_recent(self):
        """SummarizationConfig can be created with custom keep_recent_messages."""
        config = SummarizationConfig(keep_recent_messages=10)

        assert config.keep_recent_messages == 10
        assert config.threshold == 40  # Default preserved

    def test_creates_with_custom_model(self):
        """SummarizationConfig can be created with custom model."""
        config = SummarizationConfig(summarization_model="gpt-4o-mini")

        assert config.summarization_model == "gpt-4o-mini"

    def test_creates_with_custom_max_length(self):
        """SummarizationConfig can be created with custom max_summary_length."""
        config = SummarizationConfig(max_summary_length=1000)

        assert config.max_summary_length == 1000

    def test_creates_with_all_custom_values(self):
        """SummarizationConfig can be created with all custom values."""
        config = SummarizationConfig(
            threshold=100,
            keep_recent_messages=15,
            summarization_model="gemini-2.0-flash",
            max_summary_length=300,
        )

        assert config.threshold == 100
        assert config.keep_recent_messages == 15
        assert config.summarization_model == "gemini-2.0-flash"
        assert config.max_summary_length == 300


class TestSummarizationServiceCreation:
    """Tests for SummarizationService instantiation."""

    def test_service_exists(self):
        """SummarizationService class exists."""
        assert SummarizationService is not None

    def test_creates_with_default_config(self):
        """SummarizationService creates with default config."""
        service = SummarizationService()

        assert service.config.threshold == 40
        assert service.config.keep_recent_messages == 7
        assert service.config.summarization_model == "gemini-2.5-flash"
        assert service.config.max_summary_length == 500

    def test_creates_with_custom_config(self):
        """SummarizationService creates with custom config."""
        config = SummarizationConfig(
            threshold=20,
            keep_recent_messages=5,
            summarization_model="gpt-4o-mini",
            max_summary_length=250,
        )
        service = SummarizationService(config=config)

        assert service.config.threshold == 20
        assert service.config.keep_recent_messages == 5
        assert service.config.summarization_model == "gpt-4o-mini"
        assert service.config.max_summary_length == 250

    def test_initializes_counters_to_zero(self):
        """SummarizationService initializes counters to zero."""
        service = SummarizationService()

        assert service._summarization_count == 0
        assert service._tokens_saved == 0


class TestSummarizationServiceShouldSummarize:
    """Tests for should_summarize method."""

    @pytest.fixture
    def service(self):
        """Create a service with default threshold."""
        return SummarizationService()

    def test_returns_false_for_empty_history(self, service):
        """should_summarize returns False for empty history."""
        result = service.should_summarize([])

        assert result is False

    def test_returns_false_below_threshold(self, service):
        """should_summarize returns False when below threshold."""
        # Default threshold is 40, create 39 messages
        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(39)
        ]

        result = service.should_summarize(history)

        assert result is False

    def test_returns_true_at_threshold(self, service):
        """should_summarize returns True when at threshold."""
        # Default threshold is 40, create 40 messages
        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(40)
        ]

        result = service.should_summarize(history)

        assert result is True

    def test_returns_true_above_threshold(self, service):
        """should_summarize returns True when above threshold."""
        # Default threshold is 40, create 50 messages
        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(50)
        ]

        result = service.should_summarize(history)

        assert result is True

    def test_respects_custom_threshold(self):
        """should_summarize respects custom threshold."""
        service = SummarizationService(config=SummarizationConfig(threshold=10))

        # Below custom threshold
        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(9)
        ]
        assert service.should_summarize(history) is False

        # At custom threshold
        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(10)
        ]
        assert service.should_summarize(history) is True


class TestSummarizationServiceGetMessagesToSummarize:
    """Tests for get_messages_to_summarize method."""

    @pytest.fixture
    def service(self):
        """Create a service with default keep_recent_messages."""
        return SummarizationService()

    def test_returns_empty_and_all_when_below_keep_count(self, service):
        """get_messages_to_summarize returns empty list when history <= keep_count."""
        # Default keep_recent_messages is 7
        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(5)
        ]

        to_summarize, to_keep = service.get_messages_to_summarize(history)

        assert to_summarize == []
        assert len(to_keep) == 5
        assert to_keep == history

    def test_returns_empty_and_all_when_at_keep_count(self, service):
        """get_messages_to_summarize returns empty list when history == keep_count."""
        # Default keep_recent_messages is 7
        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(7)
        ]

        to_summarize, to_keep = service.get_messages_to_summarize(history)

        assert to_summarize == []
        assert len(to_keep) == 7
        assert to_keep == history

    def test_splits_when_above_keep_count(self, service):
        """get_messages_to_summarize splits messages correctly."""
        # Default keep_recent_messages is 7, create 10 messages
        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(10)
        ]

        to_summarize, to_keep = service.get_messages_to_summarize(history)

        # First 3 should be summarized
        assert len(to_summarize) == 3
        assert to_summarize[0].content == "Message 0"
        assert to_summarize[2].content == "Message 2"

        # Last 7 should be kept
        assert len(to_keep) == 7
        assert to_keep[0].content == "Message 3"
        assert to_keep[6].content == "Message 9"

    def test_respects_custom_keep_count(self):
        """get_messages_to_summarize respects custom keep_recent_messages."""
        service = SummarizationService(
            config=SummarizationConfig(keep_recent_messages=3)
        )

        history = [
            ChatMessage(role="user", content=f"Message {i}", parts=[])
            for i in range(10)
        ]

        to_summarize, to_keep = service.get_messages_to_summarize(history)

        # First 7 should be summarized
        assert len(to_summarize) == 7
        # Last 3 should be kept
        assert len(to_keep) == 3
        assert to_keep[0].content == "Message 7"
        assert to_keep[2].content == "Message 9"


class TestSummarizationServiceFormatHistoryForSummary:
    """Tests for format_history_for_summary method."""

    @pytest.fixture
    def service(self):
        """Create a service instance."""
        return SummarizationService()

    def test_formats_empty_list(self, service):
        """format_history_for_summary handles empty list."""
        result = service.format_history_for_summary([])

        assert result == ""

    def test_formats_single_message(self, service):
        """format_history_for_summary formats single message."""
        messages = [
            ChatMessage(role="user", content="Hello world", parts=[])
        ]

        result = service.format_history_for_summary(messages)

        assert result == "[USER]: Hello world"

    def test_formats_multiple_messages(self, service):
        """format_history_for_summary formats multiple messages."""
        messages = [
            ChatMessage(role="user", content="Hello", parts=[]),
            ChatMessage(role="assistant", content="Hi there", parts=[]),
            ChatMessage(role="user", content="How are you?", parts=[]),
        ]

        result = service.format_history_for_summary(messages)

        expected = "[USER]: Hello\n[ASSISTANT]: Hi there\n[USER]: How are you?"
        assert result == expected

    def test_uppercases_role(self, service):
        """format_history_for_summary uppercases role names."""
        messages = [
            ChatMessage(role="user", content="Test", parts=[]),
            ChatMessage(role="assistant", content="Response", parts=[]),
        ]

        result = service.format_history_for_summary(messages)

        assert "[USER]:" in result
        assert "[ASSISTANT]:" in result

    def test_handles_none_content(self, service):
        """format_history_for_summary handles None content."""
        messages = [
            ChatMessage(role="user", content=None, parts=[])
        ]

        result = service.format_history_for_summary(messages)

        assert result == "[USER]: "

    def test_truncates_long_messages(self, service):
        """format_history_for_summary truncates messages over 500 chars."""
        long_content = "A" * 600
        messages = [
            ChatMessage(role="user", content=long_content, parts=[])
        ]

        result = service.format_history_for_summary(messages)

        # Should be truncated to 500 chars + "..."
        assert len(result) == len("[USER]: ") + 500 + 3
        assert result.endswith("...")


class TestSummarizationServiceCreateSummaryPrompt:
    """Tests for create_summary_prompt method."""

    @pytest.fixture
    def service(self):
        """Create a service instance."""
        return SummarizationService()

    def test_includes_summary_instructions(self, service):
        """create_summary_prompt includes summarization instructions."""
        result = service.create_summary_prompt("Some history")

        assert "Summarize" in result
        assert "concisely" in result
        assert "Keep key topics" in result
        assert "decisions" in result
        assert "important context" in result

    def test_includes_max_length(self, service):
        """create_summary_prompt includes max length."""
        result = service.create_summary_prompt("Some history")

        assert "500 tokens" in result

    def test_includes_formatted_history(self, service):
        """create_summary_prompt includes formatted history."""
        history = "[USER]: Hello\n[ASSISTANT]: Hi"
        result = service.create_summary_prompt(history)

        assert history in result

    def test_respects_custom_max_length(self):
        """create_summary_prompt respects custom max_summary_length."""
        service = SummarizationService(
            config=SummarizationConfig(max_summary_length=1000)
        )

        result = service.create_summary_prompt("Some history")

        assert "1000 tokens" in result
        assert "500 tokens" not in result


class TestSummarizationServiceCreateSummaryMessage:
    """Tests for create_summary_message method."""

    @pytest.fixture
    def service(self):
        """Create a service instance."""
        return SummarizationService()

    def test_returns_chat_message(self, service):
        """create_summary_message returns ChatMessage."""
        result = service.create_summary_message("Test summary")

        assert isinstance(result, ChatMessage)

    def test_sets_role_to_user(self, service):
        """create_summary_message sets role to user."""
        result = service.create_summary_message("Test summary")

        assert result.role == "user"

    def test_wraps_summary_text(self, service):
        """create_summary_message wraps summary in brackets."""
        result = service.create_summary_message("Discussed AI models")

        assert result.content == "[Previous conversation summary: Discussed AI models]"

    def test_includes_parts(self, service):
        """create_summary_message includes parts with text."""
        result = service.create_summary_message("Test summary")

        assert len(result.parts) == 1
        assert result.parts[0]["text"] == "[Previous conversation summary: Test summary]"


class TestSummarizationServiceRecordSummarization:
    """Tests for record_summarization method."""

    @pytest.fixture
    def service(self):
        """Create a service instance."""
        return SummarizationService()

    def test_increments_summarization_count(self, service):
        """record_summarization increments counter."""
        initial_count = service._summarization_count

        service.record_summarization(10, 1000)

        assert service._summarization_count == initial_count + 1

    def test_adds_tokens_saved(self, service):
        """record_summarization adds to tokens_saved."""
        service.record_summarization(10, 1000)
        service.record_summarization(5, 500)

        assert service._tokens_saved == 1500

    def test_accumulates_multiple_records(self, service):
        """record_summarization accumulates across calls."""
        for i in range(5):
            service.record_summarization(10, 200)

        assert service._summarization_count == 5
        assert service._tokens_saved == 1000


class TestSummarizationServiceGetStats:
    """Tests for get_stats method."""

    @pytest.fixture
    def service(self):
        """Create a service instance."""
        return SummarizationService()

    def test_returns_dict(self, service):
        """get_stats returns a dictionary."""
        result = service.get_stats()

        assert isinstance(result, dict)

    def test_includes_summarization_count(self, service):
        """get_stats includes summarization_count."""
        service.record_summarization(10, 1000)
        service.record_summarization(5, 500)

        result = service.get_stats()

        assert result["summarization_count"] == 2

    def test_includes_tokens_saved(self, service):
        """get_stats includes tokens_saved."""
        service.record_summarization(10, 1000)
        service.record_summarization(5, 500)

        result = service.get_stats()

        assert result["tokens_saved"] == 1500

    def test_includes_threshold(self, service):
        """get_stats includes threshold."""
        result = service.get_stats()

        assert result["threshold"] == 40

    def test_includes_keep_recent(self, service):
        """get_stats includes keep_recent."""
        result = service.get_stats()

        assert result["keep_recent"] == 7

    def test_reflects_custom_config(self):
        """get_stats reflects custom config."""
        service = SummarizationService(
            config=SummarizationConfig(threshold=100, keep_recent_messages=15)
        )

        result = service.get_stats()

        assert result["threshold"] == 100
        assert result["keep_recent"] == 15


class TestSummarizationServiceEstimateTokens:
    """Tests for estimate_tokens method."""

    @pytest.fixture
    def service(self):
        """Create a service instance."""
        return SummarizationService()

    def test_returns_zero_for_empty_list(self, service):
        """estimate_tokens returns 0 for empty list."""
        result = service.estimate_tokens([])

        assert result == 0

    def test_estimates_single_message(self, service):
        """estimate_tokens estimates single message."""
        # "Hello world" is 11 chars, 11 // 4 = 2 tokens
        messages = [
            ChatMessage(role="user", content="Hello world", parts=[])
        ]

        result = service.estimate_tokens(messages)

        assert result == 2  # "Hello world" is 11 chars, 11 // 4 = 2

    def test_sum_multiple_messages(self, service):
        """estimate_tokens sums across messages."""
        messages = [
            ChatMessage(role="user", content="First", parts=[]),  # 5 chars
            ChatMessage(role="user", content="Second", parts=[]),  # 6 chars
            ChatMessage(role="user", content="Third", parts=[]),  # 5 chars
        ]
        # Total: 16 chars // 4 = 4 tokens

        result = service.estimate_tokens(messages)

        assert result == 4

    def test_handles_none_content(self, service):
        """estimate_tokens handles None content."""
        messages = [
            ChatMessage(role="user", content=None, parts=[])
        ]

        result = service.estimate_tokens(messages)

        assert result == 0

    def test_uses_integer_division(self, service):
        """estimate_tokens uses integer division (rounds down)."""
        # 7 chars // 4 = 1 token (not 1.75)
        messages = [
            ChatMessage(role="user", content="1234567", parts=[])
        ]

        result = service.estimate_tokens(messages)

        assert result == 1

    def test_approximates_larger_content(self, service):
        """estimate_tokens approximates larger content."""
        # 1000 chars // 4 = 250 tokens
        messages = [
            ChatMessage(role="user", content="A" * 1000, parts=[])
        ]

        result = service.estimate_tokens(messages)

        assert result == 250

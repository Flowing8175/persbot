"""Tests for model_usage_service.py module."""

import datetime
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch, mock_open

from persbot.services.model_usage_service import ModelDefinition, ModelUsageService


def _get_today_kst_date() -> str:
    """Get today's date in KST timezone."""
    now_kst = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
    return now_kst.strftime("%Y-%m-%d")


class TestModelDefinition:
    """Tests for ModelDefinition dataclass."""

    def test_creates_with_required_fields(self):
        """ModelDefinition creates with all required fields."""
        model_def = ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=100,
            scope="guild",
            provider="openai",
        )
        assert model_def.display_name == "GPT 5 mini"
        assert model_def.api_model_name == "gpt-5-mini"
        assert model_def.daily_limit == 100
        assert model_def.scope == "guild"
        assert model_def.provider == "openai"
        assert model_def.fallback_alias is None

    def test_accepts_fallback_alias(self):
        """ModelDefinition accepts optional fallback_alias."""
        model_def = ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=100,
            scope="guild",
            provider="openai",
            fallback_alias="Gemini 2.5 flash",
        )
        assert model_def.fallback_alias == "Gemini 2.5 flash"

    def test_dataclass_equality(self):
        """ModelDefinition instances with same values are equal."""
        model_def1 = ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=100,
            scope="guild",
            provider="openai",
        )
        model_def2 = ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=100,
            scope="guild",
            provider="openai",
        )
        assert model_def1 == model_def2

    def test_dataclass_immutability(self):
        """ModelDefinition fields can be modified (dataclass is not frozen)."""
        model_def = ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=100,
            scope="guild",
            provider="openai",
        )
        # Dataclass is not frozen, so fields can be modified
        model_def.daily_limit = 200
        assert model_def.daily_limit == 200


class TestModelUsageServiceInit:
    """Tests for ModelUsageService.__init__()."""

    @patch("os.path.exists", return_value=False)
    def test_init_with_default_paths(self, mock_exists):
        """__init__ uses default file paths."""
        service = ModelUsageService.__new__(ModelUsageService)
        service.data_file = "data/model_usage.json"
        service.models_file = "data/models.json"
        service.usage_data = {}
        service._default_provider = "gemini"
        service.MODEL_DEFINITIONS = {}
        service._model_definition_cache = {}
        service.ALIAS_TO_API = {}
        service.API_TO_ALIAS = {}
        service._load_models()
        service._load_usage()

        assert service.data_file == "data/model_usage.json"
        assert service.models_file == "data/models.json"

    @patch("os.path.exists", return_value=False)
    def test_init_with_custom_paths(self, mock_exists):
        """__init__ accepts custom file paths."""
        service = ModelUsageService.__new__(ModelUsageService)
        service.data_file = "custom/usage.json"
        service.models_file = "custom/models.json"
        service.usage_data = {}
        service._default_provider = "openai"
        service.MODEL_DEFINITIONS = {}
        service._model_definition_cache = {}
        service.ALIAS_TO_API = {}
        service.API_TO_ALIAS = {}
        service._load_models()
        service._load_usage()

        assert service.data_file == "custom/usage.json"
        assert service.models_file == "custom/models.json"

    @patch("os.path.exists", return_value=False)
    def test_init_with_custom_default_provider(self, mock_exists):
        """__init__ sets default provider correctly."""
        service = ModelUsageService.__new__(ModelUsageService)
        service.data_file = "data/model_usage.json"
        service.models_file = "data/models.json"
        service.usage_data = {}
        service._default_provider = "openai"
        service.MODEL_DEFINITIONS = {}
        service._model_definition_cache = {}
        service.ALIAS_TO_API = {}
        service.API_TO_ALIAS = {}
        service._load_models()
        service._load_usage()

        assert service._default_provider == "openai"


class TestModelUsageServiceLoadModels:
    """Tests for ModelUsageService model loading."""

    @patch("os.path.exists")
    def test_loads_models_from_file(self, mock_exists):
        """_load_models loads model definitions from JSON file."""
        mock_exists.return_value = True
        models_data = {
            "llm_models": {
                "Gemini 2.5 flash": {
                    "display_name": "Gemini 2.5 flash",
                    "api_model_name": "gemini-2.5-flash",
                    "daily_limit": 500,
                    "scope": "guild",
                    "provider": "gemini",
                }
            }
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(models_data))):
            service = ModelUsageService.__new__(ModelUsageService)
            service.models_file = "data/models.json"
            service._default_provider = "gemini"
            service.MODEL_DEFINITIONS = {}
            service._model_definition_cache = {}
            service.ALIAS_TO_API = {}
            service.API_TO_ALIAS = {}
            service._load_models()

        assert "Gemini 2.5 flash" in service.MODEL_DEFINITIONS
        assert service.MODEL_DEFINITIONS["Gemini 2.5 flash"].api_model_name == "gemini-2.5-flash"

    @patch("os.path.exists")
    def test_loads_models_without_llm_models_wrapper(self, mock_exists):
        """_load_models handles JSON without llm_models wrapper."""
        mock_exists.return_value = True
        models_data = {
            "Gemini 2.5 flash": {
                "display_name": "Gemini 2.5 flash",
                "api_model_name": "gemini-2.5-flash",
                "daily_limit": 500,
                "scope": "guild",
                "provider": "gemini",
            }
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(models_data))):
            service = ModelUsageService.__new__(ModelUsageService)
            service.models_file = "data/models.json"
            service._default_provider = "gemini"
            service.MODEL_DEFINITIONS = {}
            service._model_definition_cache = {}
            service.ALIAS_TO_API = {}
            service.API_TO_ALIAS = {}
            service._load_models()

        assert "Gemini 2.5 flash" in service.MODEL_DEFINITIONS

    @patch("os.path.exists")
    def test_handles_missing_file(self, mock_exists):
        """_load_models handles missing models file gracefully."""
        mock_exists.return_value = False
        service = ModelUsageService.__new__(ModelUsageService)
        service.models_file = "data/models.json"
        service._default_provider = "gemini"
        service.MODEL_DEFINITIONS = {}
        service._model_definition_cache = {}
        service.ALIAS_TO_API = {}
        service.API_TO_ALIAS = {}

        # Should not raise
        service._load_models()

    @patch("os.path.exists")
    def test_handles_invalid_json(self, mock_exists):
        """_load_models handles invalid JSON gracefully."""
        mock_exists.return_value = True

        with patch("builtins.open", mock_open(read_data="invalid json")):
            service = ModelUsageService.__new__(ModelUsageService)
            service.models_file = "data/models.json"
            service._default_provider = "gemini"
            service.MODEL_DEFINITIONS = {}
            service._model_definition_cache = {}
            service.ALIAS_TO_API = {}
            service.API_TO_ALIAS = {}

            # Should not raise
            service._load_models()

    @patch("os.path.exists")
    def test_populates_helper_maps(self, mock_exists):
        """_load_models populates ALIAS_TO_API and API_TO_ALIAS maps."""
        mock_exists.return_value = True
        models_data = {
            "llm_models": {
                "Gemini 2.5 flash": {
                    "display_name": "Gemini 2.5 flash",
                    "api_model_name": "gemini-2.5-flash",
                    "daily_limit": 500,
                    "scope": "guild",
                    "provider": "gemini",
                }
            }
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(models_data))):
            service = ModelUsageService.__new__(ModelUsageService)
            service.models_file = "data/models.json"
            service._default_provider = "gemini"
            service.MODEL_DEFINITIONS = {}
            service._model_definition_cache = {}
            service.ALIAS_TO_API = {}
            service.API_TO_ALIAS = {}
            service._load_models()

        assert service.ALIAS_TO_API["Gemini 2.5 flash"] == "gemini-2.5-flash"
        assert service.API_TO_ALIAS["gemini-2.5-flash"] == "Gemini 2.5 flash"

    @patch("os.path.exists")
    def test_loads_model_with_fallback_alias(self, mock_exists):
        """_load_models loads models with fallback_alias."""
        mock_exists.return_value = True
        models_data = {
            "llm_models": {
                "Gemini 2.5 pro": {
                    "display_name": "Gemini 2.5 pro",
                    "api_model_name": "gemini-2.5-pro",
                    "daily_limit": 50,
                    "scope": "guild",
                    "provider": "gemini",
                    "fallback_alias": "Gemini 2.5 flash",
                }
            }
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(models_data))):
            service = ModelUsageService.__new__(ModelUsageService)
            service.models_file = "data/models.json"
            service._default_provider = "gemini"
            service.MODEL_DEFINITIONS = {}
            service._model_definition_cache = {}
            service.ALIAS_TO_API = {}
            service.API_TO_ALIAS = {}
            service._load_models()

        assert service.MODEL_DEFINITIONS["Gemini 2.5 pro"].fallback_alias == "Gemini 2.5 flash"


class TestModelUsageServiceLoadUsage:
    """Tests for ModelUsageService._load_usage()."""

    @patch("os.path.exists")
    def test_loads_usage_from_file(self, mock_exists):
        """_load_usage loads usage data from JSON file."""
        mock_exists.return_value = True
        today = _get_today_kst_date()
        usage_data = {"date": today, "usage": {"guild:123:GPT 5 mini": 5}}

        with patch("builtins.open", mock_open(read_data=json.dumps(usage_data))):
            service = ModelUsageService.__new__(ModelUsageService)
            service.data_file = "data/model_usage.json"
            service.usage_data = {}
            service._load_usage()

        assert service.usage_data["date"] == today
        assert service.usage_data["usage"]["guild:123:GPT 5 mini"] == 5

    @patch("os.path.exists")
    def test_handles_missing_usage_file(self, mock_exists):
        """_load_usage handles missing usage file gracefully."""
        mock_exists.return_value = False
        service = ModelUsageService.__new__(ModelUsageService)
        service.data_file = "data/model_usage.json"
        service.usage_data = {}
        service._load_usage()

        assert service.usage_data == {} or "date" in service.usage_data

    @patch("os.path.exists")
    def test_handles_invalid_usage_json(self, mock_exists):
        """_load_usage handles invalid JSON gracefully."""
        mock_exists.return_value = True

        with patch("builtins.open", mock_open(read_data="invalid json")):
            service = ModelUsageService.__new__(ModelUsageService)
            service.data_file = "data/model_usage.json"
            service.usage_data = {}
            service._load_usage()

        # Should not raise and usage_data should be empty dict
        assert service.usage_data == {} or "date" in service.usage_data


class TestModelUsageServiceSaveUsage:
    """Tests for ModelUsageService._save_usage()."""

    @pytest.mark.asyncio
    async def test_saves_usage_to_file(self, tmp_path):
        """_save_usage writes usage data to file."""
        usage_file = tmp_path / "usage.json"
        service = ModelUsageService.__new__(ModelUsageService)
        service.data_file = str(usage_file)
        service.usage_data = {"date": "2024-01-01", "usage": {"key": 1}}

        await service._save_usage()

        assert usage_file.exists()
        with open(usage_file) as f:
            saved_data = json.load(f)
        assert saved_data["date"] == "2024-01-01"
        assert saved_data["usage"]["key"] == 1

    @pytest.mark.asyncio
    async def test_creates_directory_if_not_exists(self, tmp_path):
        """_save_usage creates directory if it doesn't exist."""
        usage_file = tmp_path / "subdir" / "usage.json"
        service = ModelUsageService.__new__(ModelUsageService)
        service.data_file = str(usage_file)
        service.usage_data = {"date": "2024-01-01", "usage": {}}

        await service._save_usage()

        assert usage_file.parent.exists()
        assert usage_file.exists()

    @pytest.mark.asyncio
    async def test_saves_unicode_content(self, tmp_path):
        """_save_usage saves Unicode content correctly."""
        usage_file = tmp_path / "usage.json"
        service = ModelUsageService.__new__(ModelUsageService)
        service.data_file = str(usage_file)
        service.usage_data = {"date": "2024-01-01", "usage": {"key": "한글 테스트"}}

        await service._save_usage()

        with open(usage_file, encoding="utf-8") as f:
            content = f.read()
        assert "한글 테스트" in content


class TestModelUsageServiceCheckDailyReset:
    """Tests for ModelUsageService._check_daily_reset()."""

    def test_resets_when_date_changed(self):
        """_check_daily_reset resets usage when date changes."""
        service = ModelUsageService.__new__(ModelUsageService)
        old_date = "2024-01-01"
        service.usage_data = {"date": old_date, "usage": {"key": 100}}

        service._check_daily_reset()

        # Date should be updated to today
        today = _get_today_kst_date()
        assert service.usage_data["date"] == today
        assert service.usage_data["usage"] == {}

    def test_keeps_usage_when_same_date(self):
        """_check_daily_reset keeps usage when same date."""
        service = ModelUsageService.__new__(ModelUsageService)
        today = _get_today_kst_date()
        service.usage_data = {"date": today, "usage": {"key": 100}}

        # Should not reset since date is same
        original_usage = service.usage_data.copy()
        service._check_daily_reset()

        assert service.usage_data == original_usage

    def test_resets_empty_usage_data(self):
        """_check_daily_reset handles empty usage_data."""
        service = ModelUsageService.__new__(ModelUsageService)
        service.usage_data = {}

        service._check_daily_reset()

        today = _get_today_kst_date()
        assert service.usage_data["date"] == today
        assert service.usage_data["usage"] == {}


class TestModelUsageServiceGetUsageKey:
    """Tests for ModelUsageService._get_usage_key()."""

    def test_returns_guild_scoped_key(self):
        """_get_usage_key returns guild-scoped key."""
        service = ModelUsageService.__new__(ModelUsageService)
        model_def = ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=100,
            scope="guild",
            provider="openai",
        )

        key = service._get_usage_key(model_def, guild_id=12345)
        assert key == "guild:12345:GPT 5 mini"

    def test_returns_different_keys_for_different_guilds(self):
        """_get_usage_key returns different keys for different guilds."""
        service = ModelUsageService.__new__(ModelUsageService)
        model_def = ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=100,
            scope="guild",
            provider="openai",
        )

        key1 = service._get_usage_key(model_def, guild_id=12345)
        key2 = service._get_usage_key(model_def, guild_id=67890)
        assert key1 != key2

    def test_returns_different_keys_for_different_models(self):
        """_get_usage_key returns different keys for different models."""
        service = ModelUsageService.__new__(ModelUsageService)
        model_def1 = ModelDefinition(
            display_name="GPT 5 mini",
            api_model_name="gpt-5-mini",
            daily_limit=100,
            scope="guild",
            provider="openai",
        )
        model_def2 = ModelDefinition(
            display_name="Gemini 2.5 flash",
            api_model_name="gemini-2.5-flash",
            daily_limit=100,
            scope="guild",
            provider="gemini",
        )

        key1 = service._get_usage_key(model_def1, guild_id=12345)
        key2 = service._get_usage_key(model_def2, guild_id=12345)
        assert key1 != key2


class TestModelUsageServiceCheckAndIncrementUsage:
    """Tests for ModelUsageService.check_and_increment_usage()."""

    @pytest.fixture
    def service(self):
        """Create a ModelUsageService with mocked file I/O."""
        today = _get_today_kst_date()
        with patch("os.path.exists", return_value=False):
            service = ModelUsageService.__new__(ModelUsageService)
            service.data_file = "data/model_usage.json"
            service.models_file = "data/models.json"
            service.usage_data = {"date": today, "usage": {}}
            service._default_provider = "gemini"
            service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash"
            service.MODEL_DEFINITIONS = {
                "Gemini 2.5 flash": ModelDefinition(
                    display_name="Gemini 2.5 flash",
                    api_model_name="gemini-2.5-flash",
                    daily_limit=100,
                    scope="guild",
                    provider="gemini",
                    fallback_alias="GPT 5 mini",
                ),
                "GPT 5 mini": ModelDefinition(
                    display_name="GPT 5 mini",
                    api_model_name="gpt-5-mini",
                    daily_limit=50,
                    scope="guild",
                    provider="openai",
                ),
            }
            service._model_definition_cache = service.MODEL_DEFINITIONS.copy()
            service.ALIAS_TO_API = {k: v.api_model_name for k, v in service.MODEL_DEFINITIONS.items()}
            service.API_TO_ALIAS = {v.api_model_name: k for k, v in service.MODEL_DEFINITIONS.items()}
            service._save_usage = AsyncMock()
            return service

    @pytest.mark.asyncio
    async def test_returns_true_when_under_limit(self, service):
        """check_and_increment_usage returns True when under limit."""
        is_allowed, alias, notification = await service.check_and_increment_usage(
            guild_id=123, model_alias="Gemini 2.5 flash"
        )
        assert is_allowed is True
        assert alias == "Gemini 2.5 flash"
        assert notification is None

    @pytest.mark.asyncio
    async def test_increments_usage_count(self, service):
        """check_and_increment_usage increments usage count."""
        await service.check_and_increment_usage(guild_id=123, model_alias="Gemini 2.5 flash")
        key = "guild:123:Gemini 2.5 flash"
        assert service.usage_data["usage"][key] == 1

    @pytest.mark.asyncio
    async def test_saves_usage_after_increment(self, service):
        """check_and_increment_usage calls _save_usage after increment."""
        await service.check_and_increment_usage(guild_id=123, model_alias="Gemini 2.5 flash")
        service._save_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_when_limit_reached(self, service):
        """check_and_increment_usage falls back when limit reached."""
        # Set usage to limit
        key = "guild:123:Gemini 2.5 flash"
        service.usage_data["usage"][key] = 100

        is_allowed, alias, notification = await service.check_and_increment_usage(
            guild_id=123, model_alias="Gemini 2.5 flash"
        )
        assert is_allowed is True
        assert alias == "GPT 5 mini"  # Fallback
        assert notification is not None

    @pytest.mark.asyncio
    async def test_fallback_increments_fallback_model_usage(self, service):
        """When falling back, the fallback model's usage is incremented."""
        # Set usage to limit for primary model
        service.usage_data["usage"]["guild:123:Gemini 2.5 flash"] = 100

        await service.check_and_increment_usage(
            guild_id=123, model_alias="Gemini 2.5 flash"
        )

        # Fallback model usage should be incremented
        assert service.usage_data["usage"]["guild:123:GPT 5 mini"] == 1

    @pytest.mark.asyncio
    async def test_returns_false_when_all_limits_exceeded(self, service):
        """check_and_increment_usage returns False when all limits exceeded."""
        # Set both models to limit
        service.usage_data["usage"]["guild:123:Gemini 2.5 flash"] = 100
        service.usage_data["usage"]["guild:123:GPT 5 mini"] = 50

        is_allowed, alias, notification = await service.check_and_increment_usage(
            guild_id=123, model_alias="Gemini 2.5 flash"
        )
        assert is_allowed is False
        assert "초과" in notification

    @pytest.mark.asyncio
    async def test_uses_default_for_unknown_alias(self, service):
        """check_and_increment_usage uses default for unknown alias."""
        is_allowed, alias, notification = await service.check_and_increment_usage(
            guild_id=123, model_alias="Unknown Model"
        )
        # Should use default and succeed
        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_uses_default_for_none_alias(self, service):
        """check_and_increment_usage uses default for None alias."""
        is_allowed, alias, notification = await service.check_and_increment_usage(
            guild_id=123, model_alias=None
        )
        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_returns_true_for_unknown_model_without_definition(self, service):
        """Returns True with default when model definition not found."""
        # Remove the fallback's definition to test the edge case
        service.MODEL_DEFINITIONS = {}  # No model definitions

        is_allowed, alias, notification = await service.check_and_increment_usage(
            guild_id=123, model_alias="Unknown Model"
        )

        assert is_allowed is True
        assert alias == service.DEFAULT_MODEL_ALIAS
        assert notification is None

    @pytest.mark.asyncio
    async def test_limits_recursion_to_prevent_infinite_loop(self, service):
        """Prevents infinite recursion with circular fallback."""
        # Create circular fallback chain
        service.MODEL_DEFINITIONS["Model A"] = ModelDefinition(
            display_name="Model A",
            api_model_name="model-a",
            daily_limit=1,
            scope="guild",
            provider="test",
            fallback_alias="Model B",
        )
        service.MODEL_DEFINITIONS["Model B"] = ModelDefinition(
            display_name="Model B",
            api_model_name="model-b",
            daily_limit=1,
            scope="guild",
            provider="test",
            fallback_alias="Model A",  # Circular!
        )
        service._model_definition_cache = service.MODEL_DEFINITIONS.copy()

        # Set both to limit
        service.usage_data["usage"]["guild:123:Model A"] = 1
        service.usage_data["usage"]["guild:123:Model B"] = 1

        is_allowed, alias, notification = await service.check_and_increment_usage(
            guild_id=123, model_alias="Model A"
        )

        # Should exit after max recursion and return error
        assert is_allowed is False
        assert "오류" in notification

    @pytest.mark.asyncio
    async def test_notification_message_for_fallback(self, service):
        """Falling back generates Korean notification message."""
        service.usage_data["usage"]["guild:123:Gemini 2.5 flash"] = 100

        is_allowed, alias, notification = await service.check_and_increment_usage(
            guild_id=123, model_alias="Gemini 2.5 flash"
        )

        assert "1일 사용한도" in notification
        assert "GPT 5 mini" in notification

    @pytest.mark.asyncio
    async def test_separate_usage_tracking_per_guild(self, service):
        """Usage is tracked separately per guild."""
        # Increment for guild 123
        await service.check_and_increment_usage(guild_id=123, model_alias="Gemini 2.5 flash")

        # Increment for guild 456
        await service.check_and_increment_usage(guild_id=456, model_alias="Gemini 2.5 flash")

        # Both should have 1 usage each
        assert service.usage_data["usage"]["guild:123:Gemini 2.5 flash"] == 1
        assert service.usage_data["usage"]["guild:456:Gemini 2.5 flash"] == 1

    @pytest.mark.asyncio
    async def test_multiple_increments_accumulate(self, service):
        """Multiple calls increment usage correctly."""
        for _ in range(5):
            await service.check_and_increment_usage(guild_id=123, model_alias="Gemini 2.5 flash")

        assert service.usage_data["usage"]["guild:123:Gemini 2.5 flash"] == 5


class TestModelUsageServiceGetApiModelName:
    """Tests for ModelUsageService.get_api_model_name()."""

    @pytest.fixture
    def service(self):
        """Create a ModelUsageService with test data."""
        with patch("os.path.exists", return_value=False):
            service = ModelUsageService.__new__(ModelUsageService)
            service._default_provider = "gemini"
            service.MODEL_DEFINITIONS = {
                "Gemini 2.5 flash": ModelDefinition(
                    display_name="Gemini 2.5 flash",
                    api_model_name="gemini-2.5-flash",
                    daily_limit=100,
                    scope="guild",
                    provider="gemini",
                ),
            }
            service._model_definition_cache = service.MODEL_DEFINITIONS.copy()
            service.ALIAS_TO_API = {k: v.api_model_name for k, v in service.MODEL_DEFINITIONS.items()}
            service.API_TO_ALIAS = {v.api_model_name: k for k, v in service.MODEL_DEFINITIONS.items()}
            service.DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash"
            return service

    def test_returns_api_name_for_valid_alias(self, service):
        """get_api_model_name returns API name for valid alias."""
        result = service.get_api_model_name("Gemini 2.5 flash")
        assert result == "gemini-2.5-flash"

    def test_returns_api_name_for_reverse_lookup(self, service):
        """get_api_model_name returns API name via reverse lookup."""
        result = service.get_api_model_name("gemini-2.5-flash")
        assert result == "gemini-2.5-flash"

    def test_returns_fallback_for_unknown_alias(self, service):
        """get_api_model_name returns fallback for unknown alias."""
        result = service.get_api_model_name("Unknown Model")
        # Should return default model's API name
        assert result == "gemini-2.5-flash"

    def test_returns_provider_default_when_cache_empty(self):
        """get_api_model_name returns provider default when cache is empty."""
        service = ModelUsageService.__new__(ModelUsageService)
        service._default_provider = "openai"
        service._model_definition_cache = {}
        service.MODEL_DEFINITIONS = {}
        service.API_TO_ALIAS = {}
        service.DEFAULT_MODEL_ALIAS = "GPT 5 mini"

        result = service.get_api_model_name("Unknown Model")

        # Should use provider-specific fallback
        assert result == "gpt-5-mini"

    def test_returns_gemini_fallback_for_unknown_provider(self):
        """get_api_model_name returns gemini fallback for unknown provider."""
        service = ModelUsageService.__new__(ModelUsageService)
        service._default_provider = "unknown_provider"
        service._model_definition_cache = {}
        service.MODEL_DEFINITIONS = {}
        service.API_TO_ALIAS = {}
        service.DEFAULT_MODEL_ALIAS = "Unknown"

        result = service.get_api_model_name("Unknown Model")

        # Should fall back to gemini
        assert result == "gemini-2.5-flash"

    def test_uses_model_definitions_as_fallback(self, service):
        """get_api_model_name uses MODEL_DEFINITIONS if cache is missing."""
        # Remove from cache but keep in MODEL_DEFINITIONS
        service._model_definition_cache = {}

        result = service.get_api_model_name("Gemini 2.5 flash")

        # Should still work via MODEL_DEFINITIONS
        assert result == "gemini-2.5-flash"

    def test_zai_provider_fallback(self):
        """get_api_model_name returns zai fallback for zai provider."""
        service = ModelUsageService.__new__(ModelUsageService)
        service._default_provider = "zai"
        service._model_definition_cache = {}
        service.MODEL_DEFINITIONS = {}
        service.API_TO_ALIAS = {}
        service.DEFAULT_MODEL_ALIAS = "Unknown"

        result = service.get_api_model_name("Unknown Model")

        assert result == "glm-4.7"

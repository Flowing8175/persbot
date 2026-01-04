
import json
import logging
import os
import datetime
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class ModelDefinition:
    display_name: str
    api_model_name: str
    daily_limit: int
    scope: str  # 'user', 'channel', or 'guild'
    provider: str # 'gemini' or 'openai'
    fallback_alias: Optional[str] = None

class ModelUsageService:
    """Service to track and enforce daily usage limits for LLM models."""

    # Define the available models and their constraints
    MODEL_DEFINITIONS: Dict[str, ModelDefinition] = {
        "Gemini 3 flash": ModelDefinition(
            display_name="Gemini 3 flash",
            api_model_name="gemini-3-flash",
            daily_limit=20,
            scope="guild",
            provider="gemini",
            fallback_alias="Gemini 2.5 flash"
        ),
        "Gemini 2.5 flash": ModelDefinition(
            display_name="Gemini 2.5 flash",
            api_model_name="gemini-2.5-flash",
            daily_limit=30,
            scope="guild",
            provider="gemini",
            fallback_alias="Gemini 2.0 flash"
        ),
        "Gemini 2.5 flash lite": ModelDefinition(
            display_name="Gemini 2.5 flash lite",
            api_model_name="gemini-2.5-flash-lite",
            daily_limit=50,
            scope="guild",
            provider="gemini",
            fallback_alias=None
        ),
        "Gemini 2.0 flash lite": ModelDefinition(
            display_name="Gemini 2.0 flash lite",
            api_model_name="gemini-2.0-flash-lite",
            daily_limit=100,
            scope="guild",
            provider="gemini",
            fallback_alias="Gemini 2.5 flash lite"
        ),
        "GPT-4.1 mini": ModelDefinition(
            display_name="GPT-4.1 mini",
            api_model_name="gpt-4.1-mini",
            daily_limit=20,
            scope="guild",
            provider="openai",
            fallback_alias="GPT-4.1 nano"
        ),
        "GPT-4.1 nano": ModelDefinition(
            display_name="GPT-4.1 nano",
            api_model_name="gpt-4.1-nano",
            daily_limit=30,
            scope="guild",
            provider="openai",
            fallback_alias=None
        ),
    }

    # Helper maps
    ALIAS_TO_API: Dict[str, str] = {k: v.api_model_name for k, v in MODEL_DEFINITIONS.items()}
    DEFAULT_MODEL_ALIAS = "Gemini 2.5 flash lite"

    def __init__(self, data_file: str = "data/model_usage.json"):
        self.data_file = data_file
        self.usage_data: Dict[str, Any] = {}
        self._load_usage()

    def _load_usage(self):
        """Load usage data from file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.usage_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model usage data: {e}")
                self.usage_data = {}
        else:
            self.usage_data = {}

        # Check date reset
        self._check_daily_reset()

    async def _save_usage(self):
        """Save usage data to file asynchronously."""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            async with aiofiles.open(self.data_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.usage_data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to save model usage data: {e}")

    def _check_daily_reset(self):
        """Reset usage if the date has changed (KST Midnight)."""
        # KST is UTC+9
        now_kst = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
        today_str = now_kst.strftime("%Y-%m-%d")

        if self.usage_data.get("date") != today_str:
            logger.info(f"Resetting model usage stats for new day: {today_str}")
            self.usage_data = {
                "date": today_str,
                "usage": {}
            }

    def _get_usage_key(self, model_def: ModelDefinition, guild_id: int) -> str:
        """Generate key based on scope."""
        # Now everything is guild scope as requested
        return f"guild:{guild_id}:{model_def.display_name}"

    async def check_and_increment_usage(self, guild_id: int, model_alias: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        """
        Check if usage is within limits. If yes, increment.
        If no, switch to fallback and recurse.

        Returns:
            (success: bool, final_model_alias: str, notification_message: Optional[str])
        """
        self._check_daily_reset()

        if not model_alias or model_alias not in self.MODEL_DEFINITIONS:
             model_alias = self.DEFAULT_MODEL_ALIAS

        current_alias = model_alias
        notification = None

        # Limit recursion loop to avoid infinite fallback
        for _ in range(5):
            model_def = self.MODEL_DEFINITIONS.get(current_alias)
            if not model_def:
                return True, self.DEFAULT_MODEL_ALIAS, None

            usage_key = self._get_usage_key(model_def, guild_id)
            current_usage = self.usage_data.get("usage", {}).get(usage_key, 0)

            if current_usage < model_def.daily_limit:
                # Increment usage
                if "usage" not in self.usage_data:
                    self.usage_data["usage"] = {}

                self.usage_data["usage"][usage_key] = current_usage + 1
                await self._save_usage()

                return True, current_alias, notification
            else:
                # Limit reached
                if model_def.fallback_alias:
                    logger.info(f"Usage limit reached for {current_alias} (Key: {usage_key}). Falling back to {model_def.fallback_alias}.")

                    if notification is None:
                        notification = f"1일 사용한도에 도달하여 🔄 {model_def.fallback_alias}모델로 변경되었습니다."
                    else:
                        notification = f"1일 사용한도에 도달하여 🔄 {model_def.fallback_alias}모델로 변경되었습니다."

                    current_alias = model_def.fallback_alias
                    continue
                else:
                    return False, current_alias, "❌ 금일 사용 가능한 모든 모델의 한도를 초과했습니다."

        return False, current_alias, "❌ 모델 전환 오류."

    def get_api_model_name(self, model_alias: str) -> str:
        def_obj = self.MODEL_DEFINITIONS.get(model_alias)
        if def_obj:
            return def_obj.api_model_name
        return "gemini-2.5-flash-lite" # Safe default (lite)

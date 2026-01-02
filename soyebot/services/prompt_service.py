import json
import os
import logging
import datetime
from typing import List, Dict, Optional

from soyebot.prompts import BOT_PERSONA_PROMPT, SUMMARY_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

class PromptService:
    def __init__(self, storage_path: str = "custom_prompts.json", usage_path: str = "prompt_usage.json"):
        self.storage_path = storage_path
        self.usage_path = usage_path
        self.prompts: List[Dict[str, str]] = []
        self.usage_data: Dict[str, Dict[str, int]] = {} # { "date": { "user_id": count } }
        self._load()
        self._load_usage()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.prompts = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load prompts: {e}")
                self.prompts = []

        # If prompts list is empty (either file missing or empty file/load error), use default
        if not self.prompts:
            self.prompts = [
                {"name": "기본값", "content": BOT_PERSONA_PROMPT}
            ]
            self._save()

    def _save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.prompts, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save prompts: {e}")

    def _load_usage(self):
        if os.path.exists(self.usage_path):
            try:
                with open(self.usage_path, "r", encoding="utf-8") as f:
                    self.usage_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load prompt usage: {e}")
                self.usage_data = {}
        else:
            self.usage_data = {}

    def _save_usage(self):
        try:
            with open(self.usage_path, "w", encoding="utf-8") as f:
                json.dump(self.usage_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save prompt usage: {e}")

    def _get_today_key(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d")

    def check_today_limit(self, user_id: int, limit: int = 2) -> bool:
        """Check if the user has reached their daily limit."""
        today = self._get_today_key()

        # Reset/Initialize daily data if needed (cleans up old data implicitly by accessing new key)
        # To avoid infinite growth, we could clear keys != today, but let's keep it simple for now.
        # Actually, for privacy/size, let's keep only the last few days or just today?
        # Let's just create today's entry if missing.
        if today not in self.usage_data:
            self.usage_data = {today: {}} # Simple reset strategy: Keep only today?
            # Or if we want to keep history, just add. But to prevent bloat, let's clear old data occasionally.
            # For simplicity in this request: Reset to today-only if today is missing (lazy reset)
            # This effectively clears history whenever the bot runs on a new day and someone creates a prompt.
            # Wait, if we reload, we might lose history? No, we loaded from file.
            # Let's just perform a cleanup of keys that are not today.
            self.usage_data = {k: v for k, v in self.usage_data.items() if k == today}
            self.usage_data.setdefault(today, {})
            self._save_usage()

        user_count = self.usage_data[today].get(str(user_id), 0)
        return user_count < limit

    def increment_today_usage(self, user_id: int):
        """Increment the usage count for the user."""
        today = self._get_today_key()
        if today not in self.usage_data:
            self.usage_data[today] = {}

        user_str = str(user_id)
        self.usage_data[today][user_str] = self.usage_data[today].get(user_str, 0) + 1
        self._save_usage()

    def add_prompt(self, name: str, content: str) -> int:
        self.prompts.append({"name": name, "content": content})
        self._save()
        return len(self.prompts) - 1

    def list_prompts(self) -> List[Dict[str, str]]:
        return self.prompts

    def get_prompt(self, index: int) -> Optional[Dict[str, str]]:
        if 0 <= index < len(self.prompts):
            return self.prompts[index]
        return None

    def get_active_assistant_prompt(self) -> str:
        """Returns the content of the currently active assistant prompt (default: index 0)."""
        if self.prompts:
            return self.prompts[0].get("content", BOT_PERSONA_PROMPT)
        return BOT_PERSONA_PROMPT

    def get_summary_prompt(self) -> str:
        """Returns the system instruction for the summarizer."""
        return SUMMARY_SYSTEM_INSTRUCTION

    def rename_prompt(self, index: int, new_name: str) -> bool:
        if 0 <= index < len(self.prompts):
            self.prompts[index]["name"] = new_name
            self._save()
            return True
        return False

    def delete_prompt(self, index: int) -> bool:
        if 0 <= index < len(self.prompts):
            self.prompts.pop(index)
            self._save()
            return True
        return False

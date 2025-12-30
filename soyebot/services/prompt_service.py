import json
import os
import logging
from typing import List, Dict, Optional

from soyebot.prompts import BOT_PERSONA_PROMPT, SUMMARY_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

class PromptService:
    def __init__(self, storage_path: str = "custom_prompts.json"):
        self.storage_path = storage_path
        self.prompts: List[Dict[str, str]] = []
        self._load()

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

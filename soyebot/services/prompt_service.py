import json
import os
import logging
from typing import List, Dict, Optional

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
        else:
            # Add default prompt if empty
            self.prompts = [
                {"name": "Default (백진우)", "content": "Default assistant prompt"} 
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

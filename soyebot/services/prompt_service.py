import json
import os
import logging
import datetime
import os
from typing import List, Dict, Optional
import asyncio

import aiofiles
from soyebot.prompts import BOT_PERSONA_PROMPT, SUMMARY_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

class PromptService:
    def __init__(self, storage_path: str = "custom_prompts.json", usage_path: str = "prompt_usage.json"):
        self.storage_path = storage_path
        self.usage_path = usage_path
        self.prompts: List[Dict[str, str]] = []
        self.usage_data: Dict[str, Dict[str, int]] = {} # { "date": { "user_id": count } }
        # NOTE: Sync load is unavoidable in __init__ if we want immediate availability.
        # However, we can use async init pattern or just accept sync read at startup.
        # Given the task requirement, we will implement async methods for runtime ops.
        # For startup, we'll keep a sync fallback or use asyncio.run (bad practice inside loop)
        # or we accept that __init__ does sync I/O once.
        # But wait, the previous code did sync I/O in __init__.
        # We'll defer loading to an async setup or use sync I/O only for init.
        # Let's provide async methods for saving/updating.
        self._load_sync()
        self._load_usage_sync()

    def _load_sync(self):
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
            # We can't await here in init, so we just set it.
            # We will save later or do sync save.
            try:
                 with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump(self.prompts, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.error(f"Failed to save default prompts: {e}")

    async def _save(self):
        try:
            async with aiofiles.open(self.storage_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(self.prompts, ensure_ascii=False, indent=4))
        except Exception as e:
            logger.error(f"Failed to save prompts: {e}")

    def _load_usage_sync(self):
        if os.path.exists(self.usage_path):
            try:
                with open(self.usage_path, "r", encoding="utf-8") as f:
                    self.usage_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load prompt usage: {e}")
                self.usage_data = {}
        else:
            self.usage_data = {}

    async def _save_usage(self):
        try:
            async with aiofiles.open(self.usage_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(self.usage_data, ensure_ascii=False, indent=4))
        except Exception as e:
            logger.error(f"Failed to save prompt usage: {e}")

    def _get_today_key(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d")

    async def check_today_limit(self, user_id: int, limit: int = 2) -> bool:
        """Check if the user has reached their daily limit."""
        today = self._get_today_key()

        if today not in self.usage_data:
            self.usage_data = {k: v for k, v in self.usage_data.items() if k == today}
            self.usage_data.setdefault(today, {})
            await self._save_usage()

        user_count = self.usage_data[today].get(str(user_id), 0)
        return user_count < limit

    async def increment_today_usage(self, user_id: int):
        """Increment the usage count for the user."""
        today = self._get_today_key()
        if today not in self.usage_data:
            self.usage_data[today] = {}

        user_str = str(user_id)
        self.usage_data[today][user_str] = self.usage_data[today].get(user_str, 0) + 1
        await self._save_usage()

    async def add_prompt(self, name: str, content: str) -> int:
        self.prompts.append({"name": name, "content": content})
        await self._save()
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

    async def rename_prompt(self, index: int, new_name: str) -> bool:
        if 0 <= index < len(self.prompts):
            self.prompts[index]["name"] = new_name
            await self._save()
            return True
        return False

    async def delete_prompt(self, index: int) -> bool:
        if 0 <= index < len(self.prompts):
            self.prompts.pop(index)
            await self._save()
            return True
        return False

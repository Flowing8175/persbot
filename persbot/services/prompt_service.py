import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles

from persbot.prompts import BOT_PERSONA_PROMPT, SUMMARY_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


class PromptService:
    def __init__(
        self, prompt_dir: str = "persbot/assets", usage_path: str = "prompt_usage.json"
    ) -> None:
        self.prompt_dir = Path(prompt_dir)
        self.usage_path = usage_path
        self.prompts: List[Dict[str, str]] = []
        self.usage_data: Dict[str, Dict[str, int]] = {}  # { "date": { "user_id": count } }

        # Ensure directory exists
        self.prompt_dir.mkdir(parents=True, exist_ok=True)

        # Load initial state synchronously for startup availability
        self._load_sync()
        self._load_usage_sync()

    def _load_sync(self) -> None:
        """Scans the assets directory for .md files and populates self.prompts."""
        self.prompts = []

        # Always check for default 'persona.md' or similar.
        # If directory is empty, we might want to create a default?
        # The instruction said "use assets/*.md".
        # We sort to ensure consistent indexing.
        md_files = sorted(self.prompt_dir.glob("*.md"))

        for file_path in md_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    name = file_path.stem  # Filename without extension
                    self.prompts.append({"name": name, "content": content, "path": str(file_path)})
            except Exception:
                logger.exception("Failed to load prompt from %s", file_path)

        # Fallback if no prompts found (though persona.md should exist)
        if not self.prompts:
            logger.warning("No .md prompts found in assets. Using default fallback.")
            self.prompts = [{"name": "기본값", "content": BOT_PERSONA_PROMPT, "path": ""}]

    async def _reload(self):
        """Asynchronous reload of prompts (for after modifications)."""
        # For simplicity, we can reuse the logic but with async file reading?
        # Or just update the list in memory if we are confident.
        # But scanning is safer for consistency.
        # Since glob is sync, we can run it in thread if many files,
        # but for a few files, it's fast. Reading content async is better.

        # Re-implement scan logic with aiofiles
        new_prompts = []
        md_files = sorted(self.prompt_dir.glob("*.md"))

        for file_path in md_files:
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    name = file_path.stem
                    new_prompts.append({"name": name, "content": content, "path": str(file_path)})
            except Exception:
                logger.exception("Failed to load prompt from %s", file_path)

        if new_prompts:
            self.prompts = new_prompts
        elif not self.prompts:  # Only fallback if absolutely nothing and prev list empty
            self.prompts = [{"name": "기본값", "content": BOT_PERSONA_PROMPT, "path": ""}]

    def _load_usage_sync(self) -> None:
        if os.path.exists(self.usage_path):
            try:
                with open(self.usage_path, "r", encoding="utf-8") as f:
                    self.usage_data = json.load(f)
            except Exception:
                logger.exception("Failed to load prompt usage")
                self.usage_data = {}
        else:
            self.usage_data = {}

    async def _save_usage(self) -> None:
        try:
            async with aiofiles.open(self.usage_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(self.usage_data, ensure_ascii=False, indent=4))
        except Exception:
            logger.exception("Failed to save prompt usage")

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

    async def increment_today_usage(self, user_id: int) -> None:
        """Increment the usage count for the user."""
        today = self._get_today_key()
        if today not in self.usage_data:
            self.usage_data[today] = {}

        user_str = str(user_id)
        self.usage_data[today][user_str] = self.usage_data[today].get(user_str, 0) + 1
        await self._save_usage()

    def _sanitize_filename(self, name: str) -> str:
        # Simple sanitization - keep spaces for display
        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
        return safe_name or "untitled"

    async def add_prompt(self, name: str, content: str) -> int:
        safe_name = self._sanitize_filename(name)
        file_path = self.prompt_dir / f"{safe_name}.md"

        # Avoid overwrite collision if possible? Or overwrite?
        # User said "add prompt", implying new.
        # If exists, maybe append suffix?
        counter = 1
        original_safe_name = safe_name
        while file_path.exists():
            safe_name = f"{original_safe_name}_{counter}"
            file_path = self.prompt_dir / f"{safe_name}.md"
            counter += 1

        try:
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

            await self._reload()

            # Find the index of the newly added prompt
            for i, p in enumerate(self.prompts):
                if p["path"] == str(file_path):
                    return i
            return len(self.prompts) - 1  # Fallback

        except Exception:
            logger.exception("Failed to add prompt file %s", file_path)
            return -1

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
        if not (0 <= index < len(self.prompts)):
            return False

        old_prompt = self.prompts[index]
        old_path = Path(old_prompt.get("path", ""))

        if not old_path.exists():
            logger.error(f"Cannot rename: File {old_path} does not exist.")
            return False

        safe_new_name = self._sanitize_filename(new_name)
        new_path = self.prompt_dir / f"{safe_new_name}.md"

        if new_path.exists():
            logger.warning(f"Cannot rename: Target {new_path} already exists.")
            return False

        try:
            # Async rename not strictly available in aiofiles/os, use os.rename or shutil.move
            # os.rename is atomic on POSIX, usually fine to call synchronously.
            # Or use asyncio.to_thread if worried about disk I/O blocking.
            await asyncio.to_thread(os.rename, old_path, new_path)
            await self._reload()
            return True
        except Exception:
            logger.exception("Failed to rename prompt")
            return False

    async def delete_prompt(self, index: int) -> bool:
        if not (0 <= index < len(self.prompts)):
            return False

        target = self.prompts[index]
        path = Path(target.get("path", ""))

        if not path.exists():
            # If path is empty (default fallback) or missing, can't delete file.
            # But we can remove from list if it was a ghost entry.
            logger.warning(f"File {path} not found for deletion.")
            return False

        try:
            await asyncio.to_thread(os.remove, path)
            await self._reload()
            return True
        except Exception:
            logger.exception("Failed to delete prompt file")
            return False

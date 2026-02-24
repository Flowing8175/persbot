"""Channel-specific prompt persistence service."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

from persbot.prompts import BOT_PERSONA_PROMPT

logger = logging.getLogger(__name__)


class ChannelPromptService:
    """Manages per-channel prompts persisted to disk.

    Prompts are stored as: data/<channel_id>/prompt.md
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self._cache: Dict[int, str] = {}  # channel_id -> prompt_content
        self._load_all_channel_prompts()

    def _load_all_channel_prompts(self) -> None:
        """Load all channel prompts from disk on startup."""
        if not self.data_dir.exists():
            logger.info("Data directory does not exist, creating it")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return

        # Find all prompt.md files in subdirectories
        for channel_dir in self.data_dir.iterdir():
            if channel_dir.is_dir() and channel_dir.name.isdigit():
                prompt_file = channel_dir / "prompt.md"
                if prompt_file.exists():
                    try:
                        content = prompt_file.read_text(encoding="utf-8")
                        channel_id = int(channel_dir.name)
                        self._cache[channel_id] = content
                        logger.info(f"Loaded prompt for channel {channel_id}")
                    except Exception:
                        logger.exception(f"Failed to load prompt from {prompt_file}")

    def get_channel_prompt(self, channel_id: int) -> Optional[str]:
        """Get the prompt for a specific channel."""
        return self._cache.get(channel_id)

    def get_all_channel_prompts(self) -> Dict[int, str]:
        """Get all channel prompts."""
        return self._cache.copy()

    async def set_channel_prompt(self, channel_id: int, prompt_content: Optional[str]) -> None:
        """Set and persist a prompt for a specific channel.

        Args:
            channel_id: The Discord channel ID
            prompt_content: The prompt content, or None to remove
        """
        channel_dir = self.data_dir / str(channel_id)
        prompt_file = channel_dir / "prompt.md"

        if prompt_content:
            # Save to disk
            try:
                channel_dir.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(prompt_file.write_text, prompt_content, encoding="utf-8")
                self._cache[channel_id] = prompt_content
                logger.info(f"Saved prompt for channel {channel_id} ({len(prompt_content)} chars)")
            except Exception:
                logger.exception(f"Failed to save prompt for channel {channel_id}")
                raise
        else:
            # Remove prompt
            if channel_id in self._cache:
                del self._cache[channel_id]
            if prompt_file.exists():
                try:
                    await asyncio.to_thread(prompt_file.unlink)
                    logger.info(f"Removed prompt for channel {channel_id}")
                    # Optionally remove empty directory
                    try:
                        await asyncio.to_thread(channel_dir.rmdir)
                    except OSError:
                        pass  # Directory not empty or other error, ignore
                except Exception:
                    logger.exception(f"Failed to remove prompt for channel {channel_id}")

    def get_effective_prompt(self, channel_id: int) -> str:
        """Get the effective prompt for a channel, falling back to default."""
        return self._cache.get(channel_id, BOT_PERSONA_PROMPT)

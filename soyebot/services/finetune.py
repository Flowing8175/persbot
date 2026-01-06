import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import aiofiles
import discord
import openai
from dotenv import set_key

from soyebot.config import AppConfig

logger = logging.getLogger(__name__)

# --- User Provided Algorithm Constants ---
SESSION_TIMEOUT_MIN = 20
MERGE_TIMEOUT_SEC = 60
MAX_CONTEXT_MSGS = 15
MIN_RESPONSE_LENGTH = 2
KEEP_RATIO_SHORT = 0.6
KEEP_RATIO_GREETING = 1  # As provided in the script

NOISE_PREFIXES = ('!', '\\', '/', 'Traceback', 'Error', '{', '}')

SYSTEM_PROMPT_PATH = Path("soyebot/assets/finetune_prompt.txt")
if not SYSTEM_PROMPT_PATH.exists():
    # Fallback/Root check
    SYSTEM_PROMPT_PATH = Path("assets/finetune_prompt.txt")

class ChatPreprocessor:
    """Encapsulates the filtering and processing logic provided by the user."""

    def __init__(self, target_user_id: int):
        self.user_map = {}
        self.user_count = 0
        self.target_user_id = str(target_user_id)

    def get_role_and_name(self, author_id, author_name):
        # ID check
        if str(author_id) == self.target_user_id:
            return "assistant", "soye"

        if author_id not in self.user_map:
            self.user_count += 1
            self.user_map[author_id] = f"User_{self.user_count}"

        return "user", self.user_map[author_id]

    def clean_content(self, text):
        if not text: return ""
        text = re.sub(r'<a?:[a-zA-Z0-9_]+:[0-9]+>', '[이모지]', text)
        text = re.sub(r'<@[0-9]+>', '@멘션', text)
        text = re.sub(r'http\S+', '[링크]', text)
        return text.strip()

    def is_valid_message(self, content):
        if not content: return False
        if content.startswith(NOISE_PREFIXES): return False
        if "```" in content: return False
        return True

    def parse_timestamp(self, ts_string):
        try:
            return datetime.fromisoformat(ts_string)
        except Exception:
            try:
                ts_string = ts_string.replace('Z', '+00:00')
                return datetime.fromisoformat(ts_string)
            except:
                return datetime.now(timezone.utc)

    def is_high_quality_sample(self, response_text):
        if len(response_text) <= MIN_RESPONSE_LENGTH:
            return random.random() < KEEP_RATIO_SHORT

        greetings = ["안녕하세요", "반가워요", "ㅎㅇ", "어서오세요", "바이", "잘가요"]
        if any(g in response_text for g in greetings) and len(response_text) < 10:
            return random.random() < KEEP_RATIO_GREETING

        return True

    async def process(self, input_file: str, output_file: str) -> bool:
        """
        Reads raw messages from input_file, applies filtering/merging,
        and writes training data to output_file.
        Returns True if successful and at least one sample was generated.
        """
        raw_data = []
        try:
            async with aiofiles.open(input_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                # Simple heuristic: check first non-whitespace char
                stripped_content = content.lstrip()
                if stripped_content.startswith('['):
                     raw_data = json.loads(content)
                else:
                    # Line-delimited JSON
                    lines = content.splitlines()
                    for line in lines:
                        if line.strip():
                            raw_data.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading input file {input_file}: {e}")
            return False

        if not raw_data:
            return False

        merged_data = []
        current_msg = None

        logger.info(f"Processing {len(raw_data)} raw messages...")

        # 2. Merge
        for msg in raw_data:
            raw_content = msg.get('content', '')
            raw_author_id = msg.get('author_id')
            raw_author_name = msg.get('author_name')
            raw_timestamp = msg.get('timestamp')

            content = self.clean_content(raw_content)

            if not self.is_valid_message(content):
                continue

            dt = self.parse_timestamp(raw_timestamp)
            role, name = self.get_role_and_name(raw_author_id, raw_author_name)

            if current_msg:
                # dt is likely timezone aware. Ensure consistency.
                if current_msg['timestamp_dt'].tzinfo is None and dt.tzinfo is not None:
                     current_msg['timestamp_dt'] = current_msg['timestamp_dt'].replace(tzinfo=dt.tzinfo)
                elif current_msg['timestamp_dt'].tzinfo is not None and dt.tzinfo is None:
                     dt = dt.replace(tzinfo=current_msg['timestamp_dt'].tzinfo)

                time_diff = (dt - current_msg['timestamp_dt']).total_seconds()
                if current_msg['name'] == name and time_diff < MERGE_TIMEOUT_SEC:
                    current_msg['content'] += " " + content
                    continue
                else:
                    merged_data.append(current_msg)

            current_msg = {
                "role": role,
                "name": name,
                "content": content,
                "timestamp_dt": dt
            }

        if current_msg:
            merged_data.append(current_msg)

        # 3. Session Split
        sessions = []
        current_session = []
        if merged_data:
            last_time = merged_data[0]['timestamp_dt']

        for msg in merged_data:
            time_diff = (msg['timestamp_dt'] - last_time).total_seconds() / 60

            if time_diff > SESSION_TIMEOUT_MIN:
                if len(current_session) > 1:
                    sessions.append(current_session)
                current_session = []

            current_session.append(msg)
            last_time = msg['timestamp_dt']

        if len(current_session) > 1:
            sessions.append(current_session)

        # 4. JSONL Generation
        sample_count = 0
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            for session in sessions:
                # Last message must be Assistant
                while session and session[-1]['role'] != 'assistant':
                    session.pop()

                if len(session) < 2:
                    continue

                if not self.is_high_quality_sample(session[-1]['content']):
                    continue

                final_history = session[-MAX_CONTEXT_MSGS:]

                # Start with User
                if final_history[0]['role'] == 'assistant':
                    final_history.pop(0)

                if not any(m['role'] == 'user' for m in final_history):
                    continue

                # Load system prompt
                system_prompt_content = "당신은 인터넷 방송인 '유소예'입니다." # Default fallback
                try:
                    async with aiofiles.open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as pf:
                        system_prompt_content = await pf.read()
                except Exception as e:
                    logger.warning(f"Failed to load finetune system prompt from file: {e}")

                entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt_content}
                    ]
                }

                for h in final_history:
                    message_dict = {
                        "role": h['role'],
                        "content": h['content']
                    }
                    entry["messages"].append(message_dict)

                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                sample_count += 1

        logger.info(f"Generated {sample_count} samples in {output_file}")
        return sample_count > 0


class FineTuneService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.state_file_path = Path("finetune_state.json")
        # Initial date provided by user
        self.initial_start_date = datetime.fromisoformat("2025-11-22T20:55:00.083000+00:00")
        self._ensure_state_file()

    async def _ensure_state_file(self):
        """Ensures the state file exists with default values."""
        if not self.state_file_path.exists():
            default_state = {
                "last_processed_date": self.initial_start_date.isoformat(),
                "current_job_id": None,
                "current_job_status": None,
                "last_check_date": datetime.now(timezone.utc).isoformat()
            }
            await self._save_state(default_state)

    async def _load_state(self) -> Dict[str, Any]:
        try:
            async with aiofiles.open(self.state_file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load fine-tune state: {e}")
            return {}

    async def _save_state(self, state: Dict[str, Any]):
        try:
            async with aiofiles.open(self.state_file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(state, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"Failed to save fine-tune state: {e}")

    async def check_due(self) -> bool:
        """Checks if a month has passed since the last processed date."""
        state = await self._load_state()
        last_date_str = state.get("last_processed_date")
        if not last_date_str:
            return False

        last_date = datetime.fromisoformat(last_date_str)
        now = datetime.now(timezone.utc)

        # Calculate one month later
        next_due_year = last_date.year + (last_date.month // 12)
        next_due_month = (last_date.month % 12) + 1

        try:
            next_due_date = last_date.replace(year=next_due_year, month=next_due_month)
        except ValueError:
            # Handle month end overflow
            if last_date.month == 1:
                 next_due_date = last_date.replace(year=next_due_year, month=next_due_month, day=28)
            else:
                 next_due_date = last_date.replace(day=28, year=next_due_year, month=next_due_month)

        logger.debug(f"Fine-tune check: Last processed {last_date}, Next due {next_due_date}, Now {now}")
        return now >= next_due_date

    async def scrape_to_file(self, bot: discord.Client, start_date: datetime, end_date: datetime) -> Optional[str]:
        """Scrapes messages and saves to a raw JSONL file."""
        if not self.config.finetune_target_user_id:
            logger.warning("Fine-tune target user ID not configured.")
            return None

        raw_data = []

        for channel_id in self.config.finetune_source_channel_ids:
            channel = bot.get_channel(channel_id)
            if not channel:
                logger.warning(f"Could not find channel {channel_id}")
                continue

            logger.info(f"Scraping channel {channel.name} from {start_date} to {end_date}")
            try:
                # Fetch history
                # We fetch oldest first implicitly by processing after fetch, or use limit=None
                # history() yields newest first by default. We collect all then reverse or sort.
                messages = []
                async for msg in channel.history(after=start_date, before=end_date, limit=None):
                    messages.append(msg)

                # Sort by time (oldest first)
                messages.sort(key=lambda x: x.created_at)

                for msg in messages:
                    # Save raw structure
                    raw_data.append({
                        "id": str(msg.id),
                        "channel_id": str(msg.channel.id),
                        "author_id": str(msg.author.id),
                        "author_name": msg.author.name,
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat()
                    })

            except Exception as e:
                logger.error(f"Error scraping channel {channel_id}: {e}")
                continue

        if not raw_data:
            logger.info("No messages found in the given period.")
            return None

        file_path = f"raw_messages_{int(end_date.timestamp())}.jsonl"
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            for item in raw_data:
                await f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(raw_data)} raw messages to {file_path}")
        return file_path

    async def run_pipeline_step(self, bot: discord.Client):
        """
        Main entry point for the recurring task.
        Checks status, scrapes if needed, processes, and submits job.
        """
        # Ensure state file initialized
        await self._ensure_state_file()
        state = await self._load_state()

        # 1. Check if job is already running
        if state.get("current_job_id"):
            await self.check_job_status()
            return

        # 2. Check if due for new scrape
        if not await self.check_due():
            return

        logger.info("Fine-tuning monthly period reached. Starting pipeline.")

        last_date_str = state.get("last_processed_date")
        start_date = datetime.fromisoformat(last_date_str)
        end_date = datetime.now(timezone.utc)

        # 3. Scrape
        raw_file = await self.scrape_to_file(bot, start_date, end_date)
        if not raw_file:
            # If no data, maybe just update date? Or try again later?
            # If strictly no data, we should probably advance the date to avoid stuck loop,
            # unless it was an error.
            # For now, let's assume if no data, we still mark it as processed to move window forward.
            # But wait, if error, we don't want to skip.
            # If scrape_to_file returns None due to no messages, it logs "No messages".
            # If error, it logs error.
            # Let's simple-mindedly advance if we tried.
            # But safer to just wait for next interval check.
            return

        # 4. Process
        train_file = f"train_sft_{int(end_date.timestamp())}.jsonl"
        preprocessor = ChatPreprocessor(self.config.finetune_target_user_id)
        success = await preprocessor.process(raw_file, train_file)

        if not success:
            logger.info("Preprocessing produced no samples. Skipping fine-tuning but advancing date.")
            # Advance date to avoid endless loop
            state["last_processed_date"] = end_date.isoformat()
            await self._save_state(state)
            return

        # 5. Start Job
        job_id = await self.start_finetuning(train_file)
        if job_id:
            # We store the end_date so we can update last_processed_date *after* success,
            # or we update it now?
            # If we update now, and job fails, we skip this month's data?
            # Better to store "pending_scrape_end_date" and update last_processed_date on success.
            state["pending_scrape_end_date"] = end_date.isoformat()
            await self._save_state(state)

    async def start_finetuning(self, file_path: str) -> Optional[str]:
        """Uploads file and starts job."""
        if not self.config.openai_api_key:
            logger.error("OpenAI API Key not set.")
            return None

        client = openai.AsyncOpenAI(api_key=self.config.openai_api_key)

        try:
            logger.info(f"Uploading file {file_path} to OpenAI...")
            with open(file_path, "rb") as f:
                response = await client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            file_id = response.id
            logger.info(f"File uploaded. ID: {file_id}")

            base_model = self.config.openai_finetuned_model or "gpt-4o-mini-2024-07-18"
            logger.info(f"Starting fine-tuning job with base model: {base_model}")

            job = await client.fine_tuning.jobs.create(
                training_file=file_id,
                model=base_model
            )

            job_id = job.id
            logger.info(f"Job started. ID: {job_id}")

            state = await self._load_state()
            state["current_job_id"] = job_id
            state["current_job_status"] = "running"
            await self._save_state(state)

            return job_id

        except Exception as e:
            logger.error(f"Error starting fine-tune job: {e}")
            return None

    async def check_job_status(self) -> str:
        """Checks the status of the current job."""
        state = await self._load_state()
        job_id = state.get("current_job_id")
        if not job_id:
            return "no_job"

        if not self.config.openai_api_key:
            return "error"

        client = openai.AsyncOpenAI(api_key=self.config.openai_api_key)

        try:
            job = await client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            logger.info(f"Fine-tune job {job_id} status: {status}")

            if status == "succeeded":
                new_model_id = job.fine_tuned_model
                logger.info(f"Fine-tuning succeeded. New model: {new_model_id}")
                self._update_env_file(new_model_id)

                # Commit the pending date
                pending_date = state.get("pending_scrape_end_date")
                if pending_date:
                    state["last_processed_date"] = pending_date
                    state.pop("pending_scrape_end_date", None)

                state["current_job_id"] = None
                state["current_job_status"] = None
                await self._save_state(state)
                return "succeeded"

            elif status in ["failed", "cancelled"]:
                logger.error(f"Fine-tuning failed or cancelled.")
                state["current_job_id"] = None
                state["current_job_status"] = None
                # Do NOT update last_processed_date, so we might retry?
                # Or should we just move on? Usually retry is better but might fail again.
                # Let's leave it; next run will try to scrape again from same date.
                await self._save_state(state)
                return status

            return status

        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return "error"

    def _update_env_file(self, new_model_id: str):
        """Updates the .env file with the new model ID."""
        env_path = Path(".env")
        # Attempt to find .env if not in current dir (though config looks in parent)
        if not env_path.exists():
            # Try parent
            env_path = Path("..") / ".env"
            if not env_path.exists():
                logger.error(".env file not found")
                return

        # We rely on soyebot being run from root usually
        # But let's be safe. Config uses parent.parent.
        # Actually config.py says: Path(__file__).resolve().parent.parent / ".env"
        # If we are in soyebot/services/, parent.parent is root.

        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"

        if not env_path.exists():
             logger.warning(f".env not found at {env_path}, trying current dir")
             env_path = Path(".env")

        success, key, value = set_key(env_path, "OPENAI_FINETUNED_MODEL", new_model_id, quote_mode="never")
        if success:
             logger.info(f"Updated .env with OPENAI_FINETUNED_MODEL={new_model_id}")
        else:
             logger.error("Failed to update .env file")

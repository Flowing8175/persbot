#!/usr/bin/env python3
"""Development runner with auto-reload for SoyeBot Discord bot.

This script watches the persbot directory for file changes and automatically
restarts the bot when changes are detected.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from watchdog.observers import Observer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BotReloader(FileSystemEventHandler):
    """Handles file system events and triggers bot reload."""

    def __init__(self, bot_process_args, debounce_seconds=1.0, ignore_patterns=None):
        """
        Initialize the BotReloader.

        Args:
            bot_process_args: List of arguments to pass to subprocess to run the bot
            debounce_seconds: Seconds to wait before restarting (prevents multiple restarts)
            ignore_patterns: List of glob patterns to ignore (e.g., ['*.pyc', '__pycache__'])
        """
        super().__init__()
        self.bot_process_args = bot_process_args
        self.debounce_seconds = debounce_seconds
        self.ignore_patterns = ignore_patterns or ["*.pyc", "__pycache__", "*.pyo", ".git"]
        self.last_restart_time = 0
        self.process = None
        self.restart_pending = False

    def _should_ignore(self, path):
        """Check if a path should be ignored based on ignore patterns."""
        for pattern in self.ignore_patterns:
            if pattern.replace("*", "") in Path(path).parts:
                return True
            if Path(path).name.endswith(pattern.replace("*", "")):
                return True
        return False

    def _is_python_file(self, path):
        """Check if the file is a Python file."""
        return path.endswith(".py")

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        if not self._is_python_file(event.src_path):
            return

        if self._should_ignore(event.src_path):
            return

        # Only restart if enough time has passed since the last restart
        current_time = time.time()
        time_since_last = current_time - self.last_restart_time

        if time_since_last < self.debounce_seconds:
            self.restart_pending = True
            return

        self._schedule_restart()

    def _schedule_restart(self):
        """Schedule a bot restart."""
        self.last_restart_time = time.time()
        self.restart_pending = False

        file_path = Path(self.bot_process_args[-1] if len(self.bot_process_args) > 1 else "").name
        logger.info(f"📝 Change detected in {file_path} - restarting bot...")
        self.restart_bot()

    def restart_bot(self):
        """Restart the bot process."""
        if self.process:
            try:
                # Terminate the existing process gracefully
                logger.info("🔄 Stopping previous bot instance...")
                self.process.terminate()
                # Give it a moment to close gracefully
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️  Bot did not stop gracefully, forcing...")
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                logger.error(f"❌ Error stopping bot: {e}")

        # Start a new bot process
        try:
            logger.info("🚀 Starting new bot instance...")
            self.process = subprocess.Popen(
                self.bot_process_args,
                stdout=None,  # Inherit stdout from parent
                stderr=None,  # Inherit stderr from parent
            )
            logger.info(f"✅ Bot started with PID: {self.process.pid}")
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")

    def stop(self):
        """Stop the bot process and observer."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"❌ Error stopping bot: {e}")


def main():
    """Main entry point for the development runner."""
    parser = argparse.ArgumentParser(
        description="Development runner with auto-reload for SoyeBot Discord bot"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="persbot",
        help="Path to watch for changes (default: persbot)",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=1.0,
        help="Debounce time in seconds before restart (default: 1.0)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current python)",
    )
    args = parser.parse_args()

    # Resolve paths
    watch_path = Path(args.path).resolve()
    if not watch_path.exists():
        logger.error(f"❌ Path does not exist: {watch_path}")
        sys.exit(1)

    # Main bot script
    main_script = Path(__file__).parent / "persbot" / "main.py"
    if not main_script.exists():
        logger.error(f"❌ Main bot script not found: {main_script}")
        sys.exit(1)

    # Bot process arguments
    bot_process_args = [args.python, str(main_script)]

    # Create and setup the reloader
    event_handler = BotReloader(
        bot_process_args=bot_process_args,
        debounce_seconds=args.debounce,
        ignore_patterns=["*.pyc", "__pycache__", "*.pyo", ".git", ".pytest_cache", "htmlcov"],
    )

    # Start the bot initially
    event_handler.restart_bot()

    # Setup file system observer
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=True)
    observer.start()

    logger.info(f"👀 Watching for changes in: {watch_path}")
    logger.info("Press Ctrl+C to stop...")

    try:
        while True:
            time.sleep(1)
            # Check if a restart was debounced and we're past the debounce window
            if event_handler.restart_pending:
                current_time = time.time()
                if current_time - event_handler.last_restart_time >= event_handler.debounce_seconds:
                    event_handler._schedule_restart()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
        observer.stop()
        event_handler.stop()
    finally:
        observer.join()
        logger.info("✅ Dev runner stopped cleanly")


if __name__ == "__main__":
    main()

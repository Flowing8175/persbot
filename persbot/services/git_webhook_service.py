"""Git Webhook Service - Handles automatic git pull via webhook."""

import asyncio
import hashlib
import hmac
import logging
import subprocess
from pathlib import Path

from aiohttp import web

from persbot.config import AppConfig

logger = logging.getLogger(__name__)


class GitWebhookService:
    """Service that listens for webhook requests and performs git pull."""

    def __init__(self, config: AppConfig, notify_callback=None):
        self.config = config
        self.notify_callback = notify_callback  # Async callback for Discord notifications
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start the webhook HTTP server."""
        if not self.config.git_webhook_enabled:
            logger.info("Git webhook is disabled")
            return

        app = web.Application()
        app.router.add_post(self.config.git_webhook_path, self._handle_webhook)

        self.runner = web.AppRunner(app)
        await self.runner.setup()

        self.site = web.TCPSite(
            self.runner,
            "0.0.0.0",
            self.config.git_webhook_port,
        )
        await self.site.start()
        self._is_running = True

        logger.info(
            "Git webhook server started on port %d at %s",
            self.config.git_webhook_port,
            self.config.git_webhook_path,
        )

    async def stop(self) -> None:
        """Stop the webhook HTTP server."""
        if self.runner:
            await self.runner.cleanup()
            self._is_running = False
            logger.info("Git webhook server stopped")

    def _verify_signature(self, payload: bytes, signature: str | None) -> bool:
        """Verify webhook signature using HMAC-SHA256."""
        if not self.config.git_webhook_secret:
            logger.warning("Git webhook secret not configured - rejecting all requests for security")
            return False

        if not signature:
            return False

        # Support both "sha256=<hash>" and raw hash formats
        if signature.startswith("sha256="):
            expected_sig = signature[7:]
        else:
            expected_sig = signature

        computed_sig = hmac.new(
            self.config.git_webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed_sig, expected_sig)

    def _validate_repo_path(self, repo_path: Path) -> bool:
        """Validate that the repo path is a valid git repository."""
        if not repo_path.exists():
            logger.error("Repo path does not exist: %s", repo_path)
            return False

        if not (repo_path / ".git").exists():
            logger.error("Path %s is not a git repository", repo_path)
            return False

        return True

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming webhook request."""
        try:
            # Check IP allowlist FIRST
            if self.config.git_webhook_allowed_ips:
                client_ip = request.remote
                if client_ip not in self.config.git_webhook_allowed_ips:
                    logger.warning("Rejected webhook from unauthorized IP: %s", client_ip)
                    return web.Response(status=403, text="Forbidden")

            payload = await request.read()

            # Check for signature in various headers
            signature = request.headers.get("X-Hub-Signature-256") or request.headers.get(
                "X-Hub-Signature"
            ) or request.headers.get("X-Gitlab-Token") or request.headers.get("Authorization")

            if not self._verify_signature(payload, signature):
                logger.warning("Invalid webhook signature")
                return web.Response(status=401, text="Unauthorized")

            # Parse the payload to get repository info (optional)
            try:
                import json

                data = json.loads(payload)
                repo_name = data.get("repository", {}).get("name", "unknown")
                ref = data.get("ref", "")
                commits = data.get("commits", [])
                commit_count = len(commits)
                logger.info(
                    "Webhook received for repo '%s' on %s (%d commits)",
                    repo_name,
                    ref,
                    commit_count,
                )
            except (json.JSONDecodeError, Exception):
                logger.info("Webhook received (could not parse payload)")

            # Perform git pull
            success, message = await self._git_pull()

            if success:
                # Send notification if callback is set
                if self.notify_callback:
                    await self.notify_callback(f"✅ Git pull successful: {message}")
                return web.Response(status=200, text=f"Success: {message}")
            else:
                if self.notify_callback:
                    await self.notify_callback(f"❌ Git pull failed: {message}")
                return web.Response(status=500, text=f"Failed: {message}")

        except Exception:
            logger.exception("Error handling webhook")
            return web.Response(status=500, text="Internal error")

    async def _git_pull(self) -> tuple[bool, str]:
        """Execute git pull and return (success, message)."""
        repo_path = Path(self.config.git_repo_path) if self.config.git_repo_path else Path.cwd()

        if not self._validate_repo_path(repo_path):
            return False, f"Invalid repository path: {repo_path}"

        try:
            # Run git pull in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "pull"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                ),
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if "Already up to date" in output or "Already up-to-date" in output:
                    return True, "Already up to date"
                return True, output or "Pull completed"
            else:
                error = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                logger.error("Git pull failed: %s", error)
                return False, error

        except subprocess.TimeoutExpired:
            return False, "Git pull timed out"
        except Exception as e:
            logger.exception("Git pull error")
            return False, str(e)

    @property
    def is_running(self) -> bool:
        """Check if the webhook server is running."""
        return self._is_running

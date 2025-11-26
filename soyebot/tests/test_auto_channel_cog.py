import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import discord

from soyebot.bot.cogs.auto_channel import AutoChannelCog
from soyebot.bot.session import SessionManager, ChatSession
from soyebot.services.llm_service import LLMService
from soyebot.config import AppConfig


class TestAutoChannelCog(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = AsyncMock()
        self.config = MagicMock(spec=AppConfig)
        self.config.auto_reply_channel_ids = [12345]
        self.llm_service = MagicMock(spec=LLMService)
        self.session_manager = MagicMock(spec=SessionManager)

        self.cog = AutoChannelCog(
            bot=self.bot,
            config=self.config,
            llm_service=self.llm_service,
            session_manager=self.session_manager,
        )

    async def test_handle_undo_command_with_permission(self):
        # Setup mock message and author
        author_with_permission = MagicMock(spec=discord.Member)
        author_with_permission.id = 1
        author_with_permission.guild_permissions.manage_guild = False

        message = AsyncMock(spec=discord.Message)
        message.channel.id = 12345
        message.content = "!@ 2"
        message.author = author_with_permission

        # Setup mock session with history
        mock_chat_session = MagicMock(spec=ChatSession)
        mock_chat_session.chat.history = [
            MagicMock(role="user", author_id=1),
            MagicMock(role="user", author_id=1),
            MagicMock(role="user", author_id=1),
            MagicMock(role="user", author_id=1),
            MagicMock(role="user", author_id=1),
            MagicMock(role="assistant", author_id=None),
        ]
        self.session_manager.sessions.get.return_value = mock_chat_session
        self.llm_service.get_user_role_name.return_value = "user"
        self.session_manager.undo_last_exchanges.return_value = True

        # Execute the command handler
        result = await self.cog._handle_undo_command(message)

        # Assertions
        self.assertTrue(result)
        self.session_manager.undo_last_exchanges.assert_called_once_with("channel:12345", 2)
        message.add_reaction.assert_called_once_with("✅")
        message.edit.assert_called_once()

    async def test_handle_undo_command_without_permission(self):
        # Setup mock message and author
        author_without_permission = MagicMock(spec=discord.Member)
        author_without_permission.id = 2
        author_without_permission.guild_permissions.manage_guild = False

        message = AsyncMock(spec=discord.Message)
        message.channel.id = 12345
        message.content = "!@"
        message.author = author_without_permission

        # Setup mock session with history
        mock_chat_session = MagicMock(spec=ChatSession)
        mock_chat_session.chat.history = [
            MagicMock(role="user", author_id=1),
            MagicMock(role="assistant", author_id=None),
        ]
        self.session_manager.sessions.get.return_value = mock_chat_session
        self.llm_service.get_user_role_name.return_value = "user"

        # Execute the command handler
        result = await self.cog._handle_undo_command(message)

        # Assertions
        self.assertTrue(result)
        self.session_manager.undo_last_exchanges.assert_not_called()
        message.add_reaction.assert_called_once_with("❌")
        message.edit.assert_not_called()

if __name__ == "__main__":
    unittest.main()

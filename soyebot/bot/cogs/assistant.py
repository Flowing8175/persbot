"""Assistant Cog for SoyeBot."""

import discord
from discord.ext import commands
import logging

from config import AppConfig
from services.gemini_service import GeminiService
from services.memory_service import MemoryService
from bot.session import SessionManager
from utils import extract_message_content, DiscordUI

logger = logging.getLogger(__name__)

class AssistantCog(commands.Cog):
    """@mention을 통한 AI 어시스턴트 기능을 처리하는 Cog"""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        gemini_service: GeminiService,
        session_manager: SessionManager,
        memory_service: MemoryService,
    ):
        self.bot = bot
        self.config = config
        self.gemini_service = gemini_service
        self.session_manager = session_manager
        self.memory_service = memory_service

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not self.bot.user.mentioned_in(message):
            return

        self.session_manager.cleanup_expired()

        user_message = extract_message_content(message)
        if not user_message:
            await message.reply("❌ 메시지 내용이 없는데요.", mention_author=False)
            return

        logger.info(f"@mention from {message.author.name}: {user_message[:50]}")

        async with message.channel.typing():
            # 세션 ID 결정: 리플라이 대상이 있으면 그 메시지 ID, 없으면 현재 메시지 ID
            session_id = message.reference.message_id if message.reference else message.id

            # Get or create user session with memory context
            chat_session, user_id = self.session_manager.get_or_create(
                user_id=message.author.id,
                username=message.author.name,
                message_id=str(session_id),
            )

            # Get function calling tools
            tools = None
            if self.config.enable_memory_system:
                tools = self.memory_service.get_gemini_function_calling_tools()

            response_result = await self.gemini_service.generate_chat_response(
                chat_session,
                user_message,
                tools=tools,
            )

            if response_result:
                response_text, response_obj = response_result

                # Handle function calls if present
                if self.config.enable_memory_system and tools and response_obj:
                    await self._handle_function_calls(
                        response_obj,
                        message.author.id,
                    )

                # Only reply if there's text content
                # (Gemini might return only function calls without text)
                if response_text:
                    await message.reply(response_text, mention_author=False)
                else:
                    # If only function calls were returned, log but don't reply
                    logger.debug(f"Response contained only function calls, no text to reply with")
            else:
                logger.warning(f"응답 생성 실패 (ID: {session_id})")
                await message.reply("❌ 응답 생성 중 오류가 발생했어요. 다시 시도해주세요.", mention_author=False)

    async def _handle_function_calls(self, response_obj, user_id: int) -> None:
        """Handle function calls from Gemini response.

        Args:
            response_obj: Full response object from Gemini
            user_id: Discord user ID
        """
        try:
            # Parse function calls from response
            function_calls = self.gemini_service.parse_function_calls(response_obj)

            if not function_calls:
                logger.debug("No function calls found in response")
                return

            user_id_str = str(user_id)
            logger.info(f"Processing {len(function_calls)} function calls for user {user_id}")

            # Execute each function call
            for func_call in function_calls:
                if not isinstance(func_call, dict):
                    logger.warning(f"Unexpected function call format: {type(func_call)}")
                    continue

                func_name = func_call.get('name', '')
                func_args = func_call.get('args', {})

                if not func_name:
                    logger.warning("Function call has no name")
                    continue

                if not isinstance(func_args, dict):
                    logger.debug(f"Function args is not a dict: {type(func_args)}, treating as empty")
                    func_args = {}

                try:
                    logger.debug(f"Executing function call: {func_name} with args: {func_args}")

                    if func_name == 'save_user_fact':
                        result = self.memory_service.handle_save_user_fact(
                            user_id_str,
                            fact=func_args.get('fact', ''),
                            category=func_args.get('category', 'other'),
                        )
                        logger.info(f"save_user_fact result: {result}")

                    elif func_name == 'save_preference':
                        result = self.memory_service.handle_save_preference(
                            user_id_str,
                            preference=func_args.get('preference', ''),
                        )
                        logger.info(f"save_preference result: {result}")

                    elif func_name == 'save_key_memory':
                        result = self.memory_service.handle_save_key_memory(
                            user_id_str,
                            memory=func_args.get('memory', ''),
                            importance=int(func_args.get('importance', 5)),
                        )
                        logger.info(f"save_key_memory result: {result}")

                    elif func_name == 'update_interaction_pattern':
                        result = self.memory_service.handle_update_interaction_pattern(
                            user_id_str,
                            topic=func_args.get('topic', ''),
                            sentiment=func_args.get('sentiment'),
                        )
                        logger.info(f"update_interaction_pattern result: {result}")

                    else:
                        logger.warning(f"Unknown function call: {func_name}")

                except Exception as e:
                    logger.error(f"Error executing function call {func_name}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Failed to handle function calls: {e}", exc_info=True)

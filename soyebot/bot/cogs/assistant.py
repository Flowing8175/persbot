"""Assistant Cog for SoyeBot."""

import logging
import time
import asyncio
import re
from typing import Optional

import discord
from discord.ext import commands

from bot.chat_handler import ChatReply, create_chat_reply, resolve_session_for_message, send_split_response
from bot.session import SessionManager, ResolvedSession
from bot.buffer import MessageBuffer
from config import AppConfig
from services.llm_service import LLMService
from services.base import ChatMessage
from services.prompt_service import PromptService
from utils import GENERIC_ERROR_MESSAGE, extract_message_content

logger = logging.getLogger(__name__)

class AssistantCog(commands.Cog):
    """@mention을 통한 AI 어시스턴트 기능을 처리하는 Cog"""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        session_manager: SessionManager,
        prompt_service: PromptService,
    ):
        self.bot = bot
        self.config = config
        self.llm_service = llm_service
        self.session_manager = session_manager
        self.prompt_service = prompt_service
        self.sending_tasks: dict[int, asyncio.Task] = {}
        
        # Track active processing tasks (LLM generation) to allow debouncing/merging
        self.processing_tasks: dict[int, asyncio.Task] = {}
        self.active_batches: dict[int, list[discord.Message]] = {}
        
        # Use config for default delay (now 0.1)
        self.message_buffer = MessageBuffer(delay=config.message_buffer_delay)

    def _should_ignore_message(self, message: discord.Message) -> bool:
        """Return True when the bot should not process the message."""

        if message.author.bot:
            return True
        # If this channel is handled by AutoChannelCog, let it handle the response
        # to avoid duplicate replies (one plain, one reply).
        if message.channel.id in self.config.auto_reply_channel_ids:
            return True
        if not self.bot.user or not self.bot.user.mentioned_in(message):
            return True
        return message.mention_everyone

    async def _send_llm_reply(self, message: discord.Message, reply: ChatReply) -> None:
        if not reply.text:
            logger.debug("LLM returned no text response for the mention.")
            return

        # If Break-Cut Mode is OFF, send normally
        if not self.config.break_cut_mode:
            reply_message = await message.reply(reply.text, mention_author=False)
            if reply_message:
                self.session_manager.link_message_to_session(str(reply_message.id), reply.session_key)
            return

        # If Break-Cut Mode is ON, use shared helper
        channel_id = message.channel.id
        if channel_id in self.sending_tasks and not self.sending_tasks[channel_id].done():
            self.sending_tasks[channel_id].cancel()

        task = asyncio.create_task(
            send_split_response(message.channel, reply, self.session_manager)
        )
        self.sending_tasks[channel_id] = task

        def _cleanup(t):
            if self.sending_tasks.get(channel_id) == t:
                self.sending_tasks.pop(channel_id, None)
        
        task.add_done_callback(_cleanup)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if self._should_ignore_message(message):
            return

        # Interrupt current sending if any (Break-Cut Mode)
        if self.config.break_cut_mode and message.channel.id in self.sending_tasks:
            task = self.sending_tasks[message.channel.id]
            if not task.done():
                logger.info(f"New message from {message.author} interrupted sending in channel {message.channel.id}")
                task.cancel()

        # Interrupt current processing (Processing/Debounce Mode)
        messages_to_prepend = []
        if message.channel.id in self.processing_tasks:
            task = self.processing_tasks[message.channel.id]
            if not task.done():
                logger.info(f"New message from {message.author} interrupted processing in channel {message.channel.id}. Merging messages.")
                messages_to_prepend = self.active_batches.get(message.channel.id, [])
                task.cancel()

        await self.message_buffer.add_message(message.channel.id, message, self._process_batch)
        
        if messages_to_prepend:
             # Ensure the list exists before prepending
             if message.channel.id in self.message_buffer.buffers:
                 self.message_buffer.buffers[message.channel.id][0:0] = messages_to_prepend

    @commands.Cog.listener()
    async def on_typing(self, channel: discord.abc.Messageable, user: discord.abc.User, when: float):
        """
        Listener for typing events.
        Extends the processing delay if a user is typing in a channel where we have pending messages.
        """
        if user.bot:
            return

        # We need the channel ID. 'channel' can be TextChannel, DMChannel, etc.
        if hasattr(channel, 'id'):
            # If Break-Cut Mode is OFF, use the old "Extend Wait" logic.
            # If ON, we do NOT extend wait (user request: "delete user input wait").
            if not self.config.break_cut_mode:
                self.message_buffer.handle_typing(channel.id, self._process_batch)

    async def _process_batch(self, messages: list[discord.Message]):
        if not messages:
            return

        # Use the first message for context/reply target, but could be improved
        primary_message = messages[0]
        channel_id = primary_message.channel.id
        
        # Register this task as the active processing task for the channel
        self.active_batches[channel_id] = messages
        self.processing_tasks[channel_id] = asyncio.current_task()

        try:
            # 1. Fetch recent context (10 messages before the primary message)
            context_messages = [
                msg async for msg in primary_message.channel.history(limit=10, before=primary_message)
            ]
            context_messages.reverse()  # Chronological order

            context_text = ""
            if context_messages:
                context_lines = []
                for msg in context_messages:
                    c_content = extract_message_content(msg)
                    if c_content:
                        context_lines.append(f"{msg.author.id}: {c_content}")

                if context_lines:
                    context_text = "=== 이전 대화 문맥 (참고용) ===\n" + "\n".join(context_lines) + "\n=== 현재 메시지 ===\n"

            # 2. Combine current batch contents
            combined_content = []
            for msg in messages:
                content = extract_message_content(msg)
                if content:
                    if len(messages) > 1 and msg.author.id:
                         combined_content.append(f"{msg.author.id}: {content}")
                    else:
                         combined_content.append(content)

            current_text = "\n".join(combined_content)

            if not current_text:
                return

            # Prepend context to the full text
            full_text = context_text + current_text

            logger.info("Processing batch of %d messages from %s: %s", len(messages), primary_message.author.name, full_text[:100])

            async with primary_message.channel.typing():
                resolution = await resolve_session_for_message(
                    primary_message,
                    full_text,
                    session_manager=self.session_manager,
                )

                if not resolution:
                    return

                reply = await create_chat_reply(
                    primary_message,
                    resolution=resolution,
                    llm_service=self.llm_service,
                    session_manager=self.session_manager,
                )

                if reply:
                    # We reply to the last message in the batch so the user sees it at the bottom
                    last_message = messages[-1]
                    await self._send_llm_reply(last_message, reply)

        except asyncio.CancelledError:
            logger.info("Batch processing cancelled for channel #%s (likely due to new message or !abort).", primary_message.channel.name)
            raise

        except asyncio.CancelledError:
            logger.info("Batch processing cancelled for channel #%s (likely due to new message or !abort).", primary_message.channel.name)
            raise

        except Exception as e:
            logger.error("메시지 처리 중 예상치 못한 오류 발생: %s", e, exc_info=True)
            await primary_message.reply(GENERIC_ERROR_MESSAGE, mention_author=False)
        
        finally:
            # Cleanup only if we are the current task
            if self.processing_tasks.get(channel_id) == asyncio.current_task():
                self.processing_tasks.pop(channel_id, None)
                self.active_batches.pop(channel_id, None)

    @commands.command(name='help', aliases=['도움말', '명령어', 'h'])
    async def help_command(self, ctx: commands.Context):
        """봇의 모든 명령어와 사용법을 안내합니다."""
        embed = discord.Embed(
            title="🤖 SoyeBot 명령어 가이드",
            description=f"접두사: `{self.config.command_prefix}` 또는 `@mention`을 사용하여 명령을 내릴 수 있습니다.",
            color=discord.Color.blue()
        )

        # 1. 대화 제어
        embed.add_field(
            name="💬 대화 제어",
            value=(
                "`!retry` (`!다시`): 마지막 답변을 지우고 다시 생성합니다.\n"
                "`!reset` (`!초기화`): 현재 채널의 대화 기록을 초기화합니다.\n"
                "`!undo [N]` (`!@`): 마지막 N개의 대화 쌍을 삭제합니다. (자동응답 채널 전용)\n"
                "`!abort` (`!중단`): 진행 중인 전송이나 AI 처리를 즉시 멈춥니다."
            ),
            inline=False
        )

        # 2. 요약 및 분석
        embed.add_field(
            name="📝 요약 및 분석",
            value=(
                "`!요약`: 최근 30분 대화를 요약합니다.\n"
                "`!요약 <시간>`: 지정 시간(예: `20분`, `1시간`) 동안의 대화를 요약합니다.\n"
                "`!요약 <ID> 이후`: 특정 메시지 이후의 대화를 요약합니다."
            ),
            inline=False
        )

        # 3. 프롬프트 관리 (Persona)
        embed.add_field(
            name="🎭 프롬프트 (페르소나) 관리",
            value=(
                "`!prompt list`: 저장된 프롬프트 목록을 보여줍니다.\n"
                "`!prompt select <번호>`: 채널에 적용할 프롬프트를 선택합니다. (생략 시 기본값)\n"
                "`!prompt new <컨셉>`: AI가 새로운 고품질 프롬프트를 자동 생성합니다.\n"
                "`!prompt show <번호>`: 프롬프트의 전체 상세 내용을 확인합니다.\n"
                "`!prompt delete <번호>`: 프롬프트를 삭제합니다."
            ),
            inline=False
        )

        # 4. 설정 및 파라미터
        embed.add_field(
            name="⚙️ 설정 및 파라미터",
            value=(
                "`!temp <0.0~2.0>`: AI의 창의성(Temperature)을 조절합니다.\n"
                "`!생각 <숫자|auto|off>`: Gemini Thinking Budget를 설정합니다.\n"
                "`!끊어치기 [on|off]`: 실시간 메시지 끊어 전송 모드를 설정합니다."
            ),
            inline=False
        )

        embed.set_footer(text="SoyeBot | Advanced Agentic Coding Assistant")
        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(name='retry', aliases=['재생성', '다시'])
    async def retry_command(self, ctx: commands.Context):
        """Re-generate the last assistant response."""
        session_key = f"channel:{ctx.channel.id}"

        # Undo the last exchange (assistant + user message)
        removed_messages: list[ChatMessage] = self.session_manager.undo_last_exchanges(session_key, 1)

        if not removed_messages:
            await ctx.reply("❌ 되돌릴 대화가 없습니다.", mention_author=False)
            return

        # Identify the user message content and the previous assistant message ID
        user_role = self.llm_service.get_user_role_name()
        assistant_role = self.llm_service.get_assistant_role_name()

        user_content = ""
        # Process removed messages to find content and IDs
        # Removed messages are chronological. Expect [User, Assistant] usually.
        for msg in removed_messages:
            if msg.role == user_role:
                user_content = msg.content
            elif msg.role == assistant_role:
                if hasattr(msg, 'message_ids') and msg.message_ids:
                    # Collect all message IDs for deletion
                    for mid in msg.message_ids:
                        try:
                            old_msg = await ctx.channel.fetch_message(int(mid))
                            await old_msg.delete()
                        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                            pass

        if not user_content:
            await ctx.send("❌ 재시도할 사용자 메시지를 찾을 수 없습니다.")
            return

        # Re-generate response
        async with ctx.channel.typing():
            resolution = ResolvedSession(session_key, user_content)

            # Create a reply using the original context author (ctx.author) which is acceptable for retry
            reply = await create_chat_reply(
                ctx.message,
                resolution=resolution,
                llm_service=self.llm_service,
                session_manager=self.session_manager,
            )

            if reply and reply.text:
                # We always send a new reply for retry now, as old ones were deleted
                await self._send_llm_reply(ctx.message, reply)
            else:
                 await ctx.send(GENERIC_ERROR_MESSAGE)

        # Attempt to delete the retry command message itself for cleanliness
        try:
            await ctx.message.delete()
        except (discord.Forbidden, discord.HTTPException):
            pass

    @commands.command(name='abort', aliases=['중단', '멈춰'])
    async def abort_command(self, ctx: commands.Context):
        """진행 중인 모든 메시지 전송 및 처리를 강제로 중단합니다."""
        channel_id = ctx.channel.id
        aborted = False

        # 1. Interrupt tasks in AssistantCog
        # Cancel sending
        if channel_id in self.sending_tasks:
            task = self.sending_tasks[channel_id]
            if not task.done():
                task.cancel()
                aborted = True

        # Cancel processing
        if channel_id in self.processing_tasks:
            task = self.processing_tasks[channel_id]
            if not task.done():
                task.cancel()
                aborted = True

        # 2. Try to interrupt tasks in AutoChannelCog
        auto_cog = self.bot.get_cog("AutoChannelCog")
        if auto_cog:
            # Cancel sending tasks
            if channel_id in auto_cog.sending_tasks:
                task = auto_cog.sending_tasks[channel_id]
                if not task.done():
                    task.cancel()
                    aborted = True
            
            # Cancel processing tasks
            if hasattr(auto_cog, 'processing_tasks') and channel_id in auto_cog.processing_tasks:
                task = auto_cog.processing_tasks[channel_id]
                if not task.done():
                    task.cancel()
                    aborted = True
        
        if aborted:
            await ctx.message.add_reaction("🛑")
            logger.info("User %s requested abort in channel %s", ctx.author.name, channel_id)
        else:
            await ctx.message.add_reaction("❓")

    @commands.command(name='초기화', aliases=['reset'])
    async def reset_session(self, ctx: commands.Context):
        """현재 채널의 대화 세션을 수동으로 초기화합니다."""

        try:
            self.session_manager.reset_session_by_channel(ctx.channel.id)
            await ctx.message.add_reaction("✅")
        except Exception as exc:
            logger.error("세션 초기화 실패: %s", exc, exc_info=True)
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

    @commands.command(name='temp')
    @commands.has_permissions(manage_guild=True)
    async def set_temperature(self, ctx: commands.Context, value: Optional[float] = None):
        """Set the temperature parameter for the LLM (0.0 - 2.0)."""
        if value is None:
            current_temp = getattr(self.config, 'temperature', 1.0)
            await ctx.reply(f"🌡️ 현재 Temperature: {current_temp}", mention_author=False)
            return

        if not (0.0 <= value <= 2.0):
            await ctx.reply("❌ Temperature는 0.0에서 2.0 사이여야 합니다.", mention_author=False)
            return

        try:
            self.llm_service.update_parameters(temperature=value)
            await ctx.message.add_reaction("✅")
        except Exception as e:
            logger.error("Temperature 설정 실패: %s", e, exc_info=True)
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

    @commands.command(name='topp')
    @commands.has_permissions(manage_guild=True)
    async def set_top_p(self, ctx: commands.Context, value: Optional[float] = None):
        """Set the top_p parameter for the LLM (0.0 - 1.0)."""
        if value is None:
            current_top_p = getattr(self.config, 'top_p', 1.0)
            await ctx.reply(f"📊 현재 Top-p: {current_top_p}", mention_author=False)
            return

        if not (0.0 <= value <= 1.0):
            await ctx.reply("❌ Top-p는 0.0에서 1.0 사이여야 합니다.", mention_author=False)
            return

        try:
            self.llm_service.update_parameters(top_p=value)
            await ctx.message.add_reaction("✅")
        except Exception as e:
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)
    
    @commands.command(name='끊어치기')
    async def toggle_break_cut(self, ctx: commands.Context, mode: Optional[str] = None):
        """끊어치기 모드를 설정합니다. (!끊어치기 [on|off], 인자 없이 사용 시 토글)"""
        if mode is None:
            # Toggle
            self.config.break_cut_mode = not self.config.break_cut_mode
        else:
            cleaned = mode.lower().strip()
            if cleaned == 'on':
                self.config.break_cut_mode = True
            elif cleaned == 'off':
                self.config.break_cut_mode = False
            else:
                await ctx.reply("사용법: !끊어치기 [on|off] (생략 시 토글)")
                return

        status = "ON" if self.config.break_cut_mode else "OFF"
        
        # Adjust buffer behavior based on mode? 
        # Actually user requested "remove wait" globally. 
        # But we did that in __init__ (delay=0.1).
        # We only gated handle_typing in on_typing.
        
        await ctx.reply(f"✂️ 끊어치기 모드가 **{status}** 상태로 변경되었습니다.")

    @commands.command(name='생각', aliases=['think'])
    @commands.has_permissions(manage_guild=True)
    async def set_thinking_budget(self, ctx: commands.Context, value: Optional[str] = None):
        """Gemini Thinking Budget를 설정합니다. (!생각 [숫자|auto|off])"""
        if value is None:
            current = getattr(self.config, 'thinking_budget', None)
            if current is None:
                display = 'OFF'
            elif current == -1:
                display = 'AUTO'
            else:
                display = str(current)
            status = f"현재 Thinking Budget: **{display}**"
            await ctx.reply(f"🧠 {status}", mention_author=False)
            return

        cleaned = value.lower().strip()
        target_value: Optional[int] = None

        if cleaned == 'off':
            target_value = None
        elif cleaned == 'auto':
            target_value = -1 # Special value for dynamic budget
        else:
            try:
                target_value = int(cleaned)
                if not (512 <= target_value <= 32768):
                    await ctx.reply("❌ Thinking Budget은 512에서 32768 사이여야 합니다.", mention_author=False)
                    return
            except ValueError:
                await ctx.reply("❌ 올바른 숫자(512~32768), 'auto', 또는 'off'를 입력해 주세요.", mention_author=False)
                return

        try:
            self.llm_service.update_parameters(thinking_budget=target_value)
            await ctx.message.add_reaction("✅")

        except Exception as e:
            logger.error("Thinking Budget 설정 실패: %s", e, exc_info=True)
            await ctx.reply(GENERIC_ERROR_MESSAGE, mention_author=False)

    @commands.group(name='prompt', invoke_without_command=True)
    async def prompt_group(self, ctx: commands.Context):
        """프롬프트 관리 명령입니다. (!prompt <new|list|show|rename|select|delete>)"""
        if ctx.invoked_subcommand is None:
            await ctx.send("사용법: `!prompt <new|list|show|rename|select|delete> [인자]`")

    @prompt_group.command(name='new')
    @commands.has_permissions(manage_guild=True)
    async def prompt_new(self, ctx: commands.Context, *, concept: str):
        """새로운 프롬프트를 자동으로 생성합니다. (!prompt new <컨셉>)"""
        status_msg = await ctx.reply("🧠 고품질 페르소나 설계 중... (약 10~20초 소요)")
        
        try:
            generated_prompt = await self.llm_service.generate_prompt_from_concept(concept)
            
            if not generated_prompt:
                await status_msg.edit(content="❌ 프롬프트 생성에 실패했습니다.")
                return

            # Extract name and content
            # Pattern: **[System Prompt: Project '{Character Name}']**
            # or sometimes without asterisks or with different casing
            name_match = re.search(r"Project\s+['\"]?(.+?)['\"]?\]", generated_prompt, re.IGNORECASE)
            name = name_match.group(1) if name_match else f"Generated ({concept[:10]}...)"
            
            # Remove the title line from the content if possible, or just keep it all
            prompt_content = generated_prompt.strip()

            idx = self.prompt_service.add_prompt(name, prompt_content)
            
            await status_msg.edit(content=f"✅ 새 페르소나 **'{name}'**이(가) 설계되었습니다! (인덱스: {idx})")
            
            await ctx.send(f"💡 페르소나 설계가 완료되었습니다. `!prompt show {idx}`로 전체 내용을 확인할 수 있습니다.")

        except Exception as e:
            logger.error(f"Error in prompt_new: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ 오류 발생: {str(e)}")

    @prompt_group.command(name='list')
    async def prompt_list(self, ctx: commands.Context):
        """저장된 프롬프트 목록을 보여줍니다."""
        prompts = self.prompt_service.list_prompts()
        if not prompts:
            await ctx.reply("저장된 프롬프트가 없습니다.")
            return

        active_content = self.session_manager.channel_prompts.get(ctx.channel.id)
        
        response = "**📋 저장된 프롬프트 목록:**\n"
        for i, p in enumerate(prompts):
            marker = "🔹"
            if active_content == p['content']:
                marker = "✅"
            response += f"{marker} **[{i}]** {p['name']}\n"
        
        await ctx.reply(response)

    @prompt_group.command(name='show')
    async def prompt_show(self, ctx: commands.Context, index: int):
        """특정 인덱스의 프롬프트 내용을 보여줍니다. (!prompt show [인덱스])"""
        prompt = self.prompt_service.get_prompt(index)
        if not prompt:
            await ctx.reply("❌ 해당 인덱스의 프롬프트를 찾을 수 없습니다.")
            return

        # Use send_split_response style or just direct messages if it's long
        content = f"**📋 프롬프트: {prompt['name']}**\n\n{prompt['content']}"
        
        if len(content) <= 2000:
            await ctx.reply(content, mention_author=False)
        else:
            # Simple chunking for Discord message limit (2000 chars)
            for i in range(0, len(content), 1900):
                await ctx.send(content[i:i+1900])

    @prompt_group.command(name='rename')
    @commands.has_permissions(manage_guild=True)
    async def prompt_rename(self, ctx: commands.Context, index: int, *, new_name: str):
        """프롬프트 이름을 변경합니다. (!prompt rename [인덱스] "새 이름")"""
        if self.prompt_service.rename_prompt(index, new_name):
            await ctx.message.add_reaction("✅")
        else:
            await ctx.reply("❌ 해당 인덱스의 프롬프트를 찾을 수 없습니다.")

    @prompt_group.command(name='delete', aliases=['삭제'])
    @commands.has_permissions(manage_guild=True)
    async def prompt_delete(self, ctx: commands.Context, index: int):
        """저장된 프롬프트를 삭제합니다. (!prompt delete [인덱스])"""
        prompt = self.prompt_service.get_prompt(index)
        if not prompt:
            await ctx.reply("❌ 해당 인덱스의 프롬프트를 찾을 수 없습니다.")
            return

        if self.prompt_service.delete_prompt(index):
            await ctx.reply(f"✅ 프롬프트 **'{prompt['name']}'**이(가) 삭제되었습니다.")
        else:
            await ctx.reply("❌ 삭제에 실패했습니다.")

    @prompt_group.command(name='select')
    @commands.has_permissions(manage_guild=True)
    async def prompt_select(self, ctx: commands.Context, index: Optional[int] = None):
        """채널에 적용할 프롬프트를 선택하거나 초기화합니다. (!prompt select [인덱스], 생략 시 초기화)"""
        if index is None:
            # Reset to default
            self.session_manager.set_channel_prompt(ctx.channel.id, None)
            await ctx.reply("✅ 채널 프롬프트가 기본값으로 초기화되었습니다.")
            return

        prompt = self.prompt_service.get_prompt(index)
        if not prompt:
            await ctx.reply("❌ 해당 인덱스의 프롬프트를 찾을 수 없습니다.")
            return

        self.session_manager.set_channel_prompt(ctx.channel.id, prompt['content'])
        await ctx.reply(f"✅ 채널 프롬프트가 **{prompt['name']}** (으)로 변경되었습니다. 대화 세션이 초기화됩니다.")

    async def cog_command_error(self, ctx: commands.Context, error: Exception):
        """Cog 내 명령어 에러 핸들러"""
        if isinstance(error, commands.MissingPermissions):
            await ctx.reply(f"❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: {', '.join(error.missing_permissions)})", mention_author=False)
        elif isinstance(error, commands.BadArgument):
            await ctx.reply("❌ 잘못된 인자가 전달되었습니다. 명령어를 다시 확인해 주세요.", mention_author=False)
        elif isinstance(error, commands.CommandOnCooldown):
            await ctx.reply(f"⏳ 쿨다운 중입니다. {error.retry_after:.1f}초 후에 다시 시도해 주세요.", mention_author=False)
        else:
            logger.error(f"Command error in {ctx.command}: {error}", exc_info=True)
            # 기본 에러 메시지는 이미 globally 처리될 수도 있지만, cog 레벨에서 한번 더 확인
            if not ctx.command.has_error_handler():
                await ctx.reply(f"❌ 명령어 실행 중 오류가 발생했습니다: {str(error)}", mention_author=False)



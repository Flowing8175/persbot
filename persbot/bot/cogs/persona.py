"""Persona/Prompt Management Cog for SoyeBot."""

import asyncio
import json
import logging
import re
from typing import Optional, List, Dict

import discord
from bot.session import SessionManager
from config import AppConfig
from discord.ext import commands
from services.llm_service import LLMService
from services.prompt_service import PromptService
from utils import send_discord_message

logger = logging.getLogger(__name__)

# --- UI Components for Prompt Manager ---


class ShowModalButton(discord.ui.View):
    """View with a button that shows a modal when clicked.

    This is needed because modals can only be sent as the first response
    to an interaction, not as a followup message.
    """

    def __init__(self, modal: discord.ui.Modal, timeout: int = 300) -> None:
        super().__init__(timeout=timeout)
        self.modal = modal

    @discord.ui.button(label="질문 답변하기", style=discord.ButtonStyle.primary, emoji="📝")
    async def show_modal(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Show the modal when button is clicked."""
        await interaction.response.send_modal(self.modal)


class PromptModeSelectView(discord.ui.View):
    """View for selecting persona creation mode."""

    def __init__(self, view: "PromptManagerView") -> None:
        super().__init__(timeout=180)
        self.parent_view = view

    @discord.ui.button(label="기본 모드", style=discord.ButtonStyle.secondary, emoji="⚡", row=0)
    async def basic_mode(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Basic mode - quick generation without questions."""
        logger.info("basic_mode button clicked - use_questions=False")
        await interaction.response.send_modal(
            PromptCreateModal(self.parent_view, use_questions=False)
        )

    @discord.ui.button(label="AI 질문 모드", style=discord.ButtonStyle.primary, emoji="🧠", row=0)
    async def qa_mode(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """AI question mode - detailed generation with AI questions."""
        logger.info("qa_mode button clicked - use_questions=True")
        await interaction.response.send_modal(
            PromptCreateModal(self.parent_view, use_questions=True)
        )

    @discord.ui.button(label="취소", style=discord.ButtonStyle.danger, emoji="❌", row=1)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Cancel mode selection."""
        await interaction.response.delete_message()

    async def on_submit(self, interaction: discord.Interaction) -> None:
        concept_str = self.concept.value
        use_qa = self.use_questions
        logger.info(
            f"PromptCreateModal.on_submit: use_questions={use_qa}, concept={concept_str[:50]}..."
        )

        if use_qa:
            # Defer and show loading, then send the questions modal
            await interaction.response.defer(ephemeral=True)

            msg = await interaction.followup.send(
                f"🧠 컨셉을 분석하여 질문 생성 중...", ephemeral=True
            )

            cog = self.view_ref.cog
            try:
                questions_json = await cog.llm_service.generate_questions_from_concept(concept_str)

                if not questions_json:
                    await msg.edit(content="❌ 질문 생성에 실패했습니다. 기본 모드로 진행합니다.")
                    # Fall through to direct generation
                    await self._generate_direct(interaction, concept_str, msg)
                    return

                # Parse JSON response
                try:
                    # Clean up the response - remove markdown code blocks if present
                    cleaned_json = questions_json.strip()
                    if cleaned_json.startswith("```"):
                        # Remove markdown code blocks
                        lines = cleaned_json.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]  # Remove opening ```
                        if lines[-1].strip() == "```":
                            lines = lines[:-1]  # Remove closing ```
                        cleaned_json = "\n".join(lines)

                    questions_data = json.loads(cleaned_json)
                    questions = questions_data.get("questions", [])

                    if not questions:
                        await msg.edit(
                            content="❌ 질문 파싱에 실패했습니다. 기본 모드로 진행합니다."
                        )
                        await self._generate_direct(interaction, concept_str, msg)
                        return

                    # Create answer modal and view with button to show it
                    # (Modals can only be sent as first response, so use a button)
                    answer_modal = PromptAnswerModal(self.view_ref, concept_str, questions)
                    modal_button_view = ShowModalButton(answer_modal)

                    # Edit the loading message to show the button
                    await msg.edit(
                        content="📝 **질문이 생성되었습니다! 아래 버튼을 눌러 답변을 입력하세요:**",
                        view=modal_button_view,
                    )

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}, Response: {questions_json[:200]}")
                    await msg.edit(content=f"❌ 질문 파싱 오류. 기본 모드로 진행합니다.")
                    await self._generate_direct(interaction, concept_str, msg)
                except Exception as e:
                    logger.error(f"Error in question modal: {e}", exc_info=True)
                    await msg.edit(content=f"❌ 오류 발생: {str(e)}")
                    await self._generate_direct(interaction, concept_str, msg)

            except Exception as e:
                logger.error(f"Error generating questions: {e}", exc_info=True)
                await msg.edit(content=f"❌ 질문 생성 오류: {str(e)}")
                await self._generate_direct(interaction, concept_str, msg)
        else:
            # Direct generation path
            await interaction.response.defer(ephemeral=True)
            msg = await interaction.followup.send(
                f"🧠 입력된 내용을 바탕으로 페르소나 설계 중...", ephemeral=True
            )
            await self._generate_direct(interaction, concept_str, msg)

    async def _generate_direct(
        self, interaction: discord.Interaction, concept_str: str, msg
    ) -> None:
        """Direct prompt generation without questions."""
        cog = self.view_ref.cog
        try:
            generated_prompt = await cog.llm_service.generate_prompt_from_concept(concept_str)

            if not generated_prompt:
                await msg.edit(content="❌ 프롬프트 생성에 실패했습니다.")
                return

            name_match = re.search(
                r"Project\s+['\"]?(.+?)['\"]?\]", generated_prompt, re.IGNORECASE
            )
            name = name_match.group(1) if name_match else f"Generated ({concept_str[:10]}...)"
            prompt_content = generated_prompt.strip()

            idx = await cog.prompt_service.add_prompt(name, prompt_content)

            # Record usage after successful creation
            await cog.prompt_service.increment_today_usage(interaction.user.id)

            await msg.edit(
                content=f"✅ 새 페르소나 **'{name}'**이(가) 설계되었습니다! (인덱스: {idx})"
            )
            await self.view_ref.refresh_view(interaction)

        except Exception as e:
            logger.error(f"Error in PromptCreateModal: {e}", exc_info=True)
            await msg.edit(content=f"❌ 오류 발생: {str(e)}")


class PromptAnswerModal(discord.ui.Modal):
    """Modal for answering LLM-generated questions about the persona."""

    def __init__(
        self,
        view: "PromptManagerView",
        concept: str,
        questions: List[Dict[str, str]],
    ) -> None:
        super().__init__(title="페르소나 질문 답변")
        self.view_ref = view
        self.concept = concept
        self.questions = questions

        # Dynamically create TextInput fields for each question
        # Use discord.ui.Label to wrap TextInput with description for full question text
        # Label text: 45 chars max, Description: 100 chars max
        for i, q in enumerate(questions[:5]):
            sample = q.get("sample_answer", "자유롭게 작성")
            # Create TextInput without label (label is deprecated, use Label wrapper)
            text_input = discord.ui.TextInput(
                style=discord.TextStyle.long,
                required=False,
                max_length=500,
                placeholder=sample[:100],
            )
            # Wrap TextInput in Label with short text + long description
            # Label text: Short identifier like "Q1" (45 chars max)
            # Description: Full question text (100 chars max, truncate if needed)
            question_text = (
                q["question"][:95] + " (비워두면 예시 사용)"
                if len(q["question"]) > 95
                else q["question"] + " (비워두면 예시 사용)"
            )
            label = discord.ui.Label(
                text=f"Q{i + 1}", description=question_text[:100], component=text_input
            )
            setattr(self, f"answer_{i}", text_input)
            setattr(self, f"label_{i}", label)
            self.add_item(label)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        # Use deferred response because generation takes time
        await interaction.response.defer(ephemeral=False)

        # Build Q&A string
        qa_pairs = []
        for i, q in enumerate(self.questions):
            answer_field = getattr(self, f"answer_{i}", None)
            if answer_field:
                answer_value = answer_field.value.strip()
                # Use user's answer, or fall back to sample answer if empty
                final_answer = answer_value if answer_value else q.get("sample_answer", "")
                qa_pairs.append(f"Q: {q['question']}\nA: {final_answer}")

        questions_and_answers = (
            "\n\n".join(qa_pairs) if qa_pairs else "No additional answers provided."
        )

        msg = await interaction.followup.send(
            f"🧠 답변을 바탕으로 페르소나 설계 중...", ephemeral=False
        )

        cog = self.view_ref.cog
        try:
            generated_prompt = await cog.llm_service.generate_prompt_from_concept_with_answers(
                self.concept, questions_and_answers
            )

            if not generated_prompt:
                await msg.edit(content="❌ 프롬프트 생성에 실패했습니다.")
                return

            name_match = re.search(
                r"Project\s+['\"]?(.+?)['\"]?\]", generated_prompt, re.IGNORECASE
            )
            name = name_match.group(1) if name_match else f"Generated ({self.concept[:10]}...)"
            prompt_content = generated_prompt.strip()

            idx = await cog.prompt_service.add_prompt(name, prompt_content)

            # Record usage after successful creation
            await cog.prompt_service.increment_today_usage(interaction.user.id)

            await msg.edit(
                content=f"✅ 새 페르소나 **'{name}'**이(가) 설계되었습니다! (인덱스: {idx})"
            )
            await self.view_ref.refresh_view(interaction)

        except Exception as e:
            logger.error(f"Error in PromptAnswerModal: {e}", exc_info=True)
            await msg.edit(content=f"❌ 오류 발생: {str(e)}")


class PromptRenameModal(discord.ui.Modal, title="페르소나 이름 변경"):
    new_name = discord.ui.TextInput(
        label="새로운 이름",
        placeholder="변경할 이름을 입력하세요",
        style=discord.TextStyle.short,
        required=True,
        max_length=50,
    )

    def __init__(self, view: "PromptManagerView", index: int, old_name: str) -> None:
        super().__init__()
        self.view_ref = view
        self.index = index
        self.new_name.default = old_name

    async def on_submit(self, interaction: discord.Interaction) -> None:
        cog = self.view_ref.cog
        if await cog.prompt_service.rename_prompt(self.index, self.new_name.value):
            await send_discord_message(
                interaction, f"✅ **{self.new_name.value}**로 변경되었습니다.", ephemeral=False
            )
            await self.view_ref.refresh_view(interaction)
        else:
            await send_discord_message(interaction, "❌ 변경 실패.", ephemeral=False)


class PromptManagerView(discord.ui.View):
    def __init__(self, cog: "PersonaCog", ctx: commands.Context) -> None:
        super().__init__(timeout=600)
        self.cog = cog
        self.ctx = ctx
        self.selected_index: Optional[int] = None
        self.message: Optional[discord.Message] = None
        self.update_components()

    def update_components(self) -> None:
        prompts = self.cog.prompt_service.list_prompts()
        self.clear_items()

        # Select Menu
        options = []
        active_content = self.cog.session_manager.channel_prompts.get(self.ctx.channel.id)

        # Limit to 25 items due to Discord Select Menu limits
        # TODO: Implement pagination if prompt list grows beyond 25
        for i, p in enumerate(prompts[:25]):
            is_active = p["content"] == active_content
            label = p["name"][:100]
            desc = "✅ 현재 적용됨" if is_active else None
            options.append(
                discord.SelectOption(
                    label=label, value=str(i), description=desc, default=(i == self.selected_index)
                )
            )

        select = discord.ui.Select(
            placeholder="관리할 페르소나를 선택하세요...",
            options=(
                options
                if options
                else [discord.SelectOption(label="저장된 프롬프트 없음", value="-1")]
            ),
            min_values=1,
            max_values=1,
            row=0,
            disabled=(not options),
        )
        select.callback = self.on_select
        self.add_item(select)

        # Row 1: New, Manual, Rename
        btn_new = discord.ui.Button(
            label="새로 만들기", style=discord.ButtonStyle.success, emoji="✨", row=1
        )
        btn_new.callback = self.on_new
        self.add_item(btn_new)

        btn_file_add = discord.ui.Button(
            label="프롬프트 추가(파일)", style=discord.ButtonStyle.secondary, emoji="📂", row=1
        )
        btn_file_add.callback = self.on_file_add
        self.add_item(btn_file_add)

        btn_rename = discord.ui.Button(
            label="이름 변경",
            style=discord.ButtonStyle.secondary,
            emoji="✏️",
            disabled=(self.selected_index is None),
            row=1,
        )
        btn_rename.callback = self.on_rename
        self.add_item(btn_rename)

        # Row 2: Apply, Delete, Close
        btn_apply = discord.ui.Button(
            label="채널에 적용",
            style=discord.ButtonStyle.primary,
            emoji="✅",
            disabled=(self.selected_index is None),
            row=2,
        )
        btn_apply.callback = self.on_apply
        self.add_item(btn_apply)

        btn_delete = discord.ui.Button(
            label="삭제",
            style=discord.ButtonStyle.danger,
            emoji="🗑️",
            disabled=(self.selected_index is None),
            row=2,
        )
        btn_delete.callback = self.on_delete
        self.add_item(btn_delete)

        btn_close = discord.ui.Button(
            label="닫기", style=discord.ButtonStyle.secondary, emoji="❌", row=2
        )
        btn_close.callback = self.on_close
        self.add_item(btn_close)

    async def refresh_view(self, interaction: Optional[discord.Interaction] = None) -> None:
        self.update_components()
        embed = self.build_embed()

        try:
            if interaction and not interaction.response.is_done():
                await interaction.response.edit_message(embed=embed, view=self)
            elif self.message:
                await self.message.edit(embed=embed, view=self)
        except Exception as e:
            logger.error(f"Failed to refresh view: {e}")

    def build_embed(self) -> discord.Embed:
        prompts = self.cog.prompt_service.list_prompts()
        embed = discord.Embed(title="🎭 페르소나 관리자", color=discord.Color.gold())

        list_text = ""
        active_content = self.cog.session_manager.channel_prompts.get(self.ctx.channel.id)

        for i, p in enumerate(prompts[:25]):
            marker = "✅" if p["content"] == active_content else "🔹"
            bold = "**" if i == self.selected_index else ""
            name_display = f"{bold}{p['name']}{bold}"
            list_text += f"{marker} `[{i}]` {name_display}\n"

        embed.description = (
            list_text or "저장된 페르소나가 없습니다. '새로 만들기'를 눌러 시작하세요."
        )

        if self.selected_index is not None and 0 <= self.selected_index < len(prompts):
            p = prompts[self.selected_index]
            embed.add_field(name="선택된 페르소나", value=f"**{p['name']}**", inline=False)
            # embed.add_field(name="미리보기", value="내용은 보안상 생략되었습니다.", inline=False)

        return embed

    async def on_select(self, interaction: discord.Interaction) -> None:
        # Permission check handled by interaction_check
        val = int(interaction.data["values"][0])
        if val == -1:
            return
        self.selected_index = val
        await interaction.response.defer()
        await self.refresh_view(interaction)

    async def on_new(self, interaction: discord.Interaction) -> None:
        if not await self.cog.prompt_service.check_today_limit(interaction.user.id):
            await send_discord_message(
                interaction,
                "❌ 오늘 생성 한도(2개)를 모두 사용하셨습니다. 내일 다시 시도해 주세요.",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title="✨ 페르소나 생성 모드 선택",
            description="원하시는 생성 모드를 선택해주세요.",
            color=discord.Color.blue(),
        )
        embed.add_field(name="⚡ 기본 모드", value="컨셉만 입력하여 빠르게 생성", inline=False)
        embed.add_field(
            name="🧠 AI 질문 모드",
            value="AI가 질문을 생성하고 답변으로 상세하게 커스텀",
            inline=False,
        )

        view = PromptModeSelectView(self)
        await send_discord_message(interaction, "", embed=embed, view=view, ephemeral=True)

    async def on_file_add(self, interaction: discord.Interaction) -> None:
        # Check permissions unless NO_CHECK_PERMISSION is set
        if not self.cog.config.no_check_permission:
            if not interaction.user.guild_permissions.manage_guild:
                await send_discord_message(
                    interaction, "❌ 이 기능을 사용할 권한(서버 관리)이 없습니다.", ephemeral=True
                )
                return

        await send_discord_message(
            interaction,
            "📂 프롬프트로 사용할 `.txt` 파일을 이 채널에 업로드해 주세요. (60초 대기)",
            ephemeral=True,
        )

        def check(m) -> bool:
            return (
                m.author.id == interaction.user.id
                and m.channel.id == interaction.channel.id
                and m.attachments
            )

        try:
            msg = await self.cog.bot.wait_for("message", check=check, timeout=60.0)

            attachment = msg.attachments[0]
            if not attachment.filename.lower().endswith(".txt"):
                await send_discord_message(
                    interaction, "❌ `.txt` 파일만 지원합니다.", ephemeral=True
                )
                return

            # Read content
            try:
                content_bytes = await attachment.read()
                content_str = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                await send_discord_message(
                    interaction,
                    "❌ 파일 인코딩 오류: UTF-8 형식이 아닙니다. [변환](https://localizely.com/text-encoding-converter/) 후 업로드해주세요.",
                    ephemeral=True,
                )
                return
            except Exception as e:
                logger.error(f"File read error: {e}")
                await send_discord_message(interaction, f"❌ 파일 읽기 오류: {e}", ephemeral=True)
                return

            name = attachment.filename.rsplit(".", 1)[0]
            idx = await self.cog.prompt_service.add_prompt(name, content_str)

            await send_discord_message(
                interaction,
                f"✅ 새 페르소나 **'{name}'**이(가) 추가되었습니다! (인덱스: {idx})",
                ephemeral=False,
            )
            await self.refresh_view(interaction)

            # Optional: Delete the user's upload message to keep channel clean?
            # await msg.delete() # Might be annoying if user wants to keep it. Leaving it.

        except asyncio.TimeoutError:
            await send_discord_message(
                interaction, "⏳ 시간 초과: 파일 업로드가 취소되었습니다.", ephemeral=True
            )

    async def on_apply(self, interaction: discord.Interaction) -> None:
        if self.selected_index is not None:
            p = self.cog.prompt_service.get_prompt(self.selected_index)
            if p:
                self.cog.session_manager.set_channel_prompt(self.ctx.channel.id, p["content"])
                await send_discord_message(
                    interaction,
                    f"✅ **{p['name']}** 페르소나가 이 채널에 적용되었습니다! (세션 초기화)",
                    ephemeral=False,
                )
                await self.refresh_view(interaction)
            else:
                await send_discord_message(
                    interaction, "❌ 페르소나를 찾을 수 없습니다.", ephemeral=True
                )

    async def on_rename(self, interaction: discord.Interaction) -> None:
        if self.selected_index is not None:
            p = self.cog.prompt_service.get_prompt(self.selected_index)
            if p:
                await interaction.response.send_modal(
                    PromptRenameModal(self, self.selected_index, p["name"])
                )

    async def on_delete(self, interaction: discord.Interaction) -> None:
        # Check permissions unless NO_CHECK_PERMISSION is set
        if not self.cog.config.no_check_permission:
            if not interaction.user.guild_permissions.manage_guild:
                await send_discord_message(
                    interaction, "❌ 이 기능을 사용할 권한(서버 관리)이 없습니다.", ephemeral=True
                )
                return
        if self.selected_index is not None:
            p = self.cog.prompt_service.get_prompt(self.selected_index)
            if p:
                if await self.cog.prompt_service.delete_prompt(self.selected_index):
                    self.selected_index = None
                    await send_discord_message(
                        interaction, f"🗑️ **{p['name']}** 삭제 완료.", ephemeral=False
                    )
                    await self.refresh_view(interaction)
                else:
                    await send_discord_message(interaction, "❌ 삭제 실패.", ephemeral=True)

    async def on_close(self, interaction: discord.Interaction) -> None:
        # Allow anyone to close the menu, or just the author? Usually anyone or author.
        # Let's delete the message.
        await interaction.message.delete()
        self.stop()

    async def _generate_direct(
        self, interaction: discord.Interaction, concept_str: str, msg
    ) -> None:
        """Direct prompt generation without questions."""
        try:
            generated_prompt = await self.cog.llm_service.generate_prompt_from_concept(concept_str)

            if not generated_prompt:
                await msg.edit(content="❌ 프롬프트 생성에 실패했습니다.")
                return

            name_match = re.search(
                r"Project\s+['\"]?(.+?)['\"]?\]", generated_prompt, re.IGNORECASE
            )
            name = name_match.group(1) if name_match else f"Generated ({concept_str[:10]}...)"
            prompt_content = generated_prompt.strip()

            idx = await self.cog.prompt_service.add_prompt(name, prompt_content)

            # Record usage after successful creation
            await self.cog.prompt_service.increment_today_usage(interaction.user.id)

            await msg.edit(
                content=f"✅ 새 페르소나 **'{name}'**이(가) 설계되었습니다! (인덱스: {idx})"
            )
            await self.refresh_view(interaction)

        except Exception as e:
            logger.error(f"Error in _generate_direct: {e}", exc_info=True)
            await msg.edit(content=f"❌ 오류 발생: {str(e)}")


class PersonaCog(commands.Cog):
    """Cog for managing Personas (Prompts) via UI."""

    def __init__(
        self,
        bot: commands.Bot,
        config: AppConfig,
        llm_service: LLMService,
        session_manager: SessionManager,
        prompt_service: PromptService,
    ) -> None:
        self.bot = bot
        self.config = config
        self.llm_service = llm_service
        self.session_manager = session_manager
        self.prompt_service = prompt_service

    @commands.hybrid_command(name="prompt", description="프롬프트(페르소나) 관리 UI를 엽니다.")
    async def prompt_command(self, ctx: commands.Context) -> None:
        """프롬프트(페르소나) 관리 UI를 엽니다."""
        view = PromptManagerView(self, ctx)
        embed = view.build_embed()
        sent_messages = await send_discord_message(
            ctx, "", embed=embed, view=view, mention_author=False
        )
        if sent_messages:
            view.message = sent_messages[0]

    async def cog_command_error(self, ctx: commands.Context, error: Exception) -> None:
        """Cog 내 명령어 에러 핸들러"""
        if isinstance(error, commands.MissingPermissions):
            await send_discord_message(
                ctx,
                f"❌ 이 명령어를 실행할 권한이 없습니다. (필요 권한: {', '.join(error.missing_permissions)})",
                mention_author=False,
            )
        elif isinstance(error, commands.BadArgument):
            await send_discord_message(
                ctx,
                "❌ 잘못된 인자가 전달되었습니다. 명령어를 다시 확인해 주세요.",
                mention_author=False,
            )
        elif isinstance(error, commands.CommandOnCooldown):
            await send_discord_message(
                ctx,
                f"⏳ 쿨다운 중입니다. {error.retry_after:.1f}초 후에 다시 시도해 주세요.",
                mention_author=False,
            )
        else:
            logger.error(f"Command error in {ctx.command}: {error}", exc_info=True)
            if not ctx.command.has_error_handler():
                await send_discord_message(
                    ctx,
                    f"❌ 명령어 실행 중 오류가 발생했습니다: {str(error)}",
                    mention_author=False,
                )

import logging
import asyncio
from discord.ext import commands, tasks

from soyebot.config import AppConfig
from soyebot.services.finetune import FineTuneService

logger = logging.getLogger(__name__)

class FineTuneCog(commands.Cog):
    def __init__(self, bot: commands.Bot, config: AppConfig):
        self.bot = bot
        self.config = config
        self.service = FineTuneService(config)
        # self.finetune_task.start()

    def cog_unload(self):
        self.finetune_task.cancel()

    @tasks.loop(hours=1)
    async def finetune_task(self):
        """Periodic task to check if fine-tuning is needed."""
        # 1달마다 파인튜닝하는 로직 주석처리 (사용자 요청)
        pass
        # # Wait until bot is ready to ensure we can access channels
        # if not self.bot.is_ready():
        #     return
        #
        # try:
        #     await self.service.run_pipeline_step(self.bot)
        # except Exception as e:
        #     logger.error(f"Error in fine-tune task: {e}")

    @finetune_task.before_loop
    async def before_finetune_task(self):
        await self.bot.wait_until_ready()

    @commands.hybrid_command(name="force_finetune_check", description="파인튜닝 점검을 강제로 실행합니다. (봇 소유자 전용)")
    @commands.is_owner()
    async def force_check(self, ctx: commands.Context):
        """Manually trigger the fine-tune check (Owner only)."""
        if ctx.interaction:
            await ctx.reply("🔄 Fine-tune check triggered.", ephemeral=True)
        else:
            await ctx.send("🔄 Fine-tune check triggered.")
        try:
            # We call the service method directly
            # Note: This might overlap with the loop if not careful,
            # but usually okay for manual trigger.
            await self.service.run_pipeline_step(self.bot)
            if ctx.interaction:
                await ctx.reply("✅ Check completed. See logs for details.", ephemeral=True)
            else:
                await ctx.send("✅ Check completed. See logs for details.")
        except Exception as e:
            if ctx.interaction:
                await ctx.reply(f"❌ Error: {e}", ephemeral=True)
            else:
                await ctx.send(f"❌ Error: {e}")

async def setup(bot: commands.Bot):
    # Retrieve config from bot (assuming it's attached or we load it)
    # The main.py usually loads config.
    # We can inspect how other cogs get config.
    # Usually passed in __init__ if we construct it manually,
    # but discord.py setup takes 'bot'.
    # We will assume bot has 'config' attribute or we load it again.

    # Check if bot has config
    config = getattr(bot, 'config', None)
    if not config:
        # Fallback: load config
        from soyebot.config import load_config
        config = load_config()

    await bot.add_cog(FineTuneCog(bot, config))

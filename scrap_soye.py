import discord
import asyncio

TOKEN = "MTQzMDc0MjEwMzU0OTI4NDM3NA.GbtxYw.W0ObJcLH7-U6xpPwVSMd-iSdXHUu6A3yJtcx-Y"      # 봇 토큰 입력
CHANNEL_ID = 1320412164221046784    # 채널 ID 입력
USER_ID = 690009286457426132        # 특정 유저 ID 입력
LIMIT = 20000                       # 불러올 메시지 수

intents = discord.Intents.default()
intents.messages = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"✅ 로그인 완료: {client.user}")
    channel = client.get_channel(CHANNEL_ID)
    if channel is None:
        print("❌ 채널을 찾을 수 없습니다.")
        await client.close()
        return

    messages = []
    async for msg in channel.history(limit=LIMIT):
        if msg.author.id == USER_ID:
            messages.append(f"{msg.content}")

    if messages:
        with open("discord_user_chat.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(reversed(messages)))
        print(f"💾 저장 완료: discord_user_chat.txt ({len(messages)}개 메시지)")
    else:
        print("❌ 해당 유저의 메시지가 없습니다.")

    await client.close()

# 기존 이벤트 루프 재사용
loop = asyncio.get_event_loop()
loop.run_until_complete(client.start(TOKEN))

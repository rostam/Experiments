# list_groups.py
from telethon import TelegramClient
import asyncio

# Replace these with your own values
api_id = 27795035
api_hash = 'ff4eb03c224de02d28480bcd0896d92d'
phone_number = '+491741666549'
# channel_username = 'nevisandegi_khalagh1'
channel_username = '1872245363'
#https://t.me/c/1872245363/209915
session_name = 'group_lister'
async def list_groups():
    async with TelegramClient(session_name, api_id, api_hash) as client:
        print(f"✅ Logged in as {await client.get_me()}\n")

        dialogs = await client.get_dialogs()

        print("📜 Groups and Channels you are in:\n")
        for dialog in dialogs:
            entity = dialog.entity
            if getattr(entity, 'megagroup', False) or getattr(entity, 'gigagroup', False):
                print(f"👥 Group: {entity.title} — ID: {entity.id} - {dialog.id}")
            elif getattr(entity, 'broadcast', False):
                print(f"📢 Channel: {entity.title} — ID: {entity.id}")
            elif getattr(entity, 'is_group', False):
                print(f"💬 Basic Group: {entity.title} — ID: {entity.id}")

        print("\n✅ Done.")

if __name__ == "__main__":
    asyncio.run(list_groups())

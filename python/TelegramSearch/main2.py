from telethon import TelegramClient, events, sync
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel
import re

# Replace these with your own values
api_id = 27795035
api_hash = 'ff4eb03c224de02d28480bcd0896d92d'
phone_number = '+491741666549'
channel_username = 'nevisandegi_khalagh1'
#https://t.me/c/1872245363/209915


client = TelegramClient('session_name', api_id, api_hash)


async def main():
    await client.start(phone_number)


    dialogs = await client.get_dialogs()

    print("Your Telegram Entities (Groups, Channels, Chats):")
    for dialog in dialogs:
        entity = dialog.entity
        name = getattr(entity, 'title', getattr(entity, 'first_name', 'No Name'))
        username = getattr(entity, 'username', 'No Username')
        entity_type = type(entity).__name__

        print(f"Name: {name}, Username: {username}, ID: {entity.id}, Type: {entity_type}")

with client:
    client.loop.run_until_complete(main())

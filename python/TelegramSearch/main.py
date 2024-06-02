from telethon import TelegramClient, events, sync
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel
import re

# Replace these with your own values
api_id = 27795035
api_hash = 'ff4eb03c224de02d28480bcd0896d92d'
phone_number = '+491741666549'
channel_username = 'nevisandegi_khalagh1'


client = TelegramClient('session_name', api_id, api_hash)


async def main():
    await client.start(phone_number)

    # Get the channel entity
    channel = await client.get_entity(channel_username)

    # Get the channel's message history
    offset_id = 0
    limit = 100  # Number of messages to fetch in one go
    all_messages = []

    while True:
        history = await client(GetHistoryRequest(
            peer=channel,
            offset_id=offset_id,
            offset_date=None,
            add_offset=0,
            limit=limit,
            max_id=0,
            min_id=0,
            hash=0
        ))

        if not history.messages:
            break

        messages = history.messages
        for message in messages:
            if hasattr(message, 'message'):
                if message.message is not None:
                    all_messages.append(message.message)

        offset_id = messages[-1].id

    print(len(all_messages))


with client:
    client.loop.run_until_complete(main())

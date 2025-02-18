import json

from telethon import TelegramClient, events, sync
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel
import re

# Replace these with your own values
api_id = 27795035
api_hash = 'ff4eb03c224de02d28480bcd0896d92d'
phone_number = '+491741666549'
# channel_username = 'nevisandegi_khalagh1'
channel_username = '1872245363'
#https://t.me/c/1872245363/209915


client = TelegramClient('session_name', api_id, api_hash)


async def main():
    await client.start(phone_number)

    # Get the channel entity
    channel = await client.get_entity(PeerChannel(int(channel_username)))
    # Get the channel's message history
    offset_id = 0
    limit = 100  # Number of messages to fetch in one go
    all_messages = []

    map_word_to_sentece = []

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
        for m in history.messages:
            if m.reply_to is not None:
                if m.reply_to.reply_to_msg_id is not None:
                    if 209915 == m.reply_to.reply_to_msg_id:
                        if '@' in m.message or '#' in m.message:
                            continue
                        splits = m.message.split("\n")

                        splits[0] = (splits[0].replace('ِ', '').replace('ُ', '')
                                     .replace('َ', '').replace(' ', '')
                                     .replace(":", "").replace("\u200C", "")
                                     .replace("!", ""))

                        if len(splits[0]) > 8:
                            continue

                        # print(splits)
                        if len(splits) == 2:
                            map_word_to_sentece.append({'word': splits[0], 'sentence': splits[1]})

                        elif len(splits) == 3:
                            map_word_to_sentece.append({'word': splits[0], 'sentence': splits[2]})
            if len(map_word_to_sentece) > 1 and len(map_word_to_sentece) % 50 == 0:
                print(len(map_word_to_sentece))

            if len(map_word_to_sentece) == 100:
                with open("crossword_vis/mydata.json", "w") as final:
                    json.dump(map_word_to_sentece, final)
                    return

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

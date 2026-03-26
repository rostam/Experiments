# gather_polls.py
from telethon import TelegramClient
import asyncio
import matplotlib.pyplot as plt
from datetime import datetime

api_id = 27795035
api_hash = 'ff4eb03c224de02d28480bcd0896d92d'

session_name = 'poll_collector'
target_group = -1002298590644

# Persian equivalents of "yes" and "no"
YES_NO_WORDS = ["بله", "بلی", "خیر"]

async def get_polls():
    async with TelegramClient(session_name, api_id, api_hash) as client:
        print(f"Connected as {await client.get_me()}")
        group = await client.get_entity(target_group)
        print(f"Fetching messages from: {group.title}")

        polls_yesno = []
        polls_other = []

        async for message in client.iter_messages(group, limit=None):
            if message.poll:
                poll = message.media.poll
                results = message.media.results

                if not results or not results.results:
                    continue

                # Convert all poll option texts to strings
                options = [str(o.text) if hasattr(o, "text") else str(o) for o in poll.answers]
                votes = [r.voters for r in results.results]
                date = message.date

                poll_data = {
                    'question': poll.question,
                    'options': options,
                    'votes': votes,
                    'date': date,
                }

                # ✅ Fixed: convert opt to string before checking substrings
                if any(any(word in str(opt) for word in YES_NO_WORDS) for opt in options):
                    polls_yesno.append(poll_data)
                else:
                    polls_other.append(poll_data)

        print(f"\n✅ Found {len(polls_yesno)} yes/no polls and {len(polls_other)} other polls.\n")

        # Rest of code unchanged ...


        # Compute vote differences over time
        def compute_vote_diff(polls):
            times, diffs = [], []
            for p in polls:
                sum = 0
                for i in range(len(p['votes'])):
                    sum = sum + p['votes'][i]
                diffs.append(sum)
                times.append(p['date'])
                # diffs.append(diff)
            return times, diffs

        # Plot Yes/No type polls
        times_yesno, diffs_yesno = compute_vote_diff(polls_yesno)
        if times_yesno:
            plt.figure(figsize=(10, 5))
            plt.plot(times_yesno, diffs_yesno, marker='o')
            plt.title("Vote Difference Over Time – Yes/No Polls")
            plt.xlabel("Date")
            plt.ylabel("Vote Difference")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Plot Other polls
        times_other, diffs_other = compute_vote_diff(polls_other)
        if times_other:
            plt.figure(figsize=(10, 5))
            plt.plot(times_other, diffs_other, marker='o', color='orange')
            plt.title("Vote Difference Over Time – Other Polls")
            plt.xlabel("Date")
            plt.ylabel("Vote Difference")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    asyncio.run(get_polls())

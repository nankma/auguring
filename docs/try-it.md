# Try Auguring

A Telegram bot that watches tech news for you — Hacker News, arXiv,
company blogs, the tech press — and either answers on demand or sends a
digest on a schedule, written as an actual summary instead of a list of
headlines.

**Code:** [github.com/nankma/auguring](https://github.com/nankma/auguring) —
the [README](https://github.com/nankma/auguring#readme) has an overview, and
[`docs/system-overview.md`](https://github.com/nankma/auguring/blob/main/docs/system-overview.md)
is the full architecture and design write-up if you want the details.

## Getting access

1. Open Telegram and message **[@mnkInfo_bot](https://t.me/mnkInfo_bot)**
   — send `/start`.
2. It's invite-gated (an access-control feature of the bot, not a
   Telegram thing) — I'll get a ping and approve you, usually quick.

## What to try

Type these as normal messages — no special syntax needed.

**Ask it something:**
- *"What's new in AI this week?"*
- *"Any recent news on NVIDIA?"*
- *"Catch me up on what's happening with robotics"*

**Tell it what you care about**, so future answers steer that way:
- *"I'm interested in climate tech and space"*
- `/interests` — see or clear what it has saved

**Not sure what to follow yet?** Say so, and it walks you through it —
showing real recent headlines and narrowing down from your reactions,
instead of making you name a topic up front:
- *"Help me figure out what to follow"*
- *"I liked that last story you sent — got more like it?"*
- *"I'm getting too much of X and not enough of Y, can we rebalance?"*

**Set a reply language:**
- *"Reply to me in Traditional Chinese from now on"*
- `/language` — see or clear it

**Get it to check in on its own, without being asked:**
- *"Send me a digest every day"*
- *"Actually make that every 6 hours"*

Everything above also works as a slash command (`/interests`,
`/language`) if you'd rather set it explicitly than ask in words — both
do the same thing.

## Feedback

Not looking for long-term usage commitment — mainly want a real second
person to poke at it and say what's confusing, wrong, or annoying.
Brutal feedback welcome, here or as a GitHub issue.

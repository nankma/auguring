# Discord Plan: Adding Discord Alongside Telegram

**Proposed 2026-10-26 — nothing here is built.** Written after an
investigation requested to grow the user base: the target audience skews
young and does not use Telegram.

Sibling of `docs/plans/multi-channel-plan.md` (LINE, on hold). Its item 2
(platform-aware identity) is the same prerequisite and is reused here
rather than re-designed.

## Why Discord is much cheaper than LINE was

| | LINE (on hold) | Discord |
|---|---|---|
| Transport | Webhook only: public HTTPS, domain, real TLS cert | Bot opens an outbound WebSocket to Discord's gateway. No public endpoint — same shape as our Telegram polling |
| Push | Free tier capped at 200 push messages/month, account-wide (the reason it was shelved) | A bot can DM a user it shares a server with; no per-month quota found. Per-route rate limits apply, handled by `discord.py` |
| Reading DMs | n/a | DM content needs no privileged intent. Only reading message content in guild channels does (and past 100 servers needs Discord's approval) |

## Current state that shapes the design

- Live data (checked 2026-10-26): 78 subscriber rows, but **72 are
  `is_test=1`; only 6 are real users**, 3 with push on. A migration only
  has to preserve those 6; there is no large dataset to protect.
- `storage/schema.py` keys the subscriber tables on `chat_id INTEGER`
  (three tables). Discord user IDs are 64-bit snowflakes, which fit, but
  the two ID spaces are unrelated and must not be mixed.
- `agent.py`, `guardrails.py` and `news_push.py` are already platform
  agnostic. `bot.py` is the Telegram glue (1070 lines) and
  `news_push.run_push_cycle` already takes a generic `send` callback.
- Replies are Telegram HTML (`agent.HTML_FORMATTING_RULES`,
  `telegram_html.py`, `split_for_telegram`). Discord renders Markdown
  (`**bold**`, `[text](url)`) and has a 2000-character message cap
  (not re-verified against the official docs yet — confirm before coding).
- Every new subscriber needs admin approval and gets free-trial caps
  (`TRIAL_AGENT_INTERACTION_LIMIT=50`, `TRIAL_PUSH_LIMIT=20`).

## The growth constraint (the part that is not code)

A bot cannot DM a user it shares no server with. A user must either
(a) share a server with the bot, or (b) install the app to their own
account (Discord "user-installable apps", which work in DMs but are
limited to responding to the user's own interactions, i.e. slash
commands — so scheduled push to a user-installed-only user may not work;
verify before relying on it).

Recommendation: run one small **home server** (rules channel, `#start`
channel) that the bot lives in. The invite link is the growth surface:
a user joins, then DMs the bot. Without this, there is no way for a new
user to find or reach the bot at all. This is an operational task, not
code, and should be decided before building.

The per-user admin approval step is also a growth bottleneck. Decide
whether Discord users are auto-approved into the trial caps (cost per
user is tiny — lifetime LLM spend to date was about $0.1) or still queue
for the Telegram admin bot.

## Design

### 1. Platform-aware identity (own PR, behavior-neutral)

Add `platform TEXT` (`"telegram"` / `"discord"`) to the subscriber tables
and make the key `(platform, chat_id)`; every `subscriber_ops` function
and caller gains a `platform` argument defaulting to `"telegram"`.
Because only 6 real rows exist, migrate in place (add column with default
`"telegram"`) instead of the full-reset alternative. Ship with all
existing tests passing unchanged — this PR must not change behavior.
Stored external IDs stay opaque; Discord IDs are stored as integers in
the same column.

### 2. Channel-aware formatting

Thread a `channel` value through `agent.py`'s existing `context` plumbing
(the way `chat_id`/`category` already travel) and select the formatting
rules per channel: keep the report structure (title, subtitle, synthesis,
sources) shared, vary only the markup vocabulary. Add a Discord splitter
(2000-char chunks, never splitting inside a Markdown link or bold span)
and a Markdown validator analogous to `telegram_html.validate`, since
`news_push` retries on invalid markup before sending.

### 3. Discord adapter

New `discord_bot.py` using `discord.py`, started from `combined_bot.py`
alongside the Telegram bots (same event loop). Handles DMs only at first:
reuse the platform-agnostic pipeline (`process_message` / agent turn) by
extracting the Telegram-specific parts of `bot.py` behind a small
interface (`reply(text)`, `typing()`), rather than copying 1000 lines.
Commands (`/interests`, `/language`, start/stop push) become Discord
slash commands that call the same handlers.

### 4. Push across platforms

`list_push_enabled_subscribers()` returns `platform`; `_push_job`
dispatches to `send_push_digest` (Telegram) or a Discord equivalent. The
Discord sender uses the same archive/record logic, a 1-second gap between
chunks, and falls back to plain text on a rejected message, mirroring the
Telegram sender.

### 5. Admin flow

Keep approvals on the existing Telegram admin bot. The request message
must name the platform, `handle_decision` must update the right
`(platform, chat_id)` row, and the approve/deny result must go back out
through the requester's own platform.

## Test plan

- Step 1: the full existing suite passes unchanged with `platform`
  threaded through; add tests that two rows with the same numeric ID on
  different platforms stay separate.
- Formatting: per-channel rule selection; Discord splitter never exceeds
  2000 characters or breaks a link/bold span; validator rejects malformed
  Markdown.
- Adapter: fake `discord.py` objects only (no real API, per repo rule);
  DM in -> agent reply out; unapproved user gets the approval flow.
- Push: dispatch by platform; Discord send failure for one subscriber
  does not stop the cycle (same isolation test Telegram has).
- Resource check before deploy: the production VM is a 1 GB free-tier
  shape; measure the container's memory with the Discord client running
  before shipping.

## Suggested build order

1. Platform-aware identity, Telegram-only behavior (own PR).
2. Channel-aware formatting + Discord splitter/validator (pure code,
   testable without Discord).
3. Extract the Telegram-specific glue from `bot.py` behind an interface.
4. `discord_bot.py` DM adapter + slash commands.
5. Push dispatch and admin flow.
6. Create the home server, register the app, measure memory, deploy.

## Open decisions

1. Home server: yes/no, and who moderates it.
2. Auto-approve Discord users into the trial caps, or keep manual
   approval.
3. DMs only (recommended first), or also answer in server channels
   (needs the Message Content privileged intent).
4. Whether user-installed apps are worth supporting for push (see above).

# Bot Features Plan

Five product features requested for the Telegram bot. Nothing here is built
yet — this doc exists to capture the goal, technical approach, and open
questions before implementation starts, same pattern as
`docs/plans/deployment-plan.md` and `docs/plans/telemetry-and-testing-plan.md`.

## Status

| # | Item | Status | Priority |
|---|------|--------|----------|
| 1 | Bot access control (admin-approval workflow) | **Done — see below** | Was urgent — bot was live and unrestricted |
| 2 | Per-user response language / translation | **Done — see below** | Built 2026-08-08 |
| 3 | Multi-user subscribers + DB-backed sessions | Partially done — approval status (#1), per-user `interests`, and now `language` (#2) are all live; sources/conversation-history persistence still missing | Normal — extend for #4 |
| 4 | Per-user search-source configuration | Not started | Normal — depends on #3 |
| 5 | Proactive news push (per-user configurable interval digest) | **Done — see below** | Was deferred, now built at the user's request (2026-08-08) |
| 6 | Free-trial usage caps (per-subscriber AI-interaction and push limits) | **Done — see below** | Built 2026-09-19 at the user's request |

## 1. Bot access control — done

`bot.py` had **no access control at all** — any Telegram user who found
`@mnkInfo_bot` could message it and consume the owner's DeepSeek API quota.
This was fixed before any cloud deployment, with an approval workflow
rather than a static allowlist (the design was upgraded from the original
plan below once it became clear a real approval flow — not just an env-var
list — was wanted).

**Design actually built: two separate bots sharing one SQLite DB.**

- **`users_db.py`** — a `subscribers` table (`chat_id`, `username`,
  `first_name`, `status` — `pending`/`approved`/`denied` —, `requested_at`,
  `decided_at`) in a SQLite file (`subscribers.db`, path configurable via
  `SUBSCRIBERS_DB_FILE`, same reasoning as other deployment-specific
  paths/endpoints elsewhere in this project being configurable rather
  than hardcoded). Shared by both bots below — this is what lets them
  agree on who's approved without talking to each other directly.
- **`bot.py`** (the public info bot) — `check_access()` runs before every
  message is handled. `ADMIN_CHAT_ID` (env var, not literally hardcoded in
  source — see "hardcode" note below) always passes. Anyone else: approved
  → proceeds normally; pending → told to wait; denied → told no; never
  seen before → a `pending` row is inserted and the admin is notified.
- **`admin_bot.py`** (new file, new bot/token) — a second, admin-only
  Telegram bot whose only job is approvals. When `bot.py` sees a new
  requester, it sends a message to `ADMIN_CHAT_ID` *via `admin_bot.py`'s
  token* with **inline-keyboard buttons** ("Approve" / "Deny",
  `callback_data="approve:<chat_id>"` / `"deny:<chat_id>"`). Tapping a
  button doesn't post a new message — it fires a `callback_query` update
  that `admin_bot.py`'s `CallbackQueryHandler` catches, which updates
  `users_db`, edits the original message to show the decision, and — using
  `bot.py`'s token this time — sends the requester a confirmation.
  `admin_bot.py` re-checks the tapper's ID against `ADMIN_CHAT_ID` on every
  callback too (defense in depth beyond "only the admin has this bot's
  link").
- **Why two bots, not admin-only commands on the one bot**: keeps the
  approval surface (buttons, `/pending`-style admin tooling later) off the
  same bot a stranger can already message — a stranger who found
  `@mnkInfo_bot` never sees `admin_bot.py` exists at all.
- **On "hardcode"**: the request was for a single fixed admin (not a
  dynamic multi-admin list) with no self-service way for anyone else to
  grant themselves access — that's exactly what's built. The ID itself is
  read from an `ADMIN_CHAT_ID` env var rather than literally written into
  the `.py` file, matching how `TELEGRAM_BOT_TOKEN` is already handled —
  avoids a personal Telegram ID sitting in git history if this repo is
  ever made public (see the earlier secret-hygiene incident in this
  project's history). The *behavior* (fixed, non-configurable-by-anyone-
  but-admin) is what "hardcode" was really asking for, and that's what
  this delivers.
- **Tests**: `tests/test_users_db.py` (DB layer), `tests/test_bot.py`
  (`check_access`'s branching — admin bypass, approved/pending/denied,
  new-request registers + notifies), `tests/test_admin_bot.py`
  (`handle_decision` — approve, deny, non-admin tap rejected). All run
  against a temp SQLite file (`isolated_subscribers_db` fixture in
  `tests/conftest.py`), no real Telegram API calls.
- **Deployment note**: the two bots need to see the *same* `subscribers.db`
  file — fine as two local processes sharing a working directory. Decided
  for containerized deployment: **`combined_bot.py`** runs both bots in one
  process/container (see `CLAUDE.md`'s "Running both bots in one process"
  section), driven by the Oracle `VM.Standard.E2.1.Micro` shape's 1GB RAM
  constraint — running `bot.py` and `admin_bot.py` as two separate OS
  processes/containers would each independently load LangChain/
  python-telegram-bot into memory. `bot.py`/`admin_bot.py` still work
  standalone (their own `main()`s are unchanged) for local dev or a future
  higher-RAM shape where splitting back into two containers might be
  preferable for isolation.
- **Original simpler plan (superseded, kept here for context)**: a static
  `TELEGRAM_ALLOWED_USER_IDS` env var, no DB, no second bot, no approval
  flow — just a fixed allowlist checked per-message. Would have worked, but
  gives no path for a friend to self-request access without the owner
  manually editing an env var and restarting the bot each time — the
  approval-workflow version handles that for free.

## 2. Translation / per-user response language — done

The LLM itself can already write in any language — this doesn't need a
translation API or library, just an instruction telling it which language to
respond in. Built 2026-08-08, once `agent.py`'s `dynamic_prompt` middleware
(built for interests/push) made the "small design decision" below moot —
per-invocation prompt injection was already the established mechanism.

- `users_db.py`: `language` column (free text, e.g. "Spanish", "Traditional
  Chinese" — same trust-the-LLM approach as interests' topic strings, not a
  constrained code list), `get_language`/`set_language`.
- `/language` command (show/set/clear), mirroring `/interests` — plus a
  natural-language surface via a new `set_language` tool and router
  category (`guardrails.py`), same command-or-conversation dual surface
  every other subscription feature here has. The router prompt explicitly
  notes that writing a message *in* a non-English language is not itself
  `set_language` — only an explicit request to change the standing
  preference is; verified live that a Chinese `set_interest` message still
  classifies as `set_interest`, not `set_language`.
- The preference is injected in `agent.py`'s `_compose_prompt` (layer 3,
  alongside interests) **regardless of category** — unlike interests, which
  only matter for `news_query`, a language preference should govern every
  reply including subscription confirmations. Also added to layer 4's
  narrow-check categories (same reasoning as `set_interest`/push: layers
  2/3 already constrain the reply's shape).
- `news_push.py`'s `write_push_digest` takes the subscriber's language
  separately and appends the same directive to its own prompt, since push
  digests don't go through `agent.py`'s `dynamic_prompt` middleware at all
  — this was the one place the "which mechanism carries the preference"
  question from the original design note below still applied.
- Persists via `users_db.py`, same as interests/push (survives restarts,
  shared across `bot.py`/`admin_bot.py`/`combined_bot.py`) — the "stateless
  v0" idea below was skipped since the DB already existed by the time this
  was built.

## 3. Multi-user subscribers + DB-backed sessions

`chat_histories: dict[int, list]` in `bot.py` (conversation history) is
still in-memory only — lost on every restart. The subscriber/approval
side of this item shipped as part of #1: **`users_db.py`'s SQLite
`subscribers` table already exists**, tracking `chat_id`, `username`,
`first_name`, `status`, `requested_at`, `decided_at`. **Per-user
interests are now built too** (see below), and so is **#2's language
preference** (own section below, done 2026-08-08) — both extended this
same table rather than a new one, as planned. Still missing: #4's
per-user source selection and persisted conversation history — those are
the only two real gaps left in this item.

- **Store: SQLite**, not a separate database server — already the choice
  made for #1's `subscribers` table, for the same reasons: a single file,
  no extra service to run/secure/scale, `sqlite3` is stdlib (no new
  dependency), matches the project's no-infra-creep pattern (e.g.
  `arize-phoenix-otel` over the full server-bundling package). A real
  client-server DB (Postgres, etc.) would be overkill for an
  owner-plus-a-few-friends subscriber list; revisit only if the user count
  grows enough for concurrent-write contention to become a real concern,
  which SQLite handles poorly.
- **Also evaluated and rejected: Oracle NoSQL Database Cloud Service**
  (Cosmos-DB-like — shard key + additional key columns, native JSON
  column type). Its Always Free allocation (3 tables, 25GB/50 RU/50 WU
  each) is genuinely generous for this project's scale, but it's
  **region-locked to Phoenix (us-phoenix-1)**, and this tenancy's Free
  tier account type is hard-capped at **one subscribed region with no
  increase path** (confirmed via Oracle's own docs and this tenancy's
  actual "exceeded maximum regions" warning when attempting to subscribe).
  Creating a table in the home region instead showed no "Always Free
  eligible" indicator and offered only Provisioned/On-Demand paid capacity
  modes — would have started incurring real charges. Not revisitable
  without a paid account tier.
- **Interests, built**: `interests` column (JSON-encoded list of topic
  strings), added via a checked `ALTER TABLE` migration since the live
  `subscribers.db` predates this column and `CREATE TABLE IF NOT EXISTS`
  alone wouldn't add it to an existing table. `get_interests()`/
  `set_interests()` in `users_db.py`; `set_interests()` upserts since a
  chat_id may have no row at all yet (the admin, who bypasses
  `request_access()` entirely via the `check_access()` fast path).
  `bot.py`'s `/interests` command lets a user show/set/clear their own
  (comma-separated topics). `handle_message()` prepends a bracketed note
  with the user's interests to the *agent-facing* copy of their message
  only — the guardrail classifiers (`docs/plans/guardrails-plan.md`) still judge
  the user's actual raw text, not a synthetic wrapper.
- **This drove a scope change beyond just storage**: the original
  `SYSTEM_PROMPT` and `docs/plans/guardrails-plan.md`'s classifiers were
  hardcoded to "AI industry" specifically, because that's the owner's own
  interest — but subscribers can care about different tech topics. Both
  were broadened to "technology industry" generally (AI included, not
  AI-only), so the guardrails don't reject the bot's own on-topic answers
  to a subscriber's non-AI tech questions.
- **"Session data lifetime"** (raised in the original request) is still an
  open question, not yet decided: does conversation history expire after N
  days of inactivity, or persist indefinitely? A personal-scale bot
  probably doesn't need aggressive expiry, but this should be a deliberate
  choice, not an accident.
- **Deployment implication:** a SQLite file needs a persistent volume — a
  bind mount for local Docker, and a Kubernetes `PersistentVolumeClaim`
  once `docs/plans/deployment-plan.md` item 2 (K8s manifests) is written. Add
  this to that doc's checklist when the DB actually gets built, since it
  wasn't accounted for when the Dockerfile/deployment plan were written.

## 4. Per-user search-source configuration

Let each user pick which of `news_sources.py`'s `SOURCE_REGISTRY` entries
`search_news` draws on for them, instead of the current global
`enabled_sources()` (env-var-gated, same for every user).

- Needs a command, e.g. `/sources` to list available sources with
  enabled/disabled state, and `/sources toggle <name>` (or similar) to
  flip one — persisted per `chat_id` in the DB from #3.
- **Real design question, not just plumbing:** `search_news` today is a
  single `@tool`-decorated function shared by one process-wide agent
  (`build_agent` is called once in `main()`/`bot.py`'s `main()`), and it
  calls `news_sources.enabled_sources()` with no notion of *which user*
  triggered the call. Per-user source lists mean `search_news` needs the
  requesting chat's context at call time. Two ways to get there, neither
  free:
  - Thread `chat_id` through into the tool call somehow (LangChain tools
    don't automatically receive caller-identity — would need e.g. a
    closure/factory that builds a bound `search_news` per request, or
    stashing "current chat_id" in a contextvar the tool reads).
  - Build one agent per user (per-chat `TOOLS` list with a bound source
    set) instead of one shared agent — more memory/setup cost per active
    user, but keeps `search_news` itself simple.
  Not resolved here; needs a decision when this item is actually built.

## 5. Proactive news push — done

Per-user digest of new tech/AI news, pushed without the user asking first
— the opposite of the bot's normal request-then-respond shape. Built
2026-08-08, after being explicitly deferred earlier in the project.

**The problem that shaped the design, found before writing any scheduler
code**: the user reported that on-demand trend queries "always return
similar news." Root cause: `news_sources.py`'s fetchers already normalized
a `published` timestamp per article, but `agent.py`'s `search_news` tool
was dropping it before it ever reached the model — so the model had no
way to judge recency or notice it was reporting the same items again. Any
push feature built on top of that would have had the same problem, just
on a timer. Fixed at the root first:

- `news_sources.py`: every fetcher now also returns `published_dt`, a
  parsed, timezone-aware `datetime` (UTC) alongside the existing raw
  `published` string — via `_parse_iso_published` (HN/NewsAPI/GNews/
  Perigon's ISO-8601-ish date fields) or `_parse_rss_published`
  (feedparser's `published_parsed`, more reliable than parsing RSS/arXiv's
  raw date string by hand).
- `agent.py`'s `search_news` tool now includes the raw `published` string
  in its output to the model, for on-demand queries.

**Design decision: push does not go through the tool-calling agent.** If
periodic push reused `run_agent`/`search_news` (as originally sketched
below), the model would decide on its own how to search, with no
guarantee it wouldn't just re-report the same top-N-by-recency results
every cycle — the exact bug above, recurring on a timer. Instead
`news_push.py` fetches deterministically via `news_sources.enabled_sources()`
directly, filters to genuinely-new articles, and only then does one plain
(non-agentic) `model.invoke()` call to write a digest from that
pre-filtered list. Guarantees no repeats instead of hoping the model
avoids them.

**Since 2026-08-24 that runs once per interest, not once per subscriber.**
A cycle walks the subscriber's interests longest-un-pushed first and sends
up to `MAX_INTERESTS_PER_PUSH` separate messages, each retrieved and
written for one interest alone. An interest with nothing new does not
consume a slot. Two limits bound the noise: `users_db.MAX_INTERESTS` on how
many a subscriber may follow, and `MAX_INTERESTS_PER_PUSH` on how many are
served per cycle — anything that does not fit waits for the next cycle
rather than being dropped.

The reason is retrieval quality, not presentation: the interest string is
the query, and one combined pool for five interests is a broader, vaguer
query than any of the five. See `docs/analysis/cluster-measurements.md`.

**"New" is judged one way, not two — updated 2026-08-19.** The filter
used to be `published_dt <= since`, with `pushed_links` as a fallback for
unparseable dates. That was wrong on two counts, both real production
failures: GNews publishes ~12h behind, so any subscriber pushed more
recently than that had every one of its articles filtered out by
`published_dt` alone, permanently; and an article that qualified but lost
`select_candidate_articles`'s `max_per_topic` cut would fall behind the
next `since` and be excluded forever, unsent and unrecorded.

`pushed_links` — the direct record of what a subscriber actually
received — is now the **only** "already seen" filter. `published_dt` is
ranking only (newest first among candidates); it filters nothing. See
`select_candidate_articles`'s own docstring in `news_push.py` for the
full account. Since 2026-08-24 this filter is scoped to one interest per
call rather than the subscriber's whole interest list — see the
per-interest push section above — but the "seen" rule itself is
unchanged.

**Schema** (`users_db.py`, all migrated via the existing `_ensure_column`
pattern): `push_interval_hours` (int, default 24), `last_push_at` (ISO
string), `pushed_links` (JSON array). New functions:
`get_push_interval_hours`/`set_push_interval_hours` (floored at 1h —
`MIN_PUSH_INTERVAL_HOURS` — so a typo or an over-eager agent can't
schedule sub-hourly pushes), `get_pushed_links`, `get_last_push_at`,
`record_push`, `list_push_enabled_subscribers`.

**Interval control is natural language**, per the router design
(`docs/plans/context-management-plan.md`), not a `/command` — same reasoning as
`push_enabled`: voice input is an eventual goal and voice has no slash
commands. A new `set_push_interval` tool joins `set_push_enabled`; the
router's `start_push` category (`guardrails.py`) now also covers "change
how often an already-enabled push sends." User-requested presets: 24h
(daily, the default), 12h, 6h, 4h — the tool accepts any integer ≥1h, the
presets are just what the agent is instructed to suggest.

**Scheduler**: `python-telegram-bot`'s `JobQueue` (wraps APScheduler, now
a real dependency — `apscheduler` added to `environment.yml` via `mamba`
per the `use-mamba-not-conda` skill; confirmed live that `Application.builder().build()`
only gets a working `job_queue` once apscheduler is importable).
`bot.register_push_job(app)` schedules a single repeating tick
(`PUSH_TICK_SECONDS = 900`, i.e. every 15 min) that calls
`news_push.run_push_cycle()` — a tick-based design, not one APScheduler
job per subscriber, because `push_interval_hours` is user-changeable at
runtime and a DB-driven due-check (`is_subscriber_due`) is simpler to
reason about than dynamically rescheduling individual jobs. Wired into
both `bot.py`'s standalone `main()` and `combined_bot.py`'s
`build_info_app()` (the actual deployed entry point), since
`Application.start()`/`.stop()` auto-start/stop the JobQueue — confirmed
via reading `python-telegram-bot`'s source, not assumed.

**Output guardrail applies to pushes too**: a subscriber's stored
`interests` are unsanitized user text that end up embedded in the digest
prompt — the same injection surface `guardrails.is_output_on_topic`
already guards against for chat replies applies here, since this is also
model output about to reach a real user unread. `run_push_cycle` runs it
before sending; if blocked, the cycle still advances `last_push_at`/
`pushed_links` (so a bad interest string doesn't cause a retry loop every
tick) but doesn't send.

**Send path reuses `bot.py`'s existing safety nets**: `send_push_digest`
calls the same `_normalize_markdown_bold` → `split_for_telegram` →
`send_message(parse_mode=HTML)` → `BadRequest` fallback-to-plain-text
pipeline as `handle_message`, since push digests go through the same
HTML-formatting prompt (`agent.HTML_FORMATTING_RULES`, extracted as a
shared constant so this and `_NEWS_QUERY_INSTRUCTIONS` can't drift apart)
and can fail the same way.

**Per-source-call dedup across subscribers watching the same topic**
(e.g. many users watching "OpenAI") is still not implemented — each
subscriber's cycle fetches independently. Noted as a future optimization,
not needed at current scale (owner + a friend or two).

## 6. Free-trial usage caps — done

Two independent, per-subscriber allowances, assigned once at approval
time (`subscriber_ops.decide`) from `trial.agent_interaction_limit`
(default 50) / `trial.push_limit` (default 20): an AI-agent interaction
count and a news-push count. Each decrements on use; hitting zero cuts
that ONE mechanism off for that ONE subscriber (not the whole bot), and
pings the admin with a "Reset" button.

**Stored as remaining-count columns, not used-count** (`subscribers.
agent_interactions_remaining`/`pushes_remaining`) — `NULL` or `-1` means
unlimited. This was a deliberate schema choice: since the two new columns
are additive (`storage/schema.py`'s `ADDITIVE_COLUMNS`), every subscriber
approved before this shipped gets `NULL` for free the moment the
migration runs, with no backfill/grandfathering code needed at all. Only
a subscriber who goes through `decide(chat_id, approved=True)` AFTER this
shipped gets a real, finite allowance.

**Where each is checked** — both deliberately as early as possible, before
any paid work happens for that request/cycle:
- AI interactions: `bot.process_message`, right after the `interest_sessions`
  continuation bypass and right before layer 2's paid router call. An
  exploration already in progress is NOT re-checked turn by turn — it
  runs to its own natural end (`interest_finder.MAX_TURNS` already bounds
  that to at most a couple more turns), rather than cutting a subscriber
  off mid-conversation. A brand-new request past the limit gets
  `TRIAL_AGENT_LIMIT_MESSAGE` and never reaches the router.
- Pushes: `news_push.run_push_cycle`, right after the due-check, before
  any candidate-article/digest work for that subscriber's cycle. One
  "push" = one cycle, not one per-interest message — consuming happens
  once regardless of how many interests get sent that cycle. A limited
  subscriber sees their remaining count appended to each digest they
  still receive (omitted entirely for an unlimited one).

**Admin notification is a deliberate, narrow exception to the 2026-08-28
Logfire-alerts split** (`news_push._push_job` dropped direct admin-Telegram
access that day because the retry loop no longer decided anything alert-
worthy). A subscriber hitting a trial limit needs a human DECISION (reset
or leave it), the same shape as `notify_admin`'s existing new-access-request
ping — not an ops-health signal, so it doesn't belong on the ops-alerts
side of that split. `admin_bot.py`'s "Reset" button
(`trial:reset_agent:{chat_id}` / `trial:reset_push:{chat_id}`) restores
the subscriber's CURRENT `trial.*_limit` setting (not whatever it was
when they were first approved), and a push reset also re-enables
`push_enabled` (hitting the limit turned it off).

## Other messaging platforms (evaluated, not pursued)

Asked in passing whether WeChat, WhatsApp, or LINE could be additional
front-ends alongside Telegram. Evaluated, not pursued for now — recorded
here so this doesn't get re-litigated from scratch later, same pattern as
`docs/current/ai-news-sources.md` documenting Reddit as considered-and-rejected.

- **WeChat** — no self-serve equivalent to Telegram's BotFather. Real-time
  auto-reply to arbitrary incoming messages needs a WeChat **Official
  Account (服务号/service account)** with **enterprise verification**
  (requires a registered business entity, ~¥300/year) — a personal-account
  path doesn't get useful API access. The alternative, unofficial personal
  automation (`itchat`, `wechaty`, etc.), works by reverse-engineering
  WeChat's protocol, **violates WeChat's ToS, and Tencent actively detects
  and bans automated personal accounts** — not something to build a
  project on.
- **WhatsApp** — has an official Meta Business Cloud API, friendlier than
  WeChat: free-form replies are allowed within a 24-hour window after a
  user messages first (fits this bot's request-then-respond shape), though
  proactive messages outside that window need pre-approved templates and
  a paid tier past a free allowance. Requires a Meta Business account and
  a dedicated WhatsApp Business phone number (can't reuse a personal
  WhatsApp number as-is).
- **LINE** — has an official, developer-friendly Messaging API (closer in
  spirit to Telegram's), free official-account creation, replies to
  inbound messages are unlimited/free, push messages have a monthly free
  quota then paid tiers. Mainly useful if target users are concentrated in
  Japan/Thailand/Taiwan, where LINE dominates.
- **The blocking factor common to both WhatsApp and LINE**: neither
  supports polling — **both require a webhook**, meaning a public HTTPS
  endpoint with a valid TLS certificate. `bot.py` deliberately uses
  Telegram's polling mode specifically to avoid needing that
  infrastructure before a cloud deployment target and domain exist (see
  `docs/plans/deployment-plan.md`). Adding WhatsApp or LINE support would force
  that decision now, ahead of schedule, rather than after the cloud
  provider (`docs/plans/deployment-plan.md` item 3) is chosen.
- **Conclusion**: not worth pursuing while the user base is "owner plus a
  few friends." Telegram already covers that need with the lowest setup
  friction. If multi-platform support becomes worth it later, the natural
  order is: pick a cloud provider → stand up a public HTTPS
  endpoint/domain (needed for webhooks anyway) → then add WhatsApp/LINE,
  not the reverse.

## Open questions

- How `admin_bot.py` and `bot.py` share `subscribers.db` once containerized
  — a mounted volume both point at via `SUBSCRIBERS_DB_FILE`, or one
  container running both processes. Not decided; needs to land in
  `docs/plans/deployment-plan.md` once the cloud provider is chosen.
- Whether `admin_bot.py` should grow a `/pending` command to list
  outstanding requests (in case a notification message is missed/deleted)
  — not built, `list_pending()` already exists in `users_db.py` to support
  it whenever it's wanted.
- Extending `subscribers` (built for #1) vs. a separate table for #2/#4/#5's
  per-user preferences (language, enabled sources, watched topics) — likely
  the same table gets new nullable columns rather than a new table, but not
  decided until one of those items is actually built.
- How `search_news`'s per-user source filtering (#4) is threaded through
  LangChain's tool-calling — the two options sketched above need a real
  decision, not just a plan-doc mention.
- Whether translation (#2) should be a fixed instruction injected per
  request, or whether it's worth maintaining one agent instance per active
  language to avoid rebuilding the instruction on every call — likely
  premature to decide before real usage data exists.

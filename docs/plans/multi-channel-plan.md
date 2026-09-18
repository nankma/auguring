# Multi-Channel Plan: Adding LINE Alongside Telegram

> A much smaller, outbound-only sibling to this plan exists:
> `docs/plans/email-digest-plan.md` (2026-09-17) adds email as a digest
> delivery option without any of this doc's webhook/identity/quota
> problems. The two plans share one open question worth deciding
> together if either is ever picked up — whether to buy the ~$11/year
> domain this doc already priced out, which would unblock LINE's TLS
> certificate AND the email plan's deliverability question at once.

**On hold as of 2026-08-09 — see "Why this is on hold" below.** Nothing
here is built. This doc still exists to capture the goal, technical
approach, and open questions for whenever this gets picked back up, same
pattern as `docs/plans/bot-features-plan.md` and `docs/plans/deployment-plan.md`.

**The ask:** since the bot already runs on a cloud VM rather than a laptop,
add LINE as a second client alongside Telegram, so users can talk to it from
either app.

## Why this is on hold

Researched before writing any code (see "Verified findings" below): LINE's
free tier caps **push** messages at 200/month, shared across the entire
account, not per-user — and push (the periodic digest, `news_push.py`'s
whole feature) is most of what makes this bot worth using over just asking
a chat app's own assistant. Reply messages (on-demand chat) are free and
unlimited, but a LINE integration that could only offer on-demand chat and
not the periodic digest isn't worth the real new infrastructure this needs
(public HTTPS webhook, a platform-aware `users_db.py` schema migration —
see below). Paid LINE tiers remove the cap (¥5,000/month for 5,000
messages) but that's a recurring cost with no revenue model behind it yet.
**Decision: hold this until there's an actual business model that would
justify either the paid LINE tier or accepting chat-only (no push) LINE
support.** Revisit this doc when that's decided — the technical research
below stays valid either way.

## Status

| # | Item | Status |
|---|------|--------|
| 1 | Public HTTPS webhook infrastructure | On hold — see above |
| 2 | Platform-aware identity in `users_db.py` | On hold — see above |
| 3 | LINE channel adapter (`line_bot.py`) | On hold — see above |
| 4 | Platform-conditional reply formatting | On hold — see above |
| 5 | Push notifications across platforms | On hold — see above; the reason this whole plan is on hold |
| 6 | Admin approval flow extension | On hold — see above |

## Why this is a bigger lift than "add a second bot token"

Telegram's design in this project was deliberately polling-based specifically
to avoid a public endpoint — `bot.py`'s own docs say "no public HTTPS
endpoint/TLS needed... same shape locally and in a future Kubernetes
Deployment." LINE's Messaging API has no polling equivalent: it is
webhook-only. LINE's servers POST events to a URL you register, over HTTPS,
with a real (non-self-signed) TLS certificate. This is new infrastructure
this project has specifically avoided until now, not a drop-in second
adapter.

Separately, `users_db.py`'s `subscribers` table is keyed on Telegram's
integer `chat_id` everywhere — interests, push, language, access control.
LINE user IDs are opaque strings (e.g. `U4af4980629...`), which don't fit an
`INTEGER PRIMARY KEY` column at all. This is a schema change, not a new
column.

The good news: `agent.py`, `guardrails.py`, and `news_push.py` already don't
know anything about Telegram specifically — `bot.py` is the only
Telegram-specific glue calling into a platform-agnostic core
(`build_agent`/`run_agent`, `classify_message`, `is_output_on_topic`,
`select_candidate_articles`/`write_push_digest`). That separation already paid off
once this session (see `combined_bot.py`'s docstring: "no changes were
needed to `agent.py`'s core logic to add this second entry point"). Adding
LINE is a new adapter plus the two items below, not a rewrite of the core.

## 1. Public HTTPS webhook infrastructure

LINE requires:
- A publicly reachable HTTPS URL, registered in the LINE Developers console
  as the channel's webhook endpoint.
- A real CA-issued TLS certificate — self-signed won't work, and **Let's
  Encrypt needs a real domain name, not a bare IP.** No domain owned yet.
  **Verified, 2026-08-09**: Porkbun confirmed directly (not a
  third-party estimate) at $11.08/year for a `.com`, flat forever — no
  bait-and-switch renewal jump like Namecheap ($10.98 first year → $18.48
  renewal). Bundles free WHOIS privacy, free DNS, and free Let's Encrypt
  SSL (redundant with the Caddy setup below, but harmless). Cloudflare
  Registrar is a close alternative (genuinely at-cost, no markup) if
  Cloudflare's DNS/proxy is wanted for other reasons, but its orange-cloud
  proxy can interfere with Caddy's automatic cert issuance unless run in
  DNS-only mode or switched to a DNS-01 challenge.
- A fast response (LINE expects an ack within a few seconds); heavy work
  (the actual agent call) should happen after acknowledging, not block the
  webhook response.

**Recommended shape**, matching this project's established "keep it on the
one small VM, avoid new paid infra" pattern (see `docs/plans/deployment-plan.md`'s
E2.1.Micro reasoning): a lightweight reverse proxy (Caddy — auto-provisions
and renews Let's Encrypt certs with a few lines of config, far less
operational overhead than manually running `certbot`) in front of a small
internal HTTP server the bot process runs (e.g. `aiohttp.web`, since
`combined_bot.py` already runs its own asyncio event loop that an
`aiohttp` server can share). Rejected alternative: an OCI Load
Balancer for TLS termination — adds real cost/complexity for a feature this
project can get for free with Caddy on the existing VM.

Same two-firewall-layer gotcha documented in `docs/plans/deployment-plan.md`
("Setup notes") will apply again: both the OCI Security List (cloud-level)
and the VM's local `iptables` (host-level) need a rule opening the HTTPS
port, or the webhook silently can't be reached — this bit the Phoenix OTLP
setup once already this session; expect to hit it again here and check both
layers immediately rather than assuming one is sufficient.

**Signature verification is the actual security boundary here**, replacing
Telegram's implicit trust model (only the bot-token holder can long-poll
Telegram's servers for updates; nobody can push fake updates to you). With a
public webhook, *anyone* can POST to the endpoint, so every request must be
verified via the `X-Line-Signature` header — base64(HMAC-SHA256(channel
secret, raw request body)) — computed and compared before touching the
payload at all. Reject anything that doesn't match. This is not optional
hardening; it's the equivalent of Telegram's guardrail layer 1 for a
completely different threat (spoofed requests, not prompt injection) and
needs the same "verify before processing" discipline.

New secrets (LINE channel secret, LINE channel access token) get the same
treatment as every other credential in this project: OCI Vault +
`docker-entrypoint.sh`'s existing `*_SECRET_OCID` pattern, never baked into
the image or passed as plain env vars in production. No new secrets-handling
design needed — the existing pattern already generalizes.

**Account setup, verified 2026-08-09**: the Messaging API is exclusively
a LINE Official Account feature — confirmed directly from LINE's docs
that a normal personal LINE account has no channel/webhook/API capability
at all, so there's no way to avoid the requirements below by using a
different account type. Creating the Official Account's "Business ID"
requires phone verification (SMS or call) — a one-time step for whoever
operates the bot, not something end users ever encounter (personal LINE
accounts dropped their own phone-number requirement in November 2023;
email/Apple/Google sign-up works fine for users messaging the bot). No
company registration needed — an individual can create one.

## 2. Platform-aware identity in `users_db.py`

Every function in `users_db.py` (`get_status`, `request_access`, `decide`,
`get_interests`/`set_interests`, `get_push_enabled`/`set_push_enabled`,
`get_language`/`set_language`, `list_push_enabled_subscribers`, etc.) is
keyed on a bare `chat_id`. Adding a second platform means every one of these
needs to know *which* platform's `chat_id` it's talking about, since the
same raw ID space could theoretically collide across platforms (unlikely in
practice given LINE's ID format, but the schema shouldn't rely on that).

**Design direction**: add a `platform` column (`TEXT`, e.g. `"telegram"` /
`"line"`) to `subscribers`, and widen `chat_id` from `INTEGER` to `TEXT`
(SQLite's dynamic typing makes this painless — existing Telegram integer
IDs still compare/store fine as text) with the primary key becoming the pair
`(platform, chat_id)`. Every `users_db.py` function signature gains a
`platform` parameter, and every caller (`bot.py`, `admin_bot.py`,
`agent.py`'s tools via `ToolRuntime.context`, `news_push.py`) needs to
thread it through — mechanical but touches nearly the whole module and
everything that calls it. Worth doing as its own isolated commit/PR before
any LINE-specific code lands, so the migration is easy to verify
independently (existing Telegram-only tests should all still pass with
`platform="telegram"` threaded through, proving the refactor didn't change
behavior before any new platform is added).

Existing production data: per this session's established stance on this
DB ("this is a demo, not worth preserving what we'll grow out of" — see
the earlier Oracle Autonomous Database discussion), a clean migration
that resets `subscribers` is acceptable rather than writing a careful
in-place `ALTER`/backfill. Confirm this is still the right call before
migrating, since real approved users and their interests currently live
there.

## 3. LINE channel adapter (`line_bot.py`)

Mirrors `bot.py`'s role: the only LINE-specific file, translating between
LINE's webhook event shape and the same platform-agnostic core `bot.py`
already calls. Per-message flow would be the same four-layer guardrail
pipeline already built (`fails_local_prefilter` → `classify_message` →
`run_agent` → `is_output_on_topic`) — none of that changes; only the
inbound (webhook event → text) and outbound (agent reply → LINE API call)
edges are new.

**Reply mechanism differs from Telegram**: LINE webhook events carry a
`replyToken`, valid once, for a short window — the initial reply must use
`replyMessage`. Anything after that window (notably: periodic push
notifications, which by definition aren't replying to a fresh inbound
event) must use LINE's `pushMessage` API against the user's ID instead.
`news_push.py`'s already-generic `send(chat_id, text)` callback shape
fits this reasonably well, but the channel adapter layer needs to pick the
right one of these two per situation.

**Verified, 2026-08-09** (via LINE's own developer docs, not a
third-party summary): reply messages are free and unlimited regardless
of plan. Push/multicast/broadcast/narrowcast messages are not — the free
"Light Plan" caps these at **200/month, for the whole LINE Official
Account, shared across every subscriber**, not per-user. Paid tiers
raise this (¥5,000/month for 5,000 messages) but that's a real recurring
cost. `news_push.py`'s per-subscriber, potentially-multiple-times-a-day
design would exhaust the free quota almost immediately with even one or
two active push subscribers. **This is the finding that put the whole
plan on hold** — see "Why this is on hold" at the top of this doc.
LY Corp is also restructuring this pricing October 1, 2026; re-verify
the exact numbers before this plan is picked back up, don't trust this
note's figures as still-current without rechecking.

## 4. Platform-conditional reply formatting

`agent.py`'s `HTML_FORMATTING_RULES`/`TREND_REPORT_STRUCTURE` explicitly
instruct Telegram HTML (`<b>`, `<a href="">`, the 📰 report structure) —
this is baked in as a Telegram-only assumption, and LINE's default text
message type doesn't render HTML at all (literal `<b>` tags would show up
as visible text, the same class of bug this project already hit and fixed
once for Markdown-vs-Telegram-HTML). LINE does support richer message
types (Flex Messages, a JSON-based layout format) but that's a materially
different output shape than an HTML string, not just a different tag
vocabulary.

**Design direction**: thread a `channel` (or similar) value through
`agent.py`'s existing `context` mechanism (the same plumbing `chat_id` and
`category` already use via `_compose_prompt`), and branch the formatting
instructions on it — a `_LINE_FORMATTING_RULES` constant alongside the
existing `HTML_FORMATTING_RULES`, selected per-channel the same way
`_LAYER2_BY_CATEGORY` is selected per-category today. Keep the actual
report *structure* (title, subtitle, synthesis, sources) shared between
both — only the markup/tag vocabulary differs.

## 5. Push notifications across platforms

`news_push.run_push_cycle`'s `send: callable` parameter is already
generic in shape, which was a deliberate design choice (kept the module
testable without a live Bot/Application — see its docstring). Extending
it to be platform-aware means `list_push_enabled_subscribers()` also
needs to return `platform` (once item 2 lands), and the caller
(`bot.register_push_job`'s `_push_job`, or wherever this ends up living
once there's more than one channel registering jobs) dispatches to the
right platform's send implementation per subscriber.

## 6. Admin approval flow extension

Recommend keeping admin notifications on the *existing* Telegram admin
bot regardless of which platform a request came from — no need to
duplicate the Approve/Deny UX on LINE too. `notify_admin`'s message needs
to say which platform the request is from, `admin_bot.py`'s
`handle_decision` needs to update the right `(platform, chat_id)` row
(once item 2 lands), and — the part actually easy to get wrong — the
outcome notification back to the requester must go out through *their*
platform's send mechanism, not always Telegram. A LINE user approved via
the Telegram admin bot needs a LINE push message telling them so, not a
Telegram message to a chat_id that doesn't exist for them.

## Suggested build order

1. Item 2 (platform-aware `users_db.py`) alone, with `platform="telegram"`
   threaded through everywhere and the full existing test suite passing
   unchanged in behavior — proves the refactor is safe before any LINE
   code exists to depend on it.
2. Item 1 (webhook infra: domain, Caddy, both firewall layers, signature
   verification) as its own milestone, testable independently with a
   trivial echo handler before any agent logic is wired to it.
3. Items 3-6 (the actual LINE adapter, formatting, push, admin flow) once
   1 and 2 are both solid.

This build order is moot until the plan comes off hold (see the top of
this doc) — the domain-name and push-quota questions that used to gate
starting are both answered now, and the answer to the push-quota one is
why nothing here is starting yet.

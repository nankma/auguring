# Email as an outbound-only digest channel

Written 2026-09-17. Status: **proposed, nothing built.**

**Scope, decided explicitly (2026-09-17): outbound only.** A subscriber can
choose to receive the periodic push digest by email instead of, or in
addition to, Telegram. Nothing about *talking to* the bot changes — adding
an interest, changing language, starting/stopping push, all of it still
happens exclusively through the Telegram conversation. This is deliberately
a much smaller scope than `docs/plans/multi-channel-plan.md`'s LINE
proposal (a full second two-way channel, on hold) — no inbound parsing, no
webhook, no second identity space.

## Why this is a much lighter lift than the LINE plan

`docs/plans/multi-channel-plan.md` is on hold specifically because LINE
requires a public HTTPS webhook (new infrastructure this project has
deliberately avoided) and its free push-message quota (200/month, shared
across the whole account) would be exhausted almost immediately. Neither
problem applies here:

- **No inbound infrastructure at all.** We only ever send. No webhook, no
  TLS certificate, no domain *required* for the mechanism to work (though
  see "Deliverability" below for why one is still worth having).
- **No second identity space.** A LINE user needed the `(platform,
  chat_id)` redesign in multi-channel-plan.md's item 2 because LINE IDs are
  a genuinely separate address space. An email address is just one more
  attribute on the *same* Telegram-identified subscriber row — Telegram's
  `chat_id` stays the one and only primary key.
- **No harsh volume ceiling.** See "Provider and cost" below — the
  candidate provider's free tier (100 emails/day) is roughly one order of
  magnitude above this project's entire current push-enabled subscriber
  count, not a wall this design runs into immediately the way LINE's
  200/month did.
- **HTML is a native fit, not a parallel formatting system.** LINE's
  plain-text default meant inventing `_LINE_FORMATTING_RULES` alongside
  the existing Telegram HTML rules. Email renders real HTML natively — the
  existing report structure likely needs only a `<html><body>` envelope,
  not a second formatting language.

## Why outbound-only still needs a verification step

An unverified, self-reported email address is a real problem even in an
outbound-only design: a typo silently sends someone else's inbox a
recurring tech-news digest forever, and repeated sends to a bad/foreign
address risk spam complaints against the sending domain's reputation
(which would degrade delivery for every subscriber, not just the one bad
address). The fix does **not** require any new public infrastructure:

**Verification loop stays entirely inside Telegram.** Subscriber says
something like "email my digest to me@example.com" → the bot generates a
short one-time code, emails it to that address, and stores the address as
pending/unverified → the subscriber pastes the code back into the Telegram
chat → `email_verified` flips true and digests start going out on that
channel. No confirmation web page, no click-through link, no second HTTP
endpoint — the same "avoid new public infra" instinct that shaped this
project's original polling-over-webhook choice for Telegram itself.

This is a different shape from `propose_interest`'s confirm-before-save
gate (`docs/plans/interest-finder-plan.md`): an interest needs grounding
(does the topic have real coverage) before a subscriber can meaningfully
agree to it; an email address needs proof of ownership, which a preview
can't provide — a real code sent to the real inbox is the only test that
actually verifies anything.

## Provider and cost

**Recommended: a transactional email API over HTTPS, not raw SMTP.**
Verified live, 2026-09-17: Oracle Cloud Infrastructure blocks outbound TCP
port 25 by default for every tenancy created after June 23, 2021 —
regardless of free or paid tier — and OCI's own Email Delivery relay
service (the sanctioned workaround) is paid-account-only. Rather than
filing an OCI service-limit exemption request for raw SMTP, every other
piece of this project that talks to an external service already goes over
HTTPS (DeepSeek, the news source APIs, Logfire) — a transactional email
API keeps that pattern instead of introducing the one exception.

**Resend** is the leading candidate, not yet decided: free tier is 100
emails/day / 3,000/month (resend.com/docs/knowledge-base/account-quotas-and-limits,
checked 2026-09-17), ships DKIM/SPF/DMARC and a plain REST API on every
tier including free. At this project's current scale (a handful of
push-enabled subscribers, each getting at most a few digests a day) this
free tier has enormous headroom — the opposite of LINE's situation, where
the free tier was the reason the whole plan stalled. SendGrid (100/day
free) and Mailgun are the obvious alternatives if Resend's terms don't
work out; not independently researched yet.

**Open question, not resolved by reading Resend's docs alone (2026-09-17):
does sending to arbitrary real-world recipients require a verified custom
domain, or does a shared/sandbox sender address work for production
traffic too?** Their docs describe "up to 3 verified domains" on the free
plan without saying whether verification is a hard prerequisite for
sending to non-account-owner addresses. Needs a real signup and a real
test send to resolve, not more documentation reading.

**This connects directly to `multi-channel-plan.md`'s own domain
research.** That plan already priced a `.com` at $11.08/year flat via
Porkbun, for LINE's Let's Encrypt certificate requirement. If a verified
domain turns out to be required for reliable email delivery too, the same
purchase would unblock both plans at once — worth deciding once, not
twice, if LINE ever comes off hold around the same time this is built.

## What changes

1. **`subscribers` schema** (`users_db.py` / `storage/`'s pluggable
   backend, per the three-layer architecture from PR #79): add `email`
   (nullable), `email_verified` (bool, default false), and a delivery
   preference — `push_channel` (`"telegram"` default / `"email"` /
   `"both"`). Existing subscribers are unaffected — the default preserves
   today's exact behavior.
2. **A small email-sending adapter** (e.g. `mail_sender.py`), mirroring
   `bot.py`'s role for Telegram: the only file that knows about the
   chosen provider's API, exposing a plain `send(to_email, subject,
   html_body) -> bool` function everything else calls. Same shape as
   `news_push.run_push_cycle`'s existing generic `send: callable`
   parameter (already designed to be provider-agnostic — see that
   module's own docstring).
3. **Verification flow**, entirely inside the existing Telegram
   conversation (see above) — likely a small addition to the
   interest-finder-style settings surface (`interest_finder.py`'s
   `set_language`-style direct-effect tools are the closest existing
   precedent, though a code-verification step is genuinely new, not just
   a copy of that tool's shape) or a dedicated simple command. Not
   designed in detail yet — flagged as its own decision in "Open
   questions" below.
4. **Email body formatting**: reuse `agent.py`'s existing report
   structure (title, subtitle, synthesis, sources) inside a minimal HTML
   envelope, rather than inventing a parallel formatting-rules constant
   the way LINE would have needed. Needs a live check against at least
   Gmail and Outlook rendering before trusting it — same "verify, don't
   assume" discipline `telegram-message-formatting`'s own skill file
   already applies to Telegram's HTML subset.
5. **`news_push.py` dispatch**: `list_push_enabled_subscribers()` gains
   `push_channel`/`email`/`email_verified` in its return shape; the
   caller picks the Telegram send callback, the new email send callback,
   or both, per subscriber's preference — a smaller version of
   multi-channel-plan.md's own item 5, without that plan's `(platform,
   chat_id)` redesign since there's still only one identity space here.
6. **Secret handling**: the provider's API key follows this project's
   existing pattern exactly — OCI Vault + `docker-entrypoint.sh`'s
   `*_SECRET_OCID` convention in production, a plain env var locally,
   never a literal value in `settings.yml`/`settings.oracle.yml` (see
   `CLAUDE.md`'s Landmines and `docs/standaloneplan/01-settings-migration.md`).
   No new secrets-handling design needed.
7. **Failure handling**: an email bounce or provider-side failure should
   probably follow the same shape as `news_push.py`'s existing
   three-strikes Telegram-unreachable handling, rather than a silently
   different failure mode for one channel — not designed in detail yet,
   and `TODO.md` already flags that the broader alerting mechanism
   (`llm_usage` tracking, admin notification on a push strike-out) isn't
   fully built regardless of this plan.

## Open questions

- **Does the chosen provider actually require a verified domain for
  production sending?** See "Provider and cost" above — this is the one
  real blocker, and it's a same-day thing to resolve (sign up, try a real
  send), not a research project.
- **Where does "set my digest email" actually live?** A natural-language
  tool inside the interest-finder agent (consistent with how
  `set_language`/`set_interest` moved there, `docs/plans/interest-finder-plan.md`'s
  front-door redesign), or a simpler dedicated flow specifically for the
  verification-code exchange? The code-based confirm step doesn't have an
  exact existing precedent to copy.
- **Is `push_channel="both"` worth supporting**, or should a subscriber
  pick exactly one? Cost is not the constraint here (unlike LINE), so
  "both" seems cheap to allow, but doubles what a bug in either path could
  affect one subscriber with.
- **Real rendering check across mail clients** — Gmail, Outlook, and
  Apple Mail at minimum — before trusting the reused HTML report
  structure email-side.
- **Should a domain be purchased now**, given it would unblock this plan's
  own open deliverability question AND multi-channel-plan.md's LINE
  TLS-certificate requirement at the same time? Worth deciding once
  either plan is actually picked up, not necessarily before.

## Suggested build order

1. Resolve the domain/verified-sender open question with a real signup
   and a real test send — this is the one finding that could reshape
   everything else here, so it goes first, same as multi-channel-plan.md
   putting its own push-quota research before any code.
2. Schema change (`email`/`email_verified`/`push_channel` columns) alone,
   default-preserving, full existing test suite passing unchanged —
   proves the migration is safe before any sending code depends on it.
3. The email-sending adapter and `news_push.py` dispatch change, testable
   with a fake `send` callable the same way Telegram push already is
   (`news_push.py`'s own docstring on why `send` is a parameter).
4. The verification flow (code generation, Telegram-side confirmation)
   and wherever "set my digest email" ends up living.
5. A real end-to-end live test: set an email, verify it, receive one real
   digest, check it in at least two real mail clients.

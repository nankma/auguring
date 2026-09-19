# TODO

Follow-ups noted during work, not yet scheduled. See CLAUDE.md's
"Design before code" -- none of these get implemented without a design
pass first, per that rule.

- [ ] **Add rate limiting for approved users.** No per-user or global
  message-rate cap exists anywhere in `bot.py`/`agent.py` today -- once
  approved, a user (or a compromised approved account) can trigger
  unlimited DeepSeek + news-source calls. Cost-control gap more than a
  security one at current scale, but worth fixing before approving more
  subscribers. See `docs/plans/security-plan.md` finding #3 (Medium,
  "Not started").

- [ ] **Cap admin-notification spam from repeated unapproved requesters.**
  `check_access()` only notifies once per distinct `chat_id`, but nothing
  stops someone from creating many Telegram accounts and messaging from
  each, generating one admin notification per account (no DeepSeek cost,
  just notification noise). Proposed fix: a global "max pending requests
  per hour" cap in `bot.py`. See `docs/plans/security-plan.md` finding #4
  (Low, "Not started -- cheap to fix, low urgency").

- [ ] **Add CI vulnerability/image scanning.** `.github/workflows/ci.yml`
  runs `pytest` only -- nothing scans `environment.yml`'s pinned packages
  or the built Docker image for known CVEs. `docker scout cves` (built
  into Docker Desktop/CLI) or Trivy are the natural options, as a CI step
  or a periodic scheduled job. See `docs/plans/security-plan.md` finding
  #7 (Medium, "Not started").

- [ ] **Add automated secrets-scanning to CI.** The project's practice of
  scanning changed files for secret-like strings before every commit is
  currently manual (established after the historical DeepSeek-key leak).
  A lightweight tool like `gitleaks` as a CI step or pre-commit hook would
  make this systematic instead of relying on remembering it every time.
  See `docs/plans/security-plan.md` finding #8 (Medium, "Not started").

- [ ] **Confirm GitHub branch protection on `main` is actually enforced.**
  Walked through the exact settings to apply (require PR, require the
  `test` status check, no admin bypass, no force-push/deletion) via
  GitHub's web UI, but applying it is a manual step never confirmed done
  -- `gh` CLI wasn't available in-session to verify programmatically.
  Carried over from `docs/plans/telemetry-and-testing-plan.md` item 4 and
  restated as `docs/plans/security-plan.md` finding #9 ("Unknown, still
  unresolved"). Needs a manual check in GitHub's repo settings.

- [ ] **Restrict SSH source IPs on both live VMs.** Port 22 is currently
  open to `0.0.0.0/0` via the default OCI security list on both the bot
  VM and the (now-stopped) second VM -- fine for a single-owner personal
  box today, but deliberately not narrowed yet because the operator's
  home IP is dynamic and a host-level `iptables` rule risks a lockout.
  The fix belongs at the OCI Security List (cloud-level, revertible from
  the console without SSH access), not `iptables`. See
  `docs/plans/security-plan.md` finding #14 ("Not done yet").

- [ ] **Back up `subscribers.db`.** It lives in a single Docker named
  volume (`myfirstagent-data`) on the bot VM with no copy anywhere else --
  if the volume is lost (disk failure, accidental `docker volume rm`, a
  bad migration), every approval decision and subscriber preference is
  gone. A simple periodic `sqlite3 .backup` copied to cloud object storage
  would cover it; cheap insurance, not urgent at "owner plus a couple of
  friends" scale. Two docs flag the same gap from different angles --
  `docs/plans/security-plan.md` finding #13 and
  `docs/plans/data-layer-plan.md` item 2 -- treat as one piece of work,
  not two.

- [ ] **Decide the trigger for migrating off SQLite to a shared
  database.** `docs/plans/data-layer-plan.md` item 1 records this as
  deliberately deferred (migrating now means choosing a target before the
  real requirements -- user count, a second host, a webhook-based channel
  -- are known), not abandoned. Revisit only when one of the doc's named
  triggers actually fires (a second host needs the data, real users with
  data worth keeping, a webhook channel changing the deployment topology,
  or real write concurrency) -- until then this is a standing decision to
  keep deferring, not a task, but worth a TODO so the trigger list doesn't
  get forgotten.

- [ ] **Build the rest of the incident-monitoring alerting mechanism:
  admin notification on a push strike-out, `llm_usage` tracking, and
  `/status`.** `docs/plans/incident-monitoring-plan.md`'s "Status: step 1
  built" section shipped `push_outcomes` and the three-strikes-disable
  action, but three pieces are still explicitly "to build": the
  `alert_state` machine (superseded in spirit by the Logfire alerts that
  now exist, but a strike-out disabling a subscriber's push still isn't
  itself alerted to the admin), a mirrored `llm_usage` table (shaped like
  the existing `api_budget` table but for LLM calls, so spend is
  attributable per-caller rather than only visible on the invoice), and
  the `/status` admin-bot command (a plain-text dashboard: subscriber
  counts, push/LLM/ingestion figures with a trailing 7-day median).

- [ ] **Add internal id pseudonymisation before spans carry raw
  `chat_id`.** `docs/plans/observability-platform-plan.md`'s "order of
  work" step 7 is still open: `chat_id` (a real Telegram user identifier)
  currently travels through every log line and would travel through every
  span if left as-is. `news_push._record` already uses
  `subscriber_ops.external_id(chat_id)` on the span specifically, but the doc
  flags the broader pseudonymisation pass as "best done before there is a
  backlog of spans carrying the raw id" -- hygiene rather than privacy
  engineering at this project's current stage, but not yet done project-wide.

- [ ] **Alert on low available memory.** No alert exists for the deploy
  target running low on memory -- worth having before the `vector_store`
  work (news_cache's pluggable `sqlite_vec` backend, 2026-09-05) loads
  more into a single SQLite file than the old one-YAML-file-per-article
  cache ever held at once. `VM.Standard.E2.1.Micro` historically had very
  little free memory to begin with (~420MB when `news_embed.py`'s
  model2vec choice was measured) -- a threshold-based Logfire alert (or a
  simple periodic `free -h`-equivalent check) would catch this before an
  OOM kill does.

- [ ] **Alert on DB size exceeding some threshold.** No alert exists for
  `subscribers.db`/the new vector-store SQLite file growing past a
  reasonable size. Directly relevant once `vector_store`'s `sqlite_vec`
  backend ships: unlike the old per-article YAML files (which the OS
  filesystem just held individually), a single growing `.db` file is
  easier to lose track of size-wise until it's already a problem.

- [ ] **Alert on total on-disk storage exceeding some threshold.**
  Broader than the DB-size alert above -- covers `NEWS_ARCHIVE_DIR`
  (already growing, unbounded, see the next item), `NEWS_CACHE_DIR`/the
  vector store, and `message_archive` together. The free-tier VM's disk
  is finite even though `news_cache.py`'s own comment currently frames it
  as "not a consideration" at 48h-retention scale (~130MB/month) -- worth
  a real number once retention windows grow past that assumption.

- [ ] **Decide a retention TTL for archived (expired) vector-store data,
  don't keep it forever.** `vector_store`'s `sqlite_vec` backend
  (2026-09-05) marks expired articles `archived_at` instead of deleting
  them (preserving their embeddings for later analysis, matching
  `NEWS_ARCHIVE_DIR`'s existing behavior for the `yaml_files` backend) --
  but neither backend's archive has ever had its own expiration. This was
  raised and deliberately left open during that design pass rather than
  decided on the spot: at minimum needs a real number (30 days? 90? tied
  to one of `docs/analysis/cluster-measurements.md`'s findings?) and a
  cleanup mechanism, before archived data quietly becomes the same kind
  of unbounded-growth risk the three alerts above exist to catch.

- [ ] **Create the two remaining HTML-validation Logfire alerts.**
  `news_push._emit_html_validation_attempt` has been emitting an
  `html_validation_attempt` span per retry attempt since the 2026-08-28
  rework, but `argus html validation retry` (low severity, `valid=false`
  AND `attempt` in (1,2)) and `argus html validation exhausted` (high
  severity, `attempt=3 AND valid=false`) were never created -- see
  `docs/plans/observability-platform-plan.md`'s 2026-08-29 section and
  `docs/current/telemetry-catalog.md`'s alert table, which both still mark
  them `*(planned)*`.

- [ ] **Deploy the arxiv/venturebeat_ai ingest fix (PR #96, merged
  2026-09-17) to INT and PROD.** Raises `REQUEST_DELAY_SECONDS` from 1.1
  to 3.0s so arxiv's 6 sequential per-cycle section calls respect arXiv's
  own documented "no more than one request every three seconds" rule
  (was causing most pulls to fail with 429/timeout), and throttles
  `venturebeat_ai` to a once-daily pull purely to monitor for when
  VentureBeat's own site-wide Vercel bot-challenge 429 gets fixed on
  their end (confirmed live: not a rate-limit issue, nothing to fix on
  our side). code-reviewer and qa-engineer both passed it, CI green on
  `main` -- just never deployed anywhere yet.

- [ ] **Investigate a possible layer 4 (`is_output_on_topic`) false
  positive, found 2026-09-10 during the front-door redesign's real-model
  QA pass.** While reproducing the stale-proposal-classification fix
  live, a qa-engineer run saw layer 4 block a legitimate mid-exploration
  reply about redefining an interest toward "quantum computing" inside
  an already-open `find_interests` session. Flagged only as a side
  observation at the time (explicitly out of scope for that fix) and
  never chased further -- worth a dedicated repro to determine whether
  it's a real guardrail-prompt gap (e.g. the output-scope prompt reading
  "quantum computing" as an off-topic/self-disclosure signal) or a one-off.

- [ ] **Decide whether `agent.models.main`/`models.guardrail` should move
  off `deepseek-v4-flash`.** The cross-model investigation behind the
  interest-finder front-door redesign (2026-09-10, see
  `docs/plans/interest-finder-plan.md`'s "front door redesign" section)
  found DeepSeek's own newest hosted release less reliable at honest
  tool-calling than an older checkpoint of the same model family hosted
  by Together.ai, and found GLM-5.3-Flash both reliable and better at
  recovering from a dead end in that one reproduction run -- one
  conversation, not a benchmark, so nothing has been changed. Also
  surfaced 2026-09-17: a purpose-built typed-decision model, Jev AI
  (jevai.org), pitches itself directly at the "LLM guardrail scoring"
  use case at a lower quoted per-token rate than DeepSeek's cache-miss
  price (though DeepSeek's cache-hit price, which our repeated-system-
  prompt guardrail calls likely benefit from, may already undercut it --
  never measured). Not started either way: blocked on applying for Jev
  AI's early access and writing a custom adapter (its API isn't
  OpenAI-wire-compatible, so `agent.build_model_from_config`'s
  `ChatOpenAI` path can't reach it as-is), then a real `tools/
  measure_guardrails.py` comparison before trusting either option over
  the current default.

- [ ] **Resolve whether OCI's "Always Free" Email Delivery service is
  actually usable on this tenancy before building
  `docs/plans/email-digest-plan.md`.** Oracle's own docs disagree with
  each other: the Always Free resources page advertises "3,000 emails/month
  free," but the official service-limits page puts a true Always-Free
  account's Email Delivery cap at 0 emails/24h -- 200/day is the Trial
  tier, 50,000/day is Pay-As-You-Go/Universal-Credits. Having a card on
  file (confirmed 2026-09-17: OCI's signup verification charge went
  through) doesn't by itself prove which of those three tiers this
  tenancy is actually in. Cheapest next step is checking directly in the
  OCI Console (Account Management shows the tenancy's real billing
  category) rather than reasoning from the docs further. Also note,
  regardless of tier: OCI Email Delivery still needs a verified Email
  Domain + Approved Sender configured first (same domain-ownership
  prerequisite `email-digest-plan.md` already flags for Resend) -- it
  isn't a zero-setup test either way. No OCI CLI is configured anywhere
  in this project (`local-infra/infrastructure.yaml`'s own `oci:` note)
  to check or configure this programmatically today. Put aside for now,
  2026-09-18 -- revisit if/when the email-digest plan is picked up.

- [ ] **Decide whether push has a total-volume cap, not just a
  frequency one.** `push_interval_hours` bounds how OFTEN a subscriber is
  pushed and `UNREACHABLE_STRIKES` (default 3) auto-disables push after
  repeated delivery FAILURES, but nothing bounds how many pushes a
  subscriber can receive in total -- a subscriber with push enabled and
  reachable keeps getting digests (each one a real `search_news`+
  DeepSeek cost) indefinitely. Related to, but distinct from, this file's
  existing "Add rate limiting for approved users" item above (that one's
  about inbound message rate, this is about outbound push volume) --
  raised 2026-09-18, not designed yet.

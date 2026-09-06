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

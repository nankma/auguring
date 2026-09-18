# Docs index

Three kinds of document, each answering a different question — this
index exists so it's clear which is which at a glance.

> A fourth kind lives in [`docs/analysis/`](analysis/README.md) — research
> and measurement: the news-ranking survey, the cross-domain
> sample-diversity survey, the measured cluster numbers, and the scripts
> that produce them. The split from `plans/` is deliberate: `plans/`
> records what we decided and built; `analysis/` records what we measured
> and what the literature says, most of which will never become code.
>
> A fifth lives at [`docs/standaloneplan/`](standaloneplan/README.md) —
> the rough, evolving plan for refactoring this into a service that runs
> standalone (no cloud dependency required) as well as on the current
> cloud deployment: settings abstraction was built and extracted to its
> own project, **[Trailsign](https://pypi.org/project/trailsign/)**,
> because the design turned out to be genuinely content-independent;
> delivery channels, management surface, telemetry pluggability, and
> storage are still rough/not started (`docs/standaloneplan/README.md`).
> This repo's own migration of its `os.environ` reads onto Trailsign is
> tracked separately, since it moves faster than the rest of the plan —
> [`docs/standaloneplan/01-settings-migration.md`](standaloneplan/01-settings-migration.md),
> storage paths done 2026-09-01, everything else not yet.

## `docs/` (top level) — kept at a stable path, linked externally

Thematically these belong in `docs/current/`/`docs/reference/` below, but
live at `docs/` directly instead: `README.md`, `tools/build_showcase.py`,
and outside links (e.g. a shared showcase page) reference them by exact
path, and a reorg that moves the file breaks those links silently. Moved
back here 2026-08-16 after the `docs/current/`/`docs/reference/` split
did exactly that.

| Doc | Covers |
|---|---|
| `system-overview.md` | Full architecture, request pipeline, design principles. No decision history — read its own Appendix B or the matching plan doc for that. Should stay accurate to the running system or it's actively misleading, not just stale |
| `try-it.md` | User-facing invite doc — how to try the live bot |

## `docs/plans/` — what we decided, what's done, what's still open

Carries history and reasoning, not just current state. Each has its own
Status table.

| Doc | Covers |
|---|---|
| `bot-features-plan.md` | Access control, translation, multi-user sessions, per-user sources, push |
| `context-management-plan.md` | Layered prompt design, the router, settings-dispatch refactor |
| `data-layer-plan.md` | SQLite vs. a shared database — deferred, why |
| `deployment-plan.md` | Containerization, cloud provider, CI/CD |
| `email-digest-plan.md` | Email as an outbound-only push-digest channel — proposed, nothing built |
| `guardrails-plan.md` | The four-layer guardrail design and its incidents |
| `incident-monitoring-plan.md` | What counts as an incident; the three criteria, criterion 1 built |
| `interest-finder-plan.md` | The "help me find my interests" conversation — why it gets an agent loop, session state, the two ceilings, and how it became the front door for adding/removing interests and switching language too |
| `interest-definition-plan.md` | Making the interest definition visible and refinable — the paragraph that actually decides what a subscriber receives |
| `observability-platform-plan.md` | Where telemetry and alerting live — moving to hosted Logfire, retiring the Phoenix VM |
| `dev-environment-plan.md` | Running the pipeline against a scratch database instead of production |
| `local-news-cache-plan.md` | Periodic ingestion + local cache, two-stage filtering |
| `model-portability-plan.md` | Swapping/routing LLM models, the guardrail-measurement harness |
| `multi-channel-plan.md` | Adding LINE — on hold |
| `security-plan.md` | The numbered security findings and their status |
| `taxonomy-and-admin-plan.md` | DB-backed category taxonomy, admin-in-the-loop growth, the admin console |
| `telemetry-and-testing-plan.md` | Test infrastructure, CI, Phoenix, what's covered vs. not |

## `docs/current/` — what actually exists, right now

No decision history — read `system-overview.md`'s own Appendix B (top
level, see above) or the matching plan doc for that. These should stay
accurate to the running system or they're actively misleading, not just
stale.

| Doc | Covers |
|---|---|
| `ai-news-sources.md` | The live source registry — what's enabled, what needs a key |
| `infrastructure.md` | VM topology, security model, deploy process shape (no real IPs/keys — see `local-infra/infrastructure.yaml`) |
| `telemetry-catalog.md` | Every span reaching Logfire — service/scope/attributes/level — and which alert (if any) reads each one |

## `docs/reference/` — how to do something

Technique and tool usage, not a decision record.

| Doc | Covers |
|---|---|
| `setup.md` | First-time environment setup, running any piece locally |
| `observability-and-debugging.md` | Diagnosing a live issue — Logfire traces, hot-patching (Phoenix section kept for historical/frozen-data queries only) |
| `local-testing-api-plan.md` | `test_api.py`'s design and how to use it (named `-plan` from before it was built; content is now a reference, not an open plan) |

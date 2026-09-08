# Telemetry & Testing Plan

Why this exists: we want CI-runnable tests for `agent.py` that hit no real LLM
and no real telemetry service — everything mocked, writing to memory or local
files only, with results compared against an expected baseline. That requires
the agent's model and its logging/telemetry to both be swappable at the point
they're constructed, rather than hardcoded.

## Status

| # | Item | Status |
|---|------|--------|
| 1 | Dependency injection (model + callbacks as parameters) | Done |
| 2 | Test infrastructure (folder, fixtures, fake LLM, fake logger) | Done |
| 3 | Telemetry service install + hook (real backend for normal runs) | Done — originally Arize Phoenix via Docker, then a second backend (Logfire) added alongside it 2026-08-21. Superseded 2026-08-24: Phoenix retired, Logfire the sole live backend. **Superseded again 2026-09-03: `logfire_logger.py` (a single hardcoded `LogfireLogger`) was itself replaced by a pluggable-provider architecture** — new top-level `telemetry.py` reads ONE settings list (`telemetry.providers`) and routes each entry to two internally-separate `TracerProvider`s (general app-events vs. LLM-call tracing) purely by the entry's own discovered class's `KIND`; providers (`otlp.py` — generic OTLP, covers Logfire/Grafana Cloud/SigNoz/OpenObserve; `file.py` — local JSON lines; `phoenix.py` — direct OTLP to a self-hosted Phoenix, no `arize-phoenix-otel` dependency) live under `telemetry_providers/`, auto-discovered the same way `news_adapters/` are. `logfire_logger.py`/`tests/test_logfire_logger.py` deleted outright. See `docs/standaloneplan/01-settings-migration.md`'s "Telemetry providers, take two/take three" sections for the full history (take two shipped a real double-export/kind-leak bug, caught by code review the same day; take three is the corrected, currently-live design) and `docs/current/telemetry-catalog.md` for what actually gets emitted. **`docs/current/infrastructure.md` was not updated alongside this and still describes the retired `logfire_logger.py`/`LOGFIRE_ENABLED` shape as current — stale, needs a pass.** |
| 4 | CI setup (test automation) | Done — GitHub Actions; branch protection pending manual confirmation |
| 5 | Test cases (actual scenarios) | Done — 735 tests as of 2026-09-05 (started at 16; see below for what's covered vs. not, and its own stale-count disclaimer). `tests/test_users_db.py` was split six ways (`tests/test_subscriber_ops.py`/`test_category_ops.py`/`test_push_outcome_ops.py`/`test_api_budget_ops.py`/`test_interest_cache_ops.py`/`test_source_state_ops.py`) alongside the `users_db.py` -> `storage/` + `*_ops.py` refactor — same bodies, same total count, no coverage lost in the split. New gap the split didn't close: `storage/postgres/__init__.py` (`PostgresStorage`) and the backend-selection dispatch itself (`storage/__init__.py`'s `_build_storage`, `storage/engine.py`'s `build_engine`) have zero test coverage — every test injects a pre-built `SqliteStorage` via `storage.reset_storage_for_tests()` (see `tests/conftest.py`'s `isolated_subscribers_db`), bypassing both files entirely. **2026-09-05: `search_news` redesigned from a LangChain tool into a plain bounded function (see item 5's own entry in `docs/plans/local-news-cache-plan.md`'s Status table for the incident) and `save_note`/`agent.NOTES_FILE` removed outright** — `tests/conftest.py`'s `isolated_notes_file` fixture (referenced in section 2 below) no longer exists, and `tests/test_agent.py`'s search_news tests (section 5 below) no longer drive it through `FakeToolCallingModel`'s tool-calling loop, they call the function directly. `agent.build_agent`/`run_agent` and their own direct tests (`test_run_agent_no_tool_call_direct_answer`, `test_run_agent_records_callback_events`) are unchanged and still pass — that machinery is dormant, not deleted. QA re-verified 2026-09-05: 735/735 passing, 90% combined coverage across `agent.py`/`bot.py`/`news_push.py`/`telegram_html.py`/`combined_bot.py` (agent.py 93%, bot.py 87%, news_push.py 100%, telegram_html.py 98%, combined_bot.py 53% pre-existing and unrelated to this change — its untested lines are `main()`/`run_both()` wiring, never exercised by any test before this change either). One gap flagged back, not written here per this doc's own division of labor: `bot._route_a_reply`'s `except Exception` branch (the `search_news` call raising, returned as `blocked_at="agent_error"`) has no test forcing that path — pre-existing (the same branch existed before this rewrite, under the old `run_agent` call), not newly introduced, but still open. **2026-09-08: `interest_finder.py` ("help me find my interests", `docs/plans/interest-finder-plan.md`) added** — 786/786 passing. `interest_finder.py` itself at 96% (3 untested branches, all low-risk string-formatting paths, not logic gaps: `list_current_interests`'s non-empty-interests return, `drop_interest`'s "now follows nothing" return, and `_compose_prompt`'s branch appending a language instruction when the subscriber has one set). The new `bot._process_find_interests` and every call site into it (the session-check-before-layer-2 branch, the multi-category-wins branch) are covered 100% — every exit (model-ended, turn-ceiling, layer-4 block, exception) has its own test in `tests/test_interest_finder.py`. `tools/measure_guardrails.py` gained a `find_interests_shapes`/`find_interests_vs_set_interest` (layer 2) and `find_interests_exploration` (layer 4) group; QA-measured baseline and a full-dataset regression check (confirming the router-prompt/output-scope-prompt widening didn't affect any existing category) are recorded in `docs/plans/guardrails-plan.md`'s "Added 2026-09-08" section, not duplicated here. `tools/run_smoke_tests.py` gained case 18 (opens an exploration, then confirms a contextless follow-up stays in it without reaching the router). |
| 6 | LLM-judged end-to-end evaluation | **Built 2026-08-16** — `tools/run_eval.py`, 11/11 passing on first real run, see below |

## 1. Dependency injection

`build_agent(model)` builds the agent from an injected model; `run_agent(agent,
messages, callbacks=None)` threads an injected callbacks list through
`agent.invoke({"messages": ...}, config={"callbacks": callbacks})`. Neither is
hardcoded — production (`main()`) passes the real `ChatDeepSeek` and no
callbacks (until item 3 exists); tests pass fakes. `create_agent`/LangChain's
Runnable interface supports both injection points natively — no custom
provider abstraction needed (considered and rejected earlier as
over-engineering for a single-file agent).

## 2. Test infrastructure

Built under `tests/`:

- ~~`tests/conftest.py` — `isolated_notes_file` fixture: monkeypatches
  `agent.NOTES_FILE` to a `tmp_path` file so `save_note` tests never touch the
  real `notes.jsonl`.~~ **Removed 2026-09-05** along with `save_note`/
  `agent.NOTES_FILE` themselves (see item 5's Status-table entry above) —
  there is no longer a notes file to isolate. (Kept here, struck through
  rather than deleted, as the historical record of why it existed: an
  ad-hoc verification test before this fixture existed wrote a real
  "hello test" entry into the actual file; had to be cleaned up by hand.)
- `tests/fakes.py`:
  - `FakeToolCallingModel` — a **custom** fake, not a LangChain built-in.
    Both `GenericFakeChatModel` and `FakeMessagesListChatModel`
    (`langchain_core.language_models.fake_chat_models`) do **not** override
    `bind_tools()`, and `create_agent` calls
    `model.bind_tools(tools, tool_choice=...)` unconditionally whenever tools
    are present — confirmed live, this raises `NotImplementedError` with
    both built-ins. `FakeToolCallingModel` overrides `bind_tools()` to
    return `self` (ignoring the schema, since responses are pre-scripted)
    and cycles through a list of scripted `AIMessage` responses, stopping on
    the last one.
  - `RecordingCallbackHandler` — in-memory `BaseCallbackHandler` recording
    LLM/tool start/end events into `self.events`, standing in for a real
    telemetry backend in tests.
- `tests/fixtures.py` — mock response payloads (HN Algolia JSON, arXiv Atom
  XML, generic RSS XML, NewsAPI/GNews/Perigon JSON) shaped to match real
  responses captured during live source verification — except Perigon,
  which is **not** independently verified (no API key available; the
  fixture only matches what `fetch_perigon` is coded to expect).
- `pytest.ini` (`pythonpath = .`) — **required**, not optional boilerplate.
  Without it, pytest's default import mode doesn't add the project root to
  `sys.path`, so `tests/conftest.py`'s `import agent` fails with
  `ModuleNotFoundError` — hit this immediately on the first `pytest --fixtures`
  run.
- HTTP mocking: `requests_mock` (pytest fixture, auto-registered by the
  `requests-mock` package — confirmed via `pytest --collect-only` showing
  `plugins: ..., requests-mock-1.12.1`).
- **Bug found while wiring up mocking, fixed in `news_sources.py`:**
  `_fetch_rss` (used by 5 of the 7 free sources) called
  `feedparser.parse(url)` directly, letting feedparser do its own internal
  HTTP fetch via `urllib` — bypassing `requests` entirely. Two problems:
  no `timeout` at all (a slow/dead feed could hang the whole `search_news`
  call), and untestable with `requests_mock` (which only intercepts
  `requests`, not `urllib`). Fixed by routing through
  `requests.get(url, timeout=10)` first, then `feedparser.parse(resp.content)`
  — same pattern `fetch_arxiv` already used. Re-verified live against the
  real OpenAI/VentureBeat feeds after the change.

Test runner is pytest (was deferred earlier as "decide later" — just went
with it since nothing else was ever proposed).

## 3. Telemetry service (real backend, for normal/non-test runs)

**Done — Arize Phoenix, but not the way originally planned.** The original
idea was `pip install arize-phoenix` running in-process with no Docker
needed. That didn't survive contact with reality:

- **The full `arize-phoenix` package can't run in this Python environment
  at all.** It unconditionally imports `pandas` (for its bundled local
  UI/session code) at `import phoenix` time, and on this machine that DLL
  load is blocked by Windows 11 **Smart App Control** — confirmed as a real,
  widely-documented phenomenon (unsigned/no-reputation compiled extensions
  get blocked; see GitHub issues on mypy, ChimeraX, Julia hitting the exact
  same error text). Not fixable by switching install tools — conda, mamba,
  and pip all install the file fine; the block happens when the DLL tries to
  *load*, not at install time.
- **The fix: split client from server.** `arize-phoenix-otel` is a separate,
  much lighter PyPI/conda-forge package — confirmed via its PyPI dependency
  metadata to have **no pandas dependency at all** (just the OpenTelemetry/
  OpenInference stack). `agent.py` uses only this lightweight package to
  *send* traces. The actual Phoenix dashboard/collector runs as its own
  Docker container instead (`arizephoenix/phoenix:latest`), which sidesteps
  the Windows-native block entirely since it's an isolated Linux
  environment. This split turns out to be the right shape for the future
  Kubernetes deployment too, not just a workaround — see
  `docs/plans/deployment-plan.md`.
- Considered and set aside: **LangSmith** (cloud-only — no fully local mode,
  requires an external account even on the free tier); **Langfuse**
  self-hosted (more full-featured — evals, prompt management — but requires
  running a Docker Compose stack with Postgres + ClickHouse, heavier than
  this project needs).

~~**How it's wired up:** `agent.py`'s `setup_telemetry()` calls
`phoenix.otel.register(endpoint=PHOENIX_ENDPOINT, project_name="myfirstagent",
protocol="grpc", auto_instrument=True)`, gated behind the `PHOENIX_ENABLED`
env var — unset (the default, including in every test and CI run) means
it's a no-op. `auto_instrument=True` means LangChain calls are traced
automatically process-wide once registered; this is **not** the same
mechanism as `run_agent`'s `callbacks` parameter — Phoenix/OTel
instrumentation is global and set up once at startup, unlike the
per-invocation `callbacks` list. `run_agent`'s `callbacks` param remains
available for the local/in-memory case (`RecordingCallbackHandler` in
tests) but Phoenix doesn't go through it.~~ **OBSOLETE — Phoenix retired,
this code path no longer exists in `agent.py` at all. See
`logfire_logger.py`/`agent.setup_telemetry()` for how it works now.**

~~**Run it:**~~ **OBSOLETE, do not run:**
<!--
```powershell
docker run -d --name phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
$env:PHOENIX_ENABLED = "true"
python agent.py
```
Dashboard was `http://localhost:6006`.
-->

**Verified end-to-end** (not just "the code runs without error") — queried
Phoenix's GraphQL API directly after a real run and confirmed 20 real spans
recorded under the `myfirstagent` project, with the expected structure:
`LangGraph` (chain) → `model` (chain) → `ChatDeepSeek`/`ChatCompletion`
(llm) → `search_news` (tool), called multiple times matching the agent's
actual multi-query behavior for that conversation.

**Tests:** `tests/test_telemetry.py` covers only the gating logic (register
not called when `PHOENIX_ENABLED` unset; called with expected args when
set) — deliberately not testing whether tracing "works," since that needs
the real Docker container and would break the test suite's zero-real-calls
guarantee. See item 6 below for how actual trace *content* gets verified.

### RESOLVED 2026-08-16 — restored and verified live

**Fixed the same day.** `docker run` redeployed with `PHOENIX_ENABLED=true`
and `PHOENIX_ENDPOINT=http://10.0.0.234:4317` restored. Verified for
real, not just "the container started without error" — `tools/
check_telemetry.py` (built the same day, see below) ran end-to-end and
confirmed **35 spans landed in Phoenix** within 90s of a real test
message through the live bot. `docker logs` also now shows the OTel
registration banner on startup, confirming the registered endpoint and
gRPC transport match what was intended.

Diagnosing this also surfaced that the Phoenix VM was never the
problem — see the section below, corrected in place: it turned out to
be a native systemd + venv install (not Docker, contrary to older docs),
confirmed healthy and reachable on port 4317 the whole time. The actual
defect really was just the two missing env vars on the bot's `docker
run`, exactly as suspected before the fix.

**A real, separate finding while verifying this**: Phoenix's own startup
banner shows `Span Processor: SimpleSpanProcessor`, with its own
explicit warning — *"strongly advised to use a BatchSpanProcessor in
production environments"*. `SimpleSpanProcessor` exports every span
synchronously the moment it ends, meaning every LLM call / tool call now
blocks on a network round-trip to the Phoenix collector before
continuing -- real latency added to every agent step, on a
`VM.Standard.E2.1.Micro` (1/8 OCPU) shape where that's not free. Not
fixed as part of this pass -- flagged here as a worthwhile follow-up
(switch to a `BatchSpanProcessor` via `register()`'s options) rather than
guessed at and changed without measuring the actual latency impact
first, same discipline as everything else in this doc.

**Original finding, retained below for the investigation trail:**

### Currently NOT connected on the live deployment — found 2026-08-16 (historical)

**The deployed `myfirstagent-bot` container is missing `PHOENIX_ENABLED`
and `PHOENIX_ENDPOINT` entirely.** `docker inspect`ing the running
container's env shows only `PHOENIX_API_KEY_SECRET_OCID` — fetching the
API key secret happens in `docker-entrypoint.sh` regardless, but that
alone doesn't turn tracing on; `setup_telemetry()` checks `PHOENIX_ENABLED`
specifically. This is a regression from the state `docs/plans/deployment-plan.md`
records as "verified end-to-end" (`PHOENIX_ENABLED=true`,
`PHOENIX_ENDPOINT=http://10.0.0.234:4317`, a real trace confirmed in
Phoenix's UI) — somewhere between that verification and the current
deployment, a `docker run` stopped including those two flags and nobody
re-added them on a later redeploy. **Also found**: the Phoenix collector
VM itself (private IP `10.0.0.234`) didn't respond to a `curl` from the
bot VM at all (connection timeout) — separate from the missing env vars,
worth checking whether that VM is still running before just restoring the
flags. **Not fixed as part of this session's work** — flagged here so it
isn't lost; restoring it means (a) confirming/reviving the Phoenix VM, (b)
re-adding both env vars to the bot's `docker run` command in
`docs/plans/deployment-plan.md` and using them on the next actual redeploy.

**Practical consequence while this stayed broken**: zero traces reached
Phoenix — not just the source-fetch spans below, but every LLM call,
router classification, and guardrail check too. **Resolved above** — see
the "RESOLVED 2026-08-16" section at the top of this finding.

### Why didn't anything alert on this? — the precise answer, found 2026-08-16

There already **is** an alerting mechanism for exactly this —
`telemetry_monitor.py`, wired into `combined_bot.py` — and it's not
broken. It never started. `combined_bot.py`'s `_start_telemetry_monitor`:

```python
if not os.environ.get("PHOENIX_ENABLED"):
    return None
```

The monitor that checks "is Phoenix's OTLP port reachable" and pages the
admin (via `admin_bot.py`'s token) on a state change is itself gated
behind the exact same env var that went missing. So the one
misconfiguration — `PHOENIX_ENABLED` absent from the deployed
container — simultaneously (a) disabled tracing and (b) prevented the
monitor built to detect (a) from ever spinning up. Not a bug in
`telemetry_monitor.py`'s logic (edge-triggered TCP reachability checks,
alert on up→down and down→up, correctly designed) — the monitor was
never given the chance to run at all. This is the single-point-of-failure
shape worth naming for next time: a self-monitoring mechanism that's
gated behind the same flag as the thing it monitors can't be relied on to
catch that flag going missing.

**Why the gating exists at all, and why it's not simply wrong**: running
a Phoenix-reachability monitor when telemetry is deliberately off (local
dev, `PHOENIX_ENABLED` intentionally unset) would be pointless noise
against `PHOENIX_ENDPOINT`'s `localhost` default. The gate is a
reasonable design choice for "don't monitor a thing that's supposed to be
off" — the actual defect is purely operational (a `docker run` silently
dropped two flags at some point), not a code bug to fix in
`telemetry_monitor.py` itself.

**What actually closes this gap, added the same day, two complementary
pieces — deliberately not one**, since each catches a different half of
"how would we have known":

1. **`tools/check_telemetry.py`** (see below the smoke-test checklist
   entry it's tied to) — an EXTERNAL, post-deploy check that doesn't run
   inside the bot process at all, so it can't be silently disabled by the
   bot's own missing env var the way `telemetry_monitor.py` was. This is
   the layer that would have caught the exact regression described
   above, on the very next deploy, regardless of what `docker run` did
   or didn't include.
2. **`healthcheck.py`** — a NEW, separate liveness check for whether
   `news_ingest.py`/`news_push.py`'s periodic jobs are still ticking at
   all, alerting the admin (reusing the same `admin_bot.py` channel
   `telemetry_monitor.py` already uses) on a change in problem state.
   Deliberately not gated behind `PHOENIX_ENABLED` or anything else
   optional — it always runs once the bot starts, so it can't fail the
   same way. Answers a different question than either `telemetry_monitor.py`
   (is Phoenix reachable) or `check_telemetry.py` (are traces actually
   landing): are the periodic jobs themselves still alive, independent of
   whether Phoenix is even in the picture.

   **RETIRED 2026-08-29** — deleted, not just superseded. Diagnosing a
   real ingest-staleness alert from it exposed its actual limitation
   beyond the "service decides and pages directly" architectural
   objection (`docs/system-overview.md` §C5): zero per-source
   granularity, so a single source silently broken for days was
   invisible to it. Replaced by `news_ingest._pull_source`'s
   `ingest_source_pull` span (structured, per-source, queryable from
   Logfire) plus four planned Logfire alerts -- see
   `docs/plans/observability-platform-plan.md`'s 2026-08-29 "healthcheck.py
   retired" section for the full design and what's still open.

### Raw source fetches — a gap auto-instrumentation can't close, added 2026-08-16

`auto_instrument=True` only wires up `openinference-instrumentation-
langchain` (the only OpenInference instrumentor in `environment.yml`) —
it traces LangChain-mediated calls (LLM invocations, `@tool`-decorated
tool calls made through `create_agent`'s loop) automatically, but has no
visibility at all into a plain `requests.get()` called from outside that
framework. `news_ingest.py`'s scheduled per-source pulls and `agent.py`'s
`search_news` both call `news_sources.py`'s fetch functions directly —
neither goes through anything LangChain instruments, so even with Phoenix
fully connected, these calls would never appear as spans.

Prompted by a real gap: diagnosing how many calls had been made against
NewsAPI/Perigon (the two budget-constrained restricted sources, see
`docs/current/ai-news-sources.md`) turned up that `search_news`'s on-demand usage
of them was invisible everywhere — not logged, not counted against any
budget, and (per the section above) not even reaching Phoenix since it
was disconnected.

**Superseded 2026-09-04 for `search_news` specifically** — it was
rewritten to read the ingested cache (`news_cache.read_all()`) instead of
calling any source live at all (`docs/plans/local-news-cache-plan.md` item
5), so it no longer calls `news_sources.py`'s fetch functions, no longer
goes through `traced_fetch`, and no longer calls
`api_budget_ops.record_api_call`. Both mechanisms below remain live and
load-bearing for `news_ingest.py`'s own scheduled pulls, which is now the
only caller of either — the rest of this section is kept as the historical
record of why they were built, not a description of `search_news`'s
current behavior.

**Fixed with two independent, complementary mechanisms** — deliberately
not just one, since Phoenix availability and the local DB are different
failure domains:

1. **`news_sources.traced_fetch(source_key, fetch, query, max_results)`**
   wraps every source fetch call (from both `news_ingest.py` and
   `search_news`) in a manual OpenTelemetry span (`trace.get_tracer(...)`),
   tagged with `source_key`, `query`, `restricted`, `article_count`, and
   `error` on failure. This is real OpenTelemetry API usage, not
   LangChain-specific — it works the moment Phoenix is connected again,
   with no further code change, and is a safe no-op right now (and in
   tests) since `get_tracer()` returns a no-op tracer when no provider is
   registered.
2. **`api_budget_ops`'s `api_budget` table** (used independently of whether Phoenix is
   up): migrated from one row per source (today's count only, overwritten
   on every date rollover — no history at all) to one row per
   `(source, date)`, so `get_api_budget_history`/`get_total_api_calls`
   give a real, persistent, queryable count regardless of tracing
   infrastructure. `search_news` now calls the new non-enforcing
   `api_budget_ops.record_api_call` for any restricted source it actually hits
   — recorded in the same table `news_ingest.py`'s budget-enforced
   `try_consume_api_budget` writes to, so a query against either source
   reflects combined usage from both call paths, not just the scheduled
   ingestion job's.

**Why both, not just Phoenix once it's reconnected**: the DB-backed count
is what `try_consume_api_budget` already needs for cap enforcement
regardless of tracing state — extending it to also serve as a queryable
log was cheap and doesn't depend on the Phoenix VM being reachable, which
it currently isn't. Phoenix (once reconnected) adds richer per-call
detail (timing, the actual query string, error text) that a plain counter
can't — the two are complementary, not redundant.

## 4. CI (test automation)

**Done — GitHub Actions.** `.github/workflows/ci.yml` runs on every push to
`main` and every PR against it: `mamba-org/setup-micromamba@v3` builds the
environment straight from `environment.yml` (no separate pip lockfile —
avoids the drift risk of maintaining two dependency files), then `pytest`.
No secrets needed — same zero-real-calls guarantee as running locally.

**Not yet confirmed:**
- Whether the workflow run actually went green on GitHub (asked to check,
  no confirmation received yet).
- Branch protection on `main` — walked through the exact settings to apply
  via GitHub's web UI (require PR, require the `test` status check, no
  admin bypass, no approval requirement, force-push/deletion disabled), but
  applying it is a manual step on GitHub's side, not something committable
  to this repo. Not confirmed done.

This is CI (test automation) only — not CD. See `docs/plans/deployment-plan.md`
for what actual deployment automation needs first (it needs the whole
deployment chain resolved, not just this).

## 5. Test cases

16 tests, all passing, all real network/LLM calls mocked out (this
snapshot is from very early in the project and is itself long superseded
by the dated entries further down this section — 735 tests as of
2026-09-05; kept as the historical starting point, not a description of
current test content):

**`tests/test_agent.py`** (the agentic loop, via `FakeToolCallingModel`):
- ~~`save_note` writes the expected JSON line to an isolated temp file and
  returns the expected confirmation — real `notes.jsonl` untouched.~~
  **`save_note` (and this test) removed outright 2026-09-05** — see the
  "search_news redesigned into a plain, bounded function" entry below.
- `search_news` aggregation with one working + one failing mocked source:
  the working source's result and an `ERROR: ...` line for the failing one
  both appear in the tool output, and the agent still reaches a final
  answer — confirms per-source error isolation actually works end-to-end,
  not just in isolated unit logic. **Superseded 2026-09-04/2026-09-05** —
  `search_news` no longer fetches from any live source or runs inside an
  agent tool-calling loop at all; see the dated entries below.
- A turn with no tool calls at all — direct answer, no `ToolMessage` in the
  result.
- `run_agent`'s `callbacks` param actually reaches the model —
  `RecordingCallbackHandler` sees `llm_start`/`llm_end` events.

**`tests/test_news_sources.py`** (individual fetchers, via `requests_mock`):
- `fetch_hackernews` — parses hits, including the `url: None` →
  `news.ycombinator.com/item?id=...` fallback link.
- `fetch_arxiv` — parses Atom entries, strips embedded newlines from title.
- `_fetch_rss` (generic — covers the shared code path behind 5 of the 7 free
  sources) — parsing and `max_results` truncation.
- `fetch_openai_blog` — confirms the wrapper hits the right URL with the
  right source name (representative spot-check; `huggingface_blog`/
  `techcrunch_ai`/`venturebeat_ai`/`mit_tech_review` share the same
  `_fetch_rss` code path and aren't each re-tested individually).
- `fetch_newsapi`, `fetch_gnews`, `fetch_perigon` — parsing plus the
  `os.environ[...]` key lookup (via `monkeypatch.setenv`).
- `enabled_sources()` — all 7 free sources always present; gated sources
  absent/present based on whether their env var is set.

**`tests/test_telemetry.py`** — the `PHOENIX_ENABLED` gating logic (see item
3), plus, added 2026-08-21: `LOGFIRE_ENABLED`/`LOGFIRE_API_KEY` gating (a
key present without the flag must not export — load-bearing, since
`LOGFIRE_API_KEY` sits in the dev shell env), the raise-when-enabled-
without-a-key contract, one shared `TracerProvider` when both backends are
on, and the token-prefix → region derivation (`logfire_traces_endpoint`).

**This "Test cases" section and its counts are stale project-wide** (the
suite is now 683 tests, not 16 — `qa-engineer` tracks the current total,
this doc doesn't attempt to stay in sync test-by-test). Notable additions
since, called out specifically because each is a new *category* of
behaviour rather than more of an existing one:

- **`tests/test_agent.py`'s multi-topic `set_interest`/`remove_interest`
  tests (2026-08-25)** — `MessageClassification.topic` (single string) →
  `topics` (list), fixing a live bug where "Add AI agent, ai coding, LLM"
  could silently collapse to one interest. Covers: multiple topics each
  stored; `known` (disambiguation context) growing across topics *within*
  one message, not just across messages; `alongside=list(known)` being an
  independent snapshot rather than a reference that a later loop
  iteration's `known[:] = after` could retroactively mutate; a cap-
  refused topic not polluting `known` for a later topic in the same
  message; and empty-`topics` handling. See
  `docs/plans/guardrails-plan.md`'s "Fixed 2026-08-25" section for the
  live-model measurement. Two gaps noted at review (2026-08-25) were
  closed the same day: `test_two_topics_normalizing_to_the_same_label_report_a_duplicate_not_a_double_add`
  and `test_removing_the_same_topic_twice_in_one_message_is_not_an_error`.

- **`tests/test_news_push.py`/`tests/test_push_outcome_ops.py`** — `push_outcomes`
  recording (`news_push._record`, `push_outcome_ops.record_push_outcome` and its
  queries), the cost-bug fix (failure paths advancing `last_push_at` only
  once generation has happened, three-strikes-and-disable for an
  unreachable chat), and the push-tick heartbeat span. See
  `docs/plans/incident-monitoring-plan.md`'s "Status: step 1 built" for
  the design this covers.
- **`tests/test_news_embed.py`, and `tests/test_news_push.py`'s near-
  duplicate collapse / relevance filter / offbeat selection tests
  (2026-08-25)** — the embedding-based "fine filter" fix for a live bug
  where "AI", "AI Agent", "AI coding" and "Large Language Model" all
  mapped to category `AI` and drew four near-identical digests from the
  same undifferentiated pool. Covers `news_embed.py`'s fail-open
  contract (no embedder, encode() failure, missing input) independently
  of `news_push.py`'s consumption of it; the `RELEVANCE_KEEP_MIN`/
  `_MAX` clamp (including the two historical bugs it was built to stop
  repeating — float-imprecision `cut_index`, and an unclamped `n_kept`
  driving a negative list index); `OFFBEAT_POOL_SIZE` matching
  `RELEVANCE_KEEP_MAX` exactly so neither is a silently-tighter dead
  ceiling; and `write_push_digest`'s dropped `[topic]`-prefix listing
  plus the "model writes an explanatory sentence instead of a literal
  empty reply" gap in the "nothing relevant" check (now gated on a real
  `<a href>`, not `digest.strip()`). See
  `docs/analysis/cluster-measurements.md` for the real-model/real-cache
  measurements behind every constant here. **One gap found at review,
  2026-08-25**: `_pick_for_topic`'s "not enough offbeat survivors past
  the relevance gate, top up with the next most recent article" branch
  (`news_push.py`, the `filler_needed > 0` block inside the offbeat arm,
  not the full-fallback one) has no test exercising it directly —
  every current offbeat test either has enough survivors to fill every
  slot or falls back to pure recency entirely. Handed back to the coding
  engineer to add, not written here.
- **`tests/test_news_push.py::test_emit_heartbeat_is_a_noop_without_a_tracer_provider`**
  and the Logfire gating tests above are this project's hermeticity check
  for telemetry specifically: nothing in `tests/` or `tests/conftest.py`
  calls `setup_telemetry()` implicitly or at import time, so a real
  `LOGFIRE_API_KEY` sitting in the dev environment cannot cause a real
  export during any test or CI run.
- **`tests/test_news_ingest.py`'s `_pull_source` tests (2026-08-29,
  `healthcheck.py` retirement)** — six new tests covering all four
  `pull.outcome` values (`not_due`, `budget_exhausted`, `success`,
  `failed`) plus `pull.expected_interval_hours` varying by source
  (default 4h vs. `perigon`'s 8h vs. `newsapi`'s 24h). The
  multi-section case (`test_pull_source_success_when_only_some_sections_of_a_multi_section_source_fail`)
  forces a real mixed pass/fail across `arxiv`'s 6 sections (one raises,
  the rest return normally) rather than asserting the trivial
  all-pass/all-fail cases only, confirming `pull.outcome` is `success`
  whenever at least one attempted section succeeds, not `failed` on a
  single transient section error. See `docs/plans/observability-
  platform-plan.md`'s 2026-08-29 "healthcheck.py retired" section for
  the design.
- **Phoenix retirement + `logfire_logger.py` (2026-08-30)** — `tests/
  test_telemetry.py` was rewritten around `setup_telemetry()` delegating
  to the new `logfire_logger.LogfireLogger.setup()` (idempotent
  provider construction, `instrument_langchain` opt-in, service-name
  ordering); `tests/test_logfire_logger.py` is new (12 tests) covering
  `LogfireLogger.log()` itself (default/explicit level, tags only when
  given, dict-vs-string `message`, exception recording + `ERROR` status,
  the printed line) and `setup()`'s idempotency/instrumentation
  branches. Eleven previously print-only `except Exception:` sites
  across `news_embed.py`, `message_archive.py`, `news_ingest.py`,
  `bot.py`, `news_classify.py`, and `guardrails.py` were converted to
  call a per-module `_events: Logger = LogfireLogger(...)` instead;
  every converted site's test now asserts the right span name, level,
  and recorded exception via a shared `tests.fakes.FakeSpan` — see
  `docs/current/telemetry-catalog.md`'s `LogfireLogger`-emitted events
  table for the full inventory. The two `guardrails.py` sites
  (`router_failed`, `output_check_failed`) stayed at `ERROR` — tied to
  the 2026-08-21 silent-fail-open incident — and their fail-open return
  values are byte-identical to before (only the `print()` inside the
  `except` block changed), so this carries no guardrail-reliability risk
  and didn't need a `measure_guardrails.py` re-run to confirm.
- **Pluggable telemetry providers (2026-09-03)** — `tests/test_telemetry.py`
  rewritten around `telemetry.setup_telemetry()`/`get_event_logger()`;
  `tests/test_telemetry_providers.py` is new, covering discovery/
  validation (including a real subprocess repro of the "unknown type
  fails the whole process at startup" contract), `otlp.py`'s
  processor-not-provider attachment and `instrument_langchain` gating
  (both the "only on the llm-side call" and "only once across repeat
  entries" cases), `file.py`'s actual JSON-line writing including a
  missing parent directory, and `phoenix.py`'s "attaches to the shared
  provider it's given, never builds its own" contract. `tests/
  test_telemetry.py`'s kind-isolation tests
  (`test_dual_kind_otlp_entry_receives_both_general_and_llm_spans_from_one_config`,
  `test_get_event_logger_llm_only_provider_never_receives_a_general_event`)
  use real `TracerProvider`/`SimpleSpanProcessor`/`InMemorySpanExporter`
  objects rather than mocks specifically because a mocked processor can't
  reveal a cross-provider leak — these are what caught the take-two
  double-export/kind-leak bug (see the Status table's row 3) and now
  guard against a regression of it. QA re-verified 2026-09-03: 708/708
  passing, 94% line coverage across `telemetry.py`/`telemetry_providers/`/
  `agent.py` combined (`telemetry_providers/otlp.py`/`phoenix.py` both
  100%). Two real gaps found and NOT yet closed (flagged back, not
  written here per this doc's own division of labor):
  - No test asserts `phoenix`'s `project_name` config value actually
    lands on the llm `TracerProvider`'s `Resource` as
    `openinference.project.name` (`telemetry.setup_telemetry`'s own
    per-entry loop that builds `resource_attrs`). Manually verified live
    (a real `Settings` + real `TracerProvider`, `llm_provider.resource.
    attributes["openinference.project.name"] == "my-proj"`) — the
    behavior is correct, just has no permanent regression test.
  - `telemetry_providers/file.py`'s `except OSError: pass` fail-open
    branch (lines 63-69) is untested — the file provider's one stated
    resilience guarantee ("a logging sink must never be the reason the
    thing it was logging about doesn't complete") has no test forcing a
    write failure (e.g. a read-only path) and confirming `log()` doesn't
    raise.

- **`search_news` rewritten onto the ingested cache (2026-09-04)** — see
  `docs/plans/local-news-cache-plan.md` item 5. `tests/test_agent.py`'s old
  single mocked-source-aggregation test was replaced by 7 tests covering:
  relevance-ranked results from `news_cache.read_all()`; exclusion of
  links already in `subscriber_ops`' shared `pushed_links` dedup memory;
  a returned result being marked shown via `mark_links_shown` *without*
  advancing `last_push_at` (the load-bearing property of the
  `mark_links_shown`/`advance_last_push_at` split — a manual search must
  never delay a subscriber's own scheduled push); the per-subscriber daily
  quota (`try_consume_search_query`) blocking a query once exhausted;
  query-definition generation + caching and reuse of an already-cached
  definition (the same `interest_cache_ops` cache `_add_one_interest`
  populates); and the no-results message. `tests/test_subscriber_ops.py`
  gained 3 matching tests for `try_consume_search_quota` itself (cap
  enforcement, date-based reset, per-subscriber scoping) plus updated
  `mark_links_shown`/`advance_last_push_at` tests for the split (including
  `news_push.py`'s three `run_push_cycle` outcome-recording call sites
  each now asserting both methods are called together, or both are
  skipped together, matching the old single `record_push`'s guard
  conditions exactly). QA re-verified 2026-09-04: 732/732 passing, 98%
  combined line coverage across `agent.py`/`subscriber_ops.py`/
  `storage/sqlite/subscriber.py`/`news_embed.py`/`news_push.py` (the only
  misses are pre-existing, unrelated to this change — `agent.py`'s CLI
  `main()` entry point and one pre-existing branch in
  `subscriber_ops._is_duplicate_topic`). No coverage gap found in the new
  code itself; two minor, non-blocking behavioral cases flagged back (not
  written here per this doc's own division of labor): no test asserts
  `SEARCH_MAX_RESULTS` actually truncates a pool bigger than 5, and no
  test asserts multiple relevant results come back newest-first.

- **`search_news` redesigned into a plain, bounded function (2026-09-05)**
  — see `docs/plans/local-news-cache-plan.md` item 5's own updated entry
  for the incident (a live INT test found the old tool-calling version
  searching one question 5-7 times) and `agent.py`'s own module note
  above `search_news`. `tests/test_agent.py`'s search_news tests no
  longer drive it through `FakeToolCallingModel`'s tool-calling loop
  (the two now-removed `_search_news_call`/`isolated_notes_file` helpers)
  — they call `agent.search_news(chat_id, query, history, model,
  guard_model, embedder)` directly, and assert against a `_RecordingModel`
  (a `FakeToolCallingModel` subclass capturing exactly what was passed to
  `model.invoke()`) rather than parsing a `ToolMessage`. New coverage
  added: the empty-candidate-pool short-circuit calls `model.invoke.
  assert_not_called()` — confirming "no is no" is a real early return,
  not just a prompt instruction; three query-rewrite cases (a successful
  rewrite using conversation history actually changing what gets
  searched, `guard_model=None` skipping the rewrite, and
  `guard_model.invoke()` raising falling back to the raw query). `agent.
  build_agent`/`run_agent` keep their own pre-existing direct tests
  unmodified (`test_run_agent_no_tool_call_direct_answer`,
  `test_run_agent_records_callback_events`) — that machinery is dormant,
  not deleted, per explicit user direction. `telegram_html.
  links_actually_sent` (moved from `news_push.py`, now shared by both
  `write_push_digest` and `search_news`) took its five existing tests
  with it into `tests/test_telegram_html.py`, unchanged in substance.
  QA re-verified 2026-09-05: 735/735 passing, 90% combined coverage
  across `agent.py`/`bot.py`/`news_push.py`/`telegram_html.py`/
  `combined_bot.py` — see item 5's Status-table entry above for the
  per-file breakdown and the one gap flagged back
  (`bot._route_a_reply`'s `except Exception` branch, pre-existing).

**Not covered yet** (candidates for later):
- ~~`agent._logfire_processor`'s actual body, and the Logfire-only wiring
  branch of `setup_telemetry`~~ — flagged 2026-08-21 by `qa-engineer`,
  closed the same day: `test_logfire_processor_targets_the_regional_endpoint_with_the_token`
  exercises the real `OTLPSpanExporter`/`BatchSpanProcessor` construction
  (mocking only those two classes, not the whole function), and
  `test_setup_telemetry_without_phoenix_builds_and_installs_its_own_provider`
  exercises the `provider is None` branch and asserts the provider is
  actually installed as the process-global one, not just constructed.
- No test compares captured events/output against a checked-in baseline file
  (snapshot testing) — current tests assert specific expected values inline
  instead. Revisit if that stops scaling.
- No multi-turn conversation test (two+ user turns in one `messages` list).
- No test of the real `ChatDeepSeek` + real API path (intentionally — that's
  what the fakes exist to avoid; covered instead by the live manual testing
  done earlier in this project's history, see `CLAUDE.md`).

## 6. LLM-judged end-to-end evaluation

**Built 2026-08-16** — `tools/run_eval.py`. A fundamentally different kind
of test than items 2/5 above, so it got its own item rather than folding
into "more test cases."

**The problem it solves:** items 2/5 test the agent's *mechanics* (does the
tool-calling loop work, does error isolation work) using a fake LLM with
scripted responses — deliberately not testing whether the *real* DeepSeek
model's actual output is any good, since that's non-deterministic and can't
be asserted against with `==`. There's currently no automated check that the
real end-to-end system (real LLM, real tools, real sources) produces
reasonable output — only the manual spot-checks done throughout this
project's history (see `CLAUDE.md`).

**What got built.** `tools/run_eval.py` runs 11 representative questions
(4 `news_query`, 6 settings categories, 1 multi-intent) through the REAL
pipeline (`bot.process_message`, real `ChatDeepSeek` models built via
`agent.build_model` — automatically evaluating whatever
`LLM_MODEL`/`LLM_MODEL_CLASSIFIER` are currently configured to, per
`docs/plans/model-portability-plan.md`), against an isolated
`SUBSCRIBERS_DB_FILE` so it never touches production subscriber data.
Judge is DeepSeek itself, reusing the classifier-tier model (no second
provider key available — same call as `docs/plans/model-portability-plan.md`'s
Level 2 plumbing), via structured output with a `reasoning`-before-
`meets_criteria` field order (same fix that raised
`guardrails.OutputCheck`'s reliability). The rubric is derived from
`agent.LAYER1_IDENTITY`/`TREND_REPORT_STRUCTURE`'s actual stated rules —
three separate rubrics (news_query / settings / multi), not one generic
"is this good" prompt.

**First real run, 2026-08-16: 11/11 passed** (4/4 news_query, 6/6
settings, 1/1 multi-intent) — a genuine first confirmation that the real
end-to-end pipeline (real DeepSeek, real news sources, real router,
real Route A/B dispatch) produces reasonable output across every category
this project supports, not just the mechanics tests items 2/5 already
covered with fakes.

**Deterministic sub-checks built** (cheaper and more reliable than the
judge where they apply), reusing `bot.py`'s own existing safety-net
helpers rather than reimplementing them:
- **Source links present** — every 🔗 line in a trend report must carry a
  real `<a href="...">`. Built without needing a `TREND_REPORT_STRUCTURE`
  change after all: it already requires "only include sources actually
  provided... never invent a URL", which is a meaningful check as-is.
- **Output format compliance** — defined concretely as: HTML tags balanced
  (reuses `bot._is_html_balanced`), no stray Markdown bold leaked through
  (reuses `bot._MARKDOWN_BOLD_RE`), and `news_query` replies start with
  the 📰 report marker (reuses `bot._TREND_REPORT_MARKER`) — the same three
  safety nets `bot.py` already applies to real traffic, now also checked
  against real model output in this harness.

**Deliberately still not built: writing pattern/style consistency.** No
rubric exists for what "style" means here — that's a product decision,
not something to invent while building the harness. Stays an open item.

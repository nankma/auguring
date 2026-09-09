# Telemetry catalog: every span reaching Logfire, and what reads it

No decision history here — read `docs/plans/observability-platform-plan.md`
for how/why this got built, and `docs/system-overview.md` §C4/§C5 for the
architecture narrative. This doc describes **what the code currently
emits**: every span this project's own source actually sends to Logfire,
its real attribute keys, and which alert (if any) reads it. Attribute
keys and `level`/`message` behavior were cross-checked against a live
`records` query (see `docs/reference/observability-and-debugging.md` for
the technique) to catch any drift from what a docstring claims, but the
inventory itself — which spans exist, from which module — is the code,
not a point-in-time query result. Keep this in sync with the code when a
span is added, renamed, or removed; a query re-check is only for
catching attribute-shape drift, not for deciding what belongs here.

## Service name

The code sets exactly one: `myfirstagent`, via
`Resource.create({"service.name": SERVICE_NAME})` in `telemetry.py`'s
`setup_telemetry()` (also forced onto `OTEL_SERVICE_NAME` before any
provider is built — see that function's own docstring for why that
ordering still matters). `agent.setup_telemetry()` is now a one-line
delegate to `telemetry.setup_telemetry()` — see that module and
`telemetry_providers/` (the pluggable-provider package, same
auto-discovery pattern as `news_adapters/`) for how the exporter/
instrumentation actually gets built; this doc only describes what gets
emitted, which is unchanged by that refactor. Every span below is under
this one service; nothing in the current codebase produces any other
`service_name`.

## `level`, `tags`, and `message` — two mechanisms, not one

Two different ways a span reaches Logfire, with different `level`/
`message` behavior:

**Plain OTel spans** (`tracer.start_as_current_span(...)` directly, no
`telemetry.EventLogger` involved — e.g. every span in the table below
except the eleven `*_failed` events) never set level explicitly. Logfire
derives `level=17` (ERROR) automatically whenever a span's
`otel_status_code` is `ERROR` (an exception propagated out of the `with`
block uncaught); everything else defaults to `level=9` (INFO). `message`
for these spans exactly equals `span_name` — Logfire's default rendering
when a span carries no explicit message template. **The real content is
in `attributes`, never in `message`**, for this kind of span.

**`EventLogger.log(...)` spans** (`telemetry.py` — the eleven `*_failed`
events below, via each module's own `_events = get_event_logger(...)`)
set `level`/`tags`/`message` explicitly, via three Logfire-recognized
attribute keys verified live against a real `records` query (not
documented in Logfire's own public docs for plain-OTel senders, so worth
recording here for the next span that wants this). These three keys are
Logfire-specific vendor behavior, unconditionally set regardless of
which `telemetry_providers/*.py` backend(s) are actually configured —
harmless extra attributes to any other OTLP-compatible receiver that
doesn't recognize them:

- `logfire.level_num` (int) — sets the native `level` column directly,
  not limited to OTel span status's OK/ERROR binary. This project uses
  `telemetry_providers.Level`'s five values (TRACE=1, INFO=9, WARN=13,
  ERROR=17, FATAL=21) — WARN is real here, unlike the plain-OTel spans
  above where it can never appear.
- `logfire.tags` (tuple/list of str) — sets the native `tags` column (a
  `List[Utf8]`), not a JSON attribute.
- `logfire.msg` (str) — overrides the `message` column with a real,
  human-readable line instead of the bare span name.

All three are stripped from the regular `attributes` JSON once
consumed — confirmed empty in probe spans that set them, so they never
clutter a `WHERE attributes->>'x'` query alongside real business
attributes.

## This project's own spans (`myfirstagent` service)

| `span_name` | `otel_scope_name` | Emitted by | Attributes (verified keys) | Cadence | Read by |
|---|---|---|---|---|---|
| `argus_heartbeat` | `argus.news_push` | `news_push._emit_heartbeat` | `heartbeat.job` (`"push_tick"`), `heartbeat.push_enabled_subscribers` (int) | Once per push cycle (every `PUSH_TICK_SECONDS`=900s), unconditionally | `argus bot liveness` reads for ANY span from the service in 30min, not this one specifically — but this is what keeps that generic dead-man's-switch satisfied even during an ingest-only outage |
| `push_outcome` | `argus.news_push` | `news_push._record` | `push.subscriber` (opaque id), `push.outcome` (`delivered`/`nothing_new`/`model_error`/others per code — only `delivered`/`nothing_new` seen in the last 30 days), `push.generated` (bool), `push.detail` (free text) | Once per subscriber per push cycle | `argus model errors` (`push.outcome='model_error'`), `argus delivery ratio` (`delivered` vs `push.generated` ratio) |
| `html_validation_attempt` | `argus.news_push` | `news_push._emit_html_validation_attempt` | `push.subscriber`, `topic`, `attempt` (1-3), `valid` (bool), `reason` (only present when `valid=false`, e.g. `"disallowed tag <hr>"`) | Once per HTML-validation retry attempt, every attempt (not just failures) | *(planned, not built)* `argus html validation retry`/`argus html validation exhausted` |
| `ingest_heartbeat` | `argus.news_ingest` | `news_ingest._emit_heartbeat` | `heartbeat.job` (`"ingest_tick"`) | Once per ingest cycle (every `INGEST_TICK_SECONDS`=900s), unconditionally, before any per-source work | `argus ingest liveness` |
| `ingest_source_pull` | `argus.news_ingest` | `news_ingest._pull_source` | `pull.source`, `pull.outcome` (`not_due`/`budget_exhausted`/`success`/`failed`), `pull.expected_interval_hours`, `pull.sections_attempted`, `pull.sections_failed` | Once per source per ingest cycle, for every registered source regardless of outcome | `argus ingest pull stalled`, `argus ingest pull failures`, `argus ingest source stale`. **Zero spans observed in the 30-day window as of 2026-08-30** — the code shipped 2026-08-29 but the ingest job's post-deploy dispatch hang (see this project's `project-ingest-hang-post-deploy-20260825` memory) has prevented it from ever running successfully since |
| `fetch_source` | `news_sources` | `news_sources.traced_fetch` | `source_key`, `section` (query-capable sources) or `query` (RSS), `restricted` (bool), `article_count` (int), `error` (only present on failure, redacted of API keys — see `_redact`) | Once per section fetch attempt — nests as a child span inside `ingest_source_pull` (or under `search_news` when called from the agent's on-demand tool) | Not read by any alert directly today — the closest is `argus ingest pull failures`, which reads the coarser `ingest_source_pull.pull.outcome='failed'` instead. This is the natural place to add a section-level failure-count alert later if `ingest pull failures`' source-level granularity ever turns out too coarse |

### `EventLogger`-emitted events (all `level`/`message` set explicitly — see above)

Eleven `except ... as exc:` sites that previously only reached
`docker logs` via a bare `print()`, converted to `_events.log(...)` calls
(`telemetry.get_event_logger(...)`, fanning out to every configured
`telemetry.providers[]` entry whose `KIND` includes `"general"` — see
`telemetry.py`) so each failure mode is independently queryable
(`WHERE span_name = '<event>'`) instead of needing a `docker logs` grep.
Every row also carries `otel.status_code=ERROR` and a recorded exception
(`span.record_exception`), on top of the `logfire.level_num` shown.

| `span_name` (`event`) | `otel_scope_name` | Emitted by | `level` | Attributes beyond `message` | Not read by any alert yet |
|---|---|---|---|---|---|
| `embedder_load_failed` | `argus.news_embed` | `news_embed.build_embedder` | ERROR | — | process-wide embedding degradation, worth its own alert if it recurs |
| `embed_batch_failed` | `argus.news_embed` | `news_embed.embed_texts` | WARN | `batch_size` | |
| `archive_write_failed` | `argus.message_archive` | `message_archive.archive_message` | WARN | `kind` | |
| `keyness_refresh_failed` | `argus.news_ingest` | `news_ingest._refresh_category_keyness` | WARN | — | |
| `category_admin_notify_failed` | `argus.bot` | `bot.review_category_proposals` | WARN | `name` | |
| `batch_classify_failed` | `argus.news_classify` | `news_classify._classify_one_batch` | WARN | `batch_size` | |
| `description_draft_failed` | `argus.news_classify` | `news_classify.draft_category_description` | WARN | `name` | |
| `interest_normalize_failed` | `argus.news_classify` | `news_classify.normalize_interest_detailed` | WARN | `text` | |
| `interest_expand_failed` | `argus.news_classify` | `news_classify.expand_interest_for_retrieval` | WARN | `interest` | |
| `router_failed` | `argus.guardrails` | `guardrails.classify_message` (layer 2) | **ERROR** | — | load-bearing: silent fail-open here is the exact 2026-08-21 incident (`docs/plans/guardrails-plan.md`) — don't let a future alert audit downgrade this to WARN |
| `output_check_failed` | `argus.guardrails` | `guardrails.is_output_on_topic` (layer 4) | **ERROR** | — | same reasoning as `router_failed`, its layer-4 mirror |
| `search_query_rewrite_failed` | `argus.agent` | `agent._rewrite_search_query` | WARN | `query` | added 2026-09-05 alongside search_news's query-rewrite step; missing from this table until now, not a new event |
| `interest_exploration_failed` | `argus.bot` | `bot._process_find_interests` | ERROR | `chat_id` | the exploration agent loop raising; the session is cleared, so the subscriber sees an error rather than being stuck in a mode |
| `interest_exploration_out_of_turns` | `argus.bot` | `bot._process_find_interests` | WARN | `chat_id`, `turns` | not an error — a subscriber who never converged. Worth watching as a RATE: frequent hits mean the elicitation method isn't working, which nothing else measures |
| `interest_turn_out_of_steps` | `argus.interest_finder` | `interest_finder.run_turn` | WARN | `chat_id`, `max_steps` | ONE turn hit the LangGraph step ceiling — measured cause is a topic with no coverage the model keeps rephrasing. Distinct from `interest_exploration_out_of_turns`, which is the whole conversation |
| `interest_saved_from_exploration` | `argus.interest_finder` | `interest_finder.execute_save` | INFO | `chat_id`, `topic`, `turns` | the outcome event — the numerator in "how many explorations produce an interest". `execute_save` is the single write path, reached from both the `save_interest` tool and `bot._execute_pending_proposal`'s deterministic confirmation gate, so this fires exactly once per real save regardless of which path triggered it |
| `interest_exploration_ended` | `argus.interest_finder` | `interest_finder.end_exploration` | INFO | `chat_id`, `reason`, `turns`, `saved_count`, `dropped_count` | fires on every model-ended exploration, INCLUDING ones that saved nothing; the denominator for the row above |
| `confirmation_check_failed` | `argus.interest_finder` | `interest_finder.classify_confirmation` | WARN | — | the affirm/decline/unclear classifier failing; fails open to "unclear", which just falls through to the normal agent turn -- never worse than before this mechanism existed |

None of these eighteen have a dedicated alert yet — they're new visibility,
not new paging. `router_failed`/`output_check_failed` are the strongest
candidates for one, given the incident they're already tied to.

### `search_news` per-call latency (`EventLogger`-emitted, `level=INFO`)

Added 2026-09-05, after a live INT test found one search_news question
taking 75-100+ seconds (the tool-calling-loop design that PR #85 replaced
— see `docs/plans/local-news-cache-plan.md` item 5's "2026-09-05
redesign" note for the full incident) and a follow-up investigation on
the fixed, deterministic pipeline still measured ~11-20s per question.
These seven events exist to make that breakdown queryable going forward
instead of needing another one-off `print()`-based diagnostic pass —
each wraps exactly one real API call (or, for `latency_cache_read_relevance_filter`,
one non-API but previously-slow step) in `agent.search_news`'s pipeline,
in the order they execute:

| `span_name` (`event`) | `otel_scope_name` | Emitted by | Attributes beyond `message` | Fires |
|---|---|---|---|---|
| `latency_layer2_classify` | `argus.bot` | `bot.process_message` | `duration_seconds` (float) | Every message, all categories — not search_news-specific |
| `latency_query_rewrite` | `argus.agent` | `agent._rewrite_search_query` | `duration_seconds` | Every search_news call with a non-None `guard_model` |
| `latency_definition_generation` | `argus.agent` | `agent.search_news` | `duration_seconds` | Only on a cache-miss for the (rewritten) topic — skipped entirely once a topic's definition is cached |
| `latency_cache_read_relevance_filter` | `argus.agent` | `agent.search_news` | `duration_seconds` | Every search_news call (news_cache.read_all() + news_embed.filter_by_relevance) |
| `latency_report_writing` | `argus.agent` | `agent.search_news` | `duration_seconds` | Every search_news call that reaches the model (i.e. the candidate pool wasn't empty) |
| `latency_search_news_total` | `argus.bot` | `bot._route_a_reply` | `duration_seconds` | Every news_query turn -- wall-clock for the whole search_news call, should roughly equal the sum of the four `agent.*` events above plus DB/quota overhead not separately timed |
| `latency_layer4_output_check` | `argus.bot` | `bot._route_a_reply` | `duration_seconds` | Every news_query turn |
| `latency_interest_turn` | `argus.bot` | `bot._process_find_interests` | `duration_seconds` | Every turn of a `find_interests` exploration — the one place in this codebase that still runs an agent LOOP, so this is where a per-turn blowup (PR #85's failure) would show up first |

**Real numbers measured live on INT, 2026-09-05** (two genuinely new
topics, each a real cache-miss so all seven events fired): 11.23s and
7.89s total. **`latency_definition_generation` was the single largest
contributor in both runs** (4.71s and 2.66s) — bigger than
`latency_report_writing` (1.88s and, in the second run, whatever the
no-results branch's write cost was), which had been the assumed
bottleneck before measuring. `latency_cache_read_relevance_filter` was
consistently ~0.1s, confirming the `vector_store`/`SqliteVecStore`
migration (this same day) closed that gap as intended. See
`docs/plans/local-news-cache-plan.md`'s latency section for the full
investigation writeup. Not yet re-measured on a topic with an
ALREADY-cached definition (`latency_definition_generation` skipped
entirely) — expected to be meaningfully faster, not yet confirmed.

## Auto-instrumented spans (not hand-written — `openinference-instrumentation-langchain`)

Everything under `otel_scope_name = 'openinference.instrumentation.langchain'`
is produced automatically by an `otlp`-type `telemetry.providers[]`
entry's `instrument_langchain: true` config wiring up the agent's
LangGraph/LangChain execution onto `telemetry.py`'s internal llm
`TracerProvider` (`telemetry_providers/otlp.py`, called from
`telemetry.setup_telemetry()`) — this project never names these spans
itself. Seen in the last 30 days:
`ChatDeepSeek`, `RunnableSequence`, `PydanticToolsParser`, `tools`,
`model`, `search_news`, `LangGraph`, `save_note`, `FakeToolCallingModel`
(test-only). None of these are read by any alert today — `argus model
errors` reads `push_outcome`'s own `model_error` outcome instead of
querying these directly, since that's already the point where a model
failure is known to have affected an actual subscriber, not just an LLM
call in isolation. Useful for manual trace debugging (following one
request's full chain through Logfire's UI) but not part of the alerting
surface as designed.

## Every alert, and the span(s)/attribute(s) it depends on

Full query text and ids: `local-infra/infrastructure.yaml`'s
`logfire.alerts` (gitignored — real ids/queries live there, not here).
This table is the quick cross-reference; that file is the source of truth.

| Alert | Span(s) | Condition |
|---|---|---|
| `argus bot liveness` | any span, `service_name='myfirstagent'` | none in 30 min — dead man's switch for the whole process, not any one job |
| `argus model errors` | `push_outcome` | `push.outcome='model_error'` in 30 min |
| `argus delivery ratio` | `push_outcome` | `delivered` count < 80% of `push.generated=true` count, 24h, only once ≥5 generated |
| `argus ingest liveness` | `ingest_heartbeat` | none in 30 min |
| `argus ingest pull stalled` | `ingest_source_pull` | no span of any outcome anywhere in 1h (fixed 2026-08-30 — a `pull.outcome='success'`-only version couldn't tell "every source correctly not due yet" apart from "the pipeline is dead," since `not_due` is the majority outcome for a fleet of 4h+-interval sources) |
| `argus ingest pull failures` | `ingest_source_pull` | more than 3 `pull.outcome='failed'` in 12h (widened 2026-08-30 from 30min/>5 — a single known-broken 8h-interval source can contribute at most 2 failures per 12h, so this tolerates one persistently-broken source without paging on it alone) |
| `argus ingest source stale` | `ingest_source_pull` | per source (`GROUP BY pull.source`, any outcome — not success-only): no `pull.outcome='success'` (or none ever) within `2 × pull.expected_interval_hours`, 3-day lookback (fixed 2026-08-30 — the original success-only `GROUP BY` made a source with zero successes in the whole window invisible instead of flagged; `perigon`'s real, ongoing 403 was the live case that caught this) |
| `argus html validation retry` *(planned)* | `html_validation_attempt` | `valid=false` AND `attempt` in (1,2) |
| `argus html validation exhausted` *(planned)* | `html_validation_attempt` | `attempt=3 AND valid=false` |

**Known gap, surfaced 2026-08-30**: no alert currently watches
`pull.outcome='budget_exhausted'` specifically. A source hitting its own
daily API cap (`news_ingest._DAILY_CAPS` — currently `perigon`, `newsapi`)
is a deliberate self-throttle, not a failure, so it's correctly excluded
from `argus ingest pull failures`. But nothing pages if a source stays
budget-capped far longer than expected either (that would eventually
surface via `argus ingest source stale` once enough time passes relative
to that source's own interval, but there's no dedicated, faster signal for
"we're throttling ourselves more than intended" specifically). Not built —
flagged here so it isn't lost, add a query on `pull.outcome='budget_exhausted'`
if this granularity turns out to matter.

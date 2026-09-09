# Local News Cache Plan

This doc captures the design, same pattern as the other `docs/*-plan.md`
files. **Every item in the Status table below is built.** One later
addition is not: see "Source sections are policy hardcoded as vocabulary"
at the end, added 2026-09-09 and still proposed.

**The ask:** stop calling news sources live on every query. Instead, pull
from all enabled sources on a schedule, cache what's fetched locally with
a rough pre-classification, and have `search_news` query the cache —
filtering first by rough category, then by content — instead of hitting
the network per request.

## Status

| # | Item | Status |
|---|---|---|
| 1 | Periodic ingestion job (pull all sources on a schedule) | **Built** — `news_ingest.py`, per-source interval + daily budget respected, wired into both `bot.py` and `combined_bot.py`'s scheduler |
| 2 | Local cache (one file per article, 2-day TTL, auto-cleanup) | **Built** — `news_cache.py`, verified live: real fetch → real classification → real YAML file, contents match this doc's spec exactly |
| 3 | Article classification (rough category tags) | **Built** — `news_classify.py`, one batched structured-output call per cycle, verified live against the real DeepSeek model (not just mocked) |
| 4 | Two-stage query filtering (category, then content) | **Built for the push path (item 6).** `search_news` (item 5) uses only the content (relevance/embedding) stage, deliberately — an ad hoc search query has no pre-classified category the way a stored interest does, so there's nothing to coarse-filter by before the relevance pass. |
| 5 | `search_news` rewritten to read the cache instead of live sources | **Built 2026-09-04, redesigned 2026-09-05.** `agent.py`'s `search_news` reads `news_cache.read_all()` (the same corpus `news_push.py` reads) and ranks by embedding similarity to a generated query definition (`news_embed.filter_by_relevance`, `news_classify.expand_interest_for_retrieval` reused via `interest_cache_ops`'s existing cache), instead of calling any source live. Also picked up a shared "already shown" dedup with push (`subscriber_ops.mark_links_shown`/`pushed_links`) and a per-subscriber daily quota (`subscriber_ops.try_consume_search_query`, default 10/day) — neither existed in the original design, added per user direction during implementation. Restricted-source gating (`RESTRICTED_SOURCES`/`get_restricted_sources_enabled`) was dropped from this function specifically: it no longer calls any source at all, live or restricted — that gate still applies to `news_ingest.py`'s own scheduled pulls and `news_push.py`'s digest filtering, unchanged. **2026-09-05 redesign**: a live INT test found the original version (`search_news` as a LangChain tool an agent loop decided whether/how often to call) searching one question 5-7 times, each a full ~15s `news_cache.read_all()` against INT's real 2036-article cache, because the gap-note instruction it inherited from `TREND_REPORT_STRUCTURE` gave the model a reason to keep searching. Fixed by making `search_news` a plain function with a fixed pipeline (query-rewrite using conversation history → cached-or-generated definition → one vector retrieval → at most one report-writing model call) instead of a tool in an open-ended loop — see `agent.py`'s own module note above `search_news` for the full incident and design. `agent.build_agent`/`run_agent` (the tool-calling machinery) are kept working but no longer used by any live route, per user direction, in case a future feature needs a genuinely open-ended agent. |
| 6 | `news_push.py` converging onto the same cache | **Built 2026-08-15** — see "Interaction with `news_push.py`" below, rewritten in place rather than left as a follow-on. |
| 7 | Since-based ingestion for query-capable sources (replacing the flat top-N cap) | **Built 2026-08-16** — see "Since-based ingestion" below |
| 8 | Article/topic embeddings — near-duplicate collapse and offbeat/novelty selection | **Built 2026-08-25** — `news_embed.py` (model2vec), computed at ingestion, consumed by `news_push.select_candidate_articles`. See `docs/analysis/cluster-measurements.md`'s "Shipped, 2026-08-25" section for the measurement behind it and what's still provisional (the offbeat gate threshold). |

**Why item 6 moved from "optional" to "built" ahead of item 5:** a real
incident forced it. `news_push.py` calling live, query-blind RSS sources
directly (Nikkei Asia's general top-stories feed, in this case) put an
Indonesia earthquake and a Japan-society piece into a subscriber's tech
digest — nothing in the old push pipeline could catch that, since no
relevance filter existed before the digest-writing prompt at all. This
was exactly the gap items 4/6 were meant to close; deferring it further
once it was visibly causing incorrect output wasn't defensible the way
deferring it pre-emptively was. `search_news` (item 5) kept its original
deferral reasoning for a while — it's synchronous and user-facing — until
it was designed and shipped 2026-09-04; see item 5's row above.

**Next step:** none outstanding for this plan's own scope — items 1–8 are
all built. (`search_news`'s daily quota and shared dedup with push, added
during item 5's implementation, are new surface with their own future
tuning questions, e.g. whether 10/day is the right default once real
usage exists — not tracked in this doc.)

## `search_news` latency investigation (2026-09-05)

The 2026-09-05 redesign (item 5's row above) fixed the *repeated-call*
problem — one question no longer triggers 5-7 `search_news` invocations
— but fixing that surfaced the next question: how long does even ONE
question actually take, and where does the time go? This section is
that follow-up investigation's writeup, per user request, so the finding
doesn't need re-deriving next time someone asks "why is search still
slow."

**Two separate bottlenecks, fixed in two separate changes:**

1. **`news_cache.read_all()` itself was slow** — ~12s against a real
   ~1600-2000 article corpus, confirmed by direct measurement (opening
   and YAML-parsing ~2000 individual files has heavy per-file overhead
   even though the actual data volume is tiny, ~6.5MB). Fixed the same
   day by making the news-cache backend pluggable (`vector_store/`) and
   adding a `SqliteVecStore` backend (SQLite + the sqlite-vec extension)
   — `read_all()` dropped to **0.09s** against the identical 1606-article
   corpus (measured on INT, real data copied from the old YAML cache, not
   synthetic) — **~137x faster**. See `vector_store/sqlite_vec_store.py`'s
   own module docstring for the full design and `TODO.md` for what this
   doesn't yet answer (a retention TTL for archived rows, memory/disk
   alerts at larger scale).

2. **Even with that fixed, one question still took ~11-20s** — because
   the disk read was never the ONLY cost, just the easiest one to
   measure first. A live per-call timing pass (`docs/current/telemetry-catalog.md`'s
   "`search_news` per-call latency" section has the full event catalog)
   found a real question runs through up to **five sequential LLM API
   calls**: the layer-2 router, `search_news`'s own query-rewrite step,
   a definition-generation call (only on a cache-miss), the report-writing
   call, and the layer-4 output guardrail. Two genuinely fresh topics,
   measured live on INT 2026-09-05:

   | Step | Run 1 | Run 2 |
   |---|---|---|
   | layer2_classify | 1.22s | — |
   | query_rewrite | 1.81s | — |
   | **definition_generation** | **4.71s** | **2.66s** |
   | cache_read_relevance_filter | 0.13s | — |
   | report_writing | 1.88s | — |
   | search_news_total | 8.65s | 5.12s |
   | layer4_output_check | 1.18s | — |
   | **wall-clock total** | **11.23s** | **7.89s** |

   **The definition-generation call was the single largest contributor in
   both runs — bigger than report-writing itself**, which had been the
   assumed bottleneck before measuring (never assume; this project's own
   repeated lesson). It only fires on a genuine cache-miss for the
   (rewritten) topic — a repeated search of the same topic skips it
   entirely, via the existing `interest_cache_ops` cache — but the FIRST
   time any given topic is searched, it's the dominant cost, not the
   report-writing call it sits next to.

   **Not yet done, flagged rather than silently skipped**: re-measuring a
   cache-HIT run (an already-searched topic) to confirm how much faster
   that path actually is with `latency_definition_generation` skipped
   entirely, and investigating whether the definition-generation call
   itself can be sped up (a smaller/faster model for this one
   sub-task? a shorter generated definition? both untested ideas, not
   decided on).

`cache_read_relevance_filter`'s ~0.1s in both runs confirms the
`vector_store` migration (item 1 above) closed its own gap completely —
none of the remaining latency traces back to it. The remaining ~8-11s is
now entirely LLM round-trip time, which is a different, separate
optimization target (reducing the number of sequential calls, or making
individual calls faster) from anything `vector_store` was built to fix.

## Since-based ingestion (item 7)

**The problem, raised by a subscriber**: pushed digests were consistently
landing around 4-6 items. Root cause traced to `news_ingest.py`'s
`MAX_RESULTS_PER_SOURCE = 5` -- a flat top-N cap applied per query per
cycle, regardless of how much was actually new since the last pull. On an
active source, anything past the first 5 was silently discarded that
cycle; the push-time `MAX_ARTICLES_PER_TOPIC` cap and the digest-writing
model's own synthesis (merging same-story coverage) then narrow an
already-thin candidate pool further. The real bottleneck was upstream, at
ingestion, not in the push-selection logic itself.

**Fix**: the 5 `forum`/`api`-class sources (the only ones with any
query/date capability at all -- see `docs/current/ai-news-sources.md`'s "Source
classes") now fetch "everything since this source's last successful
pull" instead of a flat 5. Two mechanisms:

1. A server-side date filter, for the 3 sources confirmed live to
   support one correctly (hackernews, arxiv, gnews) -- an efficiency
   optimization, smaller payloads.
2. A client-side filter in `news_ingest.py`, applied to all 5 regardless
   of #1: drop any article with `published_dt` at or before the cutoff.
   This is what's actually authoritative, and the only mechanism for
   newsapi/perigon, whose server-side date params turned out to be
   respectively counterproductive (a live-confirmed ~24-36h free-tier
   delay that would make a 24h since-window frequently return nothing)
   and unverified (no API key to test against). See
   `docs/current/ai-news-sources.md`'s "Since-based ingestion" section for the
   full per-source live-verification table.

**The cutoff itself was corrected the same day.** Originally implemented
using `last_pulled_at` (when the ingestion job last ran) -- wrong, per a
design review: that value advances every cycle regardless of whether
anything new was found, so an article a source indexes with a delay
(exactly NewsAPI's confirmed ~24-36h delay above) could fall behind a
since-cutoff that already moved past it, and get silently skipped
forever rather than just delayed. Fixed by tracking a separate per-source
value instead -- `users_db.get_source_last_article_dt`/
`set_source_last_article_dt`, the newest article's own `published_dt`
actually observed, which only ever advances to what's genuinely been
seen. `last_pulled_at` still exists and still drives the per-source
due-check (`_is_source_due`) -- an unrelated question ("how often do we
poll this source") that the fix didn't touch.

`rss`-class sources (16 of the 21 registered) have no query or date-range
parameter to ask for "since X" at all, so there's nothing to switch them
to -- but their own flat cap was still part of the same complaint, and got
raised the same day: `MAX_RESULTS_PER_SOURCE_RSS` went from 5 to **200**,
since the original 5 was arbitrary and cut real digests down regardless of
how much a feed actually had. This created a second problem needing its
own fix: at 200/feed, most of a cycle's pull is typically the same items
as last cycle, so **every currently-cached link is loaded once per cycle
and skipped from classification** if a fetched article matches one --
otherwise a redundant, paid DeepSeek classification call would run on
~195 unchanged articles every 4 hours for nothing. New-vs-already-cached
counts are logged per source and per cycle so this cap can be tuned again
from real data instead of guessed at a second time. A new, higher
`MAX_RESULTS_PER_SOURCE_SINCE_LAST_PULL` (50) is a safety ceiling for the
5 time-filterable sources, not their real limit anymore -- that's now
"however much is genuinely new."

## Item 7's unintended consequence: classification stopped working (found 2026-08-19)

Raising the RSS cap from 5 to 200 grew each cycle's batch from ~100
articles to 100-1000+. `news_classify.classify_articles` made **one
structured-output call per cycle**, and that call fails all-or-nothing
above roughly 110 articles -- almost certainly the response exceeding the
model's output token limit, since it emits one entry per article.

Measured from production by grouping cached articles by the tick that
wrote them: batches of 1/60/109 were 100%/95%/96% categorized; batches of
113/139/147/171/281/1085 were **0%**.

Net effect: **92.8% of the cache (2,099 of 2,262 articles) carried no
categories**, and `news_push.select_candidate_articles` excludes
uncategorized articles whenever the subscriber's topic has real
categories -- so most of the cache was invisible to most subscribers.

It ran for three days undetected because `classify_articles` caught every
exception and returned `{}` silently. The article still got cached, just
uncategorized, which is indistinguishable from "the classifier found
nothing that applies."

**Fixed**: batches are chunked at `MAX_ARTICLES_PER_CALL = 50` (well under
the observed cliff, since the cliff moves with prompt and model changes),
failures now print, and a failed chunk costs one chunk rather than the
whole cycle. See `docs/analysis/cluster-measurements.md` for the full
measurement.

**The generalizable lesson**, and the reason this is recorded rather than
just fixed: fail-open is the right behavior here -- a classification
hiccup should not block caching an article -- but *fail-open and silent*
is not. The same instinct that makes the system robust made a total
outage indistinguishable from normal operation. Fail open, but say so.

## Why this is worth doing (not just "instead of on-demand")

Two things converged to motivate this, both surfaced this session:

**1. Most sources can't tell you "no match" — they just dump their latest
items regardless of the query.** Confirmed directly in `news_sources.py`:
of the 21 currently-enabled sources, only `hackernews` and `arxiv` (plus
the three dormant key-gated ones) actually filter by the search query. The
other 16+ are `rss`-class — they return their latest N items no matter
what was asked. When a subscriber asked about "AAOI" (a small fiber-optics
company), `search_news` reported "30 articles found across 7 sources," but
none were actually about AAOI — the count was padded by AI-blog sources
returning their latest posts regardless of relevance. **The only thing
that caught this was the model reading 30 titles and judging none of them
matched** — an unstructured, per-call, non-reproducible judgment, not a
code-level signal.

**2. A pre-classified local cache turns that into a code-level check.**
If every cached article already carries rough category tags, "does this
plausibly cover topic X" becomes a filter over structured data instead of
the model eyeballing a live dump on every single query. This is the same
move this project's guardrail history already learned to make the hard
way (`docs/system-overview.md` Appendix B.1): things that must be judged
correctly belong in code, not in a model's free-text reading of raw
content, wherever that's achievable.

**A secondary but real benefit:** 21 sources called live on every single
`search_news` invocation is real per-message latency and load on other
people's infrastructure. Pulling on a schedule instead means one round of
network calls per cycle, not one per user message.

## Proposed architecture

```
[APScheduler tick, every N hours]
        |
        v
  for each enabled source in news_sources.SOURCE_REGISTRY:
        fetch latest articles
        |
        v
  [one batched structured-output LLM call per cycle]
        classify all newly-fetched articles -> category tags per article
        |
        v
  write one YAML file per article to the cache directory
        |
        v
  [cleanup sweep] delete any cached file older than 2 days (by fetched_at)


[a chat message or push cycle needs articles]
        |
        v
  Stage 1 -- category filter (deterministic, in code)
        narrow the cache to files tagged with a plausibly-relevant category
        |
        v
  Stage 2 -- content filter (same mechanism as today)
        the model reads titles/summaries in the narrowed set and picks real matches
        |
        v
  synthesize a reply / digest from what's left
```

The key shift: **stage 1 happens before the model ever sees anything.**
Today the model reads through everything every source returned, including
whatever the 16 query-blind RSS sources dumped. After this change, it only
ever reads a pool that's already been narrowed by category — which is
smaller, faster, and doesn't require the model to silently discard
irrelevant AI-blog noise on every single call the way it does today.

## Cache format

**File naming:** `{source}-{id}.yaml`, where `id` is a short hash of the
article's `link` — e.g. `bbc_business-a3f9c21e.yaml`. Hashing the link
(rather than an incrementing counter or the source's own ID field, which
not every source provides) means re-fetching the same article in a later
cycle produces the same filename and just overwrites harmlessly, instead
of creating a duplicate — free deduplication, no separate tracking state
needed.

**File contents (YAML):**

```yaml
source: bbc_business
title: "Nvidia in talks to invest in CoreWeave amid cloud expansion"
link: "https://www.bbc.co.uk/news/articles/..."
summary: null              # most RSS sources don't provide one; null is fine
published: "Thu, 13 Aug 2026 10:00:00 GMT"   # raw string, as today
published_dt: "2026-08-13T10:00:00+00:00"    # parsed, as today
fetched_at: "2026-08-13T12:00:05+00:00"      # when THIS system pulled it
categories: [IT, Hardware, Finance, Stock]
```

**Retention is based on `fetched_at`, not `published_dt`.** Some sources'
dates don't parse (already a known gap — see `_parse_rss_published`
returning `None`), but `fetched_at` is always known and controlled by our
own code, so the 2-day cleanup sweep has a value it can always trust.

**Storage:** flat files in a directory, matching the user's stated
preference for now. Following the same configurability convention as
`users_db.py`'s `SUBSCRIBERS_DB_FILE` — a `NEWS_CACHE_DIR` env var,
defaulting to a local relative path, pointed at the mounted `/data`
volume in the deployed container so it survives restarts the same way
`subscribers.db` does.

**Real storage estimate**, so this isn't a guess: 21 sources × ~5 articles
per pull × 6 pulls/day (if the interval is 4h) ≈ 630 files/day, ~1,260 at
steady state under a 2-day TTL. At roughly 1–2 KB per YAML file, that's
**under 3 MB total.** Checked the deployed VM directly: 35 GB free of 45
GB. This is not a real constraint at this scale, and isn't worth
optimizing around.

## Category taxonomy (proposed)

Multi-label, not single-label — the worked example from this request
("Nvidia investigates CoreWeave and its stock goes up") should plausibly
carry several tags at once, not be forced into one bucket.

| Category | Covers |
|---|---|
| **AI** | AI models, research, agents, LLMs |
| **Software** | Software products, dev tools, programming |
| **Hardware** | Chips, semiconductors, devices, infrastructure hardware |
| **IT** | Enterprise IT, cloud, infrastructure, enterprise software |
| **Startups** | Funding rounds, new companies, venture capital |
| **Finance** | Business/financial industry news, economics, corporate deals |
| **Stock** | Stock price moves, market reactions — distinct from Finance: this is specifically about market/price impact, not business news generally |
| **Policy** | Regulation, government, legal, antitrust |
| **Security** | Cybersecurity, breaches, vulnerabilities |
| **Research** | Academic papers, science (arXiv-flavored) |
| **Consumer** | Consumer gadgets, reviews, product launches for individual users |
| **Robotics** | Robotics specifically — called out separately because it's already a real subscriber interest (`機器人科技`), not something "AI" or "Hardware" alone captures well |
| **Crypto** | Cryptocurrency/blockchain — also already a real subscriber interest (`bitcoin`) |

Worked example, checked against this table: "Nvidia in talks to invest in
CoreWeave, stock reacts" → `IT` (cloud infrastructure), `Hardware` (chips),
`Finance` (the deal itself), `Stock` (the price reaction). All four
present, matching the request's own example.

**Resolved as a v1, not a final answer.** Confirmed this table is coarser
than it will eventually need to be — but the fix is refining it later
against real classified data, not blocking implementation on getting it
right up front. A category taxonomy is cheap to revise (relabel/resplit
existing cache entries, no schema migration, no code path depends on the
exact category names beyond string matching) — better to ship this
version, see what real articles actually get tagged, and refine from
observed gaps than to keep guessing categories against no data at all.

## Classification mechanism

**Proposed: one batched structured-output LLM call per ingestion cycle**,
not one call per article. Send the cycle's full batch of newly-fetched
articles (title + summary + source) in a single call, get back a list of
`{article_id, categories: [...]}`.

**Why batched, not per-article:** at ~630 articles/day, per-article calls
would mean 630 LLM calls/day just for classification — real, avoidable
cost. Batching by cycle means one call per pull (4–6 calls/day at a 4–6
hour interval), each classifying everything that cycle fetched at once.

**Why a cheap model fits here:** this is exactly the shape
`docs/plans/model-portability-plan.md`'s "Level 2 — per-stage model routing"
already anticipated — a small/fast model is sufficient as long as it
supports structured output, same as the router and the output-check
already use. No new infrastructure decision, just another consumer of a
seam that already exists.

**What this doc is NOT proposing:** keyword/heuristic tagging instead of
an LLM call. It was considered — zero marginal cost, no model dependency
— but rejected as the primary mechanism: this project's own guardrail
history is a direct lesson that hand-rolled text-matching heuristics are
exactly the kind of thing that quietly breaks in ways that are hard to
notice (`docs/system-overview.md` Appendix B.1). A cheap structured-output
model call is more reliable for the same reason it already won for the
router and the output check. Per **P4** (accuracy raised by
post-deployment testing, not assumption), classification accuracy should
still be spot-checked against real articles before being trusted at
scale — that's an implementation-phase task, not a planning-phase one.

## Two-stage query filtering, in more detail

**Stage 1 (category filter) needs to know which categories to look at.**
Two different call sites, two different mechanisms — reusing
infrastructure that already exists in both cases, rather than adding a
new classification pass:

- **Push digests** (`news_push.py`): a subscriber's `interests` are
  already stored and stable. Classify each interest into categories once,
  when it's set (`update_interests`/`/interests`), and cache that mapping
  — cheap, since it only runs on change, not per push cycle.
- **On-demand chat queries** (`agent.py`'s `search_news` tool): most real
  queries are one-off natural language, not tied to a stored interest at
  all (the AAOI question itself never touched `/interests`). Extend the
  router's existing structured-output classification
  (`guardrails.classify_message`) to also emit likely categories for the
  query — one more field on a call that already happens for every
  message, not a new call.

**Stage 2 (content filter) is unchanged from today** — the model reads
titles/summaries within the now-much-smaller, already-fresh candidate
pool and picks real matches, exactly like it does now. The difference is
what it's reading: a category-narrowed slice of a 2-day cache, not a live
dump padded by 16 sources that ignored the query entirely.

## What this doesn't solve, and the resulting recommendation

**Important limitation to be honest about:** a periodic pull of "latest N
from each source's homepage/category feed" does not guarantee coverage of
a low-profile, specific-company query. If AAOI simply never appears in
any of the 21 sources' latest items during a given 2-day window, the cache
won't have it either — no different from today. The cache makes relevance
*filtering* correct and cheap; it doesn't manufacture coverage that
doesn't exist.

**Resolved design: per-source pull frequency, one mechanism, not two.**
Earlier drafts of this doc proposed reserving budget-limited sources
(Perigon) for reactive on-demand fallback only, excluded from the
periodic pull entirely. Revised: **every source participates in the same
periodic ingestion job, each at its own safe frequency** — simpler than
maintaining two separate mechanisms (scheduled pull for free sources,
reactive-only fallback for capped ones), and it means a capped source's
scarce calls proactively broaden the cache instead of only reacting to a
miss.

| Source | Frequency | Why |
|---|---|---|
| Unrestricted (RSS, `hackernews`, `arxiv`) | Every 4h | No real budget constraint |
| GNews | Every 4h | 100/day budget comfortably covers 6 pulls/day |
| Perigon | 3x/day | 150/month budget — see below |
| NewsAPI | 1x/day | Individual-use judgment — see below |

A live, reactive fallback for a specific on-demand query that still comes
up empty after all of the above (an AAOI-style edge case) remains a
possible later addition, but isn't required for v1 — proactively pulling
from Perigon and NewsAPI several times a day already broadens the
cache's real-search coverage well beyond what the free-tier RSS sources
alone provide, which should make a total miss meaningfully rarer than it
is today.

**Every pull — scheduled, any source — writes into the same shared
cache**, not a private result for whoever triggered it. This is the
actual reason a budget-limited source is worth having in the cache
architecture at all: **one Perigon call can satisfy every subscriber
whose interests match it for the next two days**, not just one query.
Without a shared cache, a source with a 150/month budget could really
only ever answer 150 individual questions/month across every user
combined — with it, those same 150 calls seed content that any number of
matching queries can draw from within each call's 2-day cache window.

**Worked example: Perigon's budget, concretely.** 150 requests/month,
free/non-commercial tier (see `docs/current/ai-news-sources.md`'s key-acquisition
section). 3 pulls/day ≈ 93/month, leaving ≈57/month headroom for testing.
When the daily cap is hit, that cycle's pull is skipped (not attempted)
and logged, same shape as `news_push.py`'s existing per-cycle outcome
logging. Tracking the daily count is a small, global piece of state
(source name, date, count — reset when the date rolls over), natural to
add to `users_db.py` alongside everything else it already tracks.

**NewsAPI — added, with its own real constraint documented so it isn't
forgotten.** NewsAPI's free "Developer" plan terms, quoted directly from
their own criteria: *"You should only use the Developer plan if your
project is in development. If your project is being used in production,
please upgrade to a paid plan."* This project's own judgment (recorded
here, not re-litigated further): a personal, unpaid, invite-gated pilot
run by an individual is a reasonable read of "in development," not
"production" in the sense that clause is aimed at — but that judgment
is worth being able to revisit, not silently forgotten. **1 pull/day**,
chosen deliberately conservative relative to whatever NewsAPI's actual
technical rate limit turns out to be (not confirmed live — see
`docs/current/ai-news-sources.md`). **Trigger to revisit:** if this project is
ever monetized, opened beyond an invite-only pilot, or otherwise starts
looking like the "production" NewsAPI's own terms describe, upgrade to a
paid plan or drop NewsAPI — don't keep running on the Developer plan past
that point.

**Sequencing risk — resolved 2026-09-04, by item 5 shipping.** The
original concern: the Perigon and NewsAPI keys exist in Vault and
`docker-entrypoint.sh` fetches both, and before this plan was built,
nothing stopped `search_news` from calling them unthrottled on every
matching on-demand query from every user. Partially mitigated 2026-08-14
(`news_sources.RESTRICTED_SOURCES` gated both behind a per-user DB flag,
defaulting to off for everyone except the admin), then closed outright
once item 5 shipped: `search_news` no longer calls any source live at
all, restricted or not, so there is nothing left for the admin's own
usage to spend against `news_ingest.py`'s budget. `get_restricted_sources_enabled`/
`RESTRICTED_SOURCES` remain in use — just for `news_ingest.py`'s own
scheduled pulls and `news_push.py`'s digest filtering, not for
`search_news` any more.

## Interaction with `news_push.py` — built 2026-08-15

`news_push.py` now reads `news_cache.read_all()` once per push cycle
(shared across every due subscriber in that tick, not re-read per
subscriber) instead of calling `news_sources.enabled_sources()` live.
Two-stage filtering, per the architecture above:

- **Stage 1 (category filter, in code)** —
  `news_push.select_candidate_articles`. Each subscriber's `interests`
  are mapped to categories via `news_push.resolve_interest_categories`,
  which reads/writes a new `users_db.interest_categories` table (global,
  not per-subscriber — the same interest text means the same categories
  for anyone) and only calls `news_classify.classify_interests` for
  interests not already cached. An interest with NO cached category
  (classifier miss) is treated as unrestricted, matching any article,
  rather than matching nothing — the alternative would silently starve a
  subscriber over a classification gap on their own stated interest. An
  article with no categories IS excluded whenever the topic has real
  categories to match against — this is the exact mechanism that keeps a
  general-news RSS item (no tech angle at all) out of a digest.
- **Stage 2 (content filter)** — folded into the existing single
  `write_push_digest` model call rather than a separate LLM call:
  `_PUSH_DIGEST_PROMPT` now explicitly tells the model the category
  filter is coarse and to omit any candidate that survived it but isn't
  genuinely relevant, up to writing nothing at all if none qualify
  (`run_push_cycle` treats an empty/whitespace-only digest the same as
  "no new articles" — advances dedup state, doesn't send).

**Real incident that forced this**, not a speculative improvement: a
subscriber's push digest included an Indonesia earthquake and a Japan-
society piece, sourced from Nikkei Asia's general top-stories RSS feed
(query-blind, like 16 of the 21 registered sources — see "Why this is
worth doing" above). The old push pipeline had zero relevance filtering
between "source returned it" and "model was told to report on it" —
exactly the gap this section's two-stage design was meant to close, just
not yet wired up for the push path specifically. Diagnosed by pulling the
feed directly (`curl https://asia.nikkei.com/rss/feed/nar`) and confirming
it's the site's whole front page, not a tech-scoped vertical.

**Restricted-source gating (NewsAPI/Perigon) is unchanged in spirit,
adapted to the cache**: `select_candidate_articles` checks each cached
article's `source_key` against `news_sources.RESTRICTED_SOURCES` and
excludes it unless the subscriber's own `restricted_sources_enabled` flag
is set — same gate as the short-lived live-fetch version shipped one day
earlier (2026-08-14), now applied to cache reads instead of live calls.

**What stayed the same**: `pushed_links` dedup (the fallback for articles
whose `published_dt` didn't parse), the `since`/`last_push_at` recency
check, and per-subscriber isolation (one subscriber's exception doesn't
stop the cycle) are all unchanged from the live-fetch version — only
*where* the candidate articles come from changed.

> **Correction, 2026-08-19.** The paragraph above is no longer true of the
> first two items, and is left in place rather than rewritten so the change
> is visible. `pushed_links` is no longer a *fallback* for unparsed dates —
> it is the **sole** "already seen" filter, checked unconditionally on every
> path. And the `since`/`last_push_at` recency check no longer filters at
> all: `published_dt` now only **ranks** (newest first), with
> `MAX_ARTICLE_AGE_HOURS` as a separate quality gate against genuinely
> ancient content.
>
> Filtering on the date was wrong for a specific reason worth keeping: an
> article that lost one cycle's `max_per_topic` cut was excluded
> permanently, even though nobody had ever seen it. GNews publishes ~12 h
> behind, so 227 of its cached articles could never reach any digest. A
> date says what is most worth showing; it can't say what has been read.
>
> Per-subscriber isolation is unchanged. See `news_push.py`'s module
> docstring and `select_candidate_articles` for the current rules.

## Open questions — all resolved, implementation cleared to start

1. ~~**Pull interval.**~~ **Resolved**: per-source, not one global
   number. See the frequency table above — unrestricted sources and
   GNews every 4h, Perigon 3x/day, NewsAPI 1x/day.
2. ~~**Category taxonomy**~~ **Resolved as a v1**: ship the 13-category
   table as-is, refine later against real classified data rather than
   block on getting it perfect up front.
3. ~~**Live fallback**~~ **Resolved, then superseded**: the original
   "reserve capped sources for reactive fallback" idea was replaced by
   giving every source its own periodic-pull frequency instead (see
   above) — simpler, one mechanism instead of two. A true on-demand
   fallback for a still-empty query remains a possible later addition,
   not required for v1.
4. ~~**Classification model**~~ **Resolved**: the current DeepSeek
   instance, now. `docs/plans/model-portability-plan.md`'s Level 2 routing can
   swap in a cheaper model later without a redesign — independent
   decision, not a blocker.
5. ~~**Cleanup mechanism**~~ **Resolved**: folded into the ingestion
   tick — delete expired, then fetch new, same cycle, one job.
6. **Storage engine, later.** Still the one open item, and deliberately
   left open rather than resolved: files for now, per the original
   request. Same reasoning pattern as `docs/plans/data-layer-plan.md`'s SQLite
   deferral — revisit only if a real trigger shows up (e.g., needing to
   query across cached articles in ways flat files make awkward), not
   preemptively.

## Non-goals for this version

- No embeddings/vector search — stage 2 stays LLM-reads-the-shortlist,
  same mechanism as today. Worth revisiting only if the category filter
  alone proves too coarse in practice.
- No move off SQLite or onto a shared cache store — this is a
  single-process, single-host cache, consistent with
  `docs/plans/data-layer-plan.md`'s current stance on the rest of this
  project's persistence.
- No retroactive backfill of historical articles — the cache starts
  empty and fills from the first ingestion cycle forward.

---

## Source sections are policy hardcoded as vocabulary (added 2026-09-09)

Status: **proposed, nothing built.** Evidence:
`docs/analysis/retrieval-quality-measurements.md` findings 1–3.

### What's wrong

`news_sources.SOURCE_SECTIONS` decides which slice of each source we pull:

```python
"newsapi":    ["technology", "business", "science", "health"],
"gnews":      ["technology", "business", "science", "world"],
"arxiv":      ["cs.AI", "cs.LG", "cs.RO", "cs.CR", "quant-ph", "physics.optics"],  # 6
"hackernews": ["front_page"],                                                       # 1
```

Its own comment justifies living in code because *"the values are dictated
by each API."* That is **half true, and the half that isn't is the
problem.** The vendor dictates the *accepted set*; this dict stores a
*chosen subset* of it. arXiv accepts ~150 subject classes and we list 6.
Algolia accepts `front_page` / `story` / `show_hn` / `ask_hn` / `poll` and
we list 1.

So the dict conflates two different things:

| | What it is | Where it belongs |
|---|---|---|
| **Capability** | which values the API accepts; an invalid one is a broken request | code, next to the adapter — a fact about the vendor |
| **Policy** | which of those we choose to pull | settings — an editorial decision |

The file's own comment already concedes the split — *"the reasoning for
pulling by section at all is ingestion policy and lives with it, in
`news_ingest._sections_for_source`"* — the reasoning moved out, the values
didn't.

**The measured consequence.** Changing our editorial mix costs a code
change, a PR and a deploy, while adding an entire new publication costs
one settings line. That inversion produced a corpus weighted **6:1 toward
preprints over community discovery** (arXiv 272 articles vs. Hacker News
72), and a fifth of the corpus spent on general-interest wire content we
explicitly requested (`science`, `world`, `health`).

### Proposed

**1. Split capability from policy.** Each adapter declares its own valid
sections (`VALID_SECTIONS`), so the vendor fact sits next to the vendor
code — which is precisely what the current central dict drifted away from.
The chosen subset moves into `news_source.*` in settings, with the full
accepted vocabulary listed **commented out** alongside it, so it is
discoverable without reading code. That comment-as-documentation shape is
already this repo's convention (`postgres:`, `sqlite_vec:`, `api:` blocks
all do it).

**2. Invalid section → log and alert, don't crash.** Skip the bad value,
emit an ERROR event carrying the source key and the rejected value.
`_events.log(..., level=Level.ERROR)` is already what Logfire alerts read,
so this needs no new plumbing. Rationale for not failing startup: a typo
should not take the bot down — but it must never be silent, which is the
failure shape this repo has been bitten by repeatedly (unset
`NEWS_CACHE_DIR` silently resetting the cache; `PHOENIX_ENABLED` dead for
weeks).

**3. Pull order becomes a setting too.** The same conflation exists one
line deeper, inside the HN adapter:

```python
endpoint = "search" if section == "front_page" else "search_by_date"
```

"Rank by popularity or by recency" is policy, hardcoded. Make it a
per-source setting (`by_date` | `by_score`), declared as supported or not
by each adapter — it is meaningless for arXiv, GNews and every RSS feed,
so those must not accept it.

**4. Retain Hacker News `points`.** `news_adapters/hackernews.py` maps
title/link/source/summary/published and **drops `points` and
`num_comments` on the floor.** HN is the only source we have that carries
a real human popularity vote (finding 3), and it is the natural signal for
"is this worth a slot" — a question similarity ranking cannot answer.

Two cautions, both measured:

- **Points are captured at their coldest.** A Show HN two hours old has 3
  points; the same story has 300 the next day. Lightpanda's 319 was
  *after the fact*. Pulling by date stores the number at its least
  informative moment and nothing refreshes it. If points are to be used
  for ranking, recent items need a refresh pass — otherwise the signal is
  worse than none.
- **HN items have `summary: None`**, so they embed from the title alone.
  That makes them systematically weaker in the Stage-2 relevance filter
  than any RSS article with a summary — worth knowing before concluding
  that HN content "doesn't rank well".

**5. Adding `show_hn` depends on 3 and 4, and is not a one-liner.**
`show_hn` by date has 557k historical hits and most Show HNs get single-
digit points. Adding it without a score floor or score-ordering would
flood the corpus with noise — and classification is **batched**, with a
measured collapse in accuracy as batch size grows (`news_classify.py`'s
own comment: *batch 113 → 0%*). Noise here would degrade classification
for everything sharing the batch, not just cost tokens.

### Deliberately not decided here

Whether to **stop** pulling `science`/`world`/`health` from the
aggregators. It is tempting — those sections produced *"your daily coffee
habit could be taking a toll on your bones"* — but one live subscriber
follows 光通訊/AAOI/AOI, and `science` may be carrying the optics and
quantum coverage they depend on. Cutting at the request is strictly
cheaper than filtering after ingestion (no API quota, no classification
cost), but it must be **measured per section** before anything is removed:
what fraction of each section's articles ever matched an interest or
reached a digest.

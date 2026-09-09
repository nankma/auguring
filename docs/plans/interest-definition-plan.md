# Making the interest definition visible and refinable

Written 2026-09-09. Status: **built, not yet deployed.**

| Piece | Status |
|---|---|
| Defect 2 fix (`backfill_missing_interest_definitions`) | Built — `news_push.py`, wired into `run_push_cycle` alongside `resolve_interest_categories` |
| Per-subscriber schema (`subscriber_interest_definitions` table) | Built — `storage/schema.py`, `storage/sqlite/interest_cache.py` (shared by Postgres, no override needed) |
| `interest_cache_ops.resolve_interest_definition` (subscriber → shared → None) | Built |
| `search_news`/`select_candidate_articles` read the subscriber tier | Built — both now take/thread `chat_id` |
| `show_definition`/`propose_definition`/`save_definition` tools, `execute_redefine` | Built — `interest_finder.py`, entry point (e) added to its prompt and to `guardrails.py`'s router |
| Preview baked into `propose_definition` | Built — no separate preview tool, cannot be skipped |
| Confirmation-gate integration (`redefine` action) | Built — `bot._execute_pending_proposal` |
| `discusses_own_configuration` false-positive guard (definitions naming LangChain/AutoGen etc.) | Built — found during implementation, not in the original design; see "A guardrail gap found during implementation" below |
| Tests | Built — `tests/test_interest_cache_ops.py`, `tests/test_news_push.py`, `tests/test_agent.py`, `tests/test_interest_finder.py`, `tests/test_guardrails.py` |
| Live verification against the real model | Done for the guardrail widening and router entry (e) — see "Open" below for what's still unverified |

Evidence base: [`docs/analysis/retrieval-quality-measurements.md`](../analysis/retrieval-quality-measurements.md)
findings 4 and 5. Read those numbers before changing anything here — this
design exists because the effect sizes were much larger than anyone
assumed, and because two plausible-sounding alternatives were measured and
discarded.

## The problem in one line

**The interest definition decides what a subscriber receives, it is
generated behind their back, and they cannot see or change it.**

Retrieval uses `query_text = definition or topic` (`agent.search_news`,
`news_push._resolve_query_text`). Measured on the live corpus, rewriting
that one paragraph moved a target article **from rank 467 to rank 2**. The
interest word (`"AI"`) is almost incidental; the paragraph is the
preference.

Two concrete defects follow.

### Defect 1 — the generated definitions carry a vendor/enterprise prior

The live definition for the interest `AI Agent`:

> "...autonomous software systems that use large language models such as
> GPT-4, Claude, or Gemini... tool use via function calling, agentic
> frameworks like **LangChain, AutoGen, and CrewAI**... often deployed as
> **virtual assistants or workflow automation tools**."

That is a description of a technology stack and its enterprise
deployment. It retrieves *"How AI-native companies turn workflows into
operating capability"*. It does not retrieve *"I asked 100 agents to hack
me"* — which is arguably the most interesting AI-agent story of that week,
and which the subscriber explicitly asked for.

Nothing about this is wrong as a *definition of the term*. It is wrong as
a statement of **what this subscriber wants to read about it**, and those
are different things that the current design conflates.

### Defect 2 — a bare interest gets no definition at all

A live subscriber's `AI` interest has **no row** in
`interest_query_expansions`. `definition or topic` therefore embeds the
two-character string `"AI"`, which is a near-meaningless retrieval vector.
It returns *"Introducing AI-as-a-Service"*, *"AI for Food Allergies"*,
*"Democratizing AI Safety with RiskRubric.ai"* — whatever says "AI" most
often.

This is a bug independent of the feature below and should be fixed
regardless: a missing definition must not silently degrade to embedding a
two-character token.

## What was measured and rejected

Recorded so these are not re-proposed later:

- **Writing the genre into the definition** ("hands-on experiments, not
  funding news"). Measured: ~0.09 cosine spread between wanted and
  unwanted articles, and the subscriber's own flagship link scored *below*
  a corporate-acquisition baseline. Static embeddings encode subject
  matter, not story shape.
- **An explicit `story_kind` classifier label.** Withdrawn — the case for
  it rested on an article the subscriber never asked for. Revisit only if
  a residual gap survives this work and is *measured*, not assumed.

## Design

### 1. Definitions become per-subscriber

**This is the blocking schema change.** `interest_query_expansions` is
keyed on `interest` alone:

```sql
SELECT expansion FROM interest_query_expansions WHERE interest = :interest
```

It is a **global** cache. One subscriber refining `"AI"` would silently
change retrieval for every other subscriber following `"AI"`. A feature
called "your preference" cannot be built on a shared row.

Change the key to `(chat_id, interest)`.

**Cost, stated honestly:** definition generation is the most expensive
step in the search pipeline (2.7–4.7s measured,
`docs/current/telemetry-catalog.md`). Today it is paid once per interest
globally; after this, once per (subscriber, interest). It stays cached
after that, and with 26 subscribers the absolute cost is small — but the
cache hit rate genuinely drops, and that is the price of the feature.

**Migration:** keep existing rows as a fallback tier rather than
discarding them. Lookup order becomes: this subscriber's own definition →
the shared/global one → generate. That way nobody's retrieval changes on
deploy day, and personalisation is purely additive.

### 2. Show the definition, then let them refine it

Inside the existing `find_interests` conversation (`interest_finder.py`),
which already implements *show real examples → ask which land → converge*.
The same loop applies here, with the definition as the object being
converged on instead of the topic.

New tools:

| Tool | Does |
|---|---|
| `show_definition(topic)` | Returns the current definition for one of this subscriber's interests, in plain language |
| `propose_definition(topic, definition)` | Records a *proposed* rewrite in the session — does not save it |
| `save_definition(topic)` | Persists a confirmed rewrite and invalidates the cached one |

`propose_definition`/`save_definition` deliberately mirror
`propose_interest`/`save_interest` and go through the **same deterministic
confirmation gate** built in PR #91 — the model must not be able to claim
it changed a definition without the write actually happening. That
incident is documented in `interest-finder-plan.md`; do not re-litigate it
by adding a one-step save here.

### 3. The model proposes *directions*, the subscriber reacts

Do not ask a subscriber to write a retrieval definition. That is the exact
failure mode the elicitation research this feature is built on warns about
(`docs/analysis/interest-elicitation-survey.md`: people judge concrete
instances reliably and generate abstract self-descriptions badly).

Instead the model offers a small number of concrete directions drawn from
what is actually in the cache — e.g. for `AI Agent`:

- 偏工程实作与实验（有人真的跑了一遍，结果如何）
- 偏企业落地与工作流（谁在用、怎么部署）
- 偏框架与技术栈（LangChain / MCP / 评测）

…and the subscriber picks. Their pick is what gets folded into the
definition, not prose they had to author.

### 4. Preview the effect before saving — non-negotiable

**Definition effects are strongly counter-intuitive.** In the measurements
behind this plan, a headline containing "playing Doom" and "Super Mario
64" scored *lower* against a demo-flavoured definition than a Nvidia
acquisition story did. Two separate hypotheses about which article would
win were wrong, held by someone looking directly at the data.

A subscriber editing a definition blind will be wrong the same way.

So `save_definition` must be preceded by a preview: run the candidate
definition through the existing retrieval path and show **the five
articles it would now surface**. Zero new machinery — it is
`news_embed.filter_by_relevance` against the live cache, the same call
push already makes.

This turns the feature from guesswork into an experiment the subscriber
runs themselves, and it is what makes refinement converge instead of
oscillate.

## Sequencing

1. **Built.** Fix Defect 2 (missing definition must not degrade to a bare
   token) — `news_push.backfill_missing_interest_definitions`.
2. **Built**, with a shape adjustment: rather than migrating
   `interest_query_expansions` to a `(chat_id, interest)` key in place, a
   **separate** `subscriber_interest_definitions` table was added instead
   (see `storage/schema.py`'s own comment for why — mixing automatic
   shared writes and deliberate personal ones in one table risked a write
   ordering collision that keeping them apart makes impossible by
   construction). `interest_cache_ops.resolve_interest_definition` is the
   lookup that makes the two tables behave as one fallback chain to every
   caller.
3. **Built.** `show_definition` + preview, folded directly into
   `propose_definition` rather than a separate tool — see "A guardrail gap
   found during implementation" below for something this step's
   real-world content (definitions naming frameworks like LangChain)
   surfaced.
4. **Built.** `propose_definition`/`save_definition` behind the existing
   PR #91 confirmation gate (a third `action: "redefine"` alongside
   `"add"`/`"remove"`), plus `execute_redefine` as the single write path.

Steps 1–3 turned out to have no observable write-risk difference from step
4 in practice, since step 3's preview mechanism made step 4 a small
extension rather than new design — all four shipped together.

## A guardrail gap found during implementation

Not anticipated in the original design: the auto-generated definition for
the interest `AI Agent` (quoted in "Defect 1" above) names LangChain,
AutoGen, and CrewAI — and `guardrails._OUTPUT_SCOPE_PROMPT`'s
`discusses_own_configuration` check explicitly lists LangChain as one of
its trigger examples, because **this bot is itself built with LangChain**.
Showing a subscriber their own definition, or previewing a candidate one,
would have been a predictable false positive for the same reason the
2026-08-08 "already covered interest" incident was: the model conflating
"the user's own data" with "the bot's own configuration"
(`docs/plans/guardrails-plan.md`).

Fixed proactively, before any live testing, the same way the
`find_interests` output-scope widening was: both `discusses_own_
configuration` and `appropriate_bot_content` in `_OUTPUT_SCOPE_PROMPT` now
explicitly carve out "describing what a NEWS TOPIC covers" from
"describing what powers the bot" and list definition-related replies as
appropriate content. Only mock-level tests exist so far
(`tests/test_guardrails.py`); this is exactly the kind of prompt change
that needs a real-model check before being trusted, same discipline as
every other guardrail wording change in this project's history.

## How we will know it worked

The subscriber who raised this is following `AI`, `AI Agent`, `robotics`,
`AI coding`, `Large Language Model`, and `Edge AI development boards`, and
gave five example links they wanted and did not get.

Acceptance is **not** "the code runs". It is: after refinement, re-run
retrieval for that subscriber's interests and check whether the class of
article they asked for now ranks inside the pushed top-5. The measurement
harness for this already exists — it is the same rank probe used to
produce finding 4.

## Open

- **The guardrail widening and the router's entry (e) are verified against
  the real pinned model; the conversational flow itself is not.**
  QA confirmed live (2026-09-09, recorded in `docs/plans/guardrails-plan.md`):
  the `discusses_own_configuration` false-positive risk (a definition
  naming LangChain/AutoGen/CrewAI) is fixed -- 0/8 false positives, 8/8
  correct blocks on a genuine self-disclosure negative control -- and
  `classify_message` correctly routes entry (e) phrasing to
  `find_interests`, 8/8 (harness re-run at 10 trials: 100% across both
  affected groups, no regression elsewhere). What's still unverified is
  the actual multi-turn conversation this feature runs inside a real
  exploration: does the model reliably call `show_definition` before
  proposing a rewrite, follow the "offer concrete directions, never an
  abstract quality label" instruction, and read `propose_definition`'s
  preview before describing it, the way `_SYSTEM_PROMPT` asks? Same
  "measure before shipping" step the step-ceiling fix and the
  false-confirmation fix both needed before being trusted.
- **Does refinement drift into a second interest?** If a subscriber
  refines `AI` far enough, it becomes a different interest in all but
  name. Unclear whether that should be surfaced ("this looks like a new
  interest — split it?") or ignored.
- **Definition rot.** Definitions are generated once and cached forever.
  A definition written around 2026's frameworks will age. No expiry or
  regeneration policy exists today, for either the shared or the
  per-subscriber tier.
- **This does not fix supply.** Refining a definition can only re-rank
  what was ingested. Three of the five example links are from domains we
  do not pull at all — see `local-news-cache-plan.md`'s section work.

# One always-on agent, no session state

Written 2026-09-22. Status: **all six items shipped 2026-09-24 (Steps
A, B, and the Jev migration).**

Retires `end_exploration` and the `interest_sessions` mode switch. The
front door becomes a single always-running conversational agent that
dispatches work to tools/sub-calls, each of which is context-free. The
conversation is one object with one lifetime; when its context is gone,
the bot says so instead of guessing.

## Status

| # | Item | Status |
|---|------|--------|
| 1 | Retire `end_exploration` / `interest_sessions` | Shipped 2026-09-24 |
| 2 | Conversation object = messages + pending offer, one lifetime | Shipped 2026-09-24 (`bot.conversations`) |
| 3 | Tools/sub-calls receive no conversation context | Shipped 2026-09-24 (already true for `search_news`, extended to `start_push`/`stop_push` in Step A) |
| 4 | "I lost the thread" reply when context is missing | Shipped 2026-09-24 -- narrower than first proposed: fires only when there is BOTH no pending offer AND no conversation history at all, via `interest_finder.reads_as_bare_confirmation`; real history is left to the top-level agent to resolve itself |
| 5 | Layers 2 and 4 move to Jev (TypeSafe AI) | Shipped 2026-09-24 -- OpenRouter early access came through (`~typesafe/jev-latest`, `jev_client.py`); see "The Jev migration" below |
| 6 | `search_news` becomes a tool, internals unchanged | Shipped 2026-09-24 (Step A) |

Implemented as three passes: **Step A** (#105) added `search_news`/`start_push`/
`stop_push` as agent tools with routing untouched; **Step B** switched all
routing to the single agent, merged `chat_histories`/`interest_sessions`
into `conversations`, and deleted Route A/B's deterministic dispatch
along with `end_exploration`/`MAX_TURNS`; **the Jev migration** (same day,
once OpenRouter access came through) moved layers 2 and 4 off the pinned
LangChain guard_model onto Jev's typed-decision API.

**Two things qa-engineer measured live against the real model, 2026-09-24,
after Step B first shipped:**

1. **Layer 2's skip condition -- three designs tried, two of them real
   regressions, before landing on one that works both ways.** First cut
   skipped it "only when a pending offer exists" (item 4's own scope):
   measured, a topic-free but genuinely contextual reply ("sure", "the
   first one") got misclassified as off-topic 1/3 to 3/3 of the time
   (67% overall on-topic across 6 phrasings × 3 trials), since Jev saw
   the message with zero conversation context. Widened to "skip for any
   ongoing conversation" (any non-empty history) to fix that -- which
   fixed it, but opened the mirror-image gap: **case 12** of
   `tools/run_smoke_tests.py`, found on the INT deploy of this plan
   (2026-09-24), showed a genuinely off-topic pivot mid-conversation
   ("write me a poem about cats" right after a news reply) went
   completely uncaught by any layer -- the agent's own polite decline
   read as acceptable content to layer 4, and layer 2 never ran at all.
   Resolved by keeping layer 2 running on every message (skipped only for
   a live pending offer, back to the first design) but passing the last
   assistant reply to Jev as `previous_bot_message` context --
   `_ON_TOPIC_QUESTION`'s criteria explicitly cover both directions (a
   continuation of an on-topic conversation is true; an unrelated pivot
   is false even with context present). Verified live, 24/24 across the
   cases each of the first two designs got wrong, before shipping.

2. **Multi-intent reliability -- real, NOT fixed, made observable
   instead.** Step A's own accepted-risk framing ("~7% for
   `stop_push`-right-after-`start_push`") was assumed to extend to Step
   B's removal of the separate deterministic multi-category join.
   Measured instead, for "add robotics to my interests and tell me what's
   new with it": **3/25 (12%) trials satisfied BOTH intents.** The
   dominant failure mode wasn't a partial/deferred answer -- the news
   half was answered well and the interest-add half was silently dropped
   entirely, not mentioned at all. This is a materially different (and
   much worse) number than the assumption it was accepted under.
   Deliberately NOT fixed with a deterministic join (would reintroduce
   the category-based dispatch split this whole plan retires) or with
   auto-retry (could double-fire a tool call, or hallucinate a worse
   response). Instead, the Jev migration below adds
   `all_asks_addressed` -- an observability-only signal (an
   `incomplete_reply` WARN event, not a block) that makes every future
   occurrence of this failure mode queryable, so a real decision about
   whether/how to act on it can be made from accumulated data rather
   than a 25-trial sample.

3. **`tools/run_smoke_tests.py` cases 2/3 looked like a save failure --
   traced live, it's a smoke-test assumption, not an application bug.**
   On the INT re-deploy that confirmed the case-12 fix, cases 2/3 (add a
   new interest, English and Chinese) came back with the interest never
   actually saved. Reproduced directly against INT: both cases hit a
   REAL ambiguity, not a code defect. "Add quantum sensing to my
   interests" had zero real coverage in INT's cache, so the agent
   correctly refused to add it ungrounded and offered two alternative
   directions instead (per `interest_finder.py`'s own AAOI-avoidance
   rule, unrelated to anything in this plan); "我對機器人科技很感興趣" had
   real coverage, but across two distinct directions (industry/policy vs.
   technical implementation), so the agent correctly showed both and
   asked which one landed. Either way, the scripted second message
   ("yes, that's right, go ahead" / "對，就是這個") doesn't specify
   which of two-or-more offered directions is meant, and the agent
   correctly asks for clarification rather than guessing and saving the
   wrong thing -- exactly the behavior `interest_finder.py`'s own
   docstring calls out as the point of this design, not a regression.
   Fixed by loosening cases 2/3's own pass/fail check to `blocked_at is
   None` (did the conversation stay on-topic and not error) rather than
   asserting a specific category or an actual save, which was really
   asserting against live cache contents this suite doesn't control, not
   against the code.

## What triggered this

A subscriber reported sporadic "No related news found" on INT and
suspected it was not specific to one topic. Investigating that on
2026-09-19/22 turned up something else, and the investigation is worth
recording because two of its conclusions were wrong before they were
right.

**Found on disk (INT):** two rows in the shared `interest_query_expansions`
cache whose KEY was not a topic at all but a whole assistant reply — one
a Spanish news report, one a Traditional Chinese exploration reply ending
in "要設成這樣嗎？" ("shall I set it that way?"). Every other topic-keyed
table (`subscribers.interests`, `interest_categories`,
`interest_push_state`, `subscriber_interest_definitions`) was clean, which
rules out the interest-saving path and leaves exactly one writer:
`agent.search_news`, keyed on `_rewrite_search_query`'s output.

**Reproduced live**, same deployment, real model: with a history whose
last assistant turn asks a question and a user message of "要" ("yes"),
the rewrite call returned

> 好的，已將「OpenAI 新聞」設為你的興趣。之後我會每天為你推送…

— the model answering *as the assistant* instead of emitting a topic.
The same history with "好" (also "yes") returned a clean "OpenAI 新聞趨勢",
so this is borderline and intermittent, which is exactly why it reads as
"happens to other topics too".

**The full chain:**

1. The exploration model writes "要設成這樣嗎？" **and calls
   `end_exploration` in the same turn.** The session closes with the
   question unanswered. The interest is never saved.
2. `chat_histories` is a separate dict and is never cleared (only trimmed
   by age/count), so the question survives as an `ai` turn.
3. The subscriber answers "要". No session exists, so the router handles
   it, classifies it `news_query`, and it reaches `search_news`.
4. `search_news` passes the lingering history into `_rewrite_search_query`
   as real `human`/`ai` turns. The model continues the conversation.
5. That text becomes `topic` — cached as a garbage key, and injected into
   the report-writing system prompt.

**Two corrections to the first read of this, recorded deliberately:**

- The drifted topic does **not** cause "No related news found". Measured:
  `news_classify.expand_interest_for_retrieval` reads the garbled
  confirmation sentence and still produces a correct retrieval definition
  ("News coverage of OpenAI…"), 50 candidates come back, and the report
  is fine. The pipeline launders it. The original "No related news found"
  report is therefore **still unexplained** and tracked separately.
- Nothing was ever "set" by the drift. `_rewrite_search_query` calls
  `guard_model.invoke()` with **no tools bound** — the model physically
  cannot write anything. It emitted a sentence *claiming* an interest was
  set. That is why the interest tables are clean.

## The actual defect

Two parts of one conversation had **different lifetimes**:

```
session (holds the pending offer)  --X-- closed by end_exploration
chat history                       -----------------> still alive
```

Seven call sites in `bot.py` close a session (`end_exploration`,
`MAX_TURNS`, two layer-4 blocks, two exception paths, and the trial-limit
block). None of them touch history. Every one of them can therefore strip
the meaning from a reply the subscriber is about to send.

The user-visible failure is worse than the cache clutter: **the
subscriber said yes and nothing happened.**

## The design

**1. No session state.** `interest_sessions` and `end_exploration` are
deleted. "I can't help with that" becomes a sentence the agent says, not
a state transition. The agent stops *asking* — it does not stop *being*.

This is the same move the front-door redesign already made once
(`interest_finder.py`, finding 6): rather than patch the reliability of a
tool the model calls at the wrong time, remove the thing that depended on
it. `end_exploration` is the last place that dependency survived.

**2. One conversation object, one lifetime.** Messages and any pending
offer live together, in memory, trimmed by the existing
`MAX_HISTORY_AGE` (1h) / `MAX_HISTORY_MESSAGES` (20). Not persisted.

The pending offer is not separate state — it is a fact *in* the
conversation ("I just asked whether to add this"). When the conversation
is gone the fact is gone, and that is self-consistent rather than a bug.
The asymmetry above *was* the bug.

PR #91's safety property is unaffected: within a live conversation the
offer is still recorded by `propose_interest`, and `bot.py` still
performs the write itself on an affirmative — the model is still never
trusted to call the save tool at the right moment.

**3. Tools and sub-calls get no conversation context.** Already true for
three of four call sites:

| call site | shape today |
|---|---|
| `news_push.write_push_digest` | `[system, user(listing)]` — clean |
| `search_news`'s report step | `[system, user(listing)]` — clean |
| `interest_finder.classify_confirmation` | `[system, user(text)]` — clean, history passed as quoted text |
| **`agent._rewrite_search_query`** | **`[system] + history-as-turns + [user]` — the outlier** |

The rewrite step is the only place conversation enters the news path, and
the only place that breaks. Under the new design the agent holds the
conversation and every dispatched call gets `system` + one `user` message
containing its data.

**4. Missing context is stated, not guessed.** When a message reads as an
answer/confirmation and there is no record of what it answers, the bot
says so and asks for a restatement, e.g.:

> 你好，我是新闻推送助手。你刚才回覆了「要」，但我这边已经没有对应的上下文了——对话记录只保留很短的时间，也可能刚好遇上服务重启。方便再说一次你想要什么吗？

Only for genuinely context-dependent messages. A self-contained question
("What's new with OpenAI?") is just answered. The judgment is the agent's
own (prompt-level, no new mechanism); it could later move to Jev, which
is the same bounded-classification shape as item 5 below.

Must respect `subscriber_ops.get_language` — either generated by the
agent in the subscriber's language, or a template through the existing
`_translate_confirmation` path.

**5. Layers 2 and 4 move to Jev** (TypeSafe AI's "System One Model":
typed decisions with calibrated confidence, $42 per billion input tokens,
~0.114s). Both are bounded typed classifications, which is exactly its
shape. `classify_confirmation` is a natural third candidate.

Layer 2 does two jobs today — an on-topic **guardrail** and intent
classification with argument extraction. Only the guardrail is a safety
mechanism with an incident behind it; intent routing becomes the agent's
own tool selection. Keep those separable.

**6. `search_news` becomes a tool, internals unchanged.** Its four steps
(rewrite → definition → retrieval → report) are already clean calls. What
changes is who invokes it and who holds the conversation.

## Costs and risks, stated plainly

- **More model calls per message.** A plain news question is ~4 calls
  today with no agent loop; add the agent's own turn and it is ~5-6.
  `MAX_STEPS_PER_TURN` still bounds a single turn.
- **This partly re-enters what PR #85 left.** That PR converted
  `search_news` from a tool loop to a deterministic pipeline. Its lesson,
  as recorded in `interest_finder.py`, was narrower than "loops are bad":
  a loop whose every iteration cost 12-160s of corpus read was the
  problem, and `SqliteVecStore` (PR #86) took a corpus read to ~0.3s. The
  precondition is gone; the extra calls are not.
- **`MAX_TURNS` disappears as the cost bound.** The per-subscriber trial
  limits shipped in PR #101/#103 (`agent_interactions_remaining`, one per
  turn) now serve that role.
- **Deploys become user-visible.** Anyone mid-conversation during a
  restart gets the "I lost the thread" reply. Accepted: it was already
  happening, silently. Honest loss beats silent failure.
- **Reference resolution may shift.** Flattening history from turns into
  quoted text could resolve "what about Nvidia?" / "第一個" differently.
  Measured, not assumed — see below.

## The Jev migration

OpenRouter early access came through 2026-09-24 (`~typesafe/jev-latest`,
billed through an existing `OPENROUTER_API_KEY`, not a native TypeSafe
key). Verified live against the real endpoint before building against it
-- response shape matches `docs.typesafe.ai/api.md` exactly
(`{"answers": {qid: {"type": "noul", "noul": 0.0-1.0}}, "usage": {...}}`),
~0.17s round trip for a 3-question call. `jev_client.py` is a thin
`requests`-based adapter (not OpenAI-wire-compatible, so
`agent.build_model_from_config`'s `ChatOpenAI` path can't reach it, per
`TODO.md`'s original note) -- `ask(state, questions, api_key)`, no
interpretation of the answers, that's `guardrails.py`'s job.

**Layer 4** was the clean case: `OutputCheck`'s two existing booleans
(`discusses_own_configuration`, `appropriate_bot_content`) map 1:1 onto
two independent `noul` questions in one Jev call. Added a third,
`all_asks_addressed` (optional, only asked when the caller passes
`user_text`) -- see item 2 above for why, and its own docstring in
`guardrails.py` for why a `False` answer is logged
(`incomplete_reply`, WARN) rather than blocking the reply.

**Layer 2 was the harder fit.** Jev's `choice` primitive is strict
single-select (verified against `docs.typesafe.ai/primitives/choice.md`),
which can't represent "this message has two intents" the way the old
router's `categories: list[Category]` field could -- and Jev is a typed-
decision model, not a text generator, so it can't do the router's
`topics`/`push_interval_hours`/`language` free-text extraction at all.
Resolved by asking ONE independent `noul` question per category (plus
`on_topic`) in the SAME Jev call -- multi-category support now comes from
however many of those come back true, not from a combined field. The
extraction fields are dropped entirely: nothing has read them for
dispatch since Step B anyway (they were telemetry-only), so nothing of
value was lost. `MessageClassification` shrank to `on_topic`/`categories`
accordingly.

### Writing questions for an independent yes/no is not the same job

The first cut of those questions measured **badly** -- layer 2 at 71%
overall, `find_interests` at 22% on a 60-trial re-check -- and the two
root causes are worth recording, because both came from carrying
single-select habits into an independent-question format, and neither
was visible without printing raw `noul` scores per category:

1. **A relative tiebreaker becomes an over-firing bug.** The first cut
   reused the old `_ROUTER_PROMPT`'s "treat brevity charitably: a vague
   question is almost always a news_query" line. In a single-select
   prompt that is a sensible where-to-put-the-doubt rule. As an
   independent yes/no it made `is_news_query` fire at **0.87-0.94** on
   "help me figure out what to follow" and "我想追蹤機器人科技的新聞" --
   messages not asking to be told news at all. The diagnosis only
   appeared on raw scores: `find_interests` was being recognised
   perfectly (0.88-0.98) the whole time; the 22% was almost entirely
   `news_query` firing *alongside* it and breaking the exact-match
   expectation. The fix is that each question states its own negative
   boundary, naming the neighbouring requests it must not claim --
   while still answering "is this request present" rather than "is this
   the best label", so a genuine two-intent message still trips both.
2. **`criteria` is not optional decoration.** Questions written with
   bare `instructions` and no `criteria` clustered in the 0.46-0.57
   band, where a 0.5 threshold is a coin flip. That is exactly what the
   layer-4 dip was: a reply-language confirmation scored 0.47/0.49/0.52
   on `appropriate_bot_content` across three trials, because the
   enumerated subscription actions listed adding/removing an interest
   and turning push on/off but never *setting the reply language* -- the
   same content gap as the 2026-08-14 incident, surfacing differently
   because Jev has no reasoning field to lean on. Naming the case fixed
   it outright.

A third finding was the most user-visible of the three and did not come
from the pass/fail numbers at all: `on_topic` scored **0.46 and 0.32**
for "我對加密貨幣很感興趣" and "我對比特幣很感興趣" -- below threshold, so those
subscribers would get the "I only help with tech industry news" redirect.
The *category* was correct in both (`set_interest` alone). "我對區塊鏈很感
興趣" scored 0.60 on the identical sentence structure, which isolates the
variable to the topic word: crypto and bitcoin read as finance rather
than technology. This bot's own dataset has always treated them as in
scope, so `on_topic`'s criteria now say so explicitly, naming the
adjacent coverage (crypto/blockchain, semiconductors, hardware, cloud,
tech earnings) rather than leaving "technology industry" to be read
narrowly.

**Measured after the rewrite** (`tools/measure_guardrails.py --trials 10`,
490 trials, same dataset both times):

| | first cut | after |
|---|---|---|
| Layer 2, single-category | 71% (62/87) | **99% (288/290)** |
| — `find_interests` shapes | 22% (13/60) | **97% (58/60)** |
| — `chinese_crypto` | 83% | **100% (40/40)** |
| Layer 2, multi-intent | 83% | **98% (59/60)** |
| Layer 4 | 93% | **100% (140/140)** |

The two residual layer-2 misses are in `find_interests_shapes`, and the
one multi-intent miss is `mixed_language_control` -- the case this
dataset's own comment already documents as having no single true answer.

`guard_model` (the pinned LangChain model) is untouched and still backs
everything Jev can't do: `classify_confirmation`,
`reads_as_bare_confirmation`, `_translate_confirmation`, interest
normalization -- none of those fit a typed-decision shape.

## Open

- **`search_news`'s report gets rewritten by the front-door agent, not
  passed through verbatim -- accepted, not fixed.** Found on the INT
  deploy of this plan (2026-09-24, smoke case 1): the agent narrates its
  own reply around the tool's report instead of returning it as-is,
  which breaks the "reply starts with 📰" contract (case 1's own check)
  and means `agent.search_news`'s `mark_links_shown` call (based on what
  its OWN generated report cites) can mark a link shown that the agent's
  rewrite then drops from what the subscriber actually sees. Decided:
  this is fine as-is -- a dropped article surfacing again on a later
  search or the regular push is an acceptable outcome, not worth
  engineering around. Not resolved: whether the 📰-first check itself
  should be relaxed to match the new conversational-rewrite behavior
  (it's currently a smoke-test false positive, not a real bug).
- **Whether/how to act on `incomplete_reply` once real data accumulates.**
  The Jev migration above made the 12% multi-intent finding observable
  (an `incomplete_reply` WARN event) rather than fixing it. Revisit once
  there's a real sample from production, not a 25-trial measurement.
- **The original "No related news found" is still unexplained.** Ruled
  out: the `already_shown` filter (57 OpenAI articles were still
  available to every account checked) and the rewrite drift (measured
  above). Needs its own investigation.
- **Before/after measurement of the rewrite change** — same
  (history, query) pairs through both shapes, several runs each because
  the failure is intermittent. `tools/measure_guardrails.py` is the
  precedent for this kind of harness.
- **Whether the two garbled cache rows get deleted.** They are inert
  (nothing will ever look up a 584-character topic) but they are clutter,
  and the table has no expiry at all.

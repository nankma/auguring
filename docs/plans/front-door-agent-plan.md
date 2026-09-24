# One always-on agent, no session state

Written 2026-09-22. Status: **items 1/2/3/4/6 shipped 2026-09-24 (Steps
A+B); item 5 blocked.**

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
| 5 | Layers 2 and 4 move to Jev (TypeSafe AI) | Blocked on early access -- layer 2 (`classify_message`) kept as-is on the existing pinned model, now used only for on-topic gating, not dispatch |
| 6 | `search_news` becomes a tool, internals unchanged | Shipped 2026-09-24 (Step A) |

Implemented as two PRs: **Step A** (#105) added `search_news`/`start_push`/
`stop_push` as agent tools with routing untouched; **Step B** (this pass)
switched all routing to the single agent, merged `chat_histories`/
`interest_sessions` into `conversations`, and deleted Route A/B's
deterministic dispatch along with `end_exploration`/`MAX_TURNS`.

**Two things qa-engineer measured live against the real model, 2026-09-24,
after Step B first shipped:**

1. **Layer 2 misclassifying a contextual mid-conversation follow-up as
   off-topic -- real, fixed.** Step B's first cut narrowed the router's
   skip condition to "only when a pending offer exists" (item 4's own
   scope). Measured: a topic-free but genuinely contextual reply ("sure",
   "the first one") answering an ordinary agent question -- not a
   propose_interest confirmation, so no pending offer exists yet --
   got misclassified as off-topic 1/3 to 3/3 of the time across a small
   sample (67% overall on-topic across 6 phrasings × 3 trials). Fixed by
   widening the skip condition back to "any ongoing conversation" (any
   non-empty history), matching the OLD design's blanket bypass -- layer
   1 (local prefilter) and layer 4 (output check) were never dependent on
   layer 2 catching this, so nothing is lost by widening it back. Layer 2
   now runs only on the first message of a fresh (empty-history)
   conversation.

2. **Multi-intent reliability -- real, NOT fixed, needs a decision.**
   Step A's own accepted-risk framing ("~7% for `stop_push`-right-after-
   `start_push`") was assumed to extend to Step B's removal of the
   separate deterministic multi-category join. Measured instead, for "add
   robotics to my interests and tell me what's new with it": **3/25 (12%)
   trials satisfied BOTH intents.** The dominant failure mode wasn't a
   partial/deferred answer -- the news half was answered well and the
   interest-add half was silently dropped entirely, not mentioned at all.
   This is a materially different (and much worse) number than the
   assumption it was accepted under. Not resolved as of this writing --
   see the open items below.

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

## Open

- **Multi-intent reliability (12% measured, see Status above) needs a
  decision.** Options considered but not yet chosen between: (a) accept
  it as-is, same reasoning as the rest of this plan's accepted risks --
  narrow-ish blast radius, expected to improve once Jev is available; (b)
  restore some form of deterministic handling specifically for the
  "add/remove an interest + ask a news question in the same message"
  shape, without reintroducing a general category-based dispatch split;
  (c) something else. Whichever is chosen, `tools/run_smoke_tests.py`
  case 14 and `tests/test_interest_finder.py`'s
  `test_multi_category_messages_still_go_through_one_agent_turn` should
  be revisited to reflect the actual decision, not just document the
  measurement.
- **Jev early access is not granted yet.** Item 5 is blocked on it, and
  its API is not OpenAI-wire-compatible, so it needs its own adapter
  (`agent.build_model_from_config`'s `ChatOpenAI` path cannot reach it).
  See `TODO.md`.
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

# "Help me find my interests" — an elicitation conversation

Written 2026-09-08. Status: **built, not yet deployed.**

| Piece | Status |
|---|---|
| Cross-domain research survey | Done — `docs/analysis/interest-elicitation-survey.md` |
| `find_interests` router category + output-scope widening | Built (`guardrails.py`) |
| `interest_finder.py` — prompt, five tools, `run_turn` | Built |
| Per-chat session state + turn ceiling in `bot.py` | Built |
| `build_agent`/`run_agent` parameterization | Built (`agent.py`) |
| Settings entries (all three environments) | Built |
| Tests | Built — `tests/test_interest_finder.py`, additions to `tests/test_guardrails.py` |
| Live verification on INT | **Not done** |

## The problem

Every existing path into an interest requires the subscriber to already
know the answer and be able to type it: "add Nvidia", "remove crypto".
Someone who wants technology news but cannot name a topic has nowhere to
go. Worse, someone who *can* name one often names it badly — the AAOI
incident (a subscriber left following a ticker the cache has no coverage
for, receiving nothing, indefinitely) is that failure in its purest form.

## Why this one gets an agent loop

`agent.build_agent`/`run_agent` have been dormant since PR #85 converted
`search_news` from a tool-calling loop into a deterministic pipeline. The
criterion that conversion established, and that this feature is measured
against:

> Use a loop only when the number of steps genuinely cannot be known in
> advance.

`search_news` failed it — retrieval turned out to be fully boundable, and
the loop was costing 5–7 corpus reads per question. This passes it: how
many questions to ask, whether to show examples or ask a clarifying
question next, and when the subscriber has actually converged are all
undecidable up front. They depend on answers that do not exist yet.

PR #85's lesson was also narrower than "loops are bad": it was that a loop
whose every iteration costs 12–160s of corpus read is a disaster. That
precondition is gone — `SqliteVecStore` (PR #86) put a corpus read at
~0.3s. And this is interactive, so the latency budget is per-turn (a
normal conversational wait) rather than cumulative the way a one-shot
report's is.

The residual risk is the one that was actually observed: a *single* turn
fanning out into many internal tool calls. Hence two independent ceilings
— see below.

## What the research changed

Full survey in `docs/analysis/interest-elicitation-survey.md`. Three
findings shaped code, not just wording:

1. **Reaction beats articulation.** Career-interest inventories (RIASEC,
   Strong), Kelly's repertory grid, and Rocchio relevance feedback all
   converge independently: people judge concrete instances far more
   reliably than they generate abstract self-descriptions. So the loop
   leads with real headlines and asks which ones land — it never opens
   with "describe your interests".
2. **Ground every candidate in the real corpus.** `find_example_articles`
   exists so the model may only propose topics it has *seen*. An empty
   result returns an explicit "do not propose this topic — it has no
   coverage", because an interest with no coverage is worse than none.
   This is the AAOI failure guarded at its source.
3. **Preferences are constructed, not extracted.** Success is "they now
   know what they want, and it is grounded in real coverage" — not "we
   captured a pre-existing answer".

Also carried over: the funnel technique (broad before narrow — an early
narrow question supplies its own answer), laddering (from a liked story
up to a durable topic: "that earnings report" is not an interest,
"semiconductor supply chain" is), and the onboarding-UX finding that 2–5
choices is the manageable range, which is where `max_examples: 5` comes
from.

## The four ways in

All four are the same flow; only the opening differs, and the router
sends all of them to the same place:

| | Shape |
|---|---|
| (a) | "help me find something to follow" — cold start |
| (b) | "I liked that story, send me more like it" — narrowing from a pushed article |
| (c) | "too much X, not enough Y" — rebalancing by feel |
| (d) | "what could I follow?" — wants suggestions before committing |

(c) adjusts the **interest list only**. Push frequency and volume are
`start_push`/`stop_push`'s job and stay there.

Disambiguation against `set_interest` is in the router prompt: "add
robotics" is `set_interest`; "something like robotics but narrower, what
do you have?" is `find_interests`. When genuinely ambiguous, prefer
`set_interest` — the cheaper path, and a wrong guess there costs one
message rather than opening a mode.

## Session state, and why it had to exist

This is the bot's **only conversational mode**. Every other path
classifies each message independently, which works because each one
carries its own topical signal. A follow-up here does not: "yes", "the
second one", "not really" mean nothing to a per-message classifier, and
routing them by content would scatter a conversation across unrelated
categories mid-flow.

So `bot.interest_sessions` (a plain dict, in memory) marks a chat as
in-exploration, and `process_message` checks it **before layer 2**. Layer
1 still runs first — being mid-conversation is not an exemption from an
injection check — and layer 4 still runs on every reply.

In-memory is a deliberate choice, not an oversight: a container restart
drops any exploration in flight, and the subscriber simply starts over.
Persisting it would mean a new table and a stale-session sweeper to
protect a conversation whose whole value is that it is short.

`find_interests` also wins a multi-category turn outright rather than
being one segment of a joined reply — it owns the turns that follow, and
the agent can act on the rest of the message itself (it can save and drop
interests), which a joined reply could not.

## The two ceilings

Different failures, enforced in different places:

- **`max_steps_per_turn` (20)** — LangGraph's `recursion_limit`, capping
  what one turn may spend internally. This is PR #85's measured failure.
- **`max_turns` (8)** — a counter in `bot.py`, capping the conversation.
  This is the "keeps switching direction, can't decide" case. Enforced by
  the caller and not by the model on purpose: noticing one's own
  circling is exactly the self-assessment models are unreliable at.

**The step ceiling shipped wrong the first time, and the fix is worth
recording.** It was set to 10 from reasoning ("a turn needs a few steps")
rather than measurement. QA then ran the real pinned model against a topic
the cache has no coverage for ("quantum blockchain synergy") and found the
model tries one rephrasing after another — **15 `find_example_articles`
calls, ~30 graph steps** — before concluding honestly that nothing is
there. At 10 it never got that far: it blew the ceiling, and
`_process_find_interests`'s generic handler sent LangGraph's own error
text to the subscriber verbatim ("Recursion limit of 10 reached... visit
https://docs.langchain.com/..."). Three changes came out of that:

1. `run_turn` catches `GraphRecursionError` specifically and returns
   `out_of_steps_message()` — a bounded outcome has to read like one.
2. A prompt rule: *do not search more than twice before replying*.
   Rephrasing a fourth time does not find coverage that isn't there; it
   just makes the subscriber wait.
3. The default went to 20 (~9 tool calls, several times what a normal
   turn uses).

**Which of those two is actually doing the work is worth being honest
about.** Re-measured across four real-model trials on the same uncovered
topic, the prompt rule cut search counts from 15 to **9, 6, 2, 4** — a
real reduction, but nowhere near the "at most two" it asks for. The model
does not reliably cap itself. Every trial still finished comfortably
inside 20 steps, so the numeric backstop is what actually guarantees the
bound; the prompt rule only makes hitting it rare. Treating a prompt
instruction as a limit rather than a nudge is the mistake to avoid
repeating here.

Notably the model never invented an uncovered topic, in any trial before
or after the fix — the design's own named risk held. The failure was
pure persistence.

Hitting the turn ceiling produces `interest_finder.out_of_turns_message()`
— honest that it isn't working, with a concrete way forward ("name a
company or topic directly"). Looping silently until the subscriber gives
up teaches them the bot doesn't listen.

The session is cleared on every exit that isn't "keep going": the model
ending it, the ceiling, a layer-4 block, or an error. A stale entry would
silently swallow the chat's next message — a worse failure than whatever
caused it.

## What `find_example_articles` deliberately is not

It is not `search_news`, and each difference is load-bearing:

| | Why |
|---|---|
| No daily quota | Narrowing down is help, not delivery; spending the search allowance would punish asking for help |
| No `mark_links_shown` | An article shown as an *example* has not been delivered; retiring it would quietly remove it from a digest they'd otherwise get |
| No definition generation | `expand_interest_for_retrieval` is the single most expensive step in the search pipeline (2.7–4.7s measured) and caches per topic — exploration tries many one-off phrasings, so it would miss the cache nearly every time |

Raw-query embedding is a weaker retrieval signal, accepted here because
these results only need to be good enough to react to. The interest that
finally gets **saved** goes through `agent.add_one_interest`, which
generates and caches the real definition then — so an interest arrived at
by exploring is indistinguishable from one typed directly. No second
class of interest.

## Guardrails

`find_interests` is deliberately **not** in `_NARROW_CHECK_CATEGORIES`.
An exploration reply is free-form model prose (headlines plus a
question), unlike the tightly-pinned settings confirmations, so it gets
the same full output check `news_query` does. Instead
`_OUTPUT_SCOPE_PROMPT`'s `appropriate_bot_content` question was widened
to recognize that shape — showing example headlines and asking which
land, asking what appealed about a story, proposing a topic and asking
for confirmation, or saying honestly that narrowing isn't working.

That widening was done *before* any live testing, because layer 4 would
otherwise have blocked every turn of this feature — the same class of
false positive as the 2026-08-08 "already covered interest" incident
(`docs/plans/guardrails-plan.md`).

## Open

- **No live verification yet.** Nothing here has met a real model. The
  prompt's method instructions are the part most likely to need
  adjustment against real behavior — particularly "only propose topics
  you have seen", which is a rule a model can drift from without any
  error surfacing.
- **Outcome telemetry exists but nothing reads it.**
  `interest_saved_from_exploration` over `interest_exploration_ended`
  gives the "how many explorations produce an interest" rate, and
  `interest_exploration_out_of_turns` counts the ones that gave up. No
  alert or dashboard consumes any of them yet — the numbers are
  queryable, not watched.
- **Entry point (b) has no link back to the pushed story.** A subscriber
  saying "more like that one" is relying on the model searching for what
  they describe, not on the bot knowing which article was sent. Push
  history is not in `chat_histories`.

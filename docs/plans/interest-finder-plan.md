# "Help me find my interests" — an elicitation conversation, and the front door it became

Written 2026-09-08, substantially revised 2026-09-10. Status: **built,
not yet deployed.**

| Piece | Status |
|---|---|
| Cross-domain research survey | Done — `docs/analysis/interest-elicitation-survey.md` |
| `find_interests` router category + output-scope widening | Built (`guardrails.py`) |
| `interest_finder.py` — prompt, tools, `run_turn` | Built |
| Per-chat session state + turn ceiling in `bot.py` | Built |
| `build_agent`/`run_agent` parameterization | Built (`agent.py`) |
| Settings entries (all three environments) | Built |
| Tests | Built — `tests/test_interest_finder.py`, `tests/test_agent.py`, `tests/test_bot.py`, `tests/test_guardrails.py` |
| Live verification on INT (2026-09-08 round) | Done — found the false-confirmation incident below on first real use |
| False-confirmation fix (`propose_interest` + deterministic confirmation gate) | Built and verified live against the real pinned model, incl. cross-language (zh-Hant/zh-Hans/es/en) and direct DB persistence checks |
| **Front-door redesign** (`set_interest`/`remove_interest`/`set_language` moved into this agent; every add now grounded) | Built and verified live against the real pinned model 2026-09-10 (entry point (f), a multi-topic add, `set_language` mid-exploration) — see its own section below. Not yet redeployed. |

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

## The ways in

All of these are the same flow; only the opening differs, and the router
sends all of them to the same place. (e) and (f) were added by the
2026-09-10 front-door redesign below; the disambiguation note at the
bottom is now about which one to seed the conversation with, not about
picking a cheaper path — there isn't one any more.

| | Shape |
|---|---|
| (a) | "help me find something to follow" — cold start |
| (b) | "I liked that story, send me more like it" — narrowing from a pushed article |
| (c) | "too much X, not enough Y" — rebalancing by feel |
| (d) | "what could I follow?" — wants suggestions before committing |
| (e) | already follows a topic, dissatisfied with what it sends — the topic's *definition* needs adjusting, not the topic itself |
| (f) | names a specific topic directly ("add robotics to my interests") — still grounded first, just usually in fewer turns |

(c) adjusts the **interest list only**. Push frequency and volume are
`start_push`/`stop_push`'s job and stay there.

The router still distinguishes `set_interest` from `find_interests` for
argument-extraction purposes ("add robotics" names a topic outright,
"something like robotics but narrower, what do you have?" doesn't), but
both dispatch to the same agent now (`agent.INTEREST_AGENT_CATEGORIES`)
— the distinction seeds where the conversation starts, not which path is
cheaper.

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

## A false confirmation, found on the first real deploy

Found 2026-09-08, minutes after the first INT deploy, by the user testing
the feature live in Traditional Chinese. The model walked them all the
way through the flow correctly — showed real headlines, narrowed down,
proposed a specific topic, asked "確認加入嗎？" (confirm adding?) — and
after they confirmed, replied "好，我已為你加入「GPU與AI加速硬體」這個
主題" (done, I've added it), naming it in a "current list" summary.

The subscriber's interests stayed empty. Confirmed three ways: the raw
`subscribers.db` row showed `interests: '[]'`; the container's full log
had **zero** occurrences of `interest_saved_from_exploration` for that
chat despite `interest_finder turn returned` firing 7+ times; the
`message_archive` transcript showed the confirmation question and the
false "done" reply back to back, with no tool-call evidence between them.
The model composed a confident success message without ever calling
`save_interest`.

This is the same shape as the step-ceiling incident above, generalized:
**a prompt instruction is a nudge, not a guarantee — this codebase's own
lesson, twice now.** The fix follows the same principle: move the
highest-stakes decision (persisting data) out of the model's hands
entirely, into a bounded, deterministic backstop.

**The fix**, in `interest_finder.py` and `bot.py`:

1. `propose_interest(topic, action)` — a new tool the model calls in the
   SAME turn it asks for confirmation. It doesn't save anything; it just
   records `session["pending_proposal"]`, a structurally visible fact the
   code can act on (unlike free text, which is exactly what failed).
2. `bot._process_find_interests` checks for a pending proposal BEFORE
   running the agent loop at all. It classifies the subscriber's reply
   with `classify_confirmation` — a small, single-purpose structured-
   output call, the same shape and reliability class as
   `guardrails.classify_message`'s router, not a new open-ended loop.
   - **affirm** → `execute_save`/`execute_drop` run directly in code.
     `run_turn` is never even invoked for this turn — there's no step at
     which the model could fail to follow through, because it isn't
     asked to.
   - **decline** → clears the proposal, falls through to a normal turn.
   - **unclear** → leaves the proposal in place, falls through to a
     normal turn. `_compose_prompt` now also surfaces a pending proposal
     to the model itself, as defense in depth: if the classifier missed
     a genuine yes, the model still gets a chance to notice and call
     `save_interest`/`drop_interest` on its own.
3. `execute_save`/`execute_drop` are the single code path that can make
   "this subscriber follows X" true — both the tools (for the case a
   message is unambiguous enough that the model saves directly, no
   confirmation round trip needed) and the deterministic gate call the
   same functions, so there's exactly one place `interest_saved_from_
   exploration` fires from regardless of which path triggered it.

Every failure mode of the new classifier itself is safe by construction:
a miss (affirm classified as unclear, or the call erroring out) simply
falls through to the pre-fix behavior — the model's own turn — never
worse than before this mechanism existed, only ever an improvement on it.
The one thing that changed forever is that "affirm" no longer needs the
model at all.

## The front door redesign (2026-09-10)

### What triggered it

A real INT conversation stalled: a subscriber asked to find something
"hot in the open source community" (their own example: a project like
"openclaw" that later got acquired). The agent searched, honestly found
no coverage of that shape, and closed with a plain "let's leave it there"
— but `end_exploration` was never called. Investigated by directly
reproducing the same conversation via `agent.run_agent` (not through
`bot.py`, so every raw tool call could be inspected) against **four
different models/providers**: DeepSeek direct on its brand-new
V4.1-Flash (released the same day — `deepseek-v4-flash` is now
transparently routed to it, confirmed via `response_metadata`), the same
model family hosted by Together.ai on an older fixed checkpoint
(`DeepSeek-V4-Flash-0731`), GLM-5.3-Flash, and gpt-oss-120b.

Findings, in order of how they changed the plan:

1. **The tool-skipping problem is worse than one incident.** On
   DeepSeek's newest hosted version, the model didn't just skip
   `end_exploration` — turns 3 and 4 of the SAME reproduction show it
   narrating "I searched again" and "I've now searched three different
   ways" with **zero actual tool calls** in either turn. It composed a
   false claim of work done, not just a missing goodbye.
2. **Retailer matters, independent of model version.** The exact same
   prompt, same tools, same conversation, run against Together.ai's
   older checkpoint of "the same" model: every turn genuinely called
   `find_example_articles`, and the narration matched the real call
   count. DeepSeek's own newest release measured LESS reliable at this
   specific behavior than a fixed checkpoint from a different host.
   gpt-oss-120b was disqualified outright on a separate axis: raw
   internal "harmony" reasoning tokens leaked into user-facing content
   (`analysisWe attempted to find example articles...`), plus a live 500
   from the host. GLM-5.3-Flash called tools honestly on every turn and,
   notably, didn't just fail to find coverage — it pivoted to a genuinely
   covered adjacent topic and proposed that instead.
3. **`end_exploration` asks the model to do the ONE thing this project
   already learned not to trust it with.** `MAX_TURNS`'s own comment
   (above) already says self-assessment of "are we going in circles" is
   unreliable — `end_exploration`'s trigger conditions ("as soon as
   they're satisfied", "if they've changed direction repeatedly") are
   exactly that judgment, contrasted with `propose_interest`'s trigger
   ("the SAME turn you name a specific topic"), which is concrete and
   immediate, not a multi-turn synthesis.

### The decision: stop patching, remove the need to trust it

Patching `end_exploration` with a `classify_confirmation`-style backstop
(what the two ceilings section above did for saves) was the obvious next
move and was explicitly **rejected**. The user's framing: this is a user-
experience problem, not a hole to patch — fix why the model doesn't call
tools reliably at the source, not the symptom of one specific tool call
going missing.

The actual fix removes the stakes instead of the unreliability: **every
interest add now goes through the same grounded show-examples-then-
confirm flow already proven for exploration**, funneled through
`propose_interest`, which bakes in a real cache preview exactly like
`propose_definition` already did. Once no unconfirmed, ungrounded add can
ever reach `subscriber_ops.add_interest`, it stops mattering whether
`end_exploration` gets called — nothing unsafe happens while a session
lingers, and `MAX_TURNS` (already built) is an adequate bound on how long
it's allowed to. `end_exploration` stays in the tool set as a courtesy
(a cleaner exit when the model does remember), explicitly reframed in its
own prompt text as non-critical.

### What changed

- **`set_interest`/`remove_interest`/`set_language` moved out of Route B**
  (`agent.ROUTE_B_CATEGORIES` shrank to `{start_push, stop_push}`) into
  the SAME agent as `find_interests`
  (`agent.INTEREST_AGENT_CATEGORIES`). The router still classifies and
  extracts arguments the same way; only the DISPATCH target changed —
  `bot.process_message` now sends any of these four categories to
  `_process_find_interests` instead of `dispatch_settings`.
- **`propose_interest(topic, definition)` is now the only way to add an
  interest.** It runs a real preview (shared helper with
  `propose_definition`) and refuses to look like a good option when
  nothing relevant surfaces — the AAOI-avoidance rule, now enforced at
  the single choke point every add passes through, including a
  subscriber naming a topic outright ("add robotics" — entry (f) below).
  `propose_remove(topic)` is the (unchanged) removal counterpart, split
  into its own tool now that `propose_interest` no longer takes an
  `action` flag.
- **`agent.add_one_interest` no longer generates a definition blindly.**
  It still normalizes/translates the topic (unchanged), but the
  definition is now a required argument — the one the subscriber already
  saw and confirmed — and it's written to the **subscriber's own tier**
  (`interest_cache_ops.set_subscriber_interest_definition`), never the
  shared/global one. Direction from the user: "interests are not
  shared" — two subscribers adding the same word now get two
  independently confirmed definitions, not one global default whoever
  types it first establishes for everyone else.
- **A sixth entry point.** (f) a subscriber names a topic directly — still
  routes through the same grounding, just usually in fewer turns (one
  search instead of several, since the topic is already known).
- **New `set_language` tool**, direct-effect, no confirmation gate —
  language switching is low-stakes and instantly reversible, unlike
  interests, so it doesn't need the same safety net, and it now works
  mid-exploration instead of requiring the subscriber to escape one
  first.
- **`guardrails._NARROW_CHECK_CATEGORIES` shrank to `{start_push,
  stop_push}`.** set_interest/remove_interest/set_language replies are
  now free-form agent prose (examples, definitions, questions), the same
  shape as `find_interests`/`news_query`, so they need the full layer-4
  check, not the narrow self-disclosure-only one.
- **`dispatch_settings` shrank to push scheduling only** and dropped its
  `model` parameter (nothing left in it makes an LLM call).
- **`tools/run_smoke_tests.py` cases 2/3/8/9/14/17 rewritten** for the new
  shape: adding/removing/already-covered checks became real multi-turn
  conversations (each on its own chat_id, to avoid stacking turns toward
  `MAX_TURNS` or mixing unrelated topics into one session); the
  multi-category join case (14) moved to `start_push` + `news_query`
  since any `INTEREST_AGENT_CATEGORIES` member now wins a multi-category
  turn outright instead of joining.

### The cost, stated plainly

Adding an interest by name ("add robotics") used to be one message. It is
now always at least two: propose (with a real preview), then confirm.
This is the direction's whole point, not a side effect — but it is a real
added round trip on what used to be the cheapest path in the bot, and is
worth remembering as the tradeoff being spent.

## Open

- **A real-model qa-engineer pass (2026-09-10) found one more instance of
  finding 6's own failure class, closed the same day.** Reproducing entry
  point (f) and a multi-topic add against the real pinned model surfaced
  this: the model can call `propose_definition`, read its own preview,
  correctly decide out loud that the change would be a no-op, and tell
  the subscriber it won't save it — all in prose, with nothing to clear
  the `pending_proposal` it had already recorded. A later, unrelated
  affirmative reply (a "yes" answering a different question entirely)
  would then still bind to that disowned proposal, because
  `classify_confirmation` only ever read the subscriber's raw reply in
  isolation. Fixed by anchoring the classification to the assistant's
  actual last message (`history[-1]`, threaded through from `bot.py`)
  instead of the bare pending-proposal dict — the classifier can now see
  when the "confirmation question" it's judging a reply against isn't
  live any more, and returns `decline` instead of letting a stray
  affirmative through. See `classify_confirmation`'s own docstring.
  Aside from this, entry point (f) (grounds before saving even when the
  topic is named outright), a multi-topic add (non-deterministic in
  shape, but correctly AAOI-avoidant), and `set_language` mid-exploration
  all matched the design exactly against the real model, with saves
  verified by direct DB read rather than trusting the reply text.
- **What model should actually drive this agent is now a genuinely open
  question, not a settled default.** The cross-model investigation found
  DeepSeek's own newest hosted release less reliable at honest tool-
  calling than an older checkpoint of the same model hosted by a
  different retailer, and found GLM-5.3-Flash reliable AND better at
  recovering from a dead end (proposing a real adjacent topic instead of
  just giving up) in the one reproduction run this session. That is one
  conversation, not a benchmark — `agent.models.main` has not been
  changed, and shouldn't be without comparing report-writing quality and
  cost too, not just this one tool-honesty behavior.
- **Adding an interest now costs a round trip it didn't before**, by
  direction, not by accident (see "The cost, stated plainly" above) — but
  it hasn't been measured whether real subscribers tolerate that, versus
  it becoming a fall-off point real users abandon at that this project's
  telemetry doesn't yet distinguish from a normal decline/unclear turn.
- The prompt's method instructions are otherwise the part most likely to
  need adjustment against real behavior — particularly "only propose
  topics you have seen", which is a rule a model can drift from without
  any error surfacing.
- **A leaked DeepSeek internal special token was observed once in ~15
  real-model turns during the confirmation-gate fix's verification**
  (`&lt;/｜｜DSML｜｜parameter&gt;` inside an otherwise normal reply,
  breaking one HTML tag). Not reproduced elsewhere, and not a delivery
  risk today — `bot.handle_message`'s existing `BadRequest` fallback to
  plain text already covers a malformed-HTML reply — but worth watching
  for recurrence; if it becomes frequent it would need its own fix rather
  than relying on the fallback.
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

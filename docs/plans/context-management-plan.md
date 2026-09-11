# Context Management Plan

> **2026-09-10 update:** Route B's membership changed after this doc was
> written. `set_interest`/`remove_interest`/`set_language` moved OUT of
> Route B into the interest_finder conversational agent (the same one
> `find_interests` already used) -- an interest can no longer be added in
> one deterministic dispatch; it always shows a grounded definition and
> real examples first, then waits for confirmation. Only `start_push`/
> `stop_push` remain genuinely Route B. The reasoning and measurements
> below are historical record of why Route B was built THIS way
> originally -- still accurate as history, just no longer describing the
> current category membership. See `docs/plans/interest-finder-plan.md`
> for the redesign and why it happened.

Goal: replace the current monolithic `SYSTEM_PROMPT` (one long string, sent
in full on every single LLM call regardless of relevance) with a layered,
conditionally-assembled prompt — tighter core identity, situational
content only included when actually needed, per-user memory injected
consistently instead of the current ad hoc string-prepending. Nothing here
is built yet — this doc captures the design and the research behind it
before implementation starts, same pattern as the other `docs/*-plan.md`
files.

## The problem with today's `SYSTEM_PROMPT`

One string, defined once in `agent.py`, passed to `create_agent(...,
system_prompt=SYSTEM_PROMPT)` at agent-construction time, resent in full
on *every* `ChatDeepSeek` call the agent makes — confirmed directly from a
real Phoenix trace this session: a single user turn produced two separate
`ChatDeepSeek` LLM spans (one before the `search_news` tool call, one
after), each carrying the complete system prompt again. It currently
mixes three things that don't belong at the same level:

1. Core identity ("you are a technology industry analyst," scope
   confinement, anti-role-play rules) — this should basically never change
   and should always be present.
2. Telegram HTML formatting rules (`<b>`, `<a href>`, escaping, structure
   template) — only relevant *because* the output happens to go to
   Telegram. Irrelevant token weight on every call regardless of what the
   turn actually needs.
3. Tool-usage notes (when to use `save_note`) and the interests-injection
   instruction — situational, not identity.

Per-user data (interests, set via `/interests`) is currently injected by
string-prepending a `[User's stated interests: ...]` note onto the raw
user message in `bot.py` — works, but is an ad hoc mechanism distinct from
how everything else reaches the model, and doesn't generalize cleanly to
more per-user preferences (e.g. "prefers deep analysis over brief
summaries," format customization) without more of the same string-hacking.

## The proposed four layers

Priority order, highest to lowest — higher layers should win in conflicts
with lower ones:

1. **System prompt** — tight. Identity + non-negotiable behavioral rules
   only (scope confinement, anti-self-disclosure, anti-role-play). Always
   present, on every call.
2. **Subsystem/workflow/state prompt** — situational. Telegram formatting
   rules, tool-usage notes, "the user seems to be trying to set a
   preference in natural language," translation instructions if/when that
   feature exists. **Only included when relevant to the current turn** —
   not sent at all otherwise.
3. **Long-term user memory prompt** — per-user persistent data (interests,
   depth-of-analysis preference, custom formatting, once those exist).
   Injected consistently for every call from that user, not per-feature
   string hacks.
4. **Current user message** — the actual turn's input. Lowest priority —
   nothing in the user's own message should be able to override rules
   from layers 1-3.

## What the research says

- **This is a real, named pattern, not a novel idea.** Two related
  framings converge on the same shape: OpenAI's Model Spec defines an
  explicit conflict-resolution order (Root > System > Developer > User >
  Tool/Data, higher wins) — directly matches this doc's 1→4 priority
  ordering. The broader industry term for *what content goes in the
  prompt, from where, assembled how* is **"context engineering"** (the
  2025 successor term to "prompt engineering") — Anthropic's own
  engineering blog frames it as "curating and maintaining the optimal set
  of tokens during inference," and LangChain's formalization breaks it
  into **write / select / compress / isolate** strategies. This doc's
  layer 1 is "write" (durable, authored); layers 2-3 are "select" (pull in
  only what's relevant).
- **A concrete existence proof, not just theory**: Claude Code's own
  system prompt is reportedly not one monolithic document but "110+
  separate instruction strings conditionally assembled based on context"
  (per third-party analysis) — i.e., a flagship production agent already
  works the way this doc proposes, not a hypothetical.
- **Where per-user long-term memory should live in the prompt — system-
  adjacent, not folded into the user's message.** Letta/MemGPT's
  three-tier memory model puts "core memory" (key facts/preferences about
  the user) as content that's "embedded inside system instructions and
  always remains in-context," distinct from recall/archival memory
  fetched only on demand. This matches layer 3's design (a distinct,
  consistently-injected segment ranked above the live user message) — and
  means the current string-prepending-onto-the-user-message approach is
  the wrong location, not just an inelegant mechanism. Flagged as
  **convention, not a formal spec** — no equivalent of the Model Spec
  mandates this placement, but it's what practitioners converged on.
- **Token budget: retrieve/include conditionally, don't always-send
  everything.** One cited benchmark: full-context approaches ran
  23,000-26,000 tokens/query vs. 600-1,500 tokens with retrieval-based
  memory inclusion — over 90% reduction with minimal quality loss. This is
  the direct justification for layer 2 being conditional rather than
  always-on like today's monolithic prompt.
- **Caveat worth taking seriously**: reliably enforcing that lower layers
  *can't* override higher ones is still an active adversarial-robustness
  research problem (prompt injection via tool output/RAG content), not
  something the layering alone guarantees. This is exactly why
  `docs/plans/guardrails-plan.md`'s four-layer input/output classification stays
  in place as a separate, independent defense — this doc's layering is
  about *organizing* the prompt well, not a replacement for the guardrails
  that actually *police* what gets through.

## Do we need LangGraph's state machine (hand-built graph)?

**No.** `create_agent` is already built on LangGraph under the hood
(confirmed directly — Phoenix traces show a top-level `LangGraph` span
wrapping every agent call) — we're already using it, just through a
convenience wrapper that hides the graph structure.

LangChain's **middleware** system is the documented, purpose-built
mechanism for exactly this: a `wrap_model_call`-style hook can inspect the
current request/state and rewrite the system message **before each model
call**, and middleware passes straight into the existing call shape —
`create_agent(model=..., middleware=[...], tools=[...])`. No graph
rewrite, no dropping `create_agent`'s loop management (which is the whole
reason this project migrated to LangChain in the first place, per
`CLAUDE.md`).

Hand-building a raw graph would only be justified if different turns
needed genuinely different *execution paths* — different tool sets,
different node sequencing, branching logic — not just different *prompt
text* wrapped around the same linear tool-calling loop. Since this bot's
shape stays "one agent, one tool-calling loop, varying instructions,"
middleware is the right-sized tool. This is a firm recommendation, not an
"it depends" — the docs and API are purpose-built for this exact case.

**Long-term memory storage**: LangGraph does have a dedicated `BaseStore`/
`InMemoryStore` API distinct from thread-scoped state, designed for
exactly this "persistent per-user facts, shared across sessions" need —
worth knowing it exists as the "official" LangGraph-native answer. Not
recommending adoption now: `users_db.py` already does this job (working,
tested, deployed, and already holds `interests`) — introducing
LangGraph's `Store` abstraction on top would be a second persistence
mechanism to keep in sync for no functional gain at this project's scale.
Layer 3's middleware can just call `users_db.get_interests()` directly.

## Verified: the middleware API this design uses

Confirmed directly against the installed `langchain==1.3.14`, not assumed:

```python
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest

@dynamic_prompt
def my_prompt(request: ModelRequest) -> str:
    # request.state (messages, etc.) and request.runtime.context are both
    # available here -- runtime.context is how per-call data like chat_id
    # gets threaded in (via create_agent's context_schema / passed at
    # invoke time), not a global.
    return "the full system prompt string for this call"

agent = create_agent(model, tools=TOOLS, middleware=[my_prompt])
```

`dynamic_prompt` replaces the whole system message per call (not
"append") — so layers 1-3 get composed into one string inside a single
`dynamic_prompt` function, computed fresh each call, rather than three
separate middleware functions each contributing a fragment. `AgentMiddleware`
also exposes `wrap_model_call`/`before_model`/`after_model` hooks directly
for anything `dynamic_prompt` doesn't cover (it's a thin convenience
wrapper around `wrap_model_call` specifically for this use case).

## The router design: one classifier call feeds both the guardrail and layer 2

**Built and live.** This section was written as a proposal; everything
described below has since shipped as `guardrails.classify_message` /
`MessageClassification` (`category: Literal["news_query", "set_interest",
"remove_interest", "start_push", "stop_push", "set_language",
"off_topic"]` — `set_language` was added after this section was
originally written, for `docs/plans/bot-features-plan.md` item 2). Left in the
original future-tense wording below since the design rationale is still
the accurate explanation of *why* it's built this way, not because it's
still pending — see `docs/plans/guardrails-plan.md`'s own Status table for the
authoritative current state.

This is the piece that ties this doc together with `docs/plans/guardrails-plan.md`
and the newly-requested natural-language subscription management
(subscribe/unsubscribe to topics, start/stop periodic push — all via plain
conversation, not just `/interests`, since voice input is an eventual goal
and voice doesn't have slash commands).

**The insight**: `guardrails.py`'s layer-2 check (`is_input_on_topic`) and
"what kind of request is this" are the same underlying question, asked
separately today. One classification call can answer both, replacing two
(or more) sequential LLM calls with one:

```python
class MessageClassification(BaseModel):
    on_topic: bool
    category: Literal[
        "news_query", "set_interest", "remove_interest",
        "start_push", "stop_push", "off_topic",
    ]
```

Built with structured output (a Pydantic model via `.with_structured_output()`,
not free-text "yes"/"no" parsing — more robust, and the natural way to
return more than one field). This becomes the **first LLM call** per turn.

**The second LLM call is the existing `create_agent` tool-calling loop —
now doing more than search+format.** `category` from the first call feeds
directly into the `dynamic_prompt` middleware's layer 2: which instructions
and which tool the agent should reach for this turn.

- `category == "news_query"` → layer 2 = search_news usage notes +
  Telegram formatting rules (today's behavior, unchanged).
- `category in ("set_interest", "remove_interest")` → layer 2 = "the user
  wants to update their interests; use the `update_interests` tool, then
  confirm conversationally what changed." Needs a **new tool** exposed to
  the agent (`update_interests(action: "add"|"remove", topic: str)` calling
  `users_db`'s existing `set_interests()`/`get_interests()`).
- `category in ("start_push", "stop_push")` → layer 2 = "the user wants to
  toggle periodic push; use the `set_push_enabled` tool." Originally
  scoped to only recognize the intent and flip a `push_enabled` flag,
  with actual scheduled sending deferred (`docs/plans/bot-features-plan.md`
  item 5) — that deferral is now resolved too: item 5 shipped in full
  (`news_push.py`, the scheduler, the cache convergence), not just the
  flag.
- `category == "off_topic"` → same as today: skip the second call
  entirely, send the redirect message.

**Why this is designed for extensibility, not just this feature**: adding
a future capability (translation, per-user source selection) means adding
one more `category` value, one more tool, and one more `dynamic_prompt`
branch — not touching the classifier's core shape or the agent's
construction. `guardrails.py`'s `is_input_on_topic` did evolve into this
richer classifier rather than staying a separate, narrower check, exactly
as anticipated — `docs/plans/guardrails-plan.md`'s own Status table reflects
this current shape already, no further doc update pending here.

**Tools stay a single fixed list, not swapped per category.** `create_agent`
takes `tools=` once at construction time; making the actual tool list
different per request would mean rebuilding the agent per call, which is
wasteful. Simpler and sufficient: keep one growing `TOOLS` list
(`search_news`, `save_note`, `update_interests`, `set_push_enabled`, ...)
always available, and let the `dynamic_prompt`-injected layer 2 text guide
*which* tool the model reaches for this turn — LLMs are generally reliable
at picking the right tool from a fixed set given clear per-turn framing,
so dynamic tool-list swapping isn't needed to get the conditional-behavior
goal.

## Planned refactor: dispatch settings routes out of the agent

**Status: built 2026-08-16.** Identified 2026-08-09 while documenting the
request pipeline for `docs/system-overview.md` §B2; see "What got
built" below for the design points this section originally left open.

**The observation.** The router (built, live) classifies every message
into a category, and the agent's layer-2 instructions and tool set are
selected from that. Functionally this works — but *every* category still
runs through the same tool-calling agent loop, differentiated only by
prompt content. That's the wrong shape for half of them.

The categories split cleanly into two kinds of work:

| Route | Categories | Work required |
|---|---|---|
| **A — Research** | `news_query` | Genuinely open-ended: search multiple sources, synthesize across them, cite links. Needs tool use and multiple model steps — an agent loop is the right tool. |
| **B — Settings** | `set_interest`, `remove_interest`, `start_push`, `stop_push`, `set_language` | A bounded state change against one subscriber's record. The router has *already* determined the intent; there is nothing left to reason about multi-step. |

**The problem.** Route B pays for an agent loop it doesn't need. Changing
a push interval is one deterministic write, but currently costs a full
agent invocation — model call, tool-selection reasoning, and a second call
to produce the confirmation. That's latency and token cost spent on a
decision the router already made.

**Proposed change.** Dispatch Route B directly: the router's category maps
to a handler that performs the state change and returns a confirmation,
without entering the tool-calling loop. Only `news_query` enters the
agent.

**Expected benefits:**

- Lower latency and cost on the highest-volume non-news operations
- The routing boundary becomes explicit in code rather than implicit in
  which prompt fragment got selected
- Route B becomes deterministic and therefore fully unit-testable — no
  model call means no non-determinism to measure (see the reliability
  discussion in `docs/plans/guardrails-plan.md`)
- **Testability in isolation — a second, independent justification added
  2026-08-14.** `docs/plans/guardrails-plan.md`'s guardrail-harness incident
  found a `set_language` failure specifically (layer 4 blocked a correct
  confirmation while the state change silently succeeded, reproduced 1/5
  with no tunnel or HTTP layer involved — the one finding from that
  incident that survived re-verification). With
  settings and research sharing one agent, a fix aimed at settings
  reliability risks regressing research behavior, and there's no clean way
  to test one without the other in the loop. A dedicated settings path
  would let `set_interest`/`remove_interest`/`start_push`/`stop_push`/
  `set_language` be measured and fixed independently of `news_query`'s own
  (separately-imperfect) reliability — this was the original efficiency
  argument's blind spot: it justified the refactor on cost, not on the
  fact that two very different reliability problems currently share one
  blast radius.

**Open design points:**

- Does Route B still need the layer-4 output check? If the confirmation
  text is generated from a template rather than by a model, there is no
  model output to verify, and that call could be dropped too — a further
  saving. If it stays model-generated for tone/language, the check stays.
- Route B must still honor the per-user reply-language preference. A
  templated confirmation would need translation handled explicitly rather
  than falling out of the prompt, which is an argument for keeping one
  small model call on that path.
- Argument extraction (e.g. *which* topic to add) currently happens inside
  the agent via tool-call arguments. Dispatching out of the agent means
  the router must return that too — expanding its structured output from
  `{on_topic, category}` to include an optional argument payload.

That last point is the real design question: it makes the router do more,
and the router is a single point of failure for every message. Worth
measuring whether a richer structured output degrades its classification
accuracy before committing — the same discipline applied in
`docs/system-overview.md` Appendix B.1, where a plausible prompt change measured
dramatically worse.

## What got built

`guardrails.MessageClassification` gained three optional fields
(`topic`, `push_interval_hours`, `language`), extracted by the router
itself -- `_ROUTER_PROMPT` absorbed the normalization instructions
(short topic label, language typo/variant correction) that used to live
in `agent.py`'s per-category prompt fragments, since the agent no longer
sees these categories at all. `agent.dispatch_settings(category, chat_id,
classification)` performs the state change directly against `users_db`
and returns an English template confirmation -- no model call, fully
unit-testable, exactly the doc's original stated goal. `bot.py`'s
`process_message` checks each category against `agent.ROUTE_B_CATEGORIES`
(`classification.categories` is a list -- see the multi-category routing
section below) and calls this instead of `run_agent` for the five
settings categories; `news_query` is now the only category that still
enters the agent loop (Route A) at all, which let `agent.py`'s
`_compose_prompt` drop its per-category branching entirely -- it's
unconditionally the news_query instructions now.

**The open design points, resolved:**

- **Layer 4 stays, but conditionally.** Route B only runs
  `is_output_on_topic` when the confirmation was actually translated (see
  next point) -- a plain English template is our own fixed string, not
  model output, so there's nothing to check. `bot.py`'s `_route_b_reply`
  also runs the same Markdown/preamble safety nets Route A applies to its
  own model output, since a translated confirmation is real LLM text with
  the same stray-Markdown risk.
- **Translation, not prompt-driven language matching.** `bot.py`'s
  `_translate_confirmation` makes one small model call, checked against
  `users_db.get_language(chat_id)` *after* `dispatch_settings` runs -- so
  a fresh `set_language` change gets its own confirmation translated into
  the language it just set, and a pre-existing preference governs every
  other Route B category's confirmation too, per
  `docs/plans/bot-features-plan.md` item 2's original requirement.
- **Argument extraction, measured before committing, per this section's
  own stated discipline.** `tools/measure_guardrails.py`'s `LAYER2_CASES`
  now checks the new fields (`expects_topic`, `expected_push_interval_hours`,
  `expected_language_contains`), not just `on_topic`/`category`. Measured
  2026-08-16: **187/190 (98%)** overall, matching the pre-change baseline
  on every group except one -- `mixed_language_control` (a single
  real-subscriber sentence bundling a push-interval request *and* a
  language-preference request in one clause) dropped to **7/10 (70%)** on
  `push_interval_hours` extraction specifically, while `category`/`on_topic`
  stayed perfect. Not fixed here: this sentence is genuinely two intents
  in one message -- exactly the multi-category open question below --
  and is expected to resolve more naturally once the router can return
  more than one category for it, rather than by further tuning a
  single-category prompt to do two things well at once.

### Splitting layer 3 and layer 4 by branch, not just by narrow/full

**Proposed 2026-08-14.** Once the router has decided which branch a
message belongs to (Route A/query vs. Route B/command), that decision
should keep shaping every layer downstream of it, not just layer 2's
routing choice — layer 3 (the instructions/tools the model is given) and
layer 4 (the output check) should each split into a command variant and a
query variant:

- **3.1 / 4.1 — command branch.** Layer 3 instructions are narrow and
  templated (per category: add/remove an interest, toggle push, change
  language). Layer 4 already does a version of this today —
  `_NARROW_CHECK_CATEGORIES` in `guardrails.py` skips
  `appropriate_bot_content` for exactly these categories — but the prompt
  text itself (`_OUTPUT_SCOPE_PROMPT`) is still one shared string for both
  branches. 4.1 would get its own prompt, tuned specifically for judging a
  short settings confirmation, instead of a general-purpose prompt with a
  category-gated field subset bolted on.
- **3.2 / 4.2 — query branch.** Layer 3 stays the open-ended `news_query`
  research instructions (search, synthesize, cite). Layer 4 keeps the full
  two-question check (self-disclosure *and* content-appropriateness),
  since query replies are free-form in a way command confirmations aren't.

**Why this matters beyond tidiness**: the `set_language` reliability gap
found in `docs/plans/guardrails-plan.md` (53%/87% pass rate on the command
branch's `discusses_own_configuration` check) is a concrete case where the
shared prompt was carrying an implicit assumption that didn't hold for one
category — the carve-out language named "interests" and "push
notifications" explicitly but never "language preference," and a shared
prompt made that omission easy to miss until measured. A per-branch (or
even per-category) prompt makes each omission a smaller, more visible
surface, and a fix to one branch's prompt can be measured
(`tools/measure_guardrails.py`) without touching the other branch at all —
the same isolation argument already made for the agent-loop split above,
just one layer earlier.

**Relationship to the agent-loop split above**: this is a refinement of
that same Route A/Route B split, not a separate idea — 3.1/4.1 is what
Route B's dispatch handler uses instead of the shared agent prompt/output
check, and 3.2/4.2 is what Route A (the `news_query` agent loop) keeps
using largely as-is. It doesn't require deciding the agent-loop dispatch
question first; the prompts can be split now, with both branches still
running through today's single agent loop, and the dispatch-out-of-the-
agent refactor above can land independently later.

### Open question: a message with both a command and a query in one turn

**Raised 2026-08-14, not resolved.** Once routing depends on which branch
a message belongs to, a message that is genuinely both — e.g. "add
robotics to my interests and tell me what's new with it" — needs an
explicit answer, not an accidental one. Three options were raised:

- **(a) Loop the pipeline**: run the command branch (3.1/4.1) first,
  execute the state change, then run the query branch (3.2/4.2) for the
  remaining intent, and send both results (either as one combined reply or
  two messages). Requires the router to detect *and preserve* the
  remainder of the message after extracting the command part, which is a
  real complication — most naturally as a router that returns a *list* of
  categories/intents for one message instead of exactly one, with each
  intent then dispatched to its own branch in sequence.
- **(b) Branch and merge**: recognize both intents up front, run both
  branches in parallel (or in either order — no dependency between them),
  and merge the two replies into one response. Avoids the sequential
  latency of (a) but adds a merge step that has to produce one coherent
  Telegram message out of two independently-generated pieces (formatting,
  ordering, avoiding a jarring tone shift between a templated confirmation
  and a free-form report).
- **(c) Ask the user to split it**: detect that a message carries more
  than one intent and reply asking them to send the command and the query
  as separate messages, rather than trying to handle both at once.
  Simplest to implement and reason about, but adds friction on a
  real, plausible phrasing (the exact "add X and tell me about X" pattern
  is a natural single sentence, not a contrived edge case) and pushes a
  system limitation onto the user as extra typing.

**Built 2026-08-16, as (a).** `guardrails.MessageClassification.category`
became `categories: list[Category]`; `_ROUTER_PROMPT` instructs a
single-element list for the overwhelmingly common single-intent case and
only a longer list when the message genuinely asks for more than one
distinct thing. `bot.py`'s `process_message` special-cases
`len(categories) == 1` to reuse the exact pre-existing single-category
code path unchanged (`_process_single_category` -- Route A keeps its full
message-list history fidelity, Route B keeps its own shape); a real
multi-category list goes through the new `_process_multi_category`, which
dispatches each category (reusing `_route_a_reply`/`_route_b_reply`, the
same logic `_process_single_category` calls) against the same original
`history`/`user_text`, joins the replies with a blank line in category
order, and persists the whole turn as one `(Human, AI)` history pair, not
one per category. **All-or-nothing**: if any segment is blocked by layer
4, the entire reply becomes `REDIRECT_MESSAGE` rather than sending a
partial result -- note that a Route B state change from an earlier
segment in the same turn still commits even if a later segment gets
blocked, since dispatch_settings's write already happened; this is a
judgment call, not something the three original options above resolved,
and is easy to revisit if partial replies turn out to be preferable.

**Measured before committing, per this section's own stated discipline.**
`tools/measure_guardrails.py` gained `LAYER2_MULTI_INTENT_CASES` (two
genuine multi-intent examples in English and Chinese, two single-intent
negative controls confirming the router doesn't over-split ordinary
messages, and one maximally-ambiguous real case, below) and
`measure_layer2_multi_intent`. Results, 10 trials/case:

| Case | Result |
|---|---|
| `[set_interest, news_query]` (English + Chinese) | 20/20 (100%) |
| `[start_push, set_language]` (English) | 10/10 (100%) |
| Single-intent negative controls (2 cases) | 20/20 (100%) — confirms no over-splitting |
| The pre-existing `mixed_language_control` real-subscriber sentence | 6/10 (60%) — see below |

The base (single-category) `LAYER2_CASES` dataset held at **180/180
(100%)** after the schema change -- no regression from the richer
structured output, the risk this section originally flagged.

**One real, unresolved finding, not swept under the rug**: the
`mixed_language_control` case (a real subscriber's sentence bundling a
push-interval request, a language-preference request, and an ambiguous
"tech/finance/bitcoin" phrase that could be read as either a topic filter
or a new interest) doesn't have one stable answer -- direct inspection
across 10 raw trials showed the model reading it as 2 *or* 3 intents,
with both the third intent's presence and the list's order varying trial
to trial. Scored with an order-independent, extras-tolerant
`expected_categories_containing = {start_push, set_language}` (added
alongside the existing exact-match `expected_categories`, for cases that
don't have one correct exact list) rather than pinned to one exact
answer. Not chased to 100% -- this is a genuinely ambiguous real sentence
where trial-to-trial variation is itself the honest finding, the same
judgment call as `docs/plans/guardrails-plan.md` not chasing
`self_disclosure`'s 98% to 100%.

## Open questions

- ~~Exact LangChain middleware API surface~~ — **verified**, see above.
- ~~How layer 2 decides what's "relevant"~~ — **mostly resolved**: the
  router's `category` output drives it directly for anything with a
  distinct intent (interests, push); content that's relevant regardless
  of category (Telegram formatting) just stays unconditional. No separate
  relevance-classifier call needed beyond the one router call itself.
- ~~Exact tool signatures for `update_interests`/`set_push_enabled`, and
  the `users_db` schema addition~~ — **resolved by building it**:
  `update_interests`/`set_push_enabled`/`set_push_interval` all live in
  `agent.py`, `push_enabled`/`push_interval_hours` are real `users_db`
  columns.
- ~~How `docs/plans/guardrails-plan.md` needs to change once `is_input_on_topic`
  becomes this router~~ — **resolved**: that doc's own Status table
  already reflects the router as built.
- Whether to formally adopt LangGraph's `Store` API later if per-user
  memory grows more complex, or keep extending `users_db.py` directly —
  leaning toward the latter unless a concrete need for `Store`-specific
  features (e.g. built-in semantic search over memories) shows up. Still
  genuinely undecided, not resolved by anything built since.
- **Net call count per turn — built, but never actually measured.** One
  router call (replacing guardrails' old input check) + the agent's own
  1-2 calls + the existing output check (layer 4 in
  `docs/plans/guardrails-plan.md`) is the theoretical shape; whether that's
  actually a wash or slight improvement over the pre-router call count is
  still an assumption, not a measured latency number. Same open item as
  `docs/plans/guardrails-plan.md`'s own "actual token/latency numbers still not
  measured" note — one measurement would answer both.

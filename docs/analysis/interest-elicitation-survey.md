# Helping a user find their interests: a cross-domain survey

Research for a proposed **"help me find my interests"** feature — a
multi-turn conversation where the bot helps a subscriber work out what
they actually want news about, instead of requiring them to already know
and type it.

Same footing as this directory's other documents: **nothing here is
built, and much of it never will be.** This is what other fields do
about the same underlying problem, what transfers, and what this
codebase specifically can reuse. The design discussion comes after.

## 1. What the problem actually is

The obvious framing — "the user knows what they want, we just need a
nicer input box" — is wrong, and every field surveyed below agrees it's
wrong.

**Belkin's Anomalous State of Knowledge (ASK)**, the foundational IR
result here, says an information seeker is *by definition* unable to
precisely formulate a request for something they don't already know:
they recognise a gap but cannot name it. Compounding it, the
**vocabulary problem** means even a user with a clear need often
doesn't share terminology with the system. This bot has already been
bitten by exactly this: the AAOI incident (`docs/current/ai-news-sources.md`,
`docs/plans/local-news-cache-plan.md`) was a subscriber asking about a
small-cap company by ticker, where the corpus simply had nothing under
that name.

A 2026 paper on agentic recommenders (**"Beyond expert users: agents
should help users construct preferences, not just elicit them"**) pushes
this further, and it's the single most important framing for this
feature: for non-expert users, preferences often **do not exist in
finished form before the conversation**. Treating the interaction as
extraction ("what are you interested in?") assumes a fixed internal
state that isn't there. Treating it as **construction** — guided
exploration where the user develops the preference through the
conversation — is a different design goal with different success
criteria.

That reframe matters for us concretely: the feature's job is not to
transcribe an answer the user already has. It's to give them enough
concrete material to react to that a preference forms.

## 2. What other domains do

Organised by *mechanism*, since several fields independently invented
the same few moves.

### 2.1 The funnel — broad open questions before narrow closed ones

Used essentially identically in **journalism interviewing**, **clinical
history-taking**, and **qualitative UX research**. Start with broad
open-ended questions, progressively narrow, close with specific/closed
questions.

The rationale is not politeness — it's **contamination control**. Ask a
narrow question first and you've supplied the frame; whatever the person
says afterward is partly your idea reflected back. NN/g's version of
this is explicit that broad-first "avoids making assumptions." Clinical
intake uses the same shape for the same reason.

**Transfers directly.** An interest-finding flow that opens with "are
you interested in semiconductors?" has already failed — it will get a
polite yes and a useless stored interest.

### 2.2 Laddering and the repertory grid — surfacing constructs the user can't name

From **personal construct psychology** (Kelly) and **means-end chain
theory** in market research.

- **Repertory grid**: present *triads* of concrete items and ask "which
  two are alike, and how is the third different?" This elicits the
  dimensions the person actually thinks in — without ever asking them to
  name a dimension, which they usually can't do on demand.
- **Laddering**: take a stated preference and repeatedly ask *"why is
  that important to you?"*, climbing attribute → consequence → value.

The key property: both extract structure through **comparison and
reaction**, never through introspective self-description.

**Transfers well.** A triad of three real headlines and "which two feel
more like what you want?" is a far better question than "describe your
interests." Laddering gives the *upward* move (from "TSMC earnings" to
"semiconductor supply chain" as the durable stored interest).

### 2.3 Motivational interviewing (OARS) and solution-focused questioning

From counselling/behaviour change. **OARS** = Open questions,
Affirmations, Reflective listening, Summaries. Notably, the therapist
uses these to let *the client* set the focus, rather than proposing one.

Two specific tools worth stealing:

- **Reflective listening / summarising**: play back what you heard, in
  your words, and let them correct it. Cheap, and it catches
  misunderstanding before it's committed to.
- **Scaling questions** (solution-focused brief therapy): "on a 0-10,
  how much is this it?" then "why not lower?" — a compact way to get
  calibration and the *reason* behind it in one turn.

**Transfers.** The summarise-and-confirm step maps exactly onto "here's
the interest I'm about to save — right?" And scaling is a cheap way to
rank candidate topics without a long conversation.

### 2.4 Career interest inventories — the closest literal analogue

RIASEC/Holland Codes, Strong Interest Inventory: instruments whose
entire purpose is "help me find my interests." Their design choice is
striking and consistent: they ask about **reactions to concrete
activities** ("would you enjoy repairing a car?"), typically
forced-choice, and *never* ask the person to describe their interests
abstractly. The abstract profile is computed from many concrete
reactions, then reflected back.

**Transfers as a principle, not a format.** We can't run a 60-item
inventory in a chat. But the underlying finding — *people are far more
reliable at reacting to concrete instances than at generating abstract
self-descriptions* — is the same insight the repertory grid and IR
relevance feedback arrive at independently. Three fields converging on
this is the strongest signal in this survey.

### 2.5 Relevance feedback and query-by-example (IR)

Rocchio-style relevance feedback: show results, let the user mark
relevant/irrelevant, move the query vector toward the good ones and away
from the bad. **Active learning** refines this by choosing which items
to show — pick the ones whose labels are most informative (uncertainty
sampling) rather than the ones you're most confident about.

**Transfers, and we're unusually well set up for it** — see §3.

### 2.6 Conversational recommender systems (CRS)

The direct academic ancestor of what's proposed. Findings worth knowing:

- Most cold-start CRS work uses **multi-armed bandits** to decide what
  to ask next, balancing explore/exploit.
- Christakopoulou et al. (Microsoft) report **~25% improvement over a
  static model after only 2 questions** — the curve is steep and early.
  Most of the value is in the first couple of turns.
- A live design axis: **ask about items** ("interested in this story?")
  vs **ask about attributes/facets** ("more interested in policy or
  hardware?"). Item questions are concrete but low-information;
  attribute questions are high-information but hit the vocabulary
  problem.
- Recent LLM work (2025-2026) focuses on generating clarifying questions
  directly, which is roughly what we'd be doing.

### 2.7 Onboarding UX practice (news/content apps)

Flipboard, Tumblr, Google News all open with a topic-picker. Reported
practice: **2-5 selections is the manageable range**, and the flow
should deliver visible payoff immediately after (Flipboard drops you
straight into stories for what you picked).

**Transfers as a constraint, not a pattern.** Their grid-of-topics UI
isn't available to us in a chat, but the "few choices, then immediate
payoff" shape is, and the dropout risk it's guarding against is real.

## 3. What this codebase already has that this feature would use

This is the part that makes the feature cheap rather than speculative.

| Existing piece | What it gives the feature |
|---|---|
| `build_agent`/`run_agent` (dormant since PR #85) | An open-ended, unknown-step tool-calling loop — **kept alive deliberately for exactly this kind of feature.** This is its justifying use case. |
| The ingested news cache + per-article embeddings | Real headlines to react to. Enables §2.4/§2.5's "concrete beats abstract" as a *first-class* mechanism, not a mock-up. |
| `news_embed.filter_by_relevance` | Vector retrieval for "show me things near this" — the machinery for relevance feedback already exists. |
| `SqliteVecStore.search_similar` (PR #86, unused by any caller) | Indexed top-K similarity. **Written, tested, and shipped, currently with zero callers** — this feature is its natural first consumer. |
| `category_ops` taxonomy | A ready-made breadth axis for "which of these areas?" without inventing one. |
| `news_classify.normalize_interest_detailed` / `expand_interest_for_retrieval` | Turns whatever phrasing the conversation lands on into a stored, retrieval-ready interest. |
| `subscriber_ops.add_interest` + `_is_duplicate_topic` | Storage and fuzzy dedup already handled, including `MAX_INTERESTS`. |

## 4. Constraints this project specifically imposes

These are not general findings — they come from this repo's own measured
history, and they should bound the design.

1. **Latency is the hard limit on turn count.** Today's instrumentation
   (`docs/current/telemetry-catalog.md`, "search_news per-call latency")
   measured a real question at **7-20s end to end**, dominated by
   sequential LLM round trips — and PROD's constrained VM made a single
   uncached corpus read take 60-160s before the sqlite_vec fix. A
   six-turn elicitation conversation at ~10s/turn is a minute-plus of
   waiting. **The CRS finding that most value lands in the first ~2
   questions is not just an efficiency note here; it's a requirement.**

2. **An interest with no corpus coverage is worse than no interest.**
   The AAOI incident: a subscriber ends up with a stored interest that
   permanently returns nothing. Any candidate topic this feature
   proposes should be **grounded in what the cache actually contains**,
   not brainstormed freely by the model. This is a real, already-observed
   failure mode, and it argues strongly for example-driven (§2.4/§2.5)
   over brainstorm-driven elicitation.

3. **The daily search quota (10/day) is shared.** If elicitation burns
   corpus lookups, it competes with the user's actual searches. Needs an
   explicit decision.

4. **Interests are already normalised, deduped, and capped at 10.** The
   feature proposes *into* an existing system with its own rules, rather
   than owning storage.

## 5. What consistently transfers — the short version

Ranked by how strongly the evidence converges:

1. **Reaction beats articulation.** Interest inventories, repertory
   grid, and IR relevance feedback independently converge on this.
   Show real headlines; ask which ones land. Don't ask for a
   self-description.
2. **Concrete before abstract.** Then *ladder up* to the durable,
   storable interest ("why did that one land?" → "semiconductor supply
   chain").
3. **Broad before narrow, or you contaminate the answer** (funnel).
4. **Very few turns.** 2-5 is both the onboarding-UX sweet spot and
   where the CRS value curve flattens — and here it's also a latency
   requirement.
5. **Summarise and confirm before committing** (OARS's S) — cheap,
   catches errors before they're stored.
6. **Preferences are constructed, not extracted.** Success is "the user
   now knows what they want and it's grounded in real coverage," not
   "we captured a pre-existing answer."

## 6. Open design questions — for discussion, not decided here

1. **Turn budget.** Given §4.1, what's the hard cap — 2 turns? 3? What
   happens when it's hit without convergence?
2. **Where candidates come from.** Live cache content (grounded, per
   §4.2) vs the `category_ops` taxonomy (cheap, stable) vs LLM
   brainstorm (fluent, ungrounded, risks AAOI). Or a mix, and in what
   order?
3. **Item questions vs attribute questions** (§2.6) — headlines to react
   to, or facets to choose between? Probably both, but which first?
4. **Cold start vs warm refine.** Same flow for "I have no interests"
   and "I have 4 and they're not quite right"? These may want different
   openings.
5. **Does it write directly, or propose?** Given `add_interest` already
   exists with dedup and a cap, a propose-then-confirm step (§2.3) seems
   right — but that's a turn, and turns are expensive.
6. **Quota accounting** (§4.3).
7. **Termination.** How does the conversation end — user says stop,
   turn cap, or the model judging convergence? What if they end up with
   zero?
8. **Does this reuse the agent loop, or is it another fixed pipeline?**
   PR #85's whole lesson was that an unbounded agent loop over an
   expensive corpus read is a latency disaster. An elicitation flow is
   genuinely more open-ended than search was — but the same trap is
   right there, and §4.1 says the budget is tight. **This is the
   decision I'd want to make most carefully.**

## Sources

- [Anomalous States of Knowledge as a Basis for Information Retrieval (Belkin)](https://www.researchgate.net/publication/238671719_Anomalous_States_Of_Knowledge_As_A_Basis_For_Information_Retrieval)
- [Beyond expert users: agents should help users construct preferences, not just elicit them (2026)](https://arxiv.org/pdf/2606.30863)
- [Towards Conversational Recommender Systems (Christakopoulou et al., Microsoft)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/rfp0063-christakopoulou.pdf)
- [Asking Clarifying Questions for Preference Elicitation With Large Language Models (2025)](https://arxiv.org/html/2510.12015v1)
- [Conversational Information Seeking (survey)](https://arxiv.org/pdf/2201.08808)
- [The Funnel Technique in Qualitative User Research (NN/g)](https://www.nngroup.com/articles/the-funnel-technique-in-qualitative-user-research/)
- [Funnel Technique in journalism interviewing](https://library.fiveable.me/key-terms/hs-journalism/funnel-technique)
- [Using the repertory grid and laddering technique to determine the user's evaluative model of search engines](https://www.emerald.com/insight/content/doi/10.1108/00220410710737213/full/html)
- [An Adaptation of the Laddering Interview Method](https://journals.sagepub.com/doi/pdf/10.1177/1094428105280118)
- [Motivational Interviewing: The Basics, OARS](https://iod.unh.edu/sites/default/files/media/2021-10/motivational-interviewing-the-basics-oars.pdf)
- [How to Use OARS Skills in Motivational Interviewing (Relias)](https://www.relias.com/blog/oars-motivational-interviewing)
- [Onboarding UX patterns and best practices (Appcues)](https://www.appcues.com/blog/user-onboarding-ui-ux-patterns)
- [Explainable Active Learning for Preference Elicitation](https://arxiv.org/pdf/2309.00356)

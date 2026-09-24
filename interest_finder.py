"""
"Help me find my interests" -- a multi-turn conversation that helps a
subscriber work out what to follow, instead of requiring them to already
know and type it.

See docs/analysis/interest-elicitation-survey.md for the cross-domain
research this design rests on. The three findings that actually shaped
the code:

1. **Reaction beats articulation.** Career-interest inventories, the
   repertory grid, and IR relevance feedback independently converge on
   this: people are far more reliable judging concrete instances than
   generating abstract self-descriptions. So the core loop is "show real
   headlines, ask which ones land", never "describe your interests".
2. **Ground every candidate in the real corpus.** The AAOI incident (a
   subscriber left holding an interest the cache has no coverage for,
   returning nothing forever) is what happens when topics are
   brainstormed freely instead of drawn from what's actually there. Hence
   find_example_articles: the model may only propose what it has seen in
   the cache.
3. **Preferences are constructed, not extracted.** Success is "the user
   now knows what they want AND it's grounded in real coverage" -- not
   "we captured a pre-existing answer".
4. **A model can narrate an action it never took.** Found live 2026-09-08:
   a real subscriber confirmed adding a topic in Traditional Chinese, the
   model replied "好，我已為你加入..." (done, I've added it), and the
   subscriber's interests stayed empty -- zero save_interest telemetry
   fired for that whole conversation. The model composed a confident
   success message without ever calling the tool that would have made it
   true. Same lesson as MAX_STEPS_PER_TURN's own history below: a prompt
   instruction is a nudge, not a guarantee. propose_interest/
   classify_confirmation/execute_save/execute_drop exist because of this
   -- the highest-stakes action (persisting data) no longer depends on
   the model remembering to call a tool at the right moment; it depends
   only on classifying one short reply as affirm/decline/unclear, the
   same bounded, already-reliable shape as guardrails.classify_message's
   router. See propose_interest's own docstring for the mechanism.
5. **The retrieval DEFINITION is a far bigger lever than the interest word
   itself, and a subscriber can't see or touch it.** Measured 2026-09-09
   (docs/analysis/retrieval-quality-measurements.md finding 4): rewriting
   one cached paragraph moved a target article from rank 467 to rank 2 in
   the same corpus. This module's show_definition/propose_definition/
   save_definition tools exist because that lever needed a handle a
   subscriber could actually turn. A tempting alternative -- classifying
   articles by "story genre" (demo vs. announcement vs. analysis) so a
   subscriber could ask for more of one -- was measured and DISCARDED
   (finding 5): a genre written into a definition barely separates wanted
   from unwanted articles (~0.09 cosine spread), because static
   embeddings encode subject matter, not story shape. The definition
   lever works through topical vocabulary, so refinement stays topical:
   concrete directions drawn from the cache ("more hands-on/experimental",
   "more enterprise/deployment"), never an abstract genre label.
6. **The single-turn "add X to my interests" fast path is retired.**
   Found live 2026-09-10, root-caused by direct reproduction against the
   real model (not assumed): a subscriber's exploration ended with the
   model narrating "let me search again" and "let's stop here" without
   ever calling find_example_articles or end_exploration on those turns.
   Reproducing the SAME conversation against four different models/
   providers (DeepSeek direct on its latest V4.1-Flash, the same model
   family hosted by Together.ai on an older fixed checkpoint, GLM-5.3-
   Flash, and gpt-oss-120b) showed this is a property of long-tool-
   calling-loop reliability in general, not one provider's bug -- and
   that end_exploration specifically asks the model to do the ONE thing
   this project already learned not to trust it with (see MAX_TURNS's
   own comment: "self-assessment models are unreliable at"). Patching
   end_exploration with yet another classifier would only be treating a
   symptom.
   The actual fix removes the need to trust it at all: EVERY interest add
   now goes through the same grounded show-examples-then-confirm flow
   that already existed for the guided exploration, funneled through
   propose_interest, which bakes in a preview exactly like
   propose_definition already does. Once no unconfirmed, ungrounded add
   can ever reach subscriber_ops.add_interest, it no longer matters
   whether a session lingers because end_exploration was never called --
   nothing unsafe can happen while it's open, and MAX_TURNS remains an
   adequate, already-existing bound on how long it lingers. This is also
   why set_interest/remove_interest/set_language moved into this same
   agent (agent.INTEREST_AGENT_CATEGORIES) instead of Route B's one-shot
   dispatch: a subscriber naming a topic outright ("add robotics") still
   needs to see it grounded before it's saved, and switching languages or
   removing an interest needs to keep working without forcing them out of
   an in-progress exploration first.

**This is the one feature in this codebase that genuinely justifies an
agent loop** (agent.build_agent/run_agent, dormant since PR #85). The
test PR #85 established: use a loop only when the number of steps can't
be known in advance. search_news failed that test -- retrieval turned out
to be fully boundable. This passes it: how many questions, when to show
examples instead of asking, and when the user has converged are all
genuinely undecidable up front.

PR #85's actual lesson was narrower than "agent loops are bad": it was
that a loop whose every iteration costs 12-160s of corpus read is a
disaster. That precondition is gone -- SqliteVecStore (PR #86) made a
corpus read ~0.3s. And unlike search_news's one-shot answer, this is
interactive, so the latency budget is per-turn (a normal conversational
wait) rather than cumulative. The residual risk is the one actually
observed back then -- a SINGLE turn ballooning into many internal tool
calls -- which is why the caller caps turns and this module keeps each
tool cheap.
"""

from typing import Literal

from langchain.agents.middleware import dynamic_prompt
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

import agent
import interest_cache_ops
import news_cache
import news_embed
import subscriber_ops
from app_settings import get_settings
from telemetry import EventLogger, get_event_logger
from telemetry_providers import Level

_events: EventLogger = get_event_logger("argus.interest_finder")

# How many headlines one find_example_articles call shows. Small on
# purpose: this lands in a Telegram message the subscriber has to read
# and react to, and the onboarding-UX literature puts the manageable
# range at 2-5 choices (docs/analysis/interest-elicitation-survey.md
# §2.7). This is a "how much can a person weigh at once" number, not a
# retrieval-quality one.
MAX_EXAMPLES = get_settings().resolved("interest_finder.max_examples", default=5)

# Hard ceiling on user turns in one exploration, enforced by the CALLER
# (bot.py), not by the model. Answers the "user keeps switching direction
# and can't decide" case: a counter is deterministic, whereas asking the
# model to notice it is exactly the kind of self-assessment models are
# unreliable at. Generous enough for a real conversation -- the research
# puts most of the value in the first 2-5 exchanges -- while still
# guaranteeing this ends.
MAX_TURNS = get_settings().resolved("interest_finder.max_turns", default=8)

# Ceiling on model/tool steps within a SINGLE turn -- a different failure
# from MAX_TURNS above, and the one PR #85 actually measured: one question
# fanning out into 5-7 internal searches, each paid for.
#
# 20, not the 10 this shipped with first. Measured against the real pinned
# model 2026-09-08: asking about a topic the cache has NO coverage for
# ("quantum blockchain synergy") made the model try one rephrasing after
# another -- 15 find_example_articles calls, ~30 graph steps -- before
# concluding honestly that there was nothing there. 10 cut that off
# mid-loop; 20 leaves roughly 9 tool calls, several times what a normal
# turn (one search, maybe a list, a reply) uses, while still capping a
# runaway. The prompt's own "stop after two empty searches" rule below is
# the real fix for that case; this is the backstop for when it doesn't
# hold, and run_turn now handles hitting it gracefully rather than
# letting LangGraph's own error text reach the subscriber.
MAX_STEPS_PER_TURN = get_settings().resolved("interest_finder.max_steps_per_turn", default=20)

_SYSTEM_PROMPT = (
    "You are helping ONE subscriber of a technology-news Telegram bot "
    "work out what topics they want to follow. They came here because "
    "they could not simply name a topic -- if they could, a different, "
    "single-turn path would have handled it. Your job is to help them "
    "ARRIVE at an interest, not to extract one they already hold.\n\n"

    "HOW TO DO THIS -- the method matters, follow it:\n"
    "- Lead with real examples, not questions about preferences. People "
    "are far better at reacting to concrete headlines than at describing "
    "what they want in the abstract. Call find_example_articles and show "
    "what actually exists, then ask which of them land.\n"
    "- Only ever propose topics you have SEEN in find_example_articles "
    "results. Never invent or brainstorm a topic the cache has no "
    "coverage for -- an interest with no coverage sends this subscriber "
    "nothing, forever, which is worse than leaving them with none.\n"
    "- Go broad before narrow. An early narrow question supplies the "
    "answer and you learn nothing.\n"
    "- When something does land, ask what it was about that story that "
    "interested them, and use the answer to name a durable topic rather "
    "than a one-off event. (\"That specific earnings report\" is not an "
    "interest; \"semiconductor supply chain\" is.)\n"
    "- Keep it short. Aim to converge within a couple of exchanges. Ask "
    "one thing at a time.\n"
    "- Do not search more than twice before replying. If two searches "
    "come back with nothing, say so plainly and ask them for a different "
    "direction -- rephrasing the same idea a third and fourth time does "
    "not find coverage that isn't there, and it makes them wait.\n"
    "- NEVER propose adding a topic without grounding it first, even "
    "when they name it directly and specifically (\"add robotics to my "
    "interests\"). Naming something outright is not permission to skip "
    "find_example_articles -- call it for that exact topic, write a "
    "definition from what you actually found, and only then propose_"
    "interest. A specific request still gets the same grounding, just "
    "with less back-and-forth to reach it (usually one search, not "
    "several).\n\n"

    "THE SIX WAYS THEY MIGHT ARRIVE, all handled here:\n"
    "(a) They want help finding interests from scratch -- start with a "
    "broad sense of what the corpus covers, then narrow.\n"
    "(b) They liked a story they were sent and want more like it -- "
    "search for it, confirm which part appealed, name the topic.\n"
    "(c) They want their existing mix adjusted by feel (\"too much X, "
    "not enough Y\") -- call list_current_interests first, then use "
    "save_interest/drop_interest to rebalance. Adjust the interest LIST "
    "only; you cannot change push frequency or volume here.\n"
    "(d) They want suggestions before committing -- show examples, let "
    "them pick.\n"
    "(e) They already follow a topic but what it sends isn't what they "
    "wanted (\"I follow AI but I never get the interesting stuff\") -- "
    "this is the topic's DEFINITION, not the topic word itself, that "
    "needs adjusting. Call show_definition to see what's currently "
    "driving their pushes, then find_example_articles a few different "
    "ways to see what the cache actually has for this topic. Offer 2-3 "
    "concrete directions drawn from what you found (e.g. \"more hands-on "
    "experiments and demos\" vs. \"more enterprise/deployment news\" vs. "
    "\"more research papers\") -- never an abstract quality label like "
    "\"more interesting\", which cannot be turned into a retrieval query. "
    "Once they pick a direction, write a candidate definition and call "
    "propose_definition with it -- this shows you (and them) exactly "
    "which real cached articles it would surface, BEFORE anything is "
    "saved. Definition effects are measured to be counter-intuitive, so "
    "always look at the preview yourself before describing it to them, "
    "and never call save_definition on a definition you haven't "
    "previewed via propose_definition.\n"
    "(f) They name a specific topic directly (\"add robotics to my "
    "interests\") -- do not save it on the spot. Call find_example_"
    "articles for that exact topic, write a definition grounded in what "
    "you found, then propose_interest with it. If it has no coverage, "
    "say so plainly instead of adding it anyway -- same AAOI-avoidance "
    "rule as everywhere else here.\n\n"

    "BEFORE SAVING, REMOVING, OR REDEFINING: call propose_interest(topic, "
    "definition), propose_remove(topic), or propose_definition(topic, "
    "definition) in the SAME turn you say plainly what you are about to "
    "change, then ask them to confirm -- do not save on a guess. "
    "propose_interest and propose_definition both run a real preview and "
    "return it to you immediately; read it before describing the change. "
    "None of the three propose calls change anything by themselves, so "
    "calling any of them is always safe, and the system uses whichever "
    "one you called to complete the change reliably once they agree even "
    "in a turn where you don't get the chance to call "
    "save_interest/drop_interest/save_definition yourself. If you are "
    "ever unsure whether a proposal you made earlier actually went "
    "through, call the matching save/drop tool again rather than just "
    "saying it worked -- only say something is saved after a tool call "
    "has actually told you so.\n\n"

    "LANGUAGE: if they want replies in a different language from now on, "
    "call set_language directly -- no need to propose or confirm first, "
    "it's a low-stakes, instantly reversible setting, not data that could "
    "leave them following something with no coverage. This works at any "
    "point in this conversation, not just at the start.\n\n"

    "ONE-OFF NEWS QUESTIONS: if they ask a self-contained news "
    "question (\"what's new with X\", \"any news on Y\") rather than "
    "trying to find something durable to follow, call search_news "
    "with a query that stands on its own -- resolve any reference to "
    "earlier in THIS conversation yourself first (e.g. \"what about "
    "that one\" -> the actual topic), since search_news gets no memory "
    "of this conversation at all. This does not end the conversation; "
    "keep going right after.\n\n"

    "PUSH SETTINGS: if they want to turn the periodic push digest on "
    "or off (optionally naming an interval in hours for on), call "
    "start_push/stop_push directly -- same no-confirmation-needed "
    "reasoning as set_language. CALL THE TOOL EVERY TIME, even if you "
    "already believe you know the current state from earlier in THIS "
    "SAME conversation (e.g. you just called start_push and they now "
    "say to turn it off) -- never reply that push is on/off, or that "
    "you've changed it, without a fresh tool call THIS turn to back it "
    "up. Saying it happened is not the same as making it happen.\n\n"

    "WHEN TO STOP: call end_exploration as soon as they are satisfied, "
    "or if they say they are done, or if they have changed direction "
    "repeatedly without converging -- in that last case tell them "
    "honestly that this is not getting anywhere and suggest they just "
    "name a company or topic directly, then end. Do not keep looping "
    "hoping it resolves. This one is a courtesy, not a safety mechanism -- "
    "nothing risky can happen just because a conversation stays open, so "
    "say a natural goodbye even in a turn where you don't call the tool "
    "for it.\n\n"

    + agent.HTML_FORMATTING_RULES
)


def _session(runtime: ToolRuntime) -> dict:
    """The caller's own mutable per-conversation record (see
    bot.py's interest_sessions). Tools write into it so the caller can
    see what happened without parsing the agent's message transcript."""
    return runtime.context["session"]


def _relevant_cached_articles(embedder, query_text: str) -> list[dict]:
    """The shared retrieval step behind both find_example_articles and
    propose_definition's preview -- same pool, same relevance filter,
    same clamp constants. Deliberately NOT search_news: three
    differences, each load-bearing.

    1. No quota. Narrowing down (or previewing a definition) must not
       spend the subscriber's daily search allowance -- they are being
       helped, not served results.
    2. No mark_links_shown. An article shown as an EXAMPLE or PREVIEW
       here has not been delivered as news; retiring it would silently
       remove it from a future digest the subscriber would otherwise
       have gotten.
    3. No definition generation. expand_interest_for_retrieval is the
       single most expensive step in the search pipeline (2.7-4.7s
       measured, see docs/current/telemetry-catalog.md) and it caches
       per topic -- but exploration tries many one-off phrasings, so it
       would miss the cache nearly every time and pay full price on
       every turn. Raw-query embedding is a weaker retrieval signal
       (news_classify.expand_interest_for_retrieval's own docstring has
       the measurement); accepted here because these results only need
       to be good enough to react to. Whatever finally gets SAVED (an
       interest via agent.add_one_interest, or a definition via
       execute_redefine) uses the real thing -- a generated definition
       for a new interest, or the subscriber's own hand-refined text for
       an existing one."""
    pool = [a for a in news_cache.read_all() if a.get("link")]
    return news_embed.filter_by_relevance(
        pool, embedder, query_text,
        keep_fraction=agent.SEARCH_RELEVANCE_KEEP_FRACTION,
        keep_min=agent.SEARCH_RELEVANCE_KEEP_MIN,
        keep_max=agent.SEARCH_RELEVANCE_KEEP_MAX,
    )


def _format_article_lines(articles: list[dict]) -> list[str]:
    return [f"- {a['title']} ({a.get('source') or a.get('source_key', '')})" for a in articles]


@tool
def find_example_articles(query: str, runtime: ToolRuntime) -> str:
    """Search the already-ingested news cache for real recent articles
    matching a topic or phrase, to show the subscriber as concrete
    examples. Use this before proposing any topic -- it is the only way
    to know the topic has real coverage."""
    examples = _relevant_cached_articles(runtime.context.get("embedder"), query)[:MAX_EXAMPLES]
    if not examples:
        return (f'Nothing in the cache matches "{query}". Do not propose this '
                "topic -- it has no coverage. Try a broader or different phrasing.")
    lines = [f"{len(examples)} real cached article(s) for \"{query}\":"]
    lines.extend(_format_article_lines(examples))
    return "\n".join(lines)


@tool
def list_current_interests(runtime: ToolRuntime) -> str:
    """What this subscriber currently follows. Call this before
    rebalancing an existing set."""
    interests = subscriber_ops.get_interests(runtime.context["chat_id"])
    if not interests:
        return "This subscriber follows nothing yet."
    return "Currently following: " + ", ".join(interests)


@tool
def show_definition(topic: str, runtime: ToolRuntime) -> str:
    """Shows this subscriber's current retrieval definition for one of
    their existing interests -- the paragraph that actually decides what
    gets pushed for it, not the topic word itself
    (docs/plans/interest-definition-plan.md). Call this before proposing
    a redefinition, so you're working from what's actually driving their
    pushes today, not guessing at it."""
    chat_id = runtime.context["chat_id"]
    definition = interest_cache_ops.resolve_interest_definition(chat_id, topic)
    if definition is None:
        return (f'No definition exists yet for "{topic}" -- retrieval currently falls back '
                "to embedding the bare topic string, which is a weak query. Any definition "
                "you propose would be a genuine improvement, not just a change.")
    return f'Current definition for "{topic}":\n\n{definition}'


def execute_save(chat_id: int, topic: str, definition: str, guard_model, session: dict) -> str:
    """Actually persists one interest, together with the `definition` the
    subscriber has already seen previewed and confirmed -- the single
    place both the save_interest tool below AND bot.py's deterministic
    confirmation gate (see propose_interest's docstring) call, so there is
    exactly ONE code path that can make "this subscriber follows X" true,
    and exactly one place the interest_saved_from_exploration event fires
    from, regardless of which path triggered it.

    agent.add_one_interest, not subscriber_ops.add_interest directly: it
    also normalizes the phrasing (translation/disambiguation), which is
    orthogonal to this module's own grounding work and stays there rather
    than being duplicated here. `definition` is required now -- see this
    module's own docstring, finding 6: a blind, subscriber-unseen
    definition is exactly what the front-door redesign eliminated, and
    there is no add path left that doesn't go through a real preview."""
    known = subscriber_ops.get_interests(chat_id)
    reply = agent.add_one_interest(chat_id, topic, guard_model, known, definition)
    session.setdefault("saved", []).append(topic)
    # The one outcome that says this feature worked. bot.py already logs
    # the failure shapes (out-of-turns, an exception); without this there
    # is no way to ask "how many explorations actually produce an
    # interest", which is the only number that matters here.
    _events.log("interest_saved_from_exploration",
                 {"message": f"exploration saved an interest: {topic}",
                  "chat_id": chat_id, "topic": topic, "turns": session.get("turns", 0)})
    return reply


def execute_drop(chat_id: int, topic: str, session: dict) -> str:
    """The drop_interest counterpart to execute_save above -- same
    reasoning, same single-code-path rule."""
    remaining = subscriber_ops.remove_interest(chat_id, topic)
    session.setdefault("dropped", []).append(topic)
    if not remaining:
        return f"Removed {topic}. They now follow nothing."
    return f"Removed {topic}. They now follow: " + ", ".join(remaining)


def execute_redefine(chat_id: int, topic: str, definition: str, session: dict) -> str:
    """The save_definition counterpart to execute_save/execute_drop above
    -- same single-code-path rule, both the tool below and bot.py's
    deterministic confirmation gate call this and only this.

    Writes to the SUBSCRIBER's own override tier
    (interest_cache_ops.set_subscriber_interest_definition), never the
    shared/global one -- a personal refinement must never change what
    OTHER subscribers following the same topic word receive. See
    docs/plans/interest-definition-plan.md."""
    interest_cache_ops.set_subscriber_interest_definition(chat_id, topic, definition)
    session.setdefault("redefined", []).append(topic)
    # Mirrors interest_saved_from_exploration -- the outcome event that
    # says this half of the feature did something real.
    _events.log("interest_definition_redefined",
                 {"message": f"exploration redefined {topic}'s retrieval definition",
                  "chat_id": chat_id, "topic": topic, "turns": session.get("turns", 0)})
    return f'Updated how "{topic}" is defined for you -- future pushes for it use this.'


@tool
def propose_interest(topic: str, definition: str, runtime: ToolRuntime) -> str:
    """Call this to propose ADDING a new interest -- in the SAME turn you
    tell the subscriber which specific topic you are about to add and ask
    them to confirm, BEFORE they have answered. This is the only way to
    add an interest; there is no shortcut that skips grounding, even when
    they named the topic outright.

    ALSO runs the preview immediately and returns which real cached
    articles `definition` would surface right now, exactly like
    propose_definition -- read the preview before describing the change;
    if it surfaces nothing relevant, do not propose adding this topic,
    say so and ask for a different one instead (same AAOI-avoidance rule
    everywhere else here). Records the proposal so the system can
    complete it reliably even in a turn where you don't get the chance to
    call save_interest yourself -- it does not change anything by itself,
    so calling it is always safe.

    Exists in this shape because of a real 2026-09-08 incident: a model
    once told a subscriber an interest was added, in its own confident
    prose, without ever calling a save tool -- see this module's own
    docstring, findings 4 and 6. The fix moved the actual write out of
    the model's hands: once this proposal is recorded, bot.py classifies
    the subscriber's NEXT reply as affirm/decline/unclear itself
    (classify_confirmation below) and, on affirm, calls execute_save
    directly -- the write no longer depends on you remembering to call a
    tool at the right moment."""
    preview = _relevant_cached_articles(runtime.context.get("embedder"), definition)[:MAX_EXAMPLES]
    _session(runtime)["pending_proposal"] = {"topic": topic, "action": "add", "definition": definition}
    if not preview:
        return (f'This definition would currently surface NOTHING relevant for "{topic}". '
                "Proposal recorded, but do not present adding this topic as a good option -- "
                "tell the subscriber it has no coverage and ask for a different direction.")
    lines = [f'This definition would currently surface, for "{topic}":']
    lines.extend(_format_article_lines(preview))
    return "\n".join(lines)


@tool
def propose_remove(topic: str, runtime: ToolRuntime) -> str:
    """Call this in the SAME turn you tell the subscriber which topic you
    are about to remove and ask them to confirm -- BEFORE they have
    answered. No definition or preview needed for a removal. Records the
    proposal so the system can complete it reliably even in a turn where
    you don't get the chance to call drop_interest yourself -- it does
    not change anything by itself, so calling it is always safe. Same
    mechanism as propose_interest, see its docstring for why this exists."""
    _session(runtime)["pending_proposal"] = {"topic": topic, "action": "remove"}
    return f"Removal proposal recorded ({topic}). Now ask the subscriber to confirm in your reply."


@tool
def save_interest(topic: str, definition: str, runtime: ToolRuntime) -> str:
    """Add a topic to this subscriber's interests, together with the
    definition they confirmed. Only call this AFTER they have confirmed
    the specific topic AND definition you previewed via propose_interest
    -- normally bot.py's own confirmation gate completes this for you, so
    you should rarely need to call this directly; it remains available
    for the case where a message already contains unambiguous
    confirmation in one go."""
    return execute_save(
        runtime.context["chat_id"], topic, definition, runtime.context.get("guard_model"), _session(runtime))


@tool
def drop_interest(topic: str, runtime: ToolRuntime) -> str:
    """Remove a topic this subscriber no longer wants. Only call this
    after they have confirmed removing it -- same caveat as save_interest
    above: bot.py's confirmation gate normally handles this for you."""
    return execute_drop(runtime.context["chat_id"], topic, _session(runtime))


@tool
def set_language(language: str, runtime: ToolRuntime) -> str:
    """Set this subscriber's reply language from now on. Call this
    directly, immediately, with no proposal or confirmation step -- it is
    low-stakes and instantly reversible, unlike adding or removing an
    interest, so it does not need the same safety net. Works at any point
    in this conversation."""
    subscriber_ops.set_language(runtime.context["chat_id"], language)
    return f"Done -- replying in {language} from now on."


@tool
def search_news(query: str, runtime: ToolRuntime) -> str:
    """Search the already-ingested news cache for `query` and return a
    ready-to-send trend report -- a one-off lookup, not a subscription;
    it does not add or change any of the subscriber's interests. Use this
    for a self-contained news question ("what's new with X", "any news on
    Y") -- find_example_articles is the tool for narrowing down what to
    FOLLOW, this is for answering a question right now.

    Pass a query that stands on its own. No conversation history reaches
    the search underneath this call -- if the subscriber's question
    depends on something said earlier ("what about the other one?"),
    resolve that yourself first (you hold the conversation; nothing
    downstream of this tool does) and call this with the resolved,
    self-contained topic. See docs/plans/front-door-agent-plan.md for why
    that boundary exists -- letting conversation reach past this point is
    exactly the mistake that produced a real incident."""
    ctx = runtime.context
    return agent.search_news(
        ctx["chat_id"], query, [], ctx.get("model"), ctx.get("guard_model"), ctx.get("embedder"))


@tool
def start_push(interval_hours: int | None = None, *, runtime: ToolRuntime) -> str:
    """Turn on the subscriber's periodic news-push digest -- optionally
    at a stated interval in hours (omit to leave their current interval,
    or the default, unchanged). Call this directly, immediately, no
    proposal or confirmation step -- same reasoning as set_language: a
    low-stakes, instantly reversible setting, not data that could leave
    them following something with no coverage."""
    return agent.enable_push(runtime.context["chat_id"], interval_hours)


@tool
def stop_push(runtime: ToolRuntime) -> str:
    """Turn off the subscriber's periodic news-push digest. Call this
    directly, same reasoning as start_push -- no proposal needed."""
    return agent.disable_push(runtime.context["chat_id"])


@tool
def propose_definition(topic: str, definition: str, runtime: ToolRuntime) -> str:
    """Call this to propose a NEW retrieval definition for one of the
    subscriber's existing interests -- in the SAME turn you tell them
    what direction you're proposing and ask them to confirm. Same
    mechanism as propose_interest, same reason (see this module's own
    docstring on the 2026-09-08 incident): the write is deterministic
    once this is recorded, so it doesn't depend on you remembering to
    call save_definition yourself.

    ALSO runs the preview immediately, exactly like propose_interest does
    -- returns which real cached articles `definition` would surface
    right now. Definition effects are measured to be strongly counter-
    intuitive (docs/analysis/retrieval-quality-measurements.md finding
    5), so the preview is baked into this call rather than left to a
    separate step you could skip. Read the preview before describing the
    change to the subscriber; if it surfaces nothing relevant, do not
    propose this definition -- try a different direction instead."""
    preview = _relevant_cached_articles(runtime.context.get("embedder"), definition)[:MAX_EXAMPLES]
    _session(runtime)["pending_proposal"] = {"topic": topic, "action": "redefine", "definition": definition}
    if not preview:
        return (f'This definition would currently surface NOTHING relevant for "{topic}". '
                "Proposal recorded, but do not present this to the subscriber as a good option -- "
                "try a different direction instead.")
    lines = [f'This definition would currently surface, for "{topic}":']
    lines.extend(_format_article_lines(preview))
    return "\n".join(lines)


@tool
def save_definition(topic: str, definition: str, runtime: ToolRuntime) -> str:
    """Save a new retrieval definition for one of the subscriber's
    existing interests. Only call this AFTER they have confirmed the
    specific definition you previewed via propose_definition -- normally
    bot.py's own confirmation gate completes this for you, so you should
    rarely need to call this directly; same caveat as save_interest."""
    return execute_redefine(
        runtime.context["chat_id"], topic, definition, _session(runtime))


@tool
def end_exploration(reason: str, runtime: ToolRuntime) -> str:
    """Call when this conversation is finished -- the subscriber is
    satisfied, has said they are done, or is going in circles without
    converging. `reason` is for the log, not shown to them."""
    session = _session(runtime)
    session["done"] = True
    session["end_reason"] = reason
    # The counterpart to interest_saved_from_exploration above: this fires
    # on every model-ended exploration, including the ones that saved
    # nothing. The two together are what make "did narrowing down work"
    # answerable at all.
    _events.log("interest_exploration_ended",
                 {"message": f"exploration ended: {reason}",
                  "chat_id": runtime.context.get("chat_id"), "reason": reason,
                  "turns": session.get("turns", 0),
                  "saved_count": len(session.get("saved", [])),
                  "dropped_count": len(session.get("dropped", []))})
    return "Exploration marked finished. Give the subscriber a short closing reply."


TOOLS = [
    find_example_articles, list_current_interests, show_definition,
    propose_interest, save_interest, propose_remove, drop_interest,
    propose_definition, save_definition, set_language, search_news,
    start_push, stop_push, end_exploration,
]


class _ConfirmationCheck(BaseModel):
    reasoning: str
    verdict: Literal["affirm", "decline", "unclear"]


_CONFIRMATION_PROMPT = (
    "A conversational assistant proposed adding, removing, or redefining "
    "ONE specific interest, and its LATEST message to the subscriber "
    "(shown below, verbatim) is what the subscriber's reply actually "
    "answers -- not the original proposal in isolation. Classify the "
    "subscriber's reply, in whatever language it's written, AS AN ANSWER "
    "TO THAT LATEST MESSAGE:\n"
    "- affirm: the latest message IS still actively asking the subscriber "
    "to confirm the change, AND they agree (e.g. \"yes\", \"sure\", \"go "
    "ahead\", \"是的\", \"好\", \"sí\", \"confirm\") -- including a short "
    "reply that ONLY confirms.\n"
    "- decline: they say no, clearly want something different instead, OR "
    "the latest message no longer reads as a live request for "
    "confirmation at all -- for example it already says the change won't "
    "be made, or it moved on to something else entirely. In that second "
    "case, classify decline regardless of how affirmative the reply "
    "sounds: there is no live question left for it to confirm.\n"
    "- unclear: anything else -- a new question, a change of direction, "
    "ambiguous phrasing, or a reply that doesn't clearly do either."
)


def classify_confirmation(
    guard_model, user_text: str, last_assistant_reply: str = "",
) -> Literal["affirm", "decline", "unclear"]:
    """Whether a reply to a propose_interest confirmation question
    agrees, declines, or is unclear. Deliberately NOT a reading of the
    model's own subsequent prose -- a model composing text that CLAIMS a
    change happened is exactly the failure this whole mechanism exists to
    route around (see propose_interest's docstring). This is instead a
    small, single-purpose, bounded classification call -- the same shape
    and reliability class as guardrails.classify_message's router, not a
    new open-ended loop. Fails open to "unclear", which is always safe:
    the caller simply falls through to the normal agent turn, exactly as
    if no proposal were pending.

    `last_assistant_reply` anchors the classification to what the
    subscriber is ACTUALLY replying to, rather than blindly re-reading a
    stale pending_proposal. Found live 2026-09-10: a model can run
    propose_definition (recording a proposal), look at its own preview,
    decide out loud that it's a no-op, and tell the subscriber it won't
    save it -- all in prose, with nothing to clear the proposal it just
    recorded. A later, unrelated affirmative reply ("yes" answering a
    completely different question) would otherwise still bind to that
    disowned proposal and get saved anyway. Passing the assistant's own
    latest message lets the classifier see whether it's still a live
    question at all, not just whether the reply sounds affirmative."""
    if guard_model is None:
        return "unclear"
    try:
        structured = guard_model.with_structured_output(_ConfirmationCheck, method="function_calling")
        result = structured.invoke([
            {"role": "system", "content": _CONFIRMATION_PROMPT},
            {"role": "user", "content": (
                f"Assistant's latest message:\n{last_assistant_reply}\n\n"
                f"Subscriber's reply:\n{user_text}"
            )},
        ])
        if result is None:
            return "unclear"
        return result.verdict
    except Exception as exc:
        _events.log("confirmation_check_failed", {"message": "pending-proposal confirmation check failed"},
                     level=Level.WARN, exc=exc)
        return "unclear"


def _compose_prompt(request) -> str:
    """Layer 1 identity + this feature's own method instructions + the
    subscriber's current interests. Same layering idea as agent.py's own
    _compose_prompt, with this module's job in place of news_query's."""
    context = request.runtime.context or {}
    parts = [agent.LAYER1_IDENTITY, _SYSTEM_PROMPT]
    chat_id = context.get("chat_id")
    if chat_id is not None:
        interests = subscriber_ops.get_interests(chat_id)
        parts.append(
            f"This subscriber currently follows: {', '.join(interests)}."
            if interests else
            "This subscriber follows nothing yet -- this is a cold start."
        )
        language = subscriber_ops.get_language(chat_id)
        if language:
            parts.append(
                f"Write your ENTIRE reply in {language}, regardless of what "
                "language their message is written in."
            )
        session = context.get("session") or {}
        pending = session.get("pending_proposal")
        if pending:
            # Reached only when bot.py's own confirmation classifier
            # returned "unclear" for the subscriber's last reply (an
            # "affirm" is handled deterministically before the model ever
            # runs -- see propose_interest's docstring). Surfacing it here
            # is defense in depth: if their reply really was a clear yes
            # that the classifier missed, the model still has a chance to
            # notice and call the matching save tool itself.
            action = pending["action"]
            tool_name = {"add": "save_interest", "remove": "drop_interest", "redefine": "save_definition"}[action]
            what = (f"redefine \"{pending['topic']}\"" if action == "redefine"
                    else f"{action} \"{pending['topic']}\"")
            # "add" and "redefine" both carry a definition now (propose_interest
            # bakes one in exactly like propose_definition does); "remove" never has one.
            call = (f"{tool_name}(\"{pending['topic']}\", \"{pending['definition']}\")"
                    if action in ("add", "redefine") else f"{tool_name} yourself")
            parts.append(
                f"You previously proposed to {what} and are waiting on "
                f"their answer. If their latest message actually confirms "
                f"it, call {call} now. If they declined or want something "
                "else, drop this proposal and continue from what they said."
            )
    return "\n\n".join(parts)


# Same split agent.py uses: the plain function stays directly callable
# (and testable) while the decorated wrapper is what create_agent gets.
@dynamic_prompt
def compose_prompt(request):
    return _compose_prompt(request)


def run_turn(chat_id: int, user_text: str, history: list, session: dict,
             model, guard_model=None, embedder=None) -> tuple[str, bool]:
    """One exchange of an ongoing exploration. Returns (reply, done).

    `session` is the caller's own mutable per-conversation dict (see
    bot.py's interest_sessions); tools write `done`/`saved`/`dropped`
    into it, which is how the caller learns the conversation finished
    without having to parse the agent's message transcript.

    `done` here is only the MODEL's signal (it called end_exploration).
    The caller enforces its own hard MAX_TURNS ceiling separately --
    that guarantee must not depend on the model choosing to stop."""
    built = agent.build_agent(model, tools=TOOLS, middleware=[compose_prompt])
    messages = history + [{"role": "user", "content": user_text}]
    try:
        result = agent.run_agent(
            built, messages,
            context={"chat_id": chat_id, "session": session, "model": model,
                     "guard_model": guard_model, "embedder": embedder},
            recursion_limit=MAX_STEPS_PER_TURN,
        )
    except GraphRecursionError:
        # Hitting the step ceiling is a bounded outcome, not a crash, so
        # it gets the same treatment as MAX_TURNS: end the exploration
        # and say something honest. Without this the caller's generic
        # handler sends LangGraph's own error text -- "Recursion limit of
        # N reached... visit https://docs.langchain.com/..." -- to a
        # Telegram subscriber verbatim. Found live 2026-09-08 by asking
        # about a topic the cache had no coverage for.
        session["done"] = True
        _events.log("interest_turn_out_of_steps",
                     {"message": "one exploration turn hit the step ceiling",
                      "chat_id": chat_id, "max_steps": MAX_STEPS_PER_TURN},
                     level=Level.WARN)
        return out_of_steps_message(), True
    return result[-1].content, bool(session.get("done"))


def out_of_steps_message() -> str:
    """What the subscriber gets when ONE turn hits MAX_STEPS_PER_TURN.
    Distinct from out_of_turns_message below: that one means the
    conversation went nowhere, this one means a single turn spent itself
    searching. Measured cause, 2026-09-08: a topic with no coverage at
    all, which the model kept rephrasing. So this says that plainly
    rather than blaming the conversation."""
    return (
        "I searched a few different ways and couldn't find anything in my feed "
        "on that — it may just not be something my sources cover. Try a "
        "different area, or name a company or topic directly (for example: "
        "<b>add Nvidia to my interests</b>) and we can go from there."
    )


def out_of_turns_message() -> str:
    """What the subscriber gets when the hard MAX_TURNS ceiling stops an
    exploration that never converged. Deliberately honest that this
    didn't work, with a concrete way forward -- the alternative (looping
    silently until they give up) wastes their time and teaches them the
    bot doesn't listen."""
    return (
        "I don't think I'm helping you narrow this down — we've gone back and "
        "forth a fair bit without landing on something. Rather than keep "
        "guessing, it's probably faster if you name a company or topic "
        "directly (for example: <b>add Nvidia to my interests</b>), and we can "
        "adjust from there."
    )

"""
Jev-based relatedness/interestingness filter, applied to the already
embedding-filtered candidate pool news_embed.filter_by_relevance
produces -- that function's own docstring calls itself "a blunt tool,
not fine-grained relevance judgment" and says "the caller's own
downstream judgment... still does the precision work." This module is
that downstream judgment: for each candidate, ask Jev how related it is
to the topic (a graded 0-5 score, not a yes/no) and how interesting it
is, drop the ones that aren't genuinely related, and rank what's left.

CRITICAL, verified live 2026-09-24 before this shipped: Jev's own
documented pattern for scoring a batch of records -- one shared
`state.articles` list, each question referencing "article #i in the
list" -- has a severe positional bias unrelated to batch size. The
IDENTICAL article scored 0.95 (clearly related) at position 0 and
0.14-0.39 (clearly NOT related) at any later position, even in a
10-item batch. The fix used here: each question embeds that article's
own title/summary text directly in its own `instructions` string; no
shared indexed list in `state` at all. Confirmed clean up to 200
articles/200 questions in one call (~0.3s, ~$0.0004): known on-topic
articles scored 0.69-0.95 regardless of position, known off-topic
articles scored 0.01 regardless of position. Latency alone stays flat
to ~500 questions (a 800-question call gets a hard 400 from the
endpoint), but per-item ACCURACY was only ever verified up to 200 --
raise search.jev_max_articles/push.jev_max_articles past that only
after re-running the same kind of live position/content check, not on
the strength of the latency numbers alone.

NEVER revert to the shared-list-plus-index pattern to save on prompt
size -- it silently produces wrong answers for every item after the
first, not a partial degradation, and Jev's own docs recommend exactly
the pattern that's broken here.

`definition`, unlike an article's own text, IS shared context common to
every question in the batch (not per-item, indexed content), so it goes
in `state` alongside `topic` rather than being repeated in every
question's instructions -- verified live 2026-09-25 with a deliberately
ambiguous bare topic ("agents") plus a disambiguating definition: an
article about a browser-automation AI agent scored 4.95/5, while
real-estate- and insurance-agent articles scored 0.09/0.14 despite the
literal word match -- state.definition genuinely gets used, this isn't
the same failure mode as the indexed-article-list bug above (that one
was about resolving PER-ITEM content by index; a single shared string in
state was already the working pattern `topic` itself relied on).
`definition` is optional -- callers that don't have one (or a search/push
topic that's never been expanded) still work, just without the extra
disambiguation.
"""

from telemetry import EventLogger, get_event_logger
from telemetry_providers import Level

import jev_client

_events: EventLogger = get_event_logger("argus.news_jev_filter")

_RELATED_CRITERIA = [
    "Not related at all",
    "Barely related",
    "Loosely related",
    "Somewhat related",
    "Related",
    "Directly and centrally related",
]
_RELATED_KEEP_THRESHOLD = 3  # keep "Somewhat related" (3) or better

_INTERESTING_CRITERIA = ["Not interesting", "Mildly interesting", "Interesting", "Very interesting"]
_EYE_OPENER_MIN_INTERESTING = len(_INTERESTING_CRITERIA) - 1  # only the top tier ("Very interesting") qualifies


def _headline(article: dict) -> str:
    title = article.get("title") or ""
    summary = article.get("summary") or ""
    return f"{title} -- {summary}" if summary else title


def score_and_rank(articles: list[dict], topic: str, jev_api_key: str, max_articles: int,
                   definition: str | None = None) -> list[dict]:
    """Cap `articles` to `max_articles` (the caller's own relevance-keep
    pool may be wider than what one Jev call should see -- see
    search.jev_max_articles/push.jev_max_articles), then ask Jev, per
    article: how related is it to `topic` (0-5, graded, not yes/no), and
    how interesting is it. `definition`, when the caller has one (the
    same cached, LLM-generated topic definition the embedding filter
    itself prefers over the bare topic string -- see
    interest_cache_ops.resolve_interest_definition /
    news_push._resolve_query_text), goes in shared state so Jev can
    disambiguate a topic word that reads differently out of context
    (e.g. "agents" -- software agents vs. real-estate agents).

    Selection: articles scoring _RELATED_KEEP_THRESHOLD (3, "Somewhat
    related") or above are kept, ranked by (related desc, interesting
    desc), ties keeping the incoming order (already similarity-ranked by
    the caller's embedding filter). On top of that, AT MOST ONE article
    that scored below the threshold is let through anyway -- an "eye-
    opener" exception -- but only if it's genuinely the top interesting
    tier (_EYE_OPENER_MIN_INTERESTING); there is no forced exception if
    nothing rejected clears that bar. This is deliberately a single,
    high bar exception, not a second-class inclusion path -- it exists
    so something genuinely surprising isn't lost just because it scores
    low on strict topic-relatedness, the same spirit as
    news_push._pick_novelty_extra's "at most one" bonus slot (a
    separate, pre-existing mechanism this doesn't replace -- see
    news_push.py's own call site for how the two coexist).

    Fails open to the (capped, unfiltered, unranked) input on any Jev
    error -- same fail-open contract as guardrails.py's own Jev-backed
    layers; a broken Jev call must degrade to "no Jev step", not break
    the caller."""
    candidates = articles[:max_articles]
    if not candidates:
        return candidates

    questions = {}
    for i, article in enumerate(candidates):
        headline = _headline(article)
        questions[f"related_{i}"] = {
            "type": "score",
            "instructions": f"Considering the topic (and its definition, if given in state), "
                             f"how related is the news item \"{headline}\" to it?",
            "criteria": _RELATED_CRITERIA,
        }
        questions[f"interesting_{i}"] = {
            "type": "score",
            "instructions": f"How interesting/noteworthy is the news item \"{headline}\" "
                             f"to a reader who follows \"{topic}\"?",
            "criteria": _INTERESTING_CRITERIA,
        }
    state = {"topic": topic}
    if definition:
        state["definition"] = definition

    try:
        answers = jev_client.ask(state, questions, jev_api_key, timeout=30.0)
        kept, rejected = [], []
        for i, article in enumerate(candidates):
            related = answers[f"related_{i}"]["score"]
            interesting = answers[f"interesting_{i}"]["score"]
            if related >= _RELATED_KEEP_THRESHOLD:
                kept.append((related, interesting, i, article))
            else:
                rejected.append((interesting, i, article))
    except Exception as exc:
        _events.log("news_jev_filter_failed", "Jev article scoring FAILED, returning candidates unfiltered",
                     level=Level.ERROR, exc=exc)
        return candidates

    kept.sort(key=lambda t: (-t[0], -t[1], t[2]))
    result = [article for _, _, _, article in kept]
    if rejected:
        best_interesting, _, best_article = max(rejected, key=lambda t: (t[0], -t[1]))
        if best_interesting >= _EYE_OPENER_MIN_INTERESTING:
            result.append(best_article)
    return result

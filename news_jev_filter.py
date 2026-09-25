"""
Jev-based relatedness/interestingness filter, applied to the already
embedding-filtered candidate pool news_embed.filter_by_relevance
produces -- that function's own docstring calls itself "a blunt tool,
not fine-grained relevance judgment" and says "the caller's own
downstream judgment... still does the precision work." This module is
that downstream judgment: for each candidate, ask Jev whether it's
genuinely related to the topic and how interesting it is, drop the
unrelated ones, and rank what's left by interestingness.

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
raise search.jev_max_articles past that only after re-running the same
kind of live position/content check, not on the strength of the
latency numbers alone.

NEVER revert to the shared-list-plus-index pattern to save on prompt
size -- it silently produces wrong answers for every item after the
first, not a partial degradation, and Jev's own docs recommend exactly
the pattern that's broken here.
"""

from telemetry import EventLogger, get_event_logger
from telemetry_providers import Level

import jev_client

_events: EventLogger = get_event_logger("argus.news_jev_filter")

_NOUL_TRUE_THRESHOLD = 0.5

_INTERESTING_CRITERIA = ["Not interesting", "Mildly interesting", "Interesting", "Very interesting"]


def _headline(article: dict) -> str:
    title = article.get("title") or ""
    summary = article.get("summary") or ""
    return f"{title} -- {summary}" if summary else title


def score_and_rank(articles: list[dict], topic: str, jev_api_key: str, max_articles: int) -> list[dict]:
    """Cap `articles` to `max_articles` (the caller's own relevance-keep
    pool may be wider than what one Jev call should see -- see
    search.jev_max_articles), then ask Jev, per article: is it
    genuinely related to `topic`, and how interesting is it. Returns
    the related ones only, ranked most-to-least interesting; ties keep
    the incoming order (already similarity-ranked by the caller's
    embedding filter, see news_embed.filter_by_relevance).

    Fails open to the (capped, unfiltered, unranked) input on any Jev
    error -- same fail-open contract as guardrails.py's own Jev-backed
    layers; a broken Jev call must degrade to "no Jev step", not break
    search_news."""
    candidates = articles[:max_articles]
    if not candidates:
        return candidates

    questions = {}
    for i, article in enumerate(candidates):
        headline = _headline(article)
        questions[f"related_{i}"] = {
            "type": "noul",
            "instructions": f"Is the news item \"{headline}\" genuinely related to the topic \"{topic}\"?",
        }
        questions[f"interesting_{i}"] = {
            "type": "score",
            "instructions": f"How interesting/noteworthy is the news item \"{headline}\" "
                             f"to a reader who follows \"{topic}\"?",
            "criteria": _INTERESTING_CRITERIA,
        }

    try:
        answers = jev_client.ask({"topic": topic}, questions, jev_api_key, timeout=30.0)
        scored = []
        for i, article in enumerate(candidates):
            if answers[f"related_{i}"]["noul"] <= _NOUL_TRUE_THRESHOLD:
                continue
            scored.append((answers[f"interesting_{i}"]["score"], i, article))
    except Exception as exc:
        _events.log("news_jev_filter_failed", "Jev article scoring FAILED, returning candidates unfiltered",
                     level=Level.ERROR, exc=exc)
        return candidates

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [article for _, _, article in scored]

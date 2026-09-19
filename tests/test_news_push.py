import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from langchain_core.messages import AIMessage

import news_cache
import news_embed
import news_push
import news_sources
import category_ops
import interest_cache_ops
import push_outcome_ops
import storage
import subscriber_ops
from sqlalchemy import text
from tests.fakes import FakeEmbedder, FakeToolCallingModel


# Selection keys on fetched_at (when we downloaded it), not published_dt --
# see news_push.select_candidate_articles. fetched_at defaults to published_dt
# so the many tests that only care about ordering stay readable; tests about
# the delay case set them apart explicitly.
def _article(link, published_dt=None, title="Some title", source="TestSource", categories=None,
             source_key="test", fetched_at=None, embedding=None):
    return {
        "title": title,
        "link": link,
        "source": source,
        "source_key": source_key,
        "summary": None,
        "published_dt": published_dt,
        "fetched_at": fetched_at if fetched_at is not None else published_dt,
        "categories": categories or [],
        "embedding": embedding,
    }


# A fixed "now" for the age guard (MAX_ARTICLE_AGE_HOURS). Tests that use
# dated fixtures pass this so they don't start failing as the wall clock
# moves past the guard -- the same reason run_push_cycle takes `now`.
NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


@pytest.fixture
def recorded_outcomes(monkeypatch):
    """Captures every news_push._record(...) call (newest first per
    chat_id) by wrapping it, not replacing it -- the real function still
    runs, so span emission and the push_consecutive_failures reset still
    happen. Replaces the pre-2026-09-04 push_outcome_ops.recent_outcomes_for
    query now that a push cycle's outcome is a span/print only, not a row
    in a queryable table (see git history)."""
    calls = []
    original = news_push._record

    def spy(chat_id, outcome, message, now, detail=None):
        calls.append((chat_id, outcome, detail))
        return original(chat_id, outcome, message, now, detail=detail)

    monkeypatch.setattr(news_push, "_record", spy)

    def outcomes_for(chat_id):
        return [outcome for cid, outcome, _detail in reversed(calls) if cid == chat_id]

    def detail_for(chat_id):
        """Most recent call's `detail` for this chat_id -- for the one test
        that needs to check what got folded into it, not just the outcome."""
        return next(detail for cid, _outcome, detail in reversed(calls) if cid == chat_id)

    outcomes_for.detail_for = detail_for
    return outcomes_for


# --- select_candidate_articles (stage 1: category filter) -----------------


def test_select_candidate_articles_ignores_dates_when_deciding_already_seen(isolated_subscribers_db):
    """A date ranks; it never filters. Both of these were published well
    before the subscriber's last push, and neither has been sent -- so both
    must come through. This is the GNews case: a source publishing ~12h
    behind was excluded outright by the old `published_dt <= since` test,
    stranding 227 cached articles that could never reach a digest."""
    since = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    old_but_unsent = _article("https://example.com/a", published_dt=NOW - timedelta(hours=30), categories=["AI"])
    older_but_unsent = _article("https://example.com/b", published_dt=NOW - timedelta(hours=40), categories=["AI"])

    result = news_push.select_candidate_articles(
        [old_but_unsent, older_but_unsent], ["AI"], {"AI": ["AI"]}, since, set(), now=NOW
    )

    assert [a["link"] for a in result] == ["https://example.com/a", "https://example.com/b"]


def test_select_candidate_articles_keeps_offering_an_unsent_article(isolated_subscribers_db):
    """An article that lost the max_per_topic cut is not "seen" -- it stays a
    candidate until it is actually sent or ages out of the cache. Under the
    previous timestamp-based filter it would have been excluded forever,
    unsent and unrecorded."""
    articles = [
        _article(f"https://example.com/{i}", published_dt=NOW - timedelta(hours=i), categories=["AI"])
        for i in range(1, 6)
    ]

    first = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(), max_per_topic=2, now=NOW
    )
    assert [a["link"] for a in first] == ["https://example.com/1", "https://example.com/2"]

    # next cycle: only what was actually sent is excluded
    second = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, {a["link"] for a in first}, max_per_topic=2, now=NOW
    )
    assert [a["link"] for a in second] == ["https://example.com/3", "https://example.com/4"]


def test_select_candidate_articles_drops_articles_older_than_the_age_guard(isolated_subscribers_db):
    """Guards the fetched_at rule against genuinely ancient content: Perigon's
    one successful fetch returned 50 articles whose newest was over a year
    old (security-plan.md finding 21). Freshly downloaded, but not news."""
    ancient = _article(
        "https://example.com/ancient",
        published_dt=NOW - timedelta(days=400),
        fetched_at=NOW,
        categories=["AI"],
    )
    recent = _article(
        "https://example.com/recent", published_dt=NOW - timedelta(hours=2), fetched_at=NOW, categories=["AI"]
    )

    result = news_push.select_candidate_articles(
        [ancient, recent], ["AI"], {"AI": ["AI"]}, None, set(), now=NOW
    )

    assert [a["link"] for a in result] == ["https://example.com/recent"]


def test_select_candidate_articles_keeps_articles_with_unparseable_published_dt(isolated_subscribers_db):
    """Fails open, same instinct as the rest of the pipeline -- an article
    whose date didn't parse isn't assumed ancient."""
    undated = _article("https://example.com/undated", published_dt=None, fetched_at=NOW, categories=["AI"])

    result = news_push.select_candidate_articles(
        [undated], ["AI"], {"AI": ["AI"]}, None, set(), now=NOW
    )

    assert [a["link"] for a in result] == ["https://example.com/undated"]


def test_select_candidate_articles_never_resends_a_pushed_link(isolated_subscribers_db):
    """already_pushed_links is now checked unconditionally, not only when the
    date is unparseable -- so it guards every path."""
    already_sent = _article(
        "https://example.com/sent", published_dt=NOW - timedelta(hours=1), fetched_at=NOW, categories=["AI"]
    )

    result = news_push.select_candidate_articles(
        [already_sent], ["AI"], {"AI": ["AI"]}, None, {"https://example.com/sent"}, now=NOW
    )

    assert result == []


def test_select_candidate_articles_falls_back_to_pushed_links_for_unparsed_dates(isolated_subscribers_db):
    seen = _article("https://example.com/seen", published_dt=None, categories=["AI"])
    unseen = _article("https://example.com/unseen", published_dt=None, categories=["AI"])

    result = news_push.select_candidate_articles(
        [seen, unseen], ["AI"], {"AI": ["AI"]}, None, {"https://example.com/seen"}
    )

    assert [a["link"] for a in result] == ["https://example.com/unseen"]


def test_select_candidate_articles_dedupes_across_topics(isolated_subscribers_db):
    article = _article("https://example.com/shared", published_dt=NOW - timedelta(hours=1), categories=["AI"])

    result = news_push.select_candidate_articles(
        [article], ["AI", "robotics"], {"AI": ["AI"], "robotics": ["AI"]}, None, set(), now=NOW
    )

    assert len(result) == 1


def test_select_candidate_articles_topic_with_no_category_mapping_gets_nothing(isolated_subscribers_db):
    """A topic with no cached category mapping (classifier miss) used to be
    treated as "unrestricted" -- matching any article regardless of its
    categories. That fail-open design let a completely off-topic article
    reach a subscriber (the 2026-08-27 Witcher 3 incident: interest
    "robotics" classified into zero of 28 taxonomy categories, and the
    resulting "unrestricted" match let a gaming article through). A topic
    that can't be matched against anything real is now skipped entirely --
    same shape as "nothing new since last time": no candidates, no
    message, no slot consumed."""
    article = _article("https://example.com/a", categories=["Policy"])

    result = news_push.select_candidate_articles([article], ["AAOI"], {}, None, set())

    assert result == []


def test_select_candidate_articles_explicit_empty_mapping_also_gets_nothing(isolated_subscribers_db):
    """Pins the other way an interest can land with no categories to match
    against: a topic present in the dict but explicitly mapped to [] --
    e.g. an interest that once mapped to a category later retired and
    never re-mapped (see storage/sqlite/category.py's migrate_split_policy
    docstring and test_category_ops.test_policy_split_does_not_touch_pre_existing_interest_category_mappings
    -- no code path currently produces this for Policy specifically, but
    nothing would stop it for a category retired in the future). Both this
    and the "topic absent from the dict" case above hit the same
    `if not topic_cats: continue` branch and are therefore
    indistinguishable to this function -- and both now get zero
    candidates rather than being silently unrestricted."""
    article = _article("https://example.com/a", categories=["Government"])

    result = news_push.select_candidate_articles(
        [article], ["legacy policy watcher"], {"legacy policy watcher": []}, None, set()
    )

    assert result == []


def test_select_candidate_articles_excludes_off_category_article(isolated_subscribers_db):
    # The Nikkei Asia incident this was built to fix: an uncategorized
    # (or off-category) article shouldn't reach a subscriber whose topic
    # DID classify into real categories.
    earthquake = _article("https://example.com/earthquake", categories=[])
    ai_topic_categories = {"AI": ["AI", "Research"]}

    result = news_push.select_candidate_articles([earthquake], ["AI"], ai_topic_categories, None, set())

    assert result == []


def test_select_candidate_articles_includes_overlapping_category_article(isolated_subscribers_db):
    article = _article("https://example.com/a", categories=["AI", "Startups"])
    topic_categories = {"AI": ["AI", "Research"]}

    result = news_push.select_candidate_articles([article], ["AI"], topic_categories, None, set())

    assert len(result) == 1


def test_select_candidate_articles_excludes_restricted_sources_by_default(isolated_subscribers_db):
    article = _article("https://example.com/a", source_key="perigon", categories=["AI"])

    result = news_push.select_candidate_articles([article], ["AI"], {"AI": ["AI"]}, None, set())

    assert result == []


def test_select_candidate_articles_includes_restricted_sources_when_enabled(isolated_subscribers_db):
    article = _article("https://example.com/a", source_key="perigon", categories=["AI"])

    result = news_push.select_candidate_articles(
        [article], ["AI"], {"AI": ["AI"]}, None, set(), include_restricted=True
    )

    assert len(result) == 1


def test_select_candidate_articles_caps_per_topic(isolated_subscribers_db):
    articles = [
        _article(f"https://example.com/{i}", published_dt=NOW - timedelta(hours=8 - i), categories=["AI"])
        for i in range(1, 8)
    ]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(), max_per_topic=3, now=NOW
    )

    assert len(result) == 3
    # newest-first
    assert result[0]["link"] == "https://example.com/7"


# --- near-duplicate collapse (2026-08-25) ----------------------------------

def _embed(text):
    return news_embed.embed_one(FakeEmbedder(), text)


def test_near_duplicate_articles_collapse_to_the_newer_one(isolated_subscribers_db):
    wire_story = _embed("Nvidia launches new GPU architecture")
    articles = [
        _article("https://a.com/1", published_dt=NOW - timedelta(hours=1),
                 title="Nvidia launches new GPU architecture", embedding=wire_story, categories=["AI"]),
        # Same story, a different outlet's syndicated copy -- near-
        # identical embedding, different link, slightly older.
        _article("https://a.com/2", published_dt=NOW - timedelta(hours=2),
                 title="Nvidia launches new GPU architecture (wire)", embedding=wire_story, categories=["AI"]),
    ]

    result = news_push.select_candidate_articles(articles, ["AI"], {"AI": ["AI"]}, None, set(), now=NOW)

    assert [a["link"] for a in result] == ["https://a.com/1"]


def test_articles_below_the_near_duplicate_threshold_both_survive(isolated_subscribers_db):
    articles = [
        _article("https://a.com/1", published_dt=NOW - timedelta(hours=1),
                 title="Nvidia launches new GPU", embedding=_embed("Nvidia launches new GPU"), categories=["AI"]),
        _article("https://a.com/2", published_dt=NOW - timedelta(hours=2),
                 title="Bitcoin price surges", embedding=_embed("Bitcoin price surges"), categories=["AI"]),
    ]

    result = news_push.select_candidate_articles(articles, ["AI"], {"AI": ["AI"]}, None, set(), now=NOW)

    assert {a["link"] for a in result} == {"https://a.com/1", "https://a.com/2"}


def test_an_article_with_no_embedding_is_never_collapsed(isolated_subscribers_db):
    """cosine_similarity(None, x) is -1.0 by construction -- an
    unembeddable article (embedder unavailable, or cached before
    news_embed existed) must never be treated as a duplicate of
    anything, fail-open like every other check in this function."""
    wire_story_embedding = _embed("Nvidia launches new GPU architecture")
    articles = [
        _article("https://a.com/1", published_dt=NOW - timedelta(hours=1),
                 title="Nvidia launches new GPU architecture", embedding=wire_story_embedding, categories=["AI"]),
        _article("https://a.com/2", published_dt=NOW - timedelta(hours=2),
                 title="Nvidia launches new GPU architecture (wire)", embedding=None, categories=["AI"]),
    ]

    result = news_push.select_candidate_articles(articles, ["AI"], {"AI": ["AI"]}, None, set(), now=NOW)

    assert {a["link"] for a in result} == {"https://a.com/1", "https://a.com/2"}


def test_a_near_duplicate_within_one_topic_stays_eligible_for_another(isolated_subscribers_db):
    """Article 1 is Hardware-only, so topic "chips" (Hardware) pools it and
    then dedup-skips Article X (same story, both Hardware AND Markets) as
    a near-duplicate WITHIN that pool. Topic "stocks" (Markets) never
    pools Article 1 at all -- it doesn't match Markets -- so Article X
    reaches topic "stocks" with nothing to be a duplicate of. If the
    dedup skip for topic "chips" had wrongly marked Article X's link as
    globally seen, topic "stocks" would incorrectly lose it too."""
    wire_story = _embed("Nvidia launches new GPU architecture")
    article_1 = _article("https://a.com/1", published_dt=NOW - timedelta(hours=1),
                         title="Nvidia launches new GPU architecture", embedding=wire_story,
                         categories=["Hardware"])
    article_x = _article("https://a.com/x", published_dt=NOW - timedelta(hours=2),
                         title="Nvidia launches new GPU architecture (wire)", embedding=wire_story,
                         categories=["Hardware", "Markets"])
    topic_categories = {"chips": ["Hardware"], "stocks": ["Markets"]}

    result = news_push.select_candidate_articles(
        [article_1, article_x], ["chips", "stocks"], topic_categories, None, set(), now=NOW)

    by_topic = {(a["link"], a["topic"]) for a in result}
    assert ("https://a.com/1", "chips") in by_topic
    assert ("https://a.com/x", "stocks") in by_topic
    # And confirm the dedup actually fired for "chips" -- otherwise this
    # test would pass even with dedup entirely disabled.
    assert ("https://a.com/x", "chips") not in by_topic


# --- query text resolution (2026-08-25) -------------------------------------
# The retrieval query used against news_embed is not always the bare topic
# string -- a cached, generated definition (interest_cache_ops.interest_query_expansions,
# populated once by agent.py's add_one_interest) measurably outranks the
# bare phrase. See news_classify.expand_interest_for_retrieval.

def test_resolve_query_text_uses_the_cached_expansion_when_present(isolated_subscribers_db):
    interest_cache_ops.set_interest_query_expansion("AI coding", "a rich generated definition")
    assert news_push._resolve_query_text(None, "AI coding") == "a rich generated definition"


def test_resolve_query_text_falls_back_to_the_bare_topic_when_nothing_cached(
    isolated_subscribers_db
):
    assert news_push._resolve_query_text(None, "AI coding") == "AI coding"


def test_resolve_query_text_prefers_the_subscribers_own_definition(isolated_subscribers_db):
    """docs/plans/interest-definition-plan.md: a personal refinement must
    outrank the shared default, not the other way around."""
    interest_cache_ops.set_interest_query_expansion("AI coding", "the shared default definition")
    interest_cache_ops.set_subscriber_interest_definition(7, "AI coding", "this subscriber's own definition")
    assert news_push._resolve_query_text(7, "AI coding") == "this subscriber's own definition"


def test_resolve_query_text_with_chat_id_still_falls_back_to_shared_default(isolated_subscribers_db):
    """A subscriber who never refined anything still gets the shared
    default, not the bare topic -- personalisation is additive, not a
    replacement for the existing behavior."""
    interest_cache_ops.set_interest_query_expansion("AI coding", "the shared default definition")
    assert news_push._resolve_query_text(7, "AI coding") == "the shared default definition"


# select_candidate_articles's wiring of _resolve_query_text's output into
# _filter_by_relevance is a one-line pass-through (query_text =
# _resolve_query_text(topic); _filter_by_relevance(raw_pool, embedder,
# query_text)), and both halves are independently tested above/below --
# no separate end-to-end test for that wiring itself. There used to be
# one here for the offbeat gate specifically, retired 2026-08-26 when
# offbeat selection stopped using embeddings/query text at all (see
# "Offbeat selection, take two" in docs/analysis/cluster-measurements.md).


# --- relevance filter (2026-08-25) ------------------------------------------
# The "AI"/"AI Agent"/"AI coding"/"Large Language Model" incident: all four
# map to category ['AI'], so the coarse category filter alone can't tell
# them apart. This is the fine filter that runs after it, using the topic
# STRING as a retrieval query rather than its category.
#
# news_embed.filter_by_relevance itself takes no defaults (each caller's
# keep_fraction/min/max depend on its own pool size and precision/recall
# tradeoff -- see that function's own docstring), so these tests call it
# through this thin wrapper pinned to news_push's own regular-digest
# constants, matching what select_candidate_articles's regular pass
# actually uses.


def _filter_by_relevance(pool, embedder, query_text):
    return news_embed.filter_by_relevance(
        pool, embedder, query_text,
        keep_fraction=news_push.RELEVANCE_KEEP_FRACTION,
        keep_min=news_push.RELEVANCE_KEEP_MIN,
        keep_max=news_push.RELEVANCE_KEEP_MAX,
    )


def test_relevance_filter_keeps_the_top_fraction_and_drops_the_clear_outlier():
    """RELEVANCE_KEEP_MIN=20 needs a pool bigger than 20 to deterministically
    exclude anything at all -- below that, the floor swallows the whole
    pool and nothing is cut, which is deliberate (see that constant's own
    comment: a narrow topic's small pool shouldn't be starved by a
    percentage of a small number). Built with 15 on-topic + 15 off-topic
    (pool=30, so n_kept=20) and, among the off-topic set, one article
    ("vintage car restoration") sharing zero hashed dimensions with
    anything else in the pool, scoring below every tie -- unambiguously
    the single worst-ranked item regardless of how FakeEmbedder's
    zero-similarity ties among the OTHER off-topic articles sort."""
    embedder = FakeEmbedder()
    on_topic = [_article(f"https://a.com/agent{i}", title=t, embedding=_embed(t))
               for i, t in enumerate([
                   "How to self-host your own AI agent",
                   "New framework for building autonomous AI agents",
                   "AI agent startup raises funding round",
                   "Open source AI agent toolkit released today",
                   "AI agent benchmark shows strong results",
                   "Enterprise AI agent adoption grows fast",
                   "AI agent memory architecture explained",
                   "AI agent orchestration platform launches",
                   "Multi agent AI system coordinates tasks",
                   "AI agent framework gets major update",
                   "Building reliable AI agent pipelines",
                   "AI agent marketplace opens to developers",
                   "How AI agents handle long running tasks",
                   "AI agent security best practices guide",
                   "Scaling AI agent deployments in production",
               ])]
    off_topic = [_article(f"https://a.com/off{i}", title=t, embedding=_embed(t))
                for i, t in enumerate([
                    "Completely unrelated gardening tips for spring",
                    "How to bake sourdough bread at home",
                    "Best hiking trails in the Pacific Northwest",
                    "Tips for growing tomatoes in containers",
                    "The history of medieval castle architecture",
                    "How to knit a scarf for beginners",
                    "Local weather patterns this autumn",
                    "A review of the newest coffee shops downtown",
                    "Tips for training a new puppy",
                    "The art of watercolor painting techniques",
                    "How to plan a backyard vegetable garden",
                    "A beginner's guide to birdwatching",
                    "The best board games for family game night",
                    "How to organize a home pantry",
                    "A guide to vintage car restoration",
                ])]
    pool = on_topic + off_topic

    result = _filter_by_relevance(pool, embedder, "AI Agent")

    result_links = {a["link"] for a in result}
    assert {a["link"] for a in on_topic} <= result_links
    assert "https://a.com/off14" not in result_links  # "vintage car restoration"


def test_relevance_filter_is_relative_not_absolute_relevance():
    """The measured reason this can't be a fixed similarity threshold:
    querying the SAME corpus with different topic strings puts
    genuinely-relevant articles at wildly different absolute cosine
    scores (real model2vec, measured this session: "AI Agent" topped
    0.72, "Large Language Model" against the same corpus topped 0.166).
    A rank-based cut has the mirror-image limitation: it has no way to
    recognize "NOTHING in this pool is relevant" and reject accordingly
    -- it always keeps its clamped count regardless of whether anything
    actually clears a meaningful bar of relevance. Demonstrated here with
    a query that shares no real vocabulary with any of the 25 articles:
    the filter still keeps RELEVANCE_KEEP_MIN=20 of them.

    25 articles, not 10 -- needs to clear RELEVANCE_KEEP_MIN=20 for this
    property to be visible at all; a pool smaller than the floor is a
    no-op regardless of the query (a separate, deliberate property,
    tested by test_relevance_filter_falls_back_with_too_few_embedded_articles
    and the offbeat test's small-pool case, not this one)."""
    embedder = FakeEmbedder()
    titles = [
        "Options traders bet on quiet Nvidia earnings reaction",
        "Stability AI raises $76 million in fresh funding",
        "AI hits entry-level jobs for younger workers hardest",
        "Taiwan charges nine people for smuggling AI servers",
        "OpenAI Broadcom custom chip is a winner",
        "Apple debuts its most powerful chip ever",
        "Granite 4.2 LLMs how they are built",
        "Claude Cowork remembers what you told the app",
        "Accel backed startup indexes the web for AI agents",
        "Portable computer is a new local AI agent",
    ] + [f"Generic tech industry news item number {i}" for i in range(15)]
    pool = [_article(f"https://a.com/{i}", title=t, embedding=_embed(t)) for i, t in enumerate(titles)]

    result = _filter_by_relevance(pool, embedder, "completely unrelated gardening tips")

    # >= gate is inclusive, so a tie AT the gate value (common here --
    # most of these titles share zero hashed dimensions with the query
    # and score exactly 0.0) can let MORE than RELEVANCE_KEEP_MIN survive.
    # The property under test is the floor, not an exact count.
    assert len(result) >= 20
    assert len(result) < len(pool), "the filter must still exclude something, not degrade to a no-op"


def test_relevance_filter_caps_at_relevance_keep_max_for_a_large_pool():
    """RELEVANCE_KEEP_MAX=50 needs a pool where 10% of it exceeds 50 (pool
    > 500) to actually engage rather than the percentage alone deciding.
    The raw pool this filter sees is no longer count-capped before
    reaching it (RELEVANCE_SAMPLE_SIZE was removed 2026-08-25), so a
    broad topic's real pool -- 999 "AI"-category articles were measured
    live at once -- can genuinely reach this range."""
    embedder = FakeEmbedder()
    words = ["agent", "framework", "platform", "toolkit", "pipeline", "deployment",
            "architecture", "orchestration", "benchmark", "adoption", "marketplace",
            "security", "scaling", "memory", "workflow", "automation", "integration",
            "monitoring", "reliability", "governance"]
    # 3000, not a few hundred -- at 10% that's 300 uncapped vs. 50 capped,
    # a gap wide enough to survive FakeEmbedder's real tie clusters at
    # scale (verified against the real formula: capped ~208 survivors,
    # uncapped ~370, on this exact fixture). A smaller pool's tie
    # clusters can absorb the capped-vs-uncapped difference entirely and
    # silently defeat this test -- caught live while writing it.
    titles = [f"AI {words[i % len(words)]} update variant {i} released today" for i in range(3000)]
    pool = [_article(f"https://a.com/{i}", title=t, embedding=_embed(t)) for i, t in enumerate(titles)]

    result = _filter_by_relevance(pool, embedder, "AI Agent")

    assert len(result) >= news_push.RELEVANCE_KEEP_MAX
    assert len(result) < 250, "the ceiling must actually be doing something on a pool this large"


def test_candidate_pool_size_matches_relevance_keep_max():
    """Deliberately the SAME value, not independently chosen -- if
    CANDIDATE_POOL_SIZE were smaller, it would silently re-truncate
    whatever _filter_by_relevance already decided to keep, making
    RELEVANCE_KEEP_MAX a dead ceiling. See CANDIDATE_POOL_SIZE's own
    comment."""
    assert news_push.CANDIDATE_POOL_SIZE == news_push.RELEVANCE_KEEP_MAX


def test_relevance_filter_falls_back_to_unfiltered_with_no_embedder():
    pool = [_article("https://a.com/1", title="x", embedding=_embed("x"))]
    assert _filter_by_relevance(pool, None, "AI Agent") == pool


def test_relevance_filter_falls_back_with_too_few_embedded_articles():
    """Only 1 embedded article -- below the floor of 2 needed for a
    meaningful median split."""
    pool = [_article("https://a.com/1", title="x", embedding=_embed("x")),
           _article("https://a.com/2", title="y", embedding=None)]
    assert _filter_by_relevance(pool, FakeEmbedder(), "AI Agent") == pool


def test_relevance_filter_never_excludes_an_article_with_no_embedding():
    """An article that couldn't be embedded (ingested before news_embed
    existed, or the embed call failed for it specifically) has nothing
    for this filter to judge it by -- same fail-open shape as near-
    duplicate collapse and every other embedding-based feature here.
    22 embedded articles, not a handful -- needs to clear
    RELEVANCE_KEEP_MIN=20 so real filtering actually happens (some of the
    22 get excluded); below that floor, EVERYTHING survives regardless of
    embedding status, which wouldn't prove this article's absence of an
    embedding is what's protecting it."""
    embedded_pool = [_article(f"https://a.com/e{i}", title=f"AI agent update number {i}",
                              embedding=_embed(f"AI agent update number {i}"))
                     for i in range(22)]
    unembedded = _article("https://a.com/no-embedding", title="Nvidia earnings today", embedding=None)
    pool = embedded_pool + [unembedded]

    result = _filter_by_relevance(pool, FakeEmbedder(), "AI Agent")

    assert len(result) < len(pool), "real filtering must have happened for this test to prove anything"
    assert unembedded["link"] in {a["link"] for a in result}


def test_relevance_filter_preserves_input_order():
    """25 articles (>= RELEVANCE_KEEP_MIN) so real filtering happens --
    below that floor everything survives trivially and this wouldn't
    prove the surviving subset keeps its original relative order."""
    embedder = FakeEmbedder()
    titles = [
        "How to self-host your own AI agent",
        "New AI agent framework release",
        "Nvidia earnings beat forecast",
        "Stability AI raises funding",
    ] + [f"AI agent update number {i}" for i in range(21)]
    pool = [_article(f"https://a.com/{i}", title=t, embedding=_embed(t)) for i, t in enumerate(titles)]

    result = _filter_by_relevance(pool, embedder, "AI Agent")

    assert len(result) < len(pool), "real filtering must have happened for this test to prove anything"
    result_links = [a["link"] for a in result]
    original_order = [a["link"] for a in pool]
    assert result_links == [link for link in original_order if link in result_links]


# --- novelty extra (2026-08-26, replacing the offbeat-slot design one day
# earlier -- see "Offbeat selection, take two" in docs/analysis/cluster-
# measurements.md). ADDITIVE, not carved out of max_per_topic: the regular
# candidates are always pool[:max_per_topic] by pure recency; the novelty
# extra, if any, is a SEPARATE pick drawn from pool[max_per_topic:] and
# appended with is_novelty_extra=True, only when it clears a real bar
# (news_keyness.NOVELTY_KEYWORDS hit, or keyness below
# NOVELTY_KEYNESS_THRESHOLD) -- never a forced "best of what's left" pick.
# -----------------------------------------------------------------------

def test_novelty_extra_appended_beyond_the_regular_max_per_topic_count(isolated_subscribers_db, fake_nltk):
    """A novelty-keyword hit (news_keyness.NOVELTY_KEYWORDS) in the
    remainder is appended as a 4th item, marked, ON TOP of the 3 regular
    (pure-recency) candidates -- not swapped in for one of them."""
    recency = ["ai daily roundup", "ai weekly digest", "ai startup raises funding round"]
    remainder = [
        "major ai model leaks ahead of launch",  # keyword hit: "leaks"
        "ai chip earnings beat expectations",     # no signal
    ]
    titles_newest_first = recency + remainder
    articles = [
        _article(f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i), title=t, categories=["AI"])
        for i, t in enumerate(titles_newest_first)
    ]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW,
    )

    titles = [a["title"] for a in result]
    assert titles == [
        "ai daily roundup",
        "ai weekly digest",
        "ai startup raises funding round",
        "major ai model leaks ahead of launch",
    ]
    assert result[-1]["is_novelty_extra"] is True
    assert "is_novelty_extra" not in result[0]


def test_novelty_extra_picks_lowest_keyness_when_no_keyword_hits(isolated_subscribers_db, fake_nltk):
    """With no keyword hit anywhere in the remainder, the novelty extra
    goes to the article whose own vocabulary is most "foreign" to the AI
    category per the precomputed keyness table (as news_ingest.py would
    have written via interest_cache_ops.set_category_keyness) -- and only because
    it clears NOVELTY_KEYNESS_THRESHOLD; "robot" (keyness +50, strongly
    topic-typical) must NOT be picked just for being the next-best thing
    in a 2-item remainder."""
    interest_cache_ops.set_category_keyness("AI", {"openai": 300.0, "robot": 50.0, "quantum": -40.0})
    recency = ["ai daily roundup", "ai weekly digest", "openai announces new update"]
    remainder = [
        "ai robot demo at conference",                # keyness +50, does not qualify
        "ai research touches on quantum computing",   # keyness -40, clears the threshold
    ]
    titles_newest_first = recency + remainder
    articles = [
        _article(f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i), title=t, categories=["AI"])
        for i, t in enumerate(titles_newest_first)
    ]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW,
    )

    assert len(result) == 4
    assert result[-1]["title"] == "ai research touches on quantum computing"
    assert result[-1]["is_novelty_extra"] is True


def test_novelty_extra_keyword_hit_outranks_keyness_score(isolated_subscribers_db, fake_nltk):
    """Both signals qualify in the same remainder -- the keyword hit wins
    the single novelty-extra slot even though a different article scores
    lower (more topic-foreign) on keyness alone."""
    interest_cache_ops.set_category_keyness("AI", {"quantum": -40.0})
    recency = ["ai daily roundup", "ai weekly digest", "ai chip earnings today"]
    remainder = [
        "ai research touches on quantum computing",  # lowest keyness, no keyword hit
        "major ai lawsuit filed over data use",       # keyword hit: "lawsuit"
    ]
    titles_newest_first = recency + remainder
    articles = [
        _article(f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i), title=t, categories=["AI"])
        for i, t in enumerate(titles_newest_first)
    ]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW,
    )

    assert result[-1]["title"] == "major ai lawsuit filed over data use"


def test_novelty_extra_below_threshold_does_not_qualify(isolated_subscribers_db, fake_nltk):
    """A keyness score that's negative but doesn't clear
    NOVELTY_KEYNESS_THRESHOLD (-5.0) must not be picked -- this is the
    exact product correction that motivated the threshold: an earlier
    design picked whatever was least-bad in the pool regardless of how
    weak the signal actually was."""
    interest_cache_ops.set_category_keyness("AI", {"mild": -1.0})  # negative, but not below -5.0
    recency = ["ai daily roundup", "ai weekly digest", "ai chip earnings today"]
    remainder = ["ai coverage with a mild angle to it"]
    titles_newest_first = recency + remainder
    articles = [
        _article(f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i), title=t, categories=["AI"])
        for i, t in enumerate(titles_newest_first)
    ]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW,
    )

    assert len(result) == 3
    assert not any(a.get("is_novelty_extra") for a in result)


def test_novelty_extra_disabled_sends_no_extra_even_with_a_qualifying_candidate(
    isolated_subscribers_db, fake_nltk
):
    recency = ["ai daily roundup", "ai weekly digest", "ai chip earnings today"]
    remainder = ["major ai model leaks ahead of launch"]
    titles_newest_first = recency + remainder
    articles = [
        _article(f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i), title=t, categories=["AI"])
        for i, t in enumerate(titles_newest_first)
    ]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW, include_novelty_extra=False,
    )

    assert [a["link"] for a in result] == ["https://a.com/0", "https://a.com/1", "https://a.com/2"]


def test_novelty_extra_none_when_no_signal_at_all(isolated_subscribers_db, fake_nltk):
    """No keyness table cached for "AI" (news_ingest.py hasn't run a
    cycle yet, or this is a fresh category) AND no keyword hits anywhere
    in the remainder -- _pick_novelty_extra returns None, and the digest
    is exactly the regular candidates, not padded with anything."""
    articles = [
        _article(f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i),
                 title=f"ai article {i}", categories=["AI"])
        for i in range(5)
    ]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW,
    )

    assert [a["link"] for a in result] == ["https://a.com/0", "https://a.com/1", "https://a.com/2"]


def test_novelty_extra_none_when_remainder_is_empty(isolated_subscribers_db, fake_nltk):
    """Pool exactly fits max_per_topic -- nothing left over for a novelty
    extra to be drawn from at all, not even a weak one."""
    articles = [
        _article(f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i),
                 title="major ai model leaks ahead of launch", categories=["AI"])
        for i in range(3)
    ]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW,
    )

    assert len(result) == 3
    assert not any(a.get("is_novelty_extra") for a in result)


def test_novelty_extra_uses_a_wider_relevance_pass_than_the_regular_digest(
    isolated_subscribers_db, fake_nltk, monkeypatch
):
    """The novelty search doesn't just inherit whatever the regular
    digest's stricter RELEVANCE_KEEP_* cut happened to leave over
    (pool[max_per_topic:]) -- it re-filters raw_pool with its own,
    wider NOVELTY_RELEVANCE_KEEP_* clamp. Pins the wiring: the regular
    pass gets news_push's own RELEVANCE_KEEP_* constants, the novelty
    pass gets the wider NOVELTY_RELEVANCE_KEEP_* ones."""
    calls = []
    original = news_embed.filter_by_relevance

    def spy(pool, embedder, query_text, **kwargs):
        calls.append(kwargs)
        return original(pool, embedder, query_text, **kwargs)

    monkeypatch.setattr(news_embed, "filter_by_relevance", spy)

    articles = [
        _article(f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i),
                 title="major ai model leaks ahead of launch" if i == 3 else f"ai article {i}",
                 categories=["AI"])
        for i in range(5)
    ]

    news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(), max_per_topic=3, now=NOW,
    )

    assert calls[0] == {
        "keep_fraction": news_push.RELEVANCE_KEEP_FRACTION,
        "keep_min": news_push.RELEVANCE_KEEP_MIN,
        "keep_max": news_push.RELEVANCE_KEEP_MAX,
    }
    assert calls[1] == {
        "keep_fraction": news_push.NOVELTY_RELEVANCE_KEEP_FRACTION,
        "keep_min": news_push.NOVELTY_RELEVANCE_KEEP_MIN,
        "keep_max": news_push.NOVELTY_RELEVANCE_KEEP_MAX,
    }


class _AxisEmbedder:
    """A hand-controlled stand-in for FakeEmbedder, used only where a test
    needs EXACT similarity values rather than FakeEmbedder's word-hash
    approximation. FakeEmbedder's 32-dim hash space collides constantly
    once a fixture needs 40+ distinct-but-related articles (two different
    titles' differing word can hash to the same dim+sign, making their
    embeddings literally identical) -- exactly the pool size this test
    needs to make NOVELTY_RELEVANCE_KEEP_MIN's percentile cut bind at
    all. Only used for the query vector here; article embeddings below
    are built directly with the same axis convention, not run through
    this at all."""

    def encode(self, texts):
        return np.array([[1.0] + [0.0] * 46 for _ in texts], dtype=np.float32)


def test_novelty_extra_excluded_when_it_fails_even_the_wider_relevance_bar(
    isolated_subscribers_db, fake_nltk
):
    """The real incident this guards against: a keyword hit alone used to
    be enough to surface an article regardless of topical relevance. A
    real embedder now sits in front of the novelty search too -- a
    candidate that shares essentially no vocabulary with the topic must
    be excluded before _pick_novelty_extra ever sees it, even though it
    contains a novelty keyword ("leak"). Needs a pool bigger than
    NOVELTY_RELEVANCE_KEEP_MIN=40 for the percentile cut to exclude
    anything at all -- a smaller pool would have the clamp keep
    everything, which would defeat the point of this test.

    Embeddings are hand-built on independent axes (see _AxisEmbedder)
    rather than derived from real text, specifically so every on-topic
    article's similarity to the query and to every OTHER on-topic
    article is exact and identical (0.7 and 0.49 respectively) -- with
    45 of them, FakeEmbedder's usual word-hash approach can't guarantee
    that without accidental near-duplicate collisions (see
    _AxisEmbedder's docstring)."""
    embedder = _AxisEmbedder()
    # Dim 0 = shared "on topic" axis; dims 1-45 = one private axis per
    # on-topic article, so distinct articles are never bit-for-bit
    # identical (which would trigger near-duplicate collapse) but every
    # pair still has the exact same, deliberately-moderate similarity to
    # each other (0.7^2 = 0.49) and to the query (0.7). Dim 46 is the
    # off-topic article's own axis, orthogonal to everything else --
    # similarity to the query and to every on-topic article is exactly 0.
    on_topic = []
    for i in range(45):
        vector = [0.0] * 47
        vector[0] = 0.7
        vector[1 + i] = (1 - 0.7 ** 2) ** 0.5
        on_topic.append(_article(
            f"https://a.com/{i}", published_dt=NOW - timedelta(hours=i + 1),
            title=f"ai update article {i}", categories=["AI"], embedding=vector,
        ))
    off_topic_vector = [0.0] * 47
    off_topic_vector[46] = 1.0
    # Oldest of all, so it's never one of the 3 regular (recency) picks.
    off_topic_keyword_hit = _article(
        "https://a.com/off-topic", published_dt=NOW - timedelta(hours=100),
        title="Major Bitcoin exchange data leak exposes users", categories=["AI"],
        embedding=off_topic_vector,
    )
    articles = on_topic + [off_topic_keyword_hit]

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW, embedder=embedder,
    )

    assert not any(a["link"] == "https://a.com/off-topic" for a in result)
    assert not any(a.get("is_novelty_extra") for a in result)


class _RankedAxisEmbedder:
    """Like _AxisEmbedder above, but with a distinct, strictly-decreasing
    similarity per article instead of one uniform value -- needed so the
    regular RELEVANCE_KEEP_MIN=20 and novelty NOVELTY_RELEVANCE_KEEP_MIN=40
    floors bind at exact, known ranks (index 19/20 and 39/40) rather than
    an all-or-nothing pass/fail."""

    def __init__(self, dims):
        self.dims = dims

    def encode(self, texts):
        return np.array([[1.0] + [0.0] * (self.dims - 1) for _ in texts], dtype=np.float32)


def test_novelty_extra_admits_a_candidate_the_regular_relevance_filter_would_have_excluded(
    isolated_subscribers_db, fake_nltk
):
    """test_novelty_extra_uses_a_wider_relevance_pass_than_the_regular_digest
    only pins that the two _filter_by_relevance calls receive different
    kwargs -- it doesn't prove the wider clamp actually admits anything the
    narrower one wouldn't. This does: 45 articles ranked by a hand-controlled,
    strictly decreasing similarity to the query (see _RankedAxisEmbedder), so
    the regular clamp (floor 20 for a 45-item pool) and novelty clamp (floor
    40) bind at exact, known boundaries. The article at similarity rank 26
    (index 25) sits outside the regular top 20 -- confirmed directly by
    calling _filter_by_relevance with the regular defaults -- but inside the
    novelty top 40, and carries a novelty-keyword hit ("leaked") so it's the
    only remainder candidate _pick_novelty_extra can qualify, isolating the
    relevance floor as what's under test rather than keyword/keyness scoring
    (already covered by the tests above)."""
    n = 45
    dims = 1 + n
    embedder = _RankedAxisEmbedder(dims)
    articles = []
    for i in range(n):
        s = 0.9 - i * 0.01  # strictly decreasing: index i has similarity rank i
        vector = [0.0] * dims
        vector[0] = s
        vector[1 + i] = (1 - s ** 2) ** 0.5
        title = "major ai model leaked ahead of launch" if i == 25 else f"ai update article {i}"
        articles.append(_article(
            f"https://a.com/{i}", published_dt=NOW - timedelta(minutes=i),
            title=title, categories=["AI"], embedding=vector,
        ))

    # Sanity check first: with these exact params, the regular per-topic
    # relevance filter really would exclude rank 26 -- proven directly, not
    # just inferred from the pipeline result below.
    regular_pass = _filter_by_relevance(articles, embedder, "AI")
    assert articles[25]["link"] not in {a["link"] for a in regular_pass}

    result = news_push.select_candidate_articles(
        articles, ["AI"], {"AI": ["AI"]}, None, set(),
        max_per_topic=3, now=NOW, embedder=embedder,
    )

    novelty_pick = next((a for a in result if a.get("is_novelty_extra")), None)
    assert novelty_pick is not None
    assert novelty_pick["link"] == "https://a.com/25"


# --- resolve_interest_categories -------------------------------------------


def test_resolve_interest_categories_uses_cache_when_available(monkeypatch):
    monkeypatch.setattr(interest_cache_ops, "get_cached_interest_categories", lambda interests: {"AI": ["AI"]})
    classify = MagicMock()
    monkeypatch.setattr(news_push.news_classify, "classify_interests", classify)

    result = news_push.resolve_interest_categories("fake-model", ["AI"])

    assert result == {"AI": ["AI"]}
    classify.assert_not_called()


def test_resolve_interest_categories_classifies_and_caches_misses(monkeypatch, isolated_subscribers_db):
    monkeypatch.setattr(interest_cache_ops, "get_cached_interest_categories", lambda interests: {})
    monkeypatch.setattr(news_push.news_classify, "classify_interests", lambda model, interests, taxonomy: {"AAOI": ["Stock"]})
    set_categories = MagicMock()
    monkeypatch.setattr(interest_cache_ops, "set_interest_categories", set_categories)

    result = news_push.resolve_interest_categories("fake-model", ["AAOI"])

    assert result == {"AAOI": ["Stock"]}
    set_categories.assert_called_once_with("AAOI", ["Stock"])


def test_resolve_interest_categories_caches_a_genuinely_empty_result(monkeypatch, isolated_subscribers_db):
    """The model answered "no category applies". That is a real answer and
    belongs in the cache -- re-classifying it every cycle would just re-pay
    for the same conclusion."""
    monkeypatch.setattr(interest_cache_ops, "get_cached_interest_categories", lambda interests: {})
    monkeypatch.setattr(news_push.news_classify, "classify_interests",
                        lambda model, interests, taxonomy: {"Some obscure ticker": []})
    set_categories = MagicMock()
    monkeypatch.setattr(interest_cache_ops, "set_interest_categories", set_categories)

    result = news_push.resolve_interest_categories("fake-model", ["Some obscure ticker"])

    assert result == {"Some obscure ticker": []}
    set_categories.assert_called_once_with("Some obscure ticker", [])


def test_resolve_interest_categories_does_not_cache_a_failure(monkeypatch, isolated_subscribers_db):
    """Regression test for a live bug. classify_interests omits an interest
    it failed on; caching that as [] made the failure permanent, and an
    empty mapping matches every article, so affected subscribers were sent
    entirely unfiltered news. On the live DB this had poisoned "AI",
    "Bitcoin" and "機器人科技" among others -- "AI" cached as [] despite AI
    being one of the 13 categories."""
    monkeypatch.setattr(interest_cache_ops, "get_cached_interest_categories", lambda interests: {})
    monkeypatch.setattr(news_push.news_classify, "classify_interests",
                        lambda model, interests, taxonomy: {})
    set_categories = MagicMock()
    monkeypatch.setattr(interest_cache_ops, "set_interest_categories", set_categories)

    result = news_push.resolve_interest_categories("fake-model", ["AI"])

    set_categories.assert_not_called()
    assert result == {}, "unresolved, so the next cycle retries it"


# --- backfill_missing_interest_definitions ----------------------------------
# docs/plans/interest-definition-plan.md "Defect 2": an interest added
# before expand_interest_for_retrieval existed (or whose one-shot
# generation at add time failed) has no cached definition, and nothing on
# the read path (_resolve_query_text) ever retries it. This backfill,
# called once per push cycle, is the fix.


def test_backfill_skips_an_interest_that_already_has_a_definition(monkeypatch, isolated_subscribers_db):
    interest_cache_ops.set_interest_query_expansion("AI", "already has one")
    expand = MagicMock()
    monkeypatch.setattr(news_push.news_classify, "expand_interest_for_retrieval", expand)

    news_push.backfill_missing_interest_definitions("fake-model", ["AI"])

    expand.assert_not_called()
    assert interest_cache_ops.get_interest_query_expansion("AI") == "already has one"


def test_backfill_generates_and_caches_a_missing_definition(monkeypatch, isolated_subscribers_db):
    monkeypatch.setattr(news_push.news_classify, "expand_interest_for_retrieval",
                        lambda model, interest: f"a generated definition of {interest}")

    news_push.backfill_missing_interest_definitions("fake-model", ["AI"])

    assert interest_cache_ops.get_interest_query_expansion("AI") == "a generated definition of AI"


def test_backfill_writes_to_the_shared_tier_not_a_subscriber_tier(monkeypatch, isolated_subscribers_db):
    """This is an automatic, global backfill -- same tier
    add_one_interest already writes to -- not a personal refinement.
    interest_cache_ops.resolve_interest_definition has no chat_id to
    scope it to here; there's no subscriber in scope at all."""
    monkeypatch.setattr(news_push.news_classify, "expand_interest_for_retrieval",
                        lambda model, interest: "the backfilled definition")

    news_push.backfill_missing_interest_definitions("fake-model", ["AI"])

    assert interest_cache_ops.get_subscriber_interest_definition(999, "AI") is None
    assert interest_cache_ops.resolve_interest_definition(999, "AI") == "the backfilled definition"


def test_backfill_does_not_cache_a_failed_generation(monkeypatch, isolated_subscribers_db):
    """Same non-poisoning discipline as resolve_interest_categories: a
    generation failure (None) must not be cached as a permanent answer,
    so the next cycle retries it instead of being stuck with the bare
    topic string forever."""
    monkeypatch.setattr(news_push.news_classify, "expand_interest_for_retrieval",
                        lambda model, interest: None)

    news_push.backfill_missing_interest_definitions("fake-model", ["AI"])

    assert interest_cache_ops.get_interest_query_expansion("AI") is None


def test_backfill_only_generates_for_interests_actually_missing_one(monkeypatch, isolated_subscribers_db):
    interest_cache_ops.set_interest_query_expansion("AI", "already cached")
    calls = []
    monkeypatch.setattr(news_push.news_classify, "expand_interest_for_retrieval",
                        lambda model, interest: calls.append(interest) or "new definition")

    news_push.backfill_missing_interest_definitions("fake-model", ["AI", "robotics"])

    assert calls == ["robotics"]


# --- write_push_digest ------------------------------------------------------


def test_write_push_digest_returns_model_output():
    model = FakeToolCallingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [{**_article("https://example.com/a"), "topic": "AI"}]

    result = news_push.write_push_digest(model, articles)

    assert result == "<b>Digest</b>"


def test_write_push_digest_includes_language_directive_when_set():
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["system_prompt"] = messages[0]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [{**_article("https://example.com/a"), "topic": "AI"}]

    news_push.write_push_digest(model, articles, language="Spanish")

    assert "Spanish" in captured["system_prompt"]


def test_write_push_digest_no_language_directive_when_unset():
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["system_prompt"] = messages[0]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [{**_article("https://example.com/a"), "topic": "AI"}]

    news_push.write_push_digest(model, articles)

    assert captured["system_prompt"] == news_push._PUSH_DIGEST_PROMPT


def test_write_push_digest_includes_retry_reason_when_set():
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["system_prompt"] = messages[0]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [{**_article("https://example.com/a"), "topic": "AI"}]

    news_push.write_push_digest(model, articles, retry_reason="unescaped & at '&T'")

    assert "unescaped & at '&T'" in captured["system_prompt"]


def test_write_push_digest_no_retry_instruction_when_unset():
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["system_prompt"] = messages[0]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [{**_article("https://example.com/a"), "topic": "AI"}]

    news_push.write_push_digest(model, articles)

    assert captured["system_prompt"] == news_push._PUSH_DIGEST_PROMPT


def test_write_push_digest_listing_carries_no_topic_prefix(monkeypatch):
    """Regression for the 2026-08-25 fix: the listing used to prefix every
    line with `[{topic}]`, which was both informationless (every line in
    one call always carried the same value, once push became one message
    per interest) and actively misleading (it visually presented a
    coarse-filter artifact as if it were a confirmed per-article
    classification, contradicting the system prompt's own instruction to
    be skeptical of exactly that)."""
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["listing"] = messages[1]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [{**_article("https://example.com/a", title="Some headline"), "topic": "AI Agent"}]

    news_push.write_push_digest(model, articles, topic="AI Agent")

    assert "[AI Agent]" not in captured["listing"]
    assert "Some headline" in captured["listing"]


def test_write_push_digest_states_the_topic_once_when_given(monkeypatch):
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["system_prompt"] = messages[0]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [{**_article("https://example.com/a"), "topic": "AI Agent"}]

    news_push.write_push_digest(model, articles, topic="AI Agent")

    prompt = captured["system_prompt"]
    assert "AI Agent" in prompt
    assert "coarse category filter" in prompt
    # The topic-framing addition is one coherent instruction block (which
    # may reasonably repeat the topic word within it for emphasis) -- the
    # thing that must NOT happen is a per-candidate repetition, which is
    # what test_write_push_digest_listing_carries_no_topic_prefix checks
    # on the listing (the user message) rather than here.


def test_write_push_digest_no_topic_framing_when_topic_is_none(monkeypatch):
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["system_prompt"] = messages[0]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [{**_article("https://example.com/a"), "topic": "AI"}]

    news_push.write_push_digest(model, articles)

    assert captured["system_prompt"] == news_push._PUSH_DIGEST_PROMPT


def test_write_push_digest_marks_the_novelty_extra_in_the_listing(monkeypatch):
    """Exactly the one candidate carrying is_novelty_extra gets the
    [EXTRA] marker in the listing text -- not every line (that was the
    2026-08-25 [topic]-prefix mistake this deliberately doesn't repeat:
    a marker on every line reads as confirmed metadata; a marker on the
    one genuinely-different item is a real formatting signal)."""
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["listing"] = messages[1]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    articles = [
        {**_article("https://example.com/a", title="Regular headline"), "topic": "AI"},
        {**_article("https://example.com/b", title="Unusual headline"), "topic": "AI", "is_novelty_extra": True},
    ]

    news_push.write_push_digest(model, articles, topic="AI")

    assert "[EXTRA] Unusual headline" in captured["listing"]
    assert "[EXTRA] Regular headline" not in captured["listing"]
    assert captured["listing"].count("[EXTRA]") == 1


def test_write_push_digest_adds_extra_instruction_only_when_present(monkeypatch):
    captured = {}

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            captured["system_prompt"] = messages[0]["content"]
            return super().invoke(messages, *args, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="<b>Digest</b>")])
    with_extra = [{**_article("https://example.com/a"), "topic": "AI", "is_novelty_extra": True}]
    without_extra = [{**_article("https://example.com/b"), "topic": "AI"}]

    news_push.write_push_digest(model, with_extra)
    assert "[EXTRA]" in captured["system_prompt"]

    news_push.write_push_digest(model, without_extra)
    assert "[EXTRA]" not in captured["system_prompt"]


def test_is_subscriber_due_true_when_never_pushed():
    assert news_push.is_subscriber_due(None, 24, datetime.now(timezone.utc)) is True


def test_is_subscriber_due_false_within_interval():
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    last_push = now - timedelta(hours=2)
    assert news_push.is_subscriber_due(last_push, 24, now) is False


def test_is_subscriber_due_true_after_interval():
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    last_push = now - timedelta(hours=25)
    assert news_push.is_subscriber_due(last_push, 24, now) is True


def _subscriber(
    chat_id,
    interests=("AI",),
    interval=24,
    last_push_at=None,
    pushed_links=(),
    language=None,
    restricted_sources_enabled=False,
):
    return {
        "chat_id": chat_id,
        "interests": list(interests),
        "push_interval_hours": interval,
        "last_push_at": last_push_at,
        "pushed_links": list(pushed_links),
        "language": language,
        "restricted_sources_enabled": restricted_sources_enabled,
    }


def _stub_cache_and_categories(monkeypatch, cached_articles=(), topic_categories=None):
    monkeypatch.setattr(news_cache, "read_all", lambda: list(cached_articles))
    monkeypatch.setattr(
        news_push, "resolve_interest_categories", lambda model, interests: topic_categories or {}
    )
    # Same reasoning as the resolve_interest_categories stub above: this
    # also calls `model` (see backfill_missing_interest_definitions's own
    # docstring), and a test using a FakeToolCallingModel with a specific
    # scripted response sequence for write_push_digest must not have one
    # silently consumed by this unrelated call.
    monkeypatch.setattr(news_push, "backfill_missing_interest_definitions", lambda model, interests: None)


def test_run_push_cycle_skips_subscriber_with_no_interests(monkeypatch, isolated_subscribers_db):
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(1, interests=[])])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    advance_last_push_at = MagicMock()
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", advance_last_push_at)
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send))

    send.assert_not_called()
    mark_links_shown.assert_not_called()
    advance_last_push_at.assert_not_called()


def test_run_push_cycle_skips_subscriber_not_due(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    recently_pushed = _subscriber(2, last_push_at=now - timedelta(hours=1), interval=24)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [recently_pushed])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    advance_last_push_at = MagicMock()
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", advance_last_push_at)
    select = MagicMock()
    monkeypatch.setattr(news_push, "select_candidate_articles", select)
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))

    select.assert_not_called()
    send.assert_not_called()
    mark_links_shown.assert_not_called()
    advance_last_push_at.assert_not_called()


def test_run_push_cycle_sends_and_records_when_new_articles_found(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(3)])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    advance_last_push_at = MagicMock()
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", advance_last_push_at)
    new_articles = [{**_article("https://example.com/new"), "topic": "AI"}]
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=new_articles))
    # the digest must actually cite the article for it to count as sent
    digest = '<b>Digest</b> 🔗 <a href="https://example.com/new">Source</a>'
    monkeypatch.setattr(news_push, "write_push_digest", MagicMock(return_value=digest))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=True))
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))

    send.assert_called_once_with(3, digest, topic="AI")
    mark_links_shown.assert_called_once_with(3, ["https://example.com/new"], now)
    advance_last_push_at.assert_called_once_with(3, now)


def test_run_push_cycle_passes_the_subscribers_chat_id_to_select_candidate_articles(
    monkeypatch, isolated_subscribers_db
):
    """select_candidate_articles needs chat_id to prefer this subscriber's
    own refined definition over the shared default
    (docs/plans/interest-definition-plan.md) -- a regression here would
    silently fall back to shared-only retrieval for everyone."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(3)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)
    select = MagicMock(return_value=[])
    monkeypatch.setattr(news_push, "select_candidate_articles", select)
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))

    assert select.call_args.kwargs["chat_id"] == 3


def test_run_push_cycle_backfills_missing_definitions_before_selecting(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(3)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    monkeypatch.setattr(news_cache, "read_all", lambda: [])
    monkeypatch.setattr(news_push, "resolve_interest_categories", lambda model, interests: {})
    backfill = MagicMock()
    monkeypatch.setattr(news_push, "backfill_missing_interest_definitions", backfill)
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))

    backfill.assert_called_once_with("fake-model", ["AI"])


def test_run_push_cycle_only_records_articles_the_digest_actually_cited(
    monkeypatch, isolated_subscribers_db
):
    """Stage 2 (the digest prompt) drops candidates it judges irrelevant. A
    dropped candidate was never seen by the subscriber, so it must stay
    eligible for a later digest rather than being retired unread."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(9)])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    candidates = [
        {**_article("https://example.com/used"), "topic": "AI"},
        {**_article("https://example.com/dropped"), "topic": "AI"},
    ]
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=candidates))
    monkeypatch.setattr(
        news_push, "write_push_digest",
        MagicMock(return_value='📰 <a href="https://example.com/used">Only this one</a>'),
    )
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=True))

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    mark_links_shown.assert_called_once_with(9, ["https://example.com/used"], now)


def test_run_push_cycle_passes_subscriber_language_to_digest(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(8, language="French")]
    )
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    new_articles = [{**_article("https://example.com/new"), "topic": "AI"}]
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=new_articles))
    write_digest = MagicMock(return_value="<b>Digest</b>")
    monkeypatch.setattr(news_push, "write_push_digest", write_digest)
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=True))

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    write_digest.assert_called_once_with("fake-model", new_articles, "AI", "French", retry_reason=None)


# --- the HTML-validation retry loop (2026-08-28) --------------------------
# Real news_push.write_push_digest (not monkeypatched) driven by a real
# FakeToolCallingModel with scripted responses, so the retry loop's actual
# interaction with it (the retry_reason round-trip, the attempt count) is
# genuinely exercised rather than assumed.


def test_run_push_cycle_retries_once_on_invalid_html_then_sends_the_valid_reply(
    monkeypatch, isolated_subscribers_db
):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(20)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    new_articles = [{**_article("https://example.com/a"), "topic": "AI"}]
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=new_articles))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=True))
    emitted = MagicMock()
    monkeypatch.setattr(news_push, "_emit_html_validation_attempt", emitted)

    prompts = []

    class RecordingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            prompts.append(messages[0]["content"])
            return super().invoke(messages, *args, **kwargs)

    valid = '<b>Digest</b> <a href="https://example.com/a">s</a>'
    model = RecordingModel(responses=[
        AIMessage(content="AT&T unveils new chip"),  # invalid: unescaped &
        AIMessage(content=valid),
    ])
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(
        model=model, send=send, now=now))

    assert len(prompts) == 2
    assert "unescaped &" in prompts[1]  # the second attempt was told what broke
    send.assert_called_once_with(20, valid, topic="AI")
    # One emission per attempt, not just a final one -- attempt 1 failed,
    # attempt 2 passed. Nothing about "is this alert-worthy" happens here;
    # this loop only ever reports the fact of each attempt.
    assert emitted.call_count == 2
    first_call, second_call = emitted.call_args_list
    first_chat_id, first_topic, first_attempt, first_reason = first_call.args
    assert (first_attempt, first_reason and "unescaped &" in first_reason) == (1, True)
    second_chat_id, second_topic, second_attempt, second_reason = second_call.args
    assert (second_attempt, second_reason) == (2, None)


def test_run_push_cycle_reports_every_attempt_and_still_sends_after_max_attempts(
    monkeypatch, isolated_subscribers_db
):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(21)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    new_articles = [{**_article("https://example.com/a"), "topic": "AI"}]
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=new_articles))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=True))
    emitted = MagicMock()
    monkeypatch.setattr(news_push, "_emit_html_validation_attempt", emitted)

    call_count = {"n": 0}

    class CountingModel(FakeToolCallingModel):
        def invoke(self, messages, *args, **kwargs):
            call_count["n"] += 1
            return super().invoke(messages, *args, **kwargs)

    always_invalid = AIMessage(
        content='AT&T <a href="https://example.com/a">unveils new chip</a>, still broken')
    # A single scripted response, not news_push.MAX_HTML_ATTEMPTS of them --
    # FakeToolCallingModel repeats the last one once exhausted, so this
    # doesn't accidentally cap the retry loop itself; the literal 3 below
    # is what actually pins the count (referencing the constant here would
    # make this test pass under any mutation of it, since both sides would
    # move together).
    model = CountingModel(responses=[always_invalid])
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model=model, send=send, now=now))

    assert call_count["n"] == 3  # no wasted extra call
    # Every attempt reported, none skipped -- deciding "3 straight failures
    # is alert-worthy" is a Logfire query over these, not this loop's job.
    assert emitted.call_count == 3
    assert [c.args[2] for c in emitted.call_args_list] == [1, 2, 3]
    assert all(c.args[3] is not None for c in emitted.call_args_list)  # every attempt invalid
    # Sent anyway, still invalid -- send_push_digest's own BadRequest
    # fallback (bot.py) is what strips it if Telegram also rejects it.
    send.assert_called_once_with(21, always_invalid.content, topic="AI")


def test_run_push_cycle_threads_chat_id_and_topic_correctly_across_subscribers(
    monkeypatch, isolated_subscribers_db
):
    """The two retry tests above pin attempt order and call count, but
    never that chat_id/topic on each call are the RIGHT ones -- a bug
    that swapped one subscriber's chat_id or topic into another's
    _emit_html_validation_attempt call would still pass both. Two
    subscribers with two different topics, each retrying once, close
    that gap: chat_id/topic must never cross-contaminate between
    subscribers, and attempt numbering must restart at 1 for each
    subscriber/topic rather than continuing across the cycle."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        subscriber_ops, "list_push_enabled_subscribers",
        lambda: [_subscriber(30, interests=("AI",)), _subscriber(40, interests=("Space",))],
    )
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)

    def select(cached, topics, *args, **kwargs):
        return [{**_article(f"https://example.com/{topics[0]}"), "topic": topics[0]}]

    monkeypatch.setattr(news_push, "select_candidate_articles", select)
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=True))
    emitted = MagicMock()
    monkeypatch.setattr(news_push, "_emit_html_validation_attempt", emitted)

    invalid = AIMessage(content="AT&T unveils new chip")  # unescaped &, always invalid
    valid = AIMessage(content="<b>Digest</b>")
    # One (invalid, valid) pair per subscriber -- each subscriber's own
    # retry loop should see attempt 1 fail, attempt 2 pass, independent
    # of the other subscriber's loop.
    model = FakeToolCallingModel(responses=[invalid, valid, invalid, valid])
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model=model, send=send, now=now))

    calls = [(c.args[0], c.args[1], c.args[2]) for c in emitted.call_args_list]
    assert calls == [
        (30, "AI", 1),
        (30, "AI", 2),
        (40, "Space", 1),
        (40, "Space", 2),
    ]


def test_run_push_cycle_passes_subscribers_own_restricted_sources_flag(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        subscriber_ops,
        "list_push_enabled_subscribers",
        lambda: [_subscriber(9, restricted_sources_enabled=True), _subscriber(10, restricted_sources_enabled=False)],
    )
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)
    select = MagicMock(return_value=[])
    monkeypatch.setattr(news_push, "select_candidate_articles", select)

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    assert select.call_args_list[0].kwargs["include_restricted"] is True
    assert select.call_args_list[1].kwargs["include_restricted"] is False


def test_run_push_cycle_reads_cache_once_and_reuses_across_subscribers(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(11), _subscriber(12)]
    )
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    read_all = MagicMock(return_value=[])
    monkeypatch.setattr(news_cache, "read_all", read_all)
    monkeypatch.setattr(news_push, "resolve_interest_categories", lambda model, interests: {})
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=[]))

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    read_all.assert_called_once()


def test_run_push_cycle_no_new_articles_records_without_sending(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(4)])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=[]))
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))

    send.assert_not_called()
    mark_links_shown.assert_called_once_with(4, [], now)


def test_run_push_cycle_empty_digest_records_without_sending(monkeypatch, isolated_subscribers_db):
    """Stage 2 (the model's own judgment inside write_push_digest) decided
    none of the stage-1 candidates were genuinely relevant, so it wrote
    nothing. Nothing was delivered, so nothing may be recorded as seen.

    This test previously asserted the opposite -- that every candidate's
    link was recorded -- and so pinned a real bug in place. pushed_links is
    the "already seen" filter, and an article that merely lost one cycle's
    relevance judgment has not been seen; recording it here retired
    articles permanently that no subscriber ever read. The three sibling
    branches (no candidates, guardrail-blocked, delivered) all handled this
    correctly; only this one didn't, while its comment claimed it matched
    them."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(13)])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    new_articles = [{**_article("https://example.com/new"), "topic": "AI"}]
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=new_articles))
    monkeypatch.setattr(news_push, "write_push_digest", MagicMock(return_value="   "))
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))

    send.assert_not_called()
    mark_links_shown.assert_called_once_with(13, [], now)


def test_run_push_cycle_blocked_by_output_guardrail_does_not_send(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(5)])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    advance_last_push_at = MagicMock()
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", advance_last_push_at)
    new_articles = [{**_article("https://example.com/new"), "topic": "AI"}]
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=new_articles))
    monkeypatch.setattr(news_push, "write_push_digest", MagicMock(return_value="off-topic drift"))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=False))
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))

    send.assert_not_called()
    # Nothing was delivered, so nothing is recorded as seen -- the articles
    # stay eligible for the next digest. last_push_at still advances, so the
    # next attempt is a full interval away rather than retrying every tick.
    mark_links_shown.assert_called_once_with(5, [], now)
    advance_last_push_at.assert_called_once_with(5, now)


def test_run_push_cycle_isolates_one_subscribers_failure(monkeypatch, isolated_subscribers_db):
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(6), _subscriber(7)]
    )
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)

    call_count = {"n": 0}

    def select_side_effect(cached_articles, topics, topic_categories, since, pushed_links,
                           include_restricted=False, now=None, embedder=None, chat_id=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return [{**_article("https://example.com/ok"), "topic": "AI"}]

    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(side_effect=select_side_effect))
    monkeypatch.setattr(news_push, "write_push_digest",
                        MagicMock(return_value='<b>Digest</b> <a href="https://example.com/ok">s</a>'))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=True))
    send = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))

    # subscriber 6 failed silently; subscriber 7 still got its digest
    send.assert_called_once_with(7, '<b>Digest</b> <a href="https://example.com/ok">s</a>', topic="AI")


# --- push outcomes: what _record reports each cycle -----------------------
#
# Each of these asserts the OUTCOME recorded, not the print. The point of
# the table is that an alarm can distinguish cases `docker logs` only ever
# rendered as prose -- above all "generated but undeliverable", which used
# to land in the catch-all as an ordinary cycle failure.


def _cycle_with(monkeypatch, chat_id=1, subscriber=None, digest=None,
                on_topic=True, send=None, now=None,
                mark_links_shown=None, advance_last_push_at=None):
    """Drives one full cycle far enough to produce a digest, so a test only
    has to say which step misbehaves.

    `mark_links_shown`/`advance_last_push_at` are each stubbed by default
    because most of these tests are about which OUTCOME was recorded; pass
    your own MagicMock for whichever one the test actually cares about --
    pushed_links content, or whether last_push_at advanced."""
    now = now or datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers",
                        lambda: [subscriber or _subscriber(chat_id)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown or MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", advance_last_push_at or MagicMock())
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles",
                        MagicMock(return_value=[{**_article("https://example.com/new"), "topic": "AI"}]))
    if digest is None:
        digest = '<b>Digest</b> 🔗 <a href="https://example.com/new">Source</a>'
    monkeypatch.setattr(news_push, "write_push_digest", MagicMock(return_value=digest))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=on_topic))
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send or AsyncMock(), now=now))
    return now


def test_run_push_cycle_records_delivered(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    _cycle_with(monkeypatch, chat_id=11)
    assert recorded_outcomes(11) == [push_outcome_ops.PUSH_DELIVERED]


def test_a_truly_empty_digest_records_not_relevant(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    _cycle_with(monkeypatch, chat_id=51, digest="")
    assert recorded_outcomes(51) == [push_outcome_ops.PUSH_NOT_RELEVANT]


def test_a_non_empty_digest_with_no_real_link_is_treated_as_not_relevant(
    monkeypatch, isolated_subscribers_db, recorded_outcomes
):
    """The gap this guards: _PUSH_DIGEST_PROMPT's "write nothing" instruction
    asks for a literal empty reply when nothing is relevant, but the model
    doesn't always comply literally -- it can write an explanatory sentence
    instead ("No genuinely relevant stories emerged..."). That string is
    not empty, so a check on `not digest.strip()` alone would treat it as
    real content and send it to the subscriber. TREND_REPORT_STRUCTURE
    requires every genuine item to carry a real <a href> link, so a
    digest with none is the actual signal, not literal emptiness."""
    send = AsyncMock()
    _cycle_with(monkeypatch, chat_id=52, send=send,
               digest="No genuinely relevant stories emerged from the candidates this cycle.")

    assert recorded_outcomes(52) == [push_outcome_ops.PUSH_NOT_RELEVANT]
    send.assert_not_called()


def test_run_push_cycle_records_blocked_digest_as_its_own_outcome(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    _cycle_with(monkeypatch, chat_id=12, on_topic=False)
    assert recorded_outcomes(12) == [push_outcome_ops.PUSH_BLOCKED]


def test_run_push_cycle_records_chat_not_found_when_delivery_is_refused(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    """The 2026-08-21 signature: generation succeeded and was billed, only
    the send failed. Must NOT read as a generic cycle failure -- criterion
    1 keys on exactly this."""
    send = AsyncMock(side_effect=Exception("Chat not found"))
    _cycle_with(monkeypatch, chat_id=13, send=send)
    assert recorded_outcomes(13) == [push_outcome_ops.PUSH_CHAT_NOT_FOUND]


def test_run_push_cycle_records_blocked_user_as_chat_not_found(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    send = AsyncMock(side_effect=Exception("Forbidden: bot was blocked by the user"))
    _cycle_with(monkeypatch, chat_id=14, send=send)
    assert recorded_outcomes(14) == [push_outcome_ops.PUSH_CHAT_NOT_FOUND]


def test_run_push_cycle_records_model_error_when_an_llm_call_raises(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    """A 402 comes out of write_push_digest, not out of the send. Classified
    by which call raised rather than by what the message says, so a
    provider rewording its errors cannot silence criterion 2."""
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(15)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles",
                        MagicMock(return_value=[{**_article("https://example.com/new"), "topic": "AI"}]))
    monkeypatch.setattr(news_push, "write_push_digest",
                        MagicMock(side_effect=RuntimeError("Error code: 402 - Insufficient Balance")))
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(),
                                         now=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)))

    assert recorded_outcomes(15) == [push_outcome_ops.PUSH_MODEL_ERROR]


def test_run_push_cycle_records_a_non_model_failure_as_cycle_failed(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    """select_candidate_articles is local filtering, not an LLM call. If it
    raises, that is a bug in our code -- it must not inflate the model-error
    count and page someone about the provider."""
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(16)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles",
                        MagicMock(side_effect=KeyError("published_dt")))
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(),
                                         now=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)))

    assert recorded_outcomes(16) == [push_outcome_ops.PUSH_CYCLE_FAILED]


def test_run_push_cycle_records_nothing_new_without_calling_the_model(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(17)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=[]))
    write = MagicMock()
    monkeypatch.setattr(news_push, "write_push_digest", write)
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(),
                                         now=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)))

    assert recorded_outcomes(17) == [push_outcome_ops.PUSH_NOTHING_NEW]
    write.assert_not_called()


def test_run_push_cycle_records_no_interests(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers",
                        lambda: [_subscriber(18, interests=[])])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock()))

    assert recorded_outcomes(18) == [push_outcome_ops.PUSH_NO_INTERESTS]


def test_run_push_cycle_does_not_record_a_not_due_subscriber(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    """Every subscriber is 'not due' on almost every tick. Recording it
    would bury the outcomes that carry signal under ~96 rows a day each."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers",
                        lambda: [_subscriber(19, last_push_at=now - timedelta(hours=1), interval=24)])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    assert recorded_outcomes(19) == []


def test_classify_send_failure_falls_back_to_cycle_failed():
    """Wrong in the safe direction: an unrecognised delivery error must not
    be read as 'this chat is dead', because criterion 1 disables
    subscribers on that verdict."""
    assert news_push._classify_send_failure(Exception("Timed out")) == push_outcome_ops.PUSH_CYCLE_FAILED
    assert news_push._classify_send_failure(Exception("Chat not found")) == push_outcome_ops.PUSH_CHAT_NOT_FOUND


# --- capping an undeliverable subscriber ----------------------------------
#
# Two defects, fixed together because they compound. (A) no failure path
# advanced last_push_at, so a failing subscriber was due again on the next
# 15-minute tick rather than after their interval. (B) nothing ever stopped
# it. Generation happens before delivery is attempted and is billed either
# way, so an unreachable chat billed three LLM calls every 15 minutes for
# as long as the row existed -- the 2026-08-21 incident.


def _failing_send(message="Chat not found"):
    return AsyncMock(side_effect=Exception(message))


def test_delivery_failure_advances_last_push_at(monkeypatch, isolated_subscribers_db):
    """(A). Without this the subscriber is due again in PUSH_TICK_SECONDS
    and pays for another digest, having already paid for this one."""
    advance_last_push_at = MagicMock()
    now = _cycle_with(monkeypatch, chat_id=21, send=_failing_send(),
                      advance_last_push_at=advance_last_push_at)

    advance_last_push_at.assert_called_once_with(21, now)


def test_delivery_failure_retires_no_links(monkeypatch, isolated_subscribers_db):
    """The digest was never seen, so its articles must stay eligible for a
    later one -- same rule as the guardrail-blocked branch."""
    mark_links_shown = MagicMock()
    _cycle_with(monkeypatch, chat_id=22, send=_failing_send(), mark_links_shown=mark_links_shown)

    assert mark_links_shown.call_args[0][1] == []


def test_three_consecutive_chat_not_found_turns_push_off(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    # Set explicitly: get_push_enabled returns False for a row that does not
    # exist, so without this the assertion below would pass without the
    # subscriber ever having been turned off.
    subscriber_ops.set_push_enabled(23, True)
    for _ in range(news_push.UNREACHABLE_STRIKES):
        _cycle_with(monkeypatch, chat_id=23, send=_failing_send(),
                    mark_links_shown=MagicMock())

    assert recorded_outcomes(23)[0] == push_outcome_ops.PUSH_DISABLED
    assert subscriber_ops.get_push_enabled(23) is False


def test_two_consecutive_chat_not_found_leaves_push_on(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    """Turning a real subscriber off is the more expensive mistake: they
    just stop getting news, with nothing to notice."""
    subscriber_ops.set_push_enabled(23, True)
    for _ in range(news_push.UNREACHABLE_STRIKES - 1):
        _cycle_with(monkeypatch, chat_id=23, send=_failing_send(),
                    mark_links_shown=MagicMock())

    assert push_outcome_ops.PUSH_DISABLED not in recorded_outcomes(23)
    assert subscriber_ops.get_push_enabled(23) is True


def test_a_successful_delivery_clears_the_strikes(monkeypatch, isolated_subscribers_db):
    """Delivery is the only positive proof the chat is reachable, so it is
    the only thing that resets the count."""
    subscriber_ops.set_push_enabled(24, True)
    _cycle_with(monkeypatch, chat_id=24, send=_failing_send(), mark_links_shown=MagicMock())
    _cycle_with(monkeypatch, chat_id=24, send=_failing_send(), mark_links_shown=MagicMock())
    _cycle_with(monkeypatch, chat_id=24, mark_links_shown=MagicMock())          # delivered
    _cycle_with(monkeypatch, chat_id=24, send=_failing_send(), mark_links_shown=MagicMock())

    assert subscriber_ops.get_push_enabled(24) is True


def test_a_quiet_cycle_between_failures_does_not_clear_the_strikes(monkeypatch, isolated_subscribers_db):
    """The policy decision this rests on. A `nothing_new` cycle attempts no
    send, so it is evidence of nothing -- if it reset the count, a dead chat
    that happens to have a quiet cycle every third tick would bill digests
    forever and never strike out."""
    subscriber_ops.set_push_enabled(25, True)
    _cycle_with(monkeypatch, chat_id=25, send=_failing_send(), mark_links_shown=MagicMock())
    _cycle_with(monkeypatch, chat_id=25, send=_failing_send(), mark_links_shown=MagicMock())

    # a cycle with no candidate articles: no send is attempted at all
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(25)])
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles", MagicMock(return_value=[]))
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(),
                                         now=datetime(2026, 8, 8, 13, 0, tzinfo=timezone.utc)))

    _cycle_with(monkeypatch, chat_id=25, send=_failing_send(), mark_links_shown=MagicMock())

    assert subscriber_ops.get_push_enabled(25) is False


# --- Free-trial push allowance (requested 2026-09-18) ---------------------
# Unlike the unreachable-strike tests above, this is a deliberate,
# business-policy cutoff, not a delivery-failure signal -- see
# news_push._stop_push_at_trial_limit's own docstring.


def test_push_stops_and_notifies_admin_when_trial_limit_reached(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    subscriber_ops.request_access(60, "sybil", "Sybil")
    subscriber_ops.decide(60, approved=True)
    subscriber_ops.set_pushes_remaining(60, 0)
    subscriber_ops.set_push_enabled(60, True)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(60)])
    _stub_cache_and_categories(monkeypatch)
    send = AsyncMock()
    notify_admin = AsyncMock()

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send,
                                         now=datetime(2026, 8, 8, 13, 0, tzinfo=timezone.utc),
                                         notify_admin=notify_admin))

    send.assert_not_called()
    assert subscriber_ops.get_push_enabled(60) is False
    assert push_outcome_ops.PUSH_TRIAL_LIMIT_REACHED in recorded_outcomes(60)
    notify_admin.assert_called_once_with(60)


def test_push_is_not_stopped_or_charged_twice_in_one_cycle_check(monkeypatch, isolated_subscribers_db):
    """A subscriber with allowance left is charged exactly once per
    cycle, regardless of how many of their interests get sent."""
    subscriber_ops.request_access(61, "trent", "Trent")
    subscriber_ops.decide(61, approved=True)
    subscriber_ops.set_pushes_remaining(61, 5)

    _cycle_with(monkeypatch, chat_id=61, mark_links_shown=MagicMock())

    assert subscriber_ops.get_pushes_remaining(61) == 4


def test_push_digest_includes_remaining_count_when_limited(monkeypatch, isolated_subscribers_db):
    subscriber_ops.request_access(62, "uma", "Uma")
    subscriber_ops.decide(62, approved=True)
    subscriber_ops.set_pushes_remaining(62, 3)
    send = AsyncMock()

    _cycle_with(monkeypatch, chat_id=62, send=send, mark_links_shown=MagicMock())

    sent_text = send.call_args[0][1]
    assert "2 push(es) remaining" in sent_text


def test_push_digest_omits_remaining_note_when_unlimited(monkeypatch, isolated_subscribers_db):
    """No decide() call for this chat_id at all -- the same state every
    subscriber approved before this feature existed is in."""
    send = AsyncMock()

    _cycle_with(monkeypatch, chat_id=63, send=send, mark_links_shown=MagicMock())

    sent_text = send.call_args[0][1]
    assert "remaining" not in sent_text


def test_push_allowance_is_not_charged_when_nothing_is_sent(monkeypatch, isolated_subscribers_db):
    """The real bug found live on INT, 2026-09-19: charging as soon as a
    subscriber was due (before knowing whether the cycle would actually
    produce anything) meant their one remaining push could be spent on a
    cycle that sent them nothing at all -- a subscriber's last unit
    disappearing with nothing to show for it. The charge now only
    happens on a genuine PUSH_DELIVERED outcome; a peek (not a consume)
    still gates entry so an exhausted subscriber's cycle never runs at
    all (see test_push_stops_and_notifies_admin_when_trial_limit_reached
    above)."""
    subscriber_ops.request_access(64, "victor", "Victor")
    subscriber_ops.decide(64, approved=True)
    subscriber_ops.set_pushes_remaining(64, 1)

    _cycle_with(monkeypatch, chat_id=64, digest="", mark_links_shown=MagicMock())

    assert subscriber_ops.get_pushes_remaining(64) == 1


def test_model_error_before_generation_does_not_advance_last_push_at(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    """Nothing was generated, so nothing was billed -- there is no reason to
    make the subscriber wait a full interval for a transient provider blip."""
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(26)])
    mark_links_shown = MagicMock()
    advance_last_push_at = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", advance_last_push_at)
    monkeypatch.setattr(news_cache, "read_all", lambda: [])
    monkeypatch.setattr(news_push, "resolve_interest_categories",
                        MagicMock(side_effect=RuntimeError("402 Insufficient Balance")))
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(),
                                         now=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)))

    assert recorded_outcomes(26) == [push_outcome_ops.PUSH_MODEL_ERROR]
    mark_links_shown.assert_not_called()
    advance_last_push_at.assert_not_called()


def test_model_error_after_generation_does_advance_last_push_at(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    """The guardrail check is an LLM call too, and by the time it runs the
    digest has already been written and billed."""
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(27)])
    mark_links_shown = MagicMock()
    advance_last_push_at = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", advance_last_push_at)
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles",
                        MagicMock(return_value=[{**_article("https://example.com/new"), "topic": "AI"}]))
    monkeypatch.setattr(news_push, "write_push_digest",
                        MagicMock(return_value='<b>Digest</b> <a href="https://example.com/new">s</a>'))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic",
                        MagicMock(side_effect=RuntimeError("rate limited")))
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(),
                                         now=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)))

    assert recorded_outcomes(27) == [push_outcome_ops.PUSH_MODEL_ERROR]
    mark_links_shown.assert_called_once()
    advance_last_push_at.assert_called_once()


def test_striking_out_leaves_the_subscriber_and_their_settings_intact(monkeypatch, isolated_subscribers_db):
    """Only push_enabled is cleared. A user who blocked the bot and later
    unblocks it turns push back on, rather than finding their interests
    gone."""
    subscriber_ops.set_interests(28, ["AI", "Robotics"])
    subscriber_ops.set_push_enabled(28, True)
    for _ in range(news_push.UNREACHABLE_STRIKES):
        _cycle_with(monkeypatch, chat_id=28, send=_failing_send(), mark_links_shown=MagicMock())

    assert subscriber_ops.get_push_enabled(28) is False
    assert subscriber_ops.get_interests(28) == ["AI", "Robotics"]


# --- heartbeat -------------------------------------------------------------


def test_push_tick_emits_a_heartbeat_even_when_nobody_is_due(monkeypatch, isolated_subscribers_db):
    """The reason this exists. Every LLM call in a push cycle sits inside
    the per-subscriber loop after the due check, so a tick where nobody is
    due emits no spans at all -- and the dead man's switch would read a
    perfectly healthy idle system as dead. See
    docs/plans/observability-platform-plan.md."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers",
                        lambda: [_subscriber(31, last_push_at=now - timedelta(hours=1), interval=24)])
    beats = []
    monkeypatch.setattr(news_push, "_emit_heartbeat", lambda n: beats.append(n))

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    assert beats == [1]


def test_push_tick_heartbeat_carries_the_subscriber_count(monkeypatch, isolated_subscribers_db):
    """Not just a pulse: the count is the number whose quiet growth was the
    2026-08-21 incident, so the liveness span answers that too."""
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers",
                        lambda: [_subscriber(32, interests=[]), _subscriber(33, interests=[])])
    beats = []
    monkeypatch.setattr(news_push, "_emit_heartbeat", lambda n: beats.append(n))

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock()))

    assert beats == [2]


def test_emit_heartbeat_is_a_noop_without_a_tracer_provider():
    """Hermeticity: with no provider configured -- every test and CI run --
    OpenTelemetry hands back a no-op tracer, so this needs no env guard and
    nothing to stub. If it ever raises here, it would raise in production
    the moment telemetry was switched off."""
    news_push._emit_heartbeat(3)


def test_a_failure_after_a_successful_send_still_retires_what_was_delivered(
        monkeypatch, isolated_subscribers_db):
    """The window between `await send(...)` returning and mark_links_shown/
    advance_last_push_at being reached contains a database write (the
    consecutive-failures reset, _record's own side effect for a `delivered`
    outcome). If it raises, the cycle lands in the catch-all handler -- and
    recording [] there would leave articles the subscriber genuinely
    received still eligible, so they would be sent a second time.

    Simulated by making that reset raise, which is the first write that
    runs after a successful send (see news_push._record)."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(41)])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles",
                        MagicMock(return_value=[{**_article("https://example.com/new"), "topic": "AI"}]))
    digest = '<b>Digest</b> 🔗 <a href="https://example.com/new">Source</a>'
    monkeypatch.setattr(news_push, "write_push_digest", MagicMock(return_value=digest))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", MagicMock(return_value=True))

    # Everything up to and including the send works; the consecutive-
    # failures reset for `delivered` does not. Transient rather than
    # permanent, because a permanently failing writer also breaks the
    # error handler's own _record call and aborts the whole tick -- a
    # separate weakness, noted in docs/plans/incident-monitoring-plan.md,
    # not what this pins.
    monkeypatch.setattr(subscriber_ops, "reset_push_consecutive_failures",
                        MagicMock(side_effect=[RuntimeError("database is locked"), None]))

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    mark_links_shown.assert_called_once_with(41, ["https://example.com/new"], now)


def test_a_failure_before_the_send_retires_nothing(monkeypatch, isolated_subscribers_db):
    """The other side of the same rule: a digest that was generated but
    never delivered must not retire its articles, or they are lost unread."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [_subscriber(42)])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)
    monkeypatch.setattr(news_push, "select_candidate_articles",
                        MagicMock(return_value=[{**_article("https://example.com/new"), "topic": "AI"}]))
    monkeypatch.setattr(news_push, "write_push_digest", MagicMock(return_value="<b>Digest</b>"))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic",
                        MagicMock(side_effect=RuntimeError("rate limited")))

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    mark_links_shown.assert_called_once_with(42, [], now)


def test_emit_html_validation_attempt_carries_attempt_and_validity(monkeypatch, isolated_subscribers_db):
    """The attributes an alert query needs to distinguish attempt 3 of 3
    (exhausted) from attempt 1 of 3 (unremarkable, expected to happen
    occasionally) -- this function itself makes no such distinction, it
    just reports the fact. Same FakeSpan pattern as the heartbeat/
    _record span tests below."""
    recorded = {}

    class FakeSpan:
        def set_attribute(self, k, v):
            recorded[k] = v
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(news_push._tracer, "start_as_current_span",
                        lambda name: FakeSpan())

    news_push._emit_html_validation_attempt(55, "AI", 3, "unescaped & at '&T'")

    assert recorded["push.subscriber"] == subscriber_ops.external_id(55)
    assert recorded["topic"] == "AI"
    assert recorded["attempt"] == 3
    assert recorded["valid"] is False
    assert recorded["reason"] == "unescaped & at '&T'"


def test_emit_html_validation_attempt_omits_reason_when_valid(monkeypatch, isolated_subscribers_db):
    recorded = {}

    class FakeSpan:
        def set_attribute(self, k, v):
            recorded[k] = v
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(news_push._tracer, "start_as_current_span",
                        lambda name: FakeSpan())

    news_push._emit_html_validation_attempt(55, "AI", 1, None)

    assert recorded["valid"] is True
    assert "reason" not in recorded


def test_heartbeat_span_carries_the_job_and_the_subscriber_count(monkeypatch):
    """The attributes, not just the call. The count is what makes this span
    useful beyond liveness -- its quiet growth was the 2026-08-21 incident
    -- so a future edit that drops it should fail here. Same FakeSpan
    pattern as test_news_sources' redaction test."""
    recorded = {}

    class FakeSpan:
        def set_attribute(self, k, v):
            recorded[k] = v
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(news_push._tracer, "start_as_current_span",
                        lambda name: FakeSpan())

    news_push._emit_heartbeat(7)

    assert recorded == {"heartbeat.job": "push_tick",
                        "heartbeat.push_enabled_subscribers": 7}


def test_record_emits_a_span_an_alert_can_query(monkeypatch, isolated_subscribers_db):
    """The third reader. Criteria 2 and 3 live in Logfire, and Logfire can
    only alert on what it receives -- the SQLite rows sit on the bot VM
    where the alerting engine cannot see them.

    `push.generated` is computed here rather than left for each alert query
    to re-derive the outcome set: the ratio's denominator has one
    definition, in push_outcome_ops, and a query that reimplements it would drift
    from it silently."""
    recorded = {}

    class FakeSpan:
        def set_attribute(self, k, v):
            recorded[k] = v
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(news_push._tracer, "start_as_current_span",
                        lambda name: FakeSpan())
    now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)

    news_push._record(55, push_outcome_ops.PUSH_CHAT_NOT_FOUND, "gone", now, detail="BadRequest")

    assert recorded["push.outcome"] == push_outcome_ops.PUSH_CHAT_NOT_FOUND
    assert recorded["push.generated"] is True   # billed: digest written before the send
    assert recorded["push.detail"] == "BadRequest"
    # The opaque id, never the Telegram one -- a span leaves this machine.
    assert "push.chat_id" not in recorded
    assert recorded["push.subscriber"] == subscriber_ops.external_id(55)
    # The raw chat_id itself must never be the recorded value -- NOT the
    # same check as "the substring '55' never appears anywhere in the
    # opaque id": that one is flaky by construction, since external_id's
    # no-subscriber-row fallback is a sha256 hex digest, and a random
    # hex string contains any given 2-character substring reasonably
    # often by pure chance, unrelated to whether anything actually leaked.
    assert recorded["push.subscriber"] != "55"
    assert recorded["push.subscriber"] != 55


def test_record_marks_a_cycle_that_cost_nothing_as_not_generated(monkeypatch, isolated_subscribers_db):
    """The other side of the denominator. A subscriber with nothing new
    costs no LLM call, and counting them would let a crowd of idle
    subscribers hide a collapsed delivery rate."""
    recorded = {}

    class FakeSpan:
        def set_attribute(self, k, v):
            recorded[k] = v
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(news_push._tracer, "start_as_current_span",
                        lambda name: FakeSpan())

    news_push._record(56, push_outcome_ops.PUSH_NOTHING_NEW, "quiet",
                      datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc))

    assert recorded["push.generated"] is False
    assert "push.detail" not in recorded      # nothing to say, so nothing sent


# --- one message per interest, rotated ------------------------------------
# The single combined digest these replace merged every interest into one
# candidate pool and one model call, which discards the specificity that
# made each interest findable in the first place -- see
# docs/analysis/cluster-measurements.md on any category layer between the
# interest and the articles costing recall.


def _per_topic_cycle(monkeypatch, chat_id, articles_by_topic, subscriber=None,
                     now=None, send=None, on_topic=True, mark_links_shown=None):
    """A cycle where each interest has its own candidate articles, so the
    per-interest loop is actually exercised rather than collapsing onto one."""
    now = now or datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    sub = subscriber or _subscriber(chat_id, interests=list(articles_by_topic))
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [sub])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown or MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)

    def fake_select(cached, topics, cats, since, already_pushed, **kwargs):
        topic = topics[0]
        return [{**_article(link), "topic": topic}
                for link in articles_by_topic.get(topic, [])
                if link not in already_pushed]

    monkeypatch.setattr(news_push, "select_candidate_articles", fake_select)
    # Echoes every candidate's link back, so links_actually_sent sees them.
    monkeypatch.setattr(
        news_push, "write_push_digest",
        lambda model, arts, topic=None, language=None, retry_reason=None: "<b>D</b> " + " ".join(
            f'<a href="{a["link"]}">s</a>' for a in arts))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic",
                        MagicMock(return_value=on_topic))
    send = send or AsyncMock()
    asyncio.run(news_push.run_push_cycle(model="fake-model", send=send, now=now))
    return send


def test_each_interest_gets_its_own_message(monkeypatch, isolated_subscribers_db):
    send = _per_topic_cycle(monkeypatch, 40, {
        "AI": ["https://e.com/ai"],
        "Robotics": ["https://e.com/robots"],
    })

    assert send.await_count == 2
    bodies = " ".join(call.args[1] for call in send.await_args_list)
    assert "https://e.com/ai" in bodies and "https://e.com/robots" in bodies


def test_several_messages_still_record_one_outcome(monkeypatch, isolated_subscribers_db, recorded_outcomes):
    """One row per subscriber per cycle, NOT one per message. The three live
    alert criteria are thresholds over this table, and a cycle that emits N
    rows instead of 1 silently rescales every one of them."""
    _per_topic_cycle(monkeypatch, 41, {
        "AI": ["https://e.com/ai"],
        "Robotics": ["https://e.com/robots"],
        "Optics": ["https://e.com/optics"],
    })

    assert recorded_outcomes(41) == [push_outcome_ops.PUSH_DELIVERED]


def test_an_interest_with_nothing_new_does_not_consume_a_slot(
    monkeypatch, isolated_subscribers_db
):
    """The cap counts messages sent, not interests examined -- otherwise a
    quiet interest sitting at the front of the rotation would spend the
    whole cycle's budget on nothing."""
    monkeypatch.setattr(news_push, "MAX_INTERESTS_PER_PUSH", 2)
    send = _per_topic_cycle(monkeypatch, 42, {
        "AI": ["https://e.com/ai"],
        "Quiet": [],
        "Robotics": ["https://e.com/robots"],
    })

    assert send.await_count == 2
    bodies = " ".join(call.args[1] for call in send.await_args_list)
    assert "https://e.com/robots" in bodies


def test_the_cap_defers_interests_rather_than_dropping_them(
    monkeypatch, isolated_subscribers_db
):
    """Longest-un-pushed first, so the queue drains over cycles instead of
    permanently starving whatever sorts last."""
    monkeypatch.setattr(news_push, "MAX_INTERESTS_PER_PUSH", 1)
    by_topic = {"AI": ["https://e.com/ai"], "Robotics": ["https://e.com/robots"]}
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)

    first = _per_topic_cycle(monkeypatch, 43, by_topic, now=now)
    assert first.await_count == 1
    assert "https://e.com/ai" in first.await_args_list[0].args[1]

    # Second cycle: AI has now been pushed, Robotics never has.
    second = _per_topic_cycle(
        monkeypatch, 43, by_topic,
        subscriber=_subscriber(43, interests=["AI", "Robotics"]),
        now=now + timedelta(hours=25))
    assert second.await_count == 1
    assert "https://e.com/robots" in second.await_args_list[0].args[1]


def test_an_article_matching_two_interests_is_sent_once(
    monkeypatch, isolated_subscribers_db
):
    """mark_links_shown only runs at the end of the cycle, so pushed_links
    cannot know what this cycle already delivered. Without the union, a
    shared article goes out twice in the same push."""
    shared = "https://e.com/shared"
    send = _per_topic_cycle(monkeypatch, 44, {"AI": [shared], "Robotics": [shared]})

    assert send.await_count == 1


def test_only_interests_actually_sent_are_marked_pushed(
    monkeypatch, isolated_subscribers_db
):
    _per_topic_cycle(monkeypatch, 45, {"AI": ["https://e.com/ai"], "Quiet": []})

    ordered = subscriber_ops.interests_by_staleness(45, ["AI", "Quiet"])
    assert ordered == ["Quiet", "AI"], "the un-served interest must still lead"


def test_a_send_failure_keeps_what_was_already_delivered(
    monkeypatch, isolated_subscribers_db, recorded_outcomes
):
    """A send can succeed and a later one fail. Recording [] there would
    leave articles the subscriber genuinely received still eligible, and
    they would be sent again."""
    mark_links_shown = MagicMock()
    send = AsyncMock(side_effect=[None, Exception("Chat not found")])
    _per_topic_cycle(monkeypatch, 46, {
        "AI": ["https://e.com/ai"],
        "Robotics": ["https://e.com/robots"],
        "Optics": ["https://e.com/optics"],
    }, send=send, mark_links_shown=mark_links_shown)

    assert send.await_count == 2, "the third interest must not be attempted"
    assert mark_links_shown.call_args.args[1] == ["https://e.com/ai"]
    assert recorded_outcomes(46) == [push_outcome_ops.PUSH_CHAT_NOT_FOUND]


def test_every_interest_blocked_records_blocked_not_delivered(
    monkeypatch, isolated_subscribers_db, recorded_outcomes
):
    _per_topic_cycle(monkeypatch, 47, {
        "AI": ["https://e.com/ai"], "Robotics": ["https://e.com/robots"],
    }, on_topic=False)

    assert recorded_outcomes(47) == [push_outcome_ops.PUSH_BLOCKED]


def test_no_interest_having_anything_new_records_nothing_new(
    monkeypatch, isolated_subscribers_db, recorded_outcomes
):
    _per_topic_cycle(monkeypatch, 48, {"AI": [], "Robotics": []})

    assert recorded_outcomes(48) == [push_outcome_ops.PUSH_NOTHING_NEW]


def test_a_partial_block_is_visible_in_the_delivered_detail(
    monkeypatch, isolated_subscribers_db, recorded_outcomes
):
    """Some interests delivered, one blocked, in the same cycle. The
    delivered-wins branch must not swallow the blocked count -- an
    intermittently-misfiring guardrail on one interest would otherwise be
    invisible for as long as the subscriber's other interests kept
    succeeding."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    sub = _subscriber(49, interests=["AI", "Robotics"])
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [sub])
    mark_links_shown = MagicMock()
    advance_last_push_at = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", advance_last_push_at)
    _stub_cache_and_categories(monkeypatch)

    by_topic = {"AI": ["https://e.com/ai"], "Robotics": ["https://e.com/robots"]}

    def fake_select(cached, topics, cats, since, already_pushed, **kwargs):
        topic = topics[0]
        return [{**_article(link), "topic": topic}
                for link in by_topic.get(topic, []) if link not in already_pushed]

    monkeypatch.setattr(news_push, "select_candidate_articles", fake_select)
    monkeypatch.setattr(
        news_push, "write_push_digest",
        lambda model, arts, topic=None, language=None, retry_reason=None: "<b>D</b> " + " ".join(
            f'<a href="{a["link"]}">s</a>' for a in arts))
    # AI's digest passes the guardrail, Robotics' is blocked.
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic",
                        lambda model, digest: "https://e.com/ai" in digest)

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    assert recorded_outcomes(49) == [push_outcome_ops.PUSH_DELIVERED]
    mark_links_shown.assert_called_once()  # not one call per interest
    advance_last_push_at.assert_called_once()  # not one call per interest
    # The detail, not just the outcome name, must show the block.
    assert recorded_outcomes.detail_for(49) == "1 interest(s), 1 blocked"


def test_a_later_interests_model_failure_still_records_delivered(
    monkeypatch, isolated_subscribers_db, recorded_outcomes
):
    """Real incident, found live on INT 2026-09-03: interest N delivered,
    interest N+1's model call then raised OpenAITimeoutError, and the
    cycle's one outcome/span came out model_error despite
    message_archive/interest_push_state both showing the successful send.
    `sent` must win over a later interest's model failure, same "sent
    wins, fold the failure into detail" rule as the blocked case above."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    sub = _subscriber(50, interests=["AI", "Robotics"])
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [sub])
    mark_links_shown = MagicMock()
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", mark_links_shown)
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)

    by_topic = {"AI": ["https://e.com/ai"], "Robotics": ["https://e.com/robots"]}

    def fake_select(cached, topics, cats, since, already_pushed, **kwargs):
        topic = topics[0]
        return [{**_article(link), "topic": topic}
                for link in by_topic.get(topic, []) if link not in already_pushed]

    monkeypatch.setattr(news_push, "select_candidate_articles", fake_select)
    # AI's digest generates fine; Robotics' model call raises.
    monkeypatch.setattr(
        news_push, "write_push_digest",
        MagicMock(side_effect=[
            '<b>D</b> <a href="https://e.com/ai">s</a>',
            RuntimeError("Request timed out."),
        ]))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", lambda model, digest: True)

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    assert recorded_outcomes(50) == [push_outcome_ops.PUSH_DELIVERED]
    mark_links_shown.assert_called_once_with(50, ["https://e.com/ai"], now)
    detail = recorded_outcomes.detail_for(50)
    assert "1 interest(s) sent" in detail
    assert "model call failed" in detail


def test_a_later_interests_model_failure_still_charges_the_trial_allowance(
    monkeypatch, isolated_subscribers_db,
):
    """The real gap qa-engineer found, 2026-09-19: PUSH_DELIVERED can also
    be recorded from the "sent wins over a later failure" exception
    branches above (same real 2026-09-03 incident this test mirrors), not
    just the normal end-of-loop branch -- a subscriber with a finite
    allowance who genuinely received a digest through THIS path was never
    being charged for it, a real free-push leak. Same setup as
    test_a_later_interests_model_failure_still_records_delivered above,
    minus the outcome/detail assertions that test already covers."""
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    subscriber_ops.request_access(65, "yolanda", "Yolanda")
    subscriber_ops.decide(65, approved=True)
    subscriber_ops.set_pushes_remaining(65, 1)
    sub = _subscriber(65, interests=["AI", "Robotics"])
    monkeypatch.setattr(subscriber_ops, "list_push_enabled_subscribers", lambda: [sub])
    monkeypatch.setattr(subscriber_ops, "mark_links_shown", MagicMock())
    monkeypatch.setattr(subscriber_ops, "advance_last_push_at", MagicMock())
    _stub_cache_and_categories(monkeypatch)

    by_topic = {"AI": ["https://e.com/ai2"], "Robotics": ["https://e.com/robots2"]}

    def fake_select(cached, topics, cats, since, already_pushed, **kwargs):
        topic = topics[0]
        return [{**_article(link), "topic": topic}
                for link in by_topic.get(topic, []) if link not in already_pushed]

    monkeypatch.setattr(news_push, "select_candidate_articles", fake_select)
    monkeypatch.setattr(
        news_push, "write_push_digest",
        MagicMock(side_effect=[
            '<b>D</b> <a href="https://e.com/ai2">s</a>',
            RuntimeError("Request timed out."),
        ]))
    monkeypatch.setattr(news_push.guardrails, "is_output_on_topic", lambda model, digest: True)

    asyncio.run(news_push.run_push_cycle(model="fake-model", send=AsyncMock(), now=now))

    assert subscriber_ops.get_pushes_remaining(65) == 0

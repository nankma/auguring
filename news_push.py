"""
Periodic news-push digest: read the shared local cache, filter to new +
relevant, write, send. See docs/plans/bot-features-plan.md item 5 and
docs/plans/local-news-cache-plan.md's "Interaction with news_push.py" section.

Reads from news_cache.py (populated on a schedule by news_ingest.py)
rather than calling news_sources.py live -- converged onto the cache
2026-08-15, per docs/plans/local-news-cache-plan.md's item 6. Before this, every
push cycle called every enabled source live for every subscriber's every
interest, with no relevance filter beyond "does the source's own query
support happen to work" -- most sources are query-blind RSS feeds that
return their latest N items regardless of query (see news_sources.py),
so a subscriber's digest could and did include whatever a broad
mainstream-press feed's front page happened to be that day (a real
incident: Nikkei Asia's general top-stories feed put an Indonesia
earthquake and a Japan-society piece into a push digest -- nothing tech-
related about either, and nothing in the old pipeline could have caught
it, since no relevance check existed before the digest-writing prompt).

Two-stage filtering, per docs/plans/local-news-cache-plan.md:
  Stage 1 (category filter, in code) -- select_candidate_articles narrows
  the shared cache to articles whose classifier-assigned categories
  overlap with a subscriber's own classified interests, before the model
  ever sees anything.
  Stage 2 (content filter) -- folded into the existing single
  write_push_digest call rather than a separate LLM call: the prompt now
  explicitly tells the model to still exercise judgment and omit
  candidates that survived the category filter but aren't genuinely
  about the topic, instead of forcing everything given into the report.

Deliberately still does NOT go through agent.py's tool-calling agent
(build_agent/run_agent): a single plain model.invoke (no tool loop) is
enough once the candidate list is already assembled in code -- there's
nothing left for tools to do.

"New" separates two things that were previously conflated, and the
distinction is load-bearing -- see select_candidate_articles:

- FILTERING is done by subscriber_ops's pushed_links alone. "Already seen" means
  "we actually sent this to this subscriber", nothing else.
- RANKING is done by published_dt, newest first, with a maximum-age gate.
  A publish date decides what's most worth showing, never what's eligible.

Filtering on the date was the earlier design and it was wrong: an article
that lost one cycle's cut was excluded permanently, without ever having
been seen by anyone.
"""

from datetime import datetime, timedelta, timezone

import agent
from app_settings import get_settings
import guardrails
import message_archive
import news_cache
import news_classify
import news_embed
import news_keyness
import news_sources
import telegram_html
import category_ops
import interest_cache_ops
import push_outcome_ops
import subscriber_ops
from opentelemetry import trace

# How many messages one push cycle may send to one subscriber -- one per
# interest, so this is also "how many interests get served this cycle".
#
# Bounded because a subscriber at subscriber_ops.MAX_INTERESTS on a 8h interval
# would otherwise receive 30 messages a day. Interests that don't fit
# aren't dropped, they wait: interests_by_staleness puts the
# longest-un-pushed first, so the queue drains over successive cycles.
MAX_INTERESTS_PER_PUSH = get_settings().resolved("push.max_interests_per_push", default=5)

MAX_ARTICLES_PER_TOPIC = get_settings().resolved("push.max_articles_per_topic", default=5)

# Whether a push includes the experimental "novelty extra" -- a single,
# clearly-separate closing note ("by the way, we noticed something
# interesting"), NOT one of MAX_ARTICLES_PER_TOPIC's regular slots. See
# select_candidate_articles's and _pick_novelty_extra's own docstrings,
# and docs/analysis/cluster-measurements.md's "offbeat score" and
# "Offbeat selection, take two" sections for the measurement history.
# Design settled 2026-08-26, after a user-directed correction to an
# earlier version that (a) carved this out of the regular 5-article
# budget instead of adding it on top, and (b) always picked SOMETHING
# even when nothing genuinely qualified -- this version does neither:
# it's additive, and NOVELTY_KEYNESS_THRESHOLD below means a cycle with
# nothing that clears the bar sends no novelty section at all, on
# purpose, not a gap to fix. False -- not 0 -- disables the feature
# entirely and falls back to a digest with no extra section, a safe
# rollback with no code change.
INCLUDE_NOVELTY_EXTRA = True

# How far below its own overall corpus-wide rate a word's presence in one
# interest's category pool must fall before an article containing it
# counts as "novel enough" for the extra section on keyness grounds alone
# (a keyword hit -- news_keyness.NOVELTY_KEYWORDS -- always counts,
# unconditionally; this threshold only gates the keyness-only case).
# First-cut, deliberately conservative rather than tuned: this is an
# explicitly experimental feature (not required to be precise -- an
# occasional pick that isn't genuinely novel is an accepted cost), and
# real measured foreign-word scores on real production data clustered
# well past this (-19 to -40 for clearly topic-foreign terms like
# "quantum" in an AI pool -- see docs/analysis/cluster-measurements.md),
# so -5.0 is comfortably permissive without also passing near-zero noise.
# Adjust against real push data once some has accumulated.
NOVELTY_KEYNESS_THRESHOLD = -5.0

# How many times write_push_digest gets re-invoked (with the previous
# failure's reason fed back in, see that function's retry_reason param)
# before giving up on getting valid Telegram HTML out of the model and
# just sending what it produced -- send_push_digest's own BadRequest
# fallback (bot.py) still strips it to plain text if Telegram also
# rejects it. 3 is a first-cut: this is a rare failure mode (an unescaped
# & in an article title is the one confirmed real-world cause so far, see
# _emit_html_validation_attempt), so the cost of a couple of wasted
# retries on the rare cycle that needs them is cheap relative to genuinely
# broken formatting reaching a subscriber.
MAX_HTML_ATTEMPTS = get_settings().resolved("push.max_html_attempts", default=3)

# news_embed.filter_by_relevance keeps at most this many of a topic's pool, ranked
# by similarity to the topic's retrieval query -- not a fixed fraction
# any more. Three real findings, all from the same 2026-08-25
# measurement session, forced this shape:
#
# 1. A fixed fraction cannot work across different pool sizes. This
#    session's measurement pulled the REAL production cache and found
#    999 live "AI"-category articles within one 48h cache TTL window at
#    once -- a fraction tuned for a small pool either starves a 999-item
#    one or barely filters it at all, depending which end you tune for.
# 2. A fixed fraction cannot work across different topic PHRASINGS
#    either, independent of pool size -- "AI Agent"'s genuinely-relevant
#    articles scored 0.53-0.72 cosine similarity on one corpus, while
#    "Large Language Model" against the SAME corpus topped out at 0.166
#    for its single best candidate. Absolute score, and therefore any
#    fraction-of-the-score-range framing, isn't comparable across
#    queries.
# 3. Recall past a certain point stops being worth chasing. Measured
#    with a carefully hand-verified 24-article ground truth on the real
#    999-article corpus (not the earlier, since-retracted 40-article
#    sample -- see docs/analysis/cluster-measurements.md): every fraction
#    from 60% to 90% recovers at least 22 of 24, and the last 2 --
#    "How Virgin Atlantic ships faster with Codex", "Asana cleared 5
#    years of engineering work in 2 weeks with Codex" -- only clear a
#    'business outcome' headline with none of the definition's technical
#    vocabulary in it, ranking near the very bottom of a 999-item pool
#    even though the content is genuinely on-topic. Chasing literally
#    every last one means keeping ~90% of the pool, which stops being a
#    filter. The project's own standard (this session, discussing where
#    to draw this exact line): missing a couple of genuine matches out
#    of a pool already offering 20+ good candidates is fine, since only
#    MAX_ARTICLES_PER_TOPIC=5 are ever pushed regardless -- a candidate
#    ranked near the bottom of a 20+ item pool was never going to be one
#    of the 5 sent anyway.
#
# So: an ABSOLUTE count, clamped between a floor and a ceiling rather
# than a percentage of a variable-sized, variable-scaled pool --
#
#   n_keep = min(RELEVANCE_KEEP_MAX, max(round(pool * RELEVANCE_KEEP_FRACTION), RELEVANCE_KEEP_MIN))
#
# RELEVANCE_KEEP_FRACTION=0.10 with a 20-50 clamp: a narrow topic's small
# pool gets the 20-item floor rather than being starved by 10% of a small
# number; a broad topic's pool (hundreds, per the 999-article measurement
# -- and uncapped by count going in, see the raw_pool loop below) gets
# capped at 50 rather than passing hundreds of candidates through to
# write_push_digest's single LLM judgment call. All three numbers are
# first-cut, not re-validated end to end -- adjust freely; they're
# independent named constants specifically so that's cheap.
RELEVANCE_KEEP_FRACTION = get_settings().resolved("push.relevance_keep.fraction", default=0.10)
RELEVANCE_KEEP_MIN = get_settings().resolved("push.relevance_keep.min", default=20)
RELEVANCE_KEEP_MAX = get_settings().resolved("push.relevance_keep.max", default=50)

# A separate, wider clamp used ONLY for the novelty-extra search (see
# _pick_novelty_extra and its call site below), not the regular digest.
# Added 2026-08-27 after a real incident (this module's docstring) where
# an off-topic article reached a subscriber via the novelty pick; the
# ROOT cause was an uncategorized topic being treated as unrestricted
# (fixed separately, see select_candidate_articles), but the incident
# also exposed that novelty eligibility had no relevance floor of its
# OWN wider than "whatever's left after the regular cut" -- a keyword or
# keyness hit could surface something the regular relevance filter had
# already ranked as barely-relevant. This keeps that floor, but
# deliberately wider than RELEVANCE_KEEP_* (roughly 2x, a first-cut
# guess like NOVELTY_KEYNESS_THRESHOLD's -5.0 -- adjust with real push
# data): novelty content is explicitly allowed to be "not so important"
# per the feature's own design (see _pick_novelty_extra), so requiring
# it to clear the SAME strict bar as a regular candidate would defeat
# the point -- this only needs to prove "still related to topic", not
# "would have made the regular cut".
NOVELTY_RELEVANCE_KEEP_FRACTION = get_settings().resolved("push.novelty_relevance_keep.fraction", default=0.20)
NOVELTY_RELEVANCE_KEEP_MIN = get_settings().resolved("push.novelty_relevance_keep.min", default=40)
NOVELTY_RELEVANCE_KEEP_MAX = get_settings().resolved("push.novelty_relevance_keep.max", default=100)

# The relevance-filtered pool is capped here again before the regular
# recency cut and novelty-extra search run -- matches RELEVANCE_KEEP_MAX
# exactly, not some smaller number, specifically so THIS cap is never the
# tighter, silently-binding one. It was 30 until 2026-08-25 (a guess made
# before RELEVANCE_KEEP_MAX existed), which would have made
# RELEVANCE_KEEP_MAX=50 a dead ceiling -- the exact bug this whole
# revision exists to stop repeating. Named for what it bounds now
# (renamed from OFFBEAT_POOL_SIZE 2026-08-26 when novelty selection
# stopped being carved out of this same pool and started drawing from
# the remainder beyond it instead).
CANDIDATE_POOL_SIZE = RELEVANCE_KEEP_MAX

# Above this cosine similarity, two articles collapse to one -- the same
# wire story cached under two links, or two outlets syndicating one
# piece. Not a title-match: docs/analysis/cluster-measurements.md found
# genuine duplicates (9 gnews copies of one syndicated story) and
# same-titled non-duplicates (BBC's "Tech Now"/"Tech Life" programme
# titles, a different episode every time) indistinguishable by title
# alone -- only content similarity tells them apart.
NEAR_DUPLICATE_SIMILARITY = get_settings().resolved("push.near_duplicate_similarity", default=0.95)

# Ceiling on how old an article's own publication date may be to still be
# pushed, regardless of when we downloaded it. Needed as of 2026-08-19,
# when candidate selection switched from published_dt to fetched_at (see
# select_candidate_articles): "new to us" is the right test for a source
# that publishes on a delay, but on its own it would also happily push
# genuinely ancient content. Real case that forced this: Perigon's one
# successful fetch returned 50 articles whose NEWEST was over a year old
# (docs/plans/security-plan.md finding 21) -- under a fetched_at-only rule
# every one of them would have been eligible.
#
# 7 days is deliberately generous rather than tight. It exists to exclude
# absurdly stale content, not to enforce freshness -- the fetched_at check
# already does that. arXiv's own indexing runs ~3 days behind
# (docs/current/ai-news-sources.md), and those papers are legitimately worth
# sending, so a tighter bound would silently drop a real source.
MAX_ARTICLE_AGE_HOURS = get_settings().resolved("push.max_article_age_hours", default=168)

_PUSH_DIGEST_PROMPT = (
    "You are a technology industry analyst writing a periodic news digest "
    "for a Telegram subscriber, covering AI and the broader tech industry. "
    "Below is a list of candidate articles that are new since their last "
    "digest, grouped by the topic they matched during a coarse category "
    "filter. That filter is not perfect -- some candidates may not "
    "actually be about the subscriber's topic, or may not be genuinely "
    "tech-industry content at all (e.g. general news that happened to "
    "come from a source that also covers tech). Use your own judgment: "
    "write a short trend report covering ONLY the candidates that are "
    "genuinely relevant, silently omitting any that aren't -- do not "
    "force an irrelevant candidate into the report just because it was "
    "in the list, and do not invent or reference anything not in the "
    "list either. If NONE of the candidates are genuinely relevant, "
    "write nothing (an empty reply) rather than reporting on off-topic "
    "content. Do not mention that this is an automated or periodic "
    "message, and do not mention the filtering process itself.\n\n"
    + agent.HTML_FORMATTING_RULES
    + "\n\n"
    + agent.TREND_REPORT_STRUCTURE
)


def backfill_missing_interest_definitions(model, interests: list[str]) -> None:
    """Stage-1 setup sibling to resolve_interest_categories, for the
    definition cache rather than the category one. Fixes a real gap
    (docs/plans/interest-definition-plan.md "Defect 2"): add_one_interest
    generates a definition only ONCE, at add time -- an interest added
    before expand_interest_for_retrieval existed, or one whose one-shot
    generation call failed back then, has no row in
    interest_query_expansions and nothing ever retries it, because
    _resolve_query_text deliberately never calls the LLM (see its own
    docstring). A subscriber measured live with `AI` in exactly this
    state: no definition, retrieval falling back to embedding the bare
    two-character string, which is a near-meaningless query
    (docs/analysis/retrieval-quality-measurements.md finding 4).

    Same cache-check-then-generate shape as resolve_interest_categories,
    same cache (interest_cache_ops's SHARED/global tier -- the same one
    add_one_interest already writes to, so this is a retroactive fill of
    that cache, not a new one). No return value: unlike categories, which
    select_candidate_articles needs threaded through as a dict, nothing
    downstream needs the definitions handed back -- _resolve_query_text
    reads the cache itself, per topic, right after this runs."""
    missing = [i for i in interests if interest_cache_ops.get_interest_query_expansion(i) is None]
    for interest in missing:
        expansion = news_classify.expand_interest_for_retrieval(model, interest)
        if expansion is not None:
            interest_cache_ops.set_interest_query_expansion(interest, expansion)


def resolve_interest_categories(model, interests: list[str]) -> dict[str, list[str]]:
    """Stage-1 setup: maps each interest to its category tags, using
    interest_cache_ops's persistent cache and classifying only what's missing --
    interest text is stable vocabulary (unlike article content), so this
    should be a cache hit for any interest that's been pushed before.

    An interest the classifier FAILED on is left out of the cache entirely,
    so the next cycle retries it. Only a real answer gets cached -- including
    a genuinely empty one, which is a valid classification and should not be
    re-paid for every cycle.

    Caching failures as [] is what poisoned the live cache during the
    2026-08-17 classification outage: "AI", "Bitcoin", "機器人科技" and
    others were all stored as [], permanently, with no retry. At the time,
    an empty mapping matched every article (see select_candidate_articles),
    so those subscribers were receiving entirely unfiltered news -- the
    exact failure the category filter exists to prevent. That
    "unrestricted" behavior was itself reversed 2026-08-27 (an empty
    mapping now excludes the topic entirely instead), but caching a
    genuinely-empty classification permanently is still correct and
    unrelated to that reversal: it's what makes a confidently-wrong
    empty answer a standing condition rather than a one-cycle blip, on
    either side of how select_candidate_articles handles it -- see that
    function's own docstring."""
    resolved = interest_cache_ops.get_cached_interest_categories(interests)
    missing = [i for i in interests if i not in resolved]
    if missing:
        taxonomy = news_classify.Taxonomy.from_rows(category_ops.get_active_categories())
        newly_classified = news_classify.classify_interests(model, missing, taxonomy)
        failed = [i for i in missing if i not in newly_classified]
        for interest, categories in newly_classified.items():
            interest_cache_ops.set_interest_categories(interest, categories)
            resolved[interest] = categories
        if failed:
            print(f"[news_push] could not classify {len(failed)} interest(s), "
                  f"will retry next cycle: {', '.join(failed)}")
    return resolved


def select_candidate_articles(
    cached_articles: list[dict],
    topics: list[str],
    topic_categories: dict[str, list[str]],
    since: datetime | None,
    already_pushed_links: set[str],
    include_restricted: bool = False,
    max_per_topic: int = MAX_ARTICLES_PER_TOPIC,
    now: datetime | None = None,
    embedder=None,
    include_novelty_extra: bool = INCLUDE_NOVELTY_EXTRA,
    chat_id: int | None = None,
) -> list[dict]:
    """Stage 1 (category filter): narrows the shared cache to one
    subscriber's candidate articles, before the digest-writing model
    (stage 2, in write_push_digest's prompt) ever sees anything.

    An article is a candidate for `topic` when its own categories (set at
    ingestion time by news_classify.py) overlap with that topic's mapped
    categories from `topic_categories`. A topic that mapped to NO
    categories at all (a classifier miss) gets NO candidates at all --
    that interest is simply skipped this cycle (same downstream shape as
    "nothing new since last time": run_push_cycle doesn't send a message
    or consume a slot for it), rather than the "unrestricted, matches any
    article" behavior this had until 2026-08-27.

    That earlier behavior was a REPEAT of an already-diagnosed failure,
    not a new one: this exact docstring already recorded, from the
    2026-08-17 classification outage, that an empty mapping matching
    every article was "the exact failure the category filter exists to
    prevent" -- the outage's own retry logic got fixed, but the
    consequence of a *confidently-empty* classification (not a retryable
    failure, a real cached "[]" the classifier itself returned) was left
    unchanged. It surfaced for real 2026-08-27: an interest literally
    named "robotics" was classified into zero of the taxonomy's 28
    categories (including "Robotics" itself -- a genuine classifier
    misjudgment, not a system bug) and, because that made it
    "unrestricted," the subscriber's push digest surfaced a Witcher 3
    game-DLC article with a keyword-hit-triggered novelty note attached,
    entirely unrelated to robotics. Skipping the topic instead is a
    real behavior tradeoff, not a strict improvement: a subscriber whose
    interest was ACTUALLY too novel/niche for the current 28-category
    taxonomy to recognize now gets nothing at all for it, silently,
    rather than an imperfectly-filtered digest -- see resolve_
    interest_categories' own docstring for why a cached "[]" is never
    retried, which makes this a standing condition, not a one-cycle
    blip. Accepted as the better failure mode: an occasional legitimate
    interest going quiet is a worse experience to eventually notice and
    report than an unrelated-content leak is to receive without warning.

    An article with no categories (the classifier found nothing that
    plausibly applies, e.g. a general-news piece with no tech angle at
    all) is excluded whenever the topic itself has real categories to
    match against -- this is the exact mechanism that would have kept
    the Nikkei Asia earthquake/society articles out of a digest, since
    neither classifies into any of the 13 tech-industry categories.

    Since 2026-08-20 the cache distinguishes "the classifier found nothing
    applicable" (category_ops.UNCLASSIFIABLE) from "never classified" (None).
    This function deliberately still treats them alike -- neither has a
    category that can match a topic, so both are excluded the same way --
    but the distinction now exists in the data for anything that needs it,
    which is how a silent classification outage becomes detectable.
    Known, accepted overlap with a separate case this can't distinguish:
    an article left uncategorized because news_classify.classify_articles
    failed for its whole ingestion batch (fails open, see that
    function's docstring) looks identical to a genuine "nothing applies"
    result here -- both get excluded. Not solved here; a batch
    classification failure already means that cycle's articles are
    "harder to find... until the next cycle re-fetches and reclassifies
    it," per that function's own docstring, so this is consistent with
    an existing accepted limitation, not a new one.

    **A date is a ranking signal, never a "have they seen it" filter.**
    Restructured 2026-08-19 around that separation:

    - **Filter** -- `already_pushed_links`, and nothing else. That set is
      the direct record of what this subscriber was actually sent, so it
      answers the question exactly rather than approximating it.
    - **Rank** -- `published_dt`, newest first. What a date is genuinely
      good for.
    - **Quality gate** -- `MAX_ARTICLE_AGE_HOURS`, which drops absurdly
      stale articles regardless of when we downloaded them. A gate on
      worth-sending, not on already-seen.

    Why this matters, from two real failures. The original code filtered on
    `published_dt <= since`, which structurally excluded every source with
    a publication delay: GNews publishes ~12h behind, so for any subscriber
    pushed more recently than that, its articles were skipped outright --
    227 of them sat in the cache, correctly fetched and classified, and not
    one could ever reach a digest. Switching that filter to `fetched_at`
    fixed the delay case but kept the deeper flaw: an article that was a
    candidate and simply lost the `max_per_topic` cut would have its
    timestamp fall behind the next `since` and be excluded forever, unsent
    and unrecorded.

    Both failures come from the same mistake -- using a timestamp as a
    proxy for "already seen" when the actual record exists. Anything not
    yet sent stays a candidate until it is sent or ages out of the cache,
    so the pool drains in publication order and nothing starves.

    Restricted-source gating is unchanged: NewsAPI/Perigon articles are
    skipped entirely unless `include_restricted`.

    `since` is retained in the signature but no longer filters -- it stays
    only because callers already pass it and removing it would be a
    breaking change for no gain. `now` is a parameter rather than read from
    the clock so the age guard is deterministic in tests, same convention
    as run_push_cycle's own `now`.

    Three more steps run per topic before the `max_per_topic` cut, all new
    2026-08-25 and all optional (a missing embedder, or an article
    missing its own embedding, degrades each one independently rather
    than failing the whole call -- see news_embed's module docstring):

    - **Near-duplicate collapse** -- the same wire story cached under two
      links, or two outlets syndicating one piece, counts once. See
      NEAR_DUPLICATE_SIMILARITY and docs/analysis/cluster-measurements.md
      on why this can't be title matching (BBC's "Tech Now"/"Tech Life"
      programme titles repeat with different content every episode; 9
      gnews copies of one syndicated story were genuine duplicates under
      DIFFERENT titles).
    - **Relevance filter** -- a fine pass after the coarse category one,
      using the TOPIC STRING itself (not its category) as a retrieval
      query. Found necessary from a live user report: "AI", "AI Agent",
      "AI coding" and "Large Language Model" all map to category `AI`,
      so all four drew from the same undifferentiated pool of generic
      AI-industry news -- the category filter alone cannot tell them
      apart. See news_embed.filter_by_relevance for why this keeps an absolute,
      clamped count of the pool's top scorers (RELEVANCE_KEEP_MIN to
      RELEVANCE_KEEP_MAX), not a fixed threshold or a fraction of the
      pool -- both were tried and measured not to generalize.
    - **Novelty extra** -- a SEPARATE, additive pick, not one of
      `max_per_topic`'s regular slots: a novelty-keyword hit or an
      article whose own vocabulary is statistically foreign to this
      topic's category pool, ONLY when one clears NOVELTY_KEYNESS_
      THRESHOLD -- no forced pick when nothing qualifies. Drawn from a
      SEPARATE, wider relevance pass over `raw_pool` (NOVELTY_RELEVANCE_
      KEEP_*, added 2026-08-27 alongside the fix above -- the "skip an
      uncategorized topic entirely" reversal alone still left novelty
      picks with no relevance floor of their own beyond "whatever the
      regular digest's stricter cut left over") -- still required to be
      topically relevant, just against a more permissive bar than the
      regular candidates, since novelty content is allowed to be minor
      but not allowed to be unrelated. See _pick_novelty_extra."""
    now = now or datetime.now(timezone.utc)
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    ordered = sorted(
        cached_articles,
        key=lambda a: a.get("published_dt") or epoch,
        reverse=True,
    )
    max_age = timedelta(hours=MAX_ARTICLE_AGE_HOURS)
    seen_links = set()
    candidates = []
    for topic in topics:
        topic_cats = set(topic_categories.get(topic, []))
        if not topic_cats:
            # No categories to match against at all -- skip this topic
            # entirely rather than treating it as "matches everything"
            # (removed 2026-08-27; see this function's own docstring for
            # the incident that motivated the reversal). Same downstream
            # shape as "nothing new since last time": no candidates, no
            # message, no slot consumed.
            continue
        # The embedding query text -- NOT necessarily `topic` itself. See
        # _resolve_query_text: a cached, LLM-generated definition of the
        # topic outperforms the bare topic string as a retrieval query,
        # measurably (docs/analysis/cluster-measurements.md). `topic`
        # itself is still what's used for the category-match check right
        # below and for tagging candidates -- only the embedding calls
        # use `query_text`. `chat_id` (this function's own parameter)
        # lets _resolve_query_text prefer THIS subscriber's own refined
        # definition over the shared default -- see
        # docs/plans/interest-definition-plan.md.
        query_text = _resolve_query_text(chat_id, topic)
        # Newest-first, deduplicated, gathered BEFORE the max_per_topic cut
        # (unlike the old single-pass loop), because offbeat selection
        # needs a pool bigger than the final cut to have anything to
        # choose from beyond what recency already picked, and the
        # relevance filter right below needs a big-enough sample to
        # compute a meaningful median over.
        #
        # Not truncated by count here (no RELEVANCE_SAMPLE_SIZE-style cap,
        # removed 2026-08-25): per-article embeddings are precomputed at
        # ingestion time (news_ingest.py), so scoring a larger pool here
        # costs a few hundred more 256-dim dot products, not more model
        # calls -- negligible. A recency pre-cut ahead of the relevance
        # filter risked discarding a genuinely more-relevant but slightly
        # older article before the filter ever saw it. The only bound on
        # this loop's input is however much `cached_articles` the caller
        # passed in -- in production, whatever news_cache.read_all()
        # returns, itself bounded by the 48h TTL (DEFAULT_TTL_HOURS) that
        # cleanup_expired enforces, not by a count.
        raw_pool = []
        for article in ordered:
            link = article.get("link")
            if not link or link in seen_links:
                continue
            if not include_restricted and article.get("source_key") in news_sources.RESTRICTED_SOURCES:
                continue
            article_cats = set(article.get("categories") or [])
            # topic_cats is always non-empty here -- the loop `continue`d
            # for this whole topic above otherwise.
            if not (article_cats & topic_cats):
                continue
            # The one and only "has this subscriber seen it" test.
            if link in already_pushed_links:
                continue
            # Not already-seen, but not worth sending either if it's ancient.
            # Articles whose published_dt didn't parse pass this guard rather
            # than being dropped -- same fail-open instinct as the rest of
            # the pipeline.
            published_dt = article.get("published_dt")
            if published_dt is not None and now - published_dt > max_age:
                continue
            # Near-duplicate collapse. An article with no embedding never
            # matches anything here -- cosine_similarity(None, x) is -1.0
            # by construction -- so it's always kept, same fail-open shape
            # as every other check in this loop. Deliberately does NOT
            # add `link` to seen_links: being a near-duplicate within
            # THIS topic's pool doesn't mean it should be unavailable to
            # a DIFFERENT topic later in this same call, which may have
            # an entirely different pool to be a duplicate within.
            embedding = article.get("embedding")
            if embedding is not None and any(
                news_embed.cosine_similarity(embedding, p.get("embedding")) >= NEAR_DUPLICATE_SIMILARITY
                for p in raw_pool
            ):
                continue
            seen_links.add(link)
            raw_pool.append(article)

        # The fine filter after the coarse one -- see
        # news_embed.filter_by_relevance. Capped at CANDIDATE_POOL_SIZE
        # afterward, same reasoning as before this existed: bigger than
        # the final max_per_topic cut, so the novelty-extra search below
        # has a remainder to draw from.
        pool = news_embed.filter_by_relevance(
            raw_pool, embedder, query_text,
            keep_fraction=RELEVANCE_KEEP_FRACTION, keep_min=RELEVANCE_KEEP_MIN, keep_max=RELEVANCE_KEEP_MAX,
        )[:CANDIDATE_POOL_SIZE]

        # Pure recency, no offbeat carve-out -- the regular digest is
        # ALWAYS up to max_per_topic articles, newest first. See the
        # module's 2026-08-26 offbeat redesign: novelty selection used
        # to reserve slots out of this cut; now it's additive instead
        # (below), so this line no longer needs to know novelty exists
        # at all.
        for article in pool[:max_per_topic]:
            candidates.append({**article, "topic": topic})

        if include_novelty_extra:
            # A SEPARATE, wider relevance pass over raw_pool (not
            # pool[max_per_topic:]) -- see NOVELTY_RELEVANCE_KEEP_* and
            # _pick_novelty_extra's own docstring for why novelty
            # eligibility needs its own, more permissive relevance floor
            # rather than only ever seeing what the regular digest's
            # stricter cut left over.
            novelty_pool = news_embed.filter_by_relevance(
                raw_pool, embedder, query_text,
                keep_fraction=NOVELTY_RELEVANCE_KEEP_FRACTION,
                keep_min=NOVELTY_RELEVANCE_KEEP_MIN,
                keep_max=NOVELTY_RELEVANCE_KEEP_MAX,
            )
            regular_links = {a["link"] for a in pool[:max_per_topic]}
            novelty_remainder = [a for a in novelty_pool if a["link"] not in regular_links]
            novelty = _pick_novelty_extra(novelty_remainder, topic_cats)
            if novelty is not None:
                candidates.append({**novelty, "topic": topic, "is_novelty_extra": True})
    return candidates


def _resolve_query_text(chat_id: int | None, topic: str) -> str:
    """The bare topic string is a weak embedding query -- measured,
    2026-08-25: for "AI coding", three genuinely relevant real articles
    ("Claude Cowork...") scored BELOW an unrelated stock-picking article,
    because model2vec's static embeddings can't connect a product name to
    a category word without shared vocabulary. A generated definition,
    rich in the concrete tool/technique vocabulary a real article on the
    subject would use, measurably ranks genuine matches much higher (the
    same worst case dropped from needing the top 83% of a pool kept down
    to 44%). See news_classify.expand_interest_for_retrieval for what's
    generated and cached, and agent.py's add_one_interest for when.

    Checks this subscriber's own refined definition first (see
    docs/plans/interest-definition-plan.md), then the shared/global one,
    then falls back to the bare `topic` -- reached only if BOTH are
    missing, which backfill_missing_interest_definitions (called once per
    cycle before this) is meant to make rare. Delegates the subscriber-
    then-shared precedence to interest_cache_ops.resolve_interest_
    definition rather than re-expressing it here -- that precedence is
    this feature's core guarantee (interest-definition-plan.md), and
    having it live in exactly one place is what keeps it from silently
    drifting if it's ever changed. `chat_id=None` (test callers with no
    real subscriber in scope) resolves the same way a chat_id that has no
    override does: SQL `chat_id = NULL` never matches, so this degrades
    to the shared tier correctly, not incorrectly. Never calls the LLM
    itself: this runs on every push cycle, and generation is deliberately
    a write-time, cached cost, not a read-time one."""
    return interest_cache_ops.resolve_interest_definition(chat_id, topic) or topic


def _novelty_sort_key(scored_item: tuple[dict, bool, tuple[float, str] | None]) -> tuple[int, float]:
    """Keyword hits always rank first (0 < 1); within a tier, lower
    keyness score (more topic-foreign) ranks first."""
    _article, has_keyword, keyness_result = scored_item
    keyness_score = keyness_result[0] if keyness_result is not None else 0.0
    return (0 if has_keyword else 1, keyness_score)


def _pick_novelty_extra(remainder: list[dict], categories: set[str] | list[str]) -> dict | None:
    """At most one "by the way, we noticed something interesting" pick,
    drawn from `remainder` -- articles that cleared select_candidate_
    articles' own WIDER novelty relevance pass (NOVELTY_RELEVANCE_KEEP_*,
    a separate and more permissive filter over raw_pool than the regular
    digest's RELEVANCE_KEEP_* one -- added 2026-08-27, see that
    constant's own comment) and aren't already one of the regular
    candidates, so this never re-suggests an article already in the
    regular list. See news_keyness.py's own
    module docstring and docs/analysis/cluster-measurements.md's
    "Offbeat selection, take two" for the full measurement history
    (replaced an earlier embedding-based centroid-distance design
    2026-08-26, after live use showed its picks read as unrelated more
    often than novel).

    Two independent signals, either of which qualifies an article:
    a novelty-keyword hit (news_keyness.NOVELTY_KEYWORDS), unconditionally;
    or its single most topic-foreign noun's keyness score
    (news_keyness.min_term_keyness) below NOVELTY_KEYNESS_THRESHOLD.
    Deliberately a real bar, not "whatever's least bad in the pool" --
    an earlier version of this always picked something, even when
    nothing in the pool had a real signal; a 2026-08-26 product
    correction made this explicitly optional instead: **returns None,
    not a weak fallback pick, when nothing in `remainder` clears either
    bar** -- callers must not force a novelty section into every push.

    `categories` is the topic's own category set (from topic_categories
    in select_candidate_articles), used to look up news_keyness's
    precomputed per-category table (interest_cache_ops.get_category_keyness) --
    merged across every category in the set, keeping the lower (more
    foreign) score for any term scored under more than one."""
    if not remainder:
        return None

    keyness: dict[str, float] = {}
    for category in categories:
        for term, score in interest_cache_ops.get_category_keyness(category).items():
            if term not in keyness or score < keyness[term]:
                keyness[term] = score

    scored = [
        (a, news_keyness.has_novelty_keyword(a.get("title"), a.get("summary")),
         news_keyness.min_term_keyness(a.get("title"), a.get("summary"), keyness))
        for a in remainder
    ]
    qualifying = [
        (a, has_kw, result) for a, has_kw, result in scored
        if has_kw or (result is not None and result[0] < NOVELTY_KEYNESS_THRESHOLD)
    ]
    if not qualifying:
        return None
    qualifying.sort(key=_novelty_sort_key)
    return qualifying[0][0]


def write_push_digest(model, articles: list[dict], topic: str | None = None,
                      language: str | None = None, retry_reason: str | None = None) -> str:
    """A single direct model call (no tool loop -- see module docstring)
    that turns a stage-1-filtered candidate list into a Telegram HTML
    digest, applying stage-2 (content) filtering itself per the prompt
    above. `language`, when set, is the subscriber's stored reply-language
    preference (subscriber_ops.get_language) -- pushes go through this module's
    own prompt rather than agent.py's dynamic_prompt middleware, so the
    preference has to be threaded in here too, not just in _compose_prompt.

    `topic` is an explicit parameter, passed by the caller (which already
    has it in scope -- see run_push_cycle's per-interest loop), NOT read
    off `articles`. Until 2026-08-25 it wasn't a parameter at all: each
    candidate carried its own `"topic"` field (stamped on by
    select_candidate_articles) and the listing repeated `[{topic}]` as a
    prefix on every single line. Two things wrong with that, found via a
    live user report of near-identical digests across different
    interests:

    1. It was a fossil. Before push became one-message-per-interest
       (2026-08-24), one call covered a subscriber's WHOLE interest list,
       and the per-line tag genuinely distinguished which of several
       DIFFERENT interests each candidate belonged to. Once every call
       covers exactly one interest, every line in one listing carries the
       identical value -- zero information content, confirmed by grep:
       no code anywhere reads `article["topic"]` except this f-string.
    2. It was actively misleading, not just useless. `_PUSH_DIGEST_PROMPT`
       explicitly tells the model to be skeptical -- "some candidates may
       not actually be about the subscriber's topic... use your own
       judgment" -- but the listing FORMAT presented `[AI Agent]` as a
       per-article, seemingly-confirmed classification sitting right next
       to each headline. The instruction said "doubt this," the
       presentation said "this is verified." Reproduced live: given 5
       generic AI-industry articles (an earnings-reaction piece, a
       funding round, a job-market study -- none actually about agents or
       coding), write_push_digest kept every single one under BOTH
       topic="AI Agent" and topic="AI coding", producing near-identical
       reports. Stage 1's category filter is coarse by design; stage 2
       (this function) is supposed to be the fine filter, and the old
       format gave it nothing to be fine WITH.

    The fix here is necessary but not sufficient: stating the topic once,
    plainly, with an explicit instruction to judge specificity rather
    than category membership, gives the model a chance to filter
    correctly -- it does not guarantee it will, since the model still has
    no signal stronger than its own semantic judgment of a headline
    against a topic word. A stricter fix (embedding-based relevance
    filtering ahead of this call, using the topic string as a retrieval
    query) shipped 2026-08-25, later moved to news_embed.filter_by_relevance -- see
    select_candidate_articles's own docstring and
    docs/analysis/cluster-measurements.md.

    `retry_reason` is set by run_push_cycle's own retry loop (added
    2026-08-28, see telegram_html.validate) when a PREVIOUS call's reply
    failed Telegram HTML validation -- appended as the last instruction so
    it's the most salient thing the model sees, telling it specifically
    what broke rather than just "try again"."""
    listing = "\n".join(
        f"- {'[EXTRA] ' if a.get('is_novelty_extra') else ''}{a['title']} "
        f"({a.get('source')}, published {a.get('published') or 'date unknown'}) — {a.get('link')}"
        for a in articles
    )
    system_prompt = _PUSH_DIGEST_PROMPT
    if any(a.get("is_novelty_extra") for a in articles):
        # At most one candidate is ever marked this way (see
        # select_candidate_articles/_pick_novelty_extra) -- unlike the
        # retired per-line [topic] prefix this replaced in spirit, this
        # marks exactly one line, for a genuinely different purpose
        # (formatting instruction, not a relevance claim the model is
        # meant to trust). The model still applies its own relevance
        # judgment to it like any other candidate -- being marked
        # [EXTRA] means "format it this way IF you include it," not
        # "this one is guaranteed good."
        system_prompt += (
            "\n\nOne candidate above is marked [EXTRA] -- a separate, "
            "experimental pick (an article that's genuinely on-topic but "
            "unusual or notable in some way), not one of the regular "
            "candidates. Apply the same relevance judgment to it as any "
            "other candidate; it is not guaranteed to belong in the "
            "report. If you do include it, do NOT blend it into the main "
            "synthesized report -- write it as a short, clearly separate "
            "closing note afterward, introduced by something like 'By "
            "the way, we noticed something interesting:' translated into "
            "the reply language, naming its title and a one-sentence "
            "summary with its own source link. The main report above it "
            "should read exactly as it would if [EXTRA] didn't exist -- "
            "including possibly being empty: if every OTHER candidate is "
            "irrelevant but [EXTRA] itself genuinely belongs, send just "
            "the closing note on its own, don't withhold it because "
            "there's nothing else to report alongside it."
        )
    if topic:
        system_prompt += (
            f"\n\nThe subscriber's interest is: {topic}. The candidates "
            "below already passed a coarse category filter, which only "
            "confirms they share a broad category with this interest -- "
            "it does NOT confirm they are actually, specifically about "
            f"{topic}. Generic industry news that merely mentions the "
            "same broad field (earnings, funding, market moves, general "
            f"research) does not count as being about {topic} unless it "
            "genuinely is. Apply this before applying your usual "
            "relevance judgment from the instructions above."
        )
    if language:
        system_prompt += (
            f"\n\nWrite your ENTIRE reply in {language}, regardless of what "
            "language the article titles/summaries above are in. If this "
            "is a specific script/variant (e.g. Traditional vs Simplified "
            "Chinese, Brazilian vs European Portuguese), use exactly that "
            "variant's script and spelling conventions throughout."
        )
    if retry_reason:
        system_prompt += (
            f"\n\nYour previous reply had invalid Telegram HTML: {retry_reason}. "
            "Fix this specific problem and reply again, following the HTML "
            "formatting rules above exactly."
        )
    response = model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": listing},
        ]
    )
    return response.content

# Liveness for the dead man's switch in
# docs/plans/observability-platform-plan.md.
#
# The alarm asks "has ANY span arrived in the last 30 minutes". Without
# this, the answer is legitimately "no" on a healthy system: every LLM call
# in a push cycle sits inside the per-subscriber loop AFTER the due check,
# so a tick where nobody is due emits nothing at all. Two subscribers on a
# 6-hour interval leave most of the 96 daily ticks silent, and an alarm
# that cries wolf on an idle Sunday is an alarm nobody reads.
#
# A no-op when no tracer provider is configured -- which is every test and
# CI run -- because OpenTelemetry's default is a no-op tracer. No env check
# needed, and nothing to remember to stub.
_tracer = trace.get_tracer("argus.news_push")


def _emit_heartbeat(subscriber_count: int) -> None:
    """One span per tick, whether or not there was any work to do.

    Carries the subscriber count as an attribute rather than just existing:
    the same span then answers "is it running" and "how many subscribers
    did it see", and the second is the number whose sudden growth was the
    2026-08-21 incident."""
    with _tracer.start_as_current_span("argus_heartbeat") as span:
        span.set_attribute("heartbeat.job", "push_tick")
        span.set_attribute("heartbeat.push_enabled_subscribers", subscriber_count)


class _ModelStageError(Exception):
    """Marks a failure that came out of an LLM call rather than out of
    local filtering or the database.

    The alternative -- deciding after the fact whether an exception "looks
    like" a model error by matching its message -- breaks the first time a
    provider rewords one, and breaks silently, in the direction of
    under-counting. Classifying by WHERE the call was made cannot drift:
    `_model_call` wraps exactly the LLM calls and nothing else."""


def _model_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise _ModelStageError(exc) from exc


# Telegram's wording for "this chat can no longer receive anything."
# Both are permanent for practical purposes and both cost a full digest
# generation per cycle, so criterion 1 treats them identically.
_UNREACHABLE_CHAT_MARKERS = (
    "chat not found",       # BadRequest -- the chat never existed or was deleted
    "bot was blocked",      # Forbidden -- the user blocked the bot
    "user is deactivated",  # Forbidden -- the account is gone
)


def _classify_send_failure(exc: Exception) -> str:
    """Narrows a delivery failure to chat_not_found where the message says
    so, and leaves everything else as a generic cycle failure.

    Matching on the message rather than the exception type because the
    type does not separate these: `BadRequest` covers both a dead chat and
    malformed HTML, and only the text tells them apart. The failure mode
    of a reworded message is that a genuinely unreachable chat records as
    `cycle_failed` -- criterion 1 goes quiet, rather than every failure
    being mislabelled as unreachable and subscribers being disabled
    wrongly. Wrong in the safe direction, deliberately."""
    text = str(exc).lower()
    if any(marker in text for marker in _UNREACHABLE_CHAT_MARKERS):
        return push_outcome_ops.PUSH_CHAT_NOT_FOUND
    return push_outcome_ops.PUSH_CYCLE_FAILED


# Consecutive undeliverable digests before push is turned off for that
# subscriber -- criterion 1 in docs/plans/incident-monitoring-plan.md.
#
# Three rather than one because delivery can fail for reasons that are not
# about the chat being gone, and turning a real subscriber off is the more
# expensive mistake of the two: they simply stop receiving news, with no
# error to notice. Three consecutive failures with no successful delivery
# between them is not a blip.
UNREACHABLE_STRIKES = get_settings().resolved("push.unreachable_strikes", default=3)


def _strike_unreachable_subscriber(chat_id: int, now: datetime) -> None:
    """Turns push off once a chat has been undeliverable UNREACHABLE_STRIKES
    times running.

    This is the bound on the 2026-08-21 failure mode. An unreachable chat
    still costs a full digest generation -- three LLM calls -- every cycle,
    because generation happens before delivery is attempted and is billed
    whether or not the send lands. Without a stop, that recurs for as long
    as the subscriber row exists: the leak was open for eight days.

    Reversible on purpose: only `push_enabled` is cleared. The subscriber,
    their interests and their language survive, so a user who blocked the
    bot and later unblocks it turns push back on and continues, rather than
    discovering their settings were deleted.

    Re-enabling does NOT reset the strike count -- only a delivered digest
    does, or the subscriber themselves stopping push (see
    subscriber_ops.reset_push_consecutive_failures). So a subscriber who
    turns push back on while still unreachable is disabled again after a
    single further failure rather than after three. That is deliberate: a
    transient outage records `cycle_failed` and accrues no strikes at all,
    so one more `chat_not_found` really does mean still unreachable, and
    granting a fresh allowance would just pay for three more undeliverable
    digests."""
    strikes = subscriber_ops.record_push_failure(chat_id)
    if strikes < UNREACHABLE_STRIKES:
        return
    subscriber_ops.set_push_enabled(chat_id, False)
    detail = f"undeliverable {strikes} cycles running"
    _record(chat_id, push_outcome_ops.PUSH_DISABLED,
            f"{detail} -- push turned off (they can turn it back on)",
            now, detail=detail)


async def _stop_push_at_trial_limit(chat_id: int, now: datetime, notify_admin) -> None:
    """Turns push off once this subscriber's free-trial allowance
    (subscriber_ops.get_pushes_remaining) is already at exactly 0 -- same
    "only push_enabled, everything else survives" reversibility as
    _strike_unreachable_subscriber above, except an admin reset
    (subscriber_ops.reset_push_limit) is what turns it back on here, not
    the subscriber themselves.

    `notify_admin` is `async def notify_admin(chat_id: int) -> None` --
    same injected-callable shape as `send` (run_push_cycle's own
    docstring), so this module still needs no live Bot/Application to
    test. None (the default) skips notification silently, same as
    `embedder=None` being an optional enhancement elsewhere in this
    module -- a deployment with no admin configured still turns push off
    correctly, it just doesn't tell anyone."""
    subscriber_ops.set_push_enabled(chat_id, False)
    _record(chat_id, push_outcome_ops.PUSH_TRIAL_LIMIT_REACHED,
            "free-trial push allowance exhausted -- push turned off", now)
    if notify_admin is not None:
        await notify_admin(chat_id)


def _record(chat_id: int, outcome: str, message: str, now: datetime,
            detail: str | None = None) -> None:
    """Reports a push outcome two ways, from one call site so they cannot
    disagree: a log line and a span. (A third way, a push_outcomes SQLite
    row, existed until 2026-09-04 -- retired as an unbounded event-log
    table with no reader that a bounded per-subscriber counter couldn't
    serve instead; see subscriber_ops.record_push_failure/
    reset_push_consecutive_failures and git history.)

    The PRINT goes to `docker logs`, where someone looks when debugging a
    specific cycle. It is free text with no timestamp of its own, it cannot
    be read from inside the container, and a deploy destroys it -- fine for
    a human reading a single cycle, useless as an alarm's input.

    The SPAN is what an ALARM can query. Criteria 2 and 3 of
    docs/plans/incident-monitoring-plan.md live in Logfire, and Logfire can
    only alert on what it has received. A no-op when no tracer provider is
    configured, which is every test and CI run.

    Also resets the consecutive-failure counter on a delivered outcome --
    a successful delivery is the only positive proof a chat is reachable,
    so it's the only thing that clears it. See
    _strike_unreachable_subscriber's own docstring for the other half."""
    print(f"[news_push] chat_id={chat_id}: {message}")
    if outcome == push_outcome_ops.PUSH_DELIVERED:
        subscriber_ops.reset_push_consecutive_failures(chat_id)
    with _tracer.start_as_current_span("push_outcome") as span:
        # Flat, low-cardinality attributes: an alert query filters on
        # `outcome` and counts, so it must not have to parse prose.
        span.set_attribute("push.outcome", outcome)
        # The opaque id, not chat_id: the log line keeps the real Telegram
        # identifier (it never leaves the VM), but a span does leave. See
        # subscriber_ops.external_id.
        span.set_attribute("push.subscriber", subscriber_ops.external_id(chat_id))
        # Whether an LLM was paid for this cycle -- criterion 3's
        # denominator, computed here rather than by re-deriving the outcome
        # set inside every alert query.
        span.set_attribute("push.generated", outcome in push_outcome_ops.PUSH_GENERATED_OUTCOMES)
        if detail:
            span.set_attribute("push.detail", detail)


def _emit_html_validation_attempt(chat_id: int, topic: str, attempt: int, reason: str | None) -> None:
    """Reports one retry-loop attempt -- print + span, deliberately NOT a
    third thing (no direct admin alert, unlike this function's
    2026-08-28 predecessor `_report_html_validation_exhausted`).

    2026-08-28, later the same day: reworked per an explicit architectural
    split -- service / log / incident / notification are four separate
    layers now, and deciding "this specific pattern of facts is an
    incident, page someone" is Logfire's job, not this module's. This
    function's only job is to report the fact (attempt N either passed or
    failed, and why) -- every attempt, not just the last one, so Logfire
    has the full sequence to query over. Whether 3 straight failures for
    the same subscriber/topic is alert-worthy, and at what severity, is
    a Logfire alert query (see docs/plans/observability-platform-plan.md);
    whether/how a human gets told is the Jira relay documented there.
    Nothing here decides either.

    Deliberately separate from _record/push_outcomes -- this isn't a
    delivery outcome (the digest still gets sent regardless, see
    run_push_cycle's retry loop), it's a content-quality signal, and
    folding it into the outcome enum would make every existing criterion
    1-3 query (incident-monitoring-plan.md) have to account for a case
    that isn't about delivery at all."""
    print(f"[news_push] chat_id={chat_id} topic={topic!r}: HTML validation attempt {attempt} "
          f"{'passed' if reason is None else f'failed: {reason}'}")
    with _tracer.start_as_current_span("html_validation_attempt") as span:
        span.set_attribute("push.subscriber", subscriber_ops.external_id(chat_id))
        span.set_attribute("topic", topic)
        span.set_attribute("attempt", attempt)
        span.set_attribute("valid", reason is None)
        if reason:
            span.set_attribute("reason", reason)


def is_subscriber_due(last_push_at: datetime | None, interval_hours: int, now: datetime) -> bool:
    if last_push_at is None:
        return True
    elapsed_hours = (now - last_push_at).total_seconds() / 3600
    return elapsed_hours >= interval_hours


async def run_push_cycle(model, send: "callable", now: datetime | None = None, embedder=None,
                         notify_admin: "callable" = None) -> None:
    """One scheduler tick: for every push-enabled, due subscriber with at
    least one interest, select candidate articles from the shared cache,
    and if there are any, write and send a digest. `send` is
    `async def send(chat_id, html_text, topic=None)` (bound to the real bot's
    send_message in production, faked in tests) -- kept generic so this
    module doesn't need a live Bot/Application to be tested. One
    subscriber's failure doesn't stop the others, same isolation pattern
    as search_news's per-source error handling -- but unlike that
    isolation, every outcome here is both printed (docker logs captures
    stdout) and recorded in push_outcomes rather than swallowed silently,
    including ticks where nobody was due -- not just when something
    actually sends. See _record for why both, and
    docs/plans/incident-monitoring-plan.md for what reads the rows.

    `notify_admin=None` (the default): `async def notify_admin(chat_id: int)
    -> None`, called only when a subscriber's free-trial push allowance
    runs out (see `_stop_push_at_trial_limit`) -- everything else here
    still reports through `_record`/Logfire, not this. None is a safe
    default (no admin configured, or a test that doesn't care), same
    "optional, degrades quietly" shape as `embedder=None`.

    `embedder=None` (the default) threads straight through to every
    select_candidate_articles call, which degrades near-duplicate
    collapse and relevance ranking to their pre-2026-08-25 behavior
    rather than failing -- see news_embed's module docstring. The
    novelty extra (2026-08-26 on) no longer depends on the embedder at
    all -- it degrades independently, sending no novelty section (not a
    forced weak pick) when neither its own keyword nor keyness signal
    clears NOVELTY_KEYNESS_THRESHOLD (see _pick_novelty_extra).

    The cache is read ONCE per cycle and reused across every subscriber
    -- matches docs/plans/local-news-cache-plan.md's stated efficiency argument
    for a shared cache ("one Perigon call can satisfy every subscriber
    whose interests match it"), and avoids N redundant directory scans
    for N due subscribers in the same tick.

    Real incident, 2026-08-09, two parts: (1) a subscriber reported never
    receiving a push despite push_outcome_ops showing a completed cycle -- there
    was no way to confirm from logs alone whether `send` actually ran,
    since nothing was ever printed either way; (2) fixing that alone
    turned out insufficient -- the container's stdout was block-buffered
    (Python's default when stdout isn't a TTY, true for any `docker run
    -d` container), so prints were never reaching `docker logs` at all
    regardless of what they said (fixed separately: PYTHONUNBUFFERED=1
    in the Dockerfile). Once both were fixed, a THIRD gap remained: the
    same subscriber asked why a nominal 30-minute interval sometimes took
    close to 45 -- unanswerable because only *due* subscribers were ever
    logged, not the tick itself or why a not-yet-due subscriber was
    skipped. This function now logs every tick's summary and every
    subscriber's due-check outcome, not just successful sends."""
    now = now or datetime.now(timezone.utc)
    # Same reasoning, same cadence -- one maintenance pass per cycle
    # rather than a glob+stat per message sent. See message_archive's
    # own module docstring for why this exists at all.
    message_archive.prune_message_archive(now)
    subscribers = subscriber_ops.list_push_enabled_subscribers()
    print(f"[news_push] tick at {now.isoformat()}: {len(subscribers)} push-enabled subscriber(s)")
    _emit_heartbeat(len(subscribers))
    cached_articles = news_cache.read_all()
    for subscriber in subscribers:
        chat_id = subscriber["chat_id"]
        interests = subscriber["interests"]
        if not interests:
            _record(chat_id, push_outcome_ops.PUSH_NO_INTERESTS,
                    "push enabled but no interests set -- skipping", now)
            continue
        last_push_at = subscriber["last_push_at"]
        interval_hours = subscriber["push_interval_hours"]
        if not is_subscriber_due(last_push_at, interval_hours, now):
            elapsed = (now - last_push_at).total_seconds() / 3600 if last_push_at else None
            print(
                f"[news_push] chat_id={chat_id}: not due yet "
                f"(interval={interval_hours}h, elapsed={elapsed:.2f}h)"
            )
            continue
        print(f"[news_push] chat_id={chat_id}: due -- checking for new articles")

        # Free-trial push allowance (requested 2026-09-18, refined
        # 2026-09-19 after live testing on INT surfaced a real problem:
        # consuming HERE, unconditionally, meant a cycle that found
        # nothing worth sending could still spend a subscriber's only
        # remaining push and leave them with nothing to show for it).
        #
        # This is now a PEEK, not a consume -- it only blocks a
        # subscriber already sitting at exactly 0; it does not decrement.
        # The actual decrement happens later, once and only once this
        # cycle, at the moment a message is genuinely delivered (see the
        # PUSH_DELIVERED branch below) -- the remaining count now means
        # "pushes you will actually receive", not "cycles the scheduler
        # is willing to attempt on your behalf". A subscriber sitting on
        # their last unit can still cost a generation call on a cycle
        # that ends up sending nothing; that is the accepted tradeoff for
        # never silently spending the one thing this number promises them.
        pushes_remaining = subscriber_ops.get_pushes_remaining(chat_id)
        if pushes_remaining == 0:
            await _stop_push_at_trial_limit(chat_id, now, notify_admin)
            continue

        # What record_push must be told, and whether it must be told at
        # all. None until an LLM has been paid to write a digest; a list
        # from then on. Load-bearing twice over:
        #
        # Not-None means the money is spent, so the next attempt must be a
        # full interval away even though this cycle failed. Leaving
        # last_push_at untouched instead -- which is what every failure
        # path used to do -- makes the subscriber due again on the very
        # next tick, and the tick is PUSH_TICK_SECONDS (15 minutes), not
        # their interval.
        #
        # Its CONTENTS matter for a narrower case: a send can succeed and a
        # later step still raise (the outcome insert, or record_push
        # itself). Recording [] there would leave articles the subscriber
        # genuinely received still eligible, and they would be sent again.
        delivered: list[str] | None = None
        # Bound before the try block, not inside it: the except handlers
        # below reference both to decide whether a later failure gets
        # folded into an otherwise-successful outcome. Both are genuinely
        # 0 at this point regardless -- nothing can have sent or been
        # blocked before interests are even resolved -- but they must be
        # bound (not merely 0) so the except clauses can't hit an
        # UnboundLocalError if resolve_interest_categories itself is what
        # raises.
        sent = 0
        blocked = 0
        try:
            topic_categories = _model_call(resolve_interest_categories, model, interests)
            _model_call(backfill_missing_interest_definitions, model, interests)
            # One message per interest, longest-un-pushed first. Two
            # reasons this replaced a single combined digest:
            #
            # Retrieval. The interest string IS the query, and merging
            # five of them into one candidate pool then asking one model
            # call to write about all of it discards the specificity that
            # made each one findable -- the same "any category layer
            # between the interest and the articles costs recall" result
            # measured in docs/analysis/cluster-measurements.md.
            #
            # Rotation. MAX_INTERESTS_PER_PUSH bounds how noisy a cycle
            # can be, and staleness ordering is what stops that bound from
            # permanently starving whatever sorts last. An interest with no
            # new articles does NOT consume one of the cycle's slots --
            # only a message that was actually sent does.
            send_failure: tuple[str, str] | None = None
            for topic in subscriber_ops.interests_by_staleness(chat_id, interests):
                if sent >= MAX_INTERESTS_PER_PUSH:
                    break
                new_articles = select_candidate_articles(
                    cached_articles,
                    [topic],
                    topic_categories,
                    subscriber["last_push_at"],
                    # What this cycle has already delivered counts as seen
                    # too. Without the union, an article that matches two
                    # of a subscriber's interests goes out twice in the
                    # same push -- record_push only runs at the end, so
                    # pushed_links cannot know about it yet.
                    set(subscriber["pushed_links"]) | set(delivered or []),
                    include_restricted=subscriber["restricted_sources_enabled"],
                    now=now,
                    embedder=embedder,
                    chat_id=chat_id,
                )
                if not new_articles:
                    continue

                # Validated BEFORE send, not just reacted to at send time
                # (bot.py's send_push_digest still has its own BadRequest
                # fallback as a last-resort net) -- retries with the
                # specific validation failure fed back to the model (see
                # write_push_digest's retry_reason param), up to
                # MAX_HTML_ATTEMPTS. Runs unconditionally: a legitimately
                # empty "nothing relevant" reply has no tags to break, so
                # telegram_html.validate("") is None and this loop exits
                # on its first pass, same cost as before this existed.
                #
                # Every attempt is reported (_emit_html_validation_attempt),
                # not just a final exhausted one -- deciding whether 3
                # straight failures is alert-worthy, and telling anyone
                # about it, is Logfire's/Jira's job now, not this loop's.
                # See that function's own docstring for the 2026-08-28
                # architectural split this reflects. If every attempt
                # fails, the digest is still sent below regardless
                # (send_push_digest's own fallback strips it to plain
                # text if Telegram also rejects it) -- this loop never
                # withholds a digest just because its HTML wasn't clean.
                reason = None
                for attempt in range(1, MAX_HTML_ATTEMPTS + 1):
                    digest = _model_call(write_push_digest, model, new_articles,
                                         topic, subscriber["language"], retry_reason=reason)
                    reason = telegram_html.validate(digest)
                    _emit_html_validation_attempt(chat_id, topic, attempt, reason)
                    if reason is None:
                        break
                # Generated: the money is spent from here on, so the next
                # attempt must be a full interval away even if the rest of
                # this fails. See the module's `delivered` reasoning.
                if delivered is None:
                    delivered = []

                # Checked by presence of a real source link, not just
                # "is the string empty" -- _PUSH_DIGEST_PROMPT's "write
                # nothing" instruction asks for a literal empty reply
                # when nothing is relevant, but the model doesn't always
                # comply with that literally; it can instead write a
                # non-empty sentence explaining that nothing was relevant
                # ("No genuinely relevant AI coding stories emerged..."),
                # which `not digest.strip()` would miss entirely and send
                # to the subscriber as if it were real content.
                # TREND_REPORT_STRUCTURE requires every genuine item to
                # carry a real <a href> link, so a digest with none is
                # equivalent to an empty one regardless of what prose
                # surrounds that fact. Reproduced live, 2026-08-25, right
                # after tightening the "be skeptical of the topic label"
                # instruction below -- a stricter prompt makes the model
                # MORE likely to reject everything, which makes this
                # exact gap more likely to fire, not less.
                if not digest or not telegram_html.links_actually_sent(digest, new_articles):
                    # Stage 2 judged none of this interest's candidates
                    # genuinely relevant -- see _PUSH_DIGEST_PROMPT's
                    # explicit "write nothing" instruction. Not an error,
                    # and nothing is retired: an article that merely lost a
                    # relevance judgment has not been seen.
                    continue

                # A stored interest is user-supplied, unsanitized text that
                # ends up embedded in the digest prompt, so the same output
                # guardrail bot.py runs on chat replies applies here.
                if not _model_call(guardrails.is_output_on_topic, model, digest):
                    blocked += 1
                    continue

                # Appended after the guardrail check, not before -- fixed,
                # non-model text has nothing to be checked, and appending
                # it earlier risks the output-topic check misreading a
                # "you have N left" sentence as off-topic self-disclosure.
                # Omitted entirely for an unlimited subscriber (None/-1).
                #
                # `pushes_remaining` above is the PRE-charge peek (the
                # actual decrement happens once, after this loop, in the
                # PUSH_DELIVERED branch) -- shown here minus 1 since a
                # send is genuinely about to happen, so the number a
                # subscriber sees always matches what it will be right
                # after this message lands, not a stale pre-charge value.
                if pushes_remaining is not None and pushes_remaining != -1:
                    digest = f"{digest}\n\nYou have {pushes_remaining - 1} push(es) remaining in your trial."

                try:
                    await send(chat_id, digest, topic=topic)
                except Exception as exc:
                    # Delivery failed, not generation. Recorded as its own
                    # outcome so "we paid to write digests nobody can
                    # receive" stays answerable -- the shape of the
                    # 2026-08-21 incident.
                    #
                    # Stops this subscriber's remaining interests rather
                    # than trying them: whatever makes one send fail
                    # (blocked bot, deleted chat, Telegram down) applies to
                    # the next one too, and retrying would multiply the
                    # failure by MAX_INTERESTS_PER_PUSH.
                    send_failure = (_classify_send_failure(exc), repr(exc))
                    break

                # Only what the subscriber actually saw is retired. A
                # candidate the model left out stays eligible for a later
                # digest -- see telegram_html.links_actually_sent.
                delivered.extend(telegram_html.links_actually_sent(digest, new_articles))
                subscriber_ops.mark_interest_pushed(chat_id, topic, now)
                sent += 1

            if send_failure is not None:
                outcome, detail = send_failure
                _record(chat_id, outcome,
                        f"{sent} message(s) sent, then delivery failed with {detail}",
                        now, detail=detail)
                if outcome == push_outcome_ops.PUSH_CHAT_NOT_FOUND:
                    _strike_unreachable_subscriber(chat_id, now)
            elif sent:
                # One outcome per subscriber per cycle, NOT one per message.
                # The three live alert criteria in
                # docs/plans/incident-monitoring-plan.md are thresholds over
                # this table; making a cycle emit N rows instead of 1 would
                # silently rescale every one of them.
                #
                # `blocked` folded into the detail rather than dropped: this
                # branch wins over the `elif blocked` one below whenever
                # ANY interest got through, so a guardrail misfiring on one
                # specific interest while the subscriber's others keep
                # succeeding would otherwise leave no trace anywhere -- not
                # in this detail, not printed, not in the span. Still
                # correctly not marked pushed and still retried next cycle;
                # this only fixes whether a human investigating later has
                # something to go on.
                detail = f"{sent} interest(s)" + (f", {blocked} blocked" if blocked else "")
                _record(chat_id, push_outcome_ops.PUSH_DELIVERED,
                        f"sent {sent} message(s), one per interest", now, detail=detail)
                # The actual free-trial charge, deferred to here (see the
                # peek-check earlier in this function): exactly one unit
                # per cycle, regardless of how many interests got sent,
                # and only now that PUSH_DELIVERED is the confirmed
                # outcome. Guaranteed to still succeed (peeked >0 or
                # unlimited before any work in this cycle started, and
                # this process is single-threaded per cycle).
                subscriber_ops.try_consume_push(chat_id)
            elif blocked:
                _record(chat_id, push_outcome_ops.PUSH_BLOCKED,
                        f"{blocked} digest(s) blocked by output guardrail, none sent", now)
            elif delivered is not None:
                _record(chat_id, push_outcome_ops.PUSH_NOT_RELEVANT,
                        "candidates found but none judged relevant -- not sending", now)
            else:
                # Nothing anywhere had new articles. last_push_at still
                # advances so the next check is a full interval away
                # instead of re-checking every tick.
                _record(chat_id, push_outcome_ops.PUSH_NOTHING_NEW,
                        "due, but no new articles -- advancing last_push_at only", now)

            subscriber_ops.mark_links_shown(chat_id, delivered or [], now)
            subscriber_ops.advance_last_push_at(chat_id, now)
        except _ModelStageError as exc:
            # A 402, a rate limit, a provider outage. Unlike everything
            # else here this is never about one subscriber -- if the model
            # is down for this chat it is down for all of them -- which is
            # why criterion 2 alerts on a single occurrence rather than on
            # a threshold.
            detail = repr(exc.__cause__ or exc)
            # `sent` wins if this interest already delivered before a LATER
            # interest's model call failed -- same "sent wins, fold the
            # failure into detail" pattern as the `blocked` case in the
            # `elif sent:` branch above. Without this, a real digest a
            # subscriber actually received gets recorded (and alerted on)
            # as a full model_error, which is the exact bug found live on
            # INT 2026-09-03: interest N delivered, interest N+1 then hit
            # OpenAITimeoutError, and the one row/span for the cycle came
            # out model_error despite message_archive/interest_push_state
            # both showing the successful send.
            if sent:
                outcome_detail = f"{sent} interest(s) sent" + (f", {blocked} blocked" if blocked else "") + \
                    f", then a later interest's model call failed with {detail}"
                _record(chat_id, push_outcome_ops.PUSH_DELIVERED,
                        f"sent {sent} message(s), then model call failed with {detail}",
                        now, detail=outcome_detail)
                # Same free-trial charge as the normal "sent wins" branch
                # above -- found missing here by qa-engineer, 2026-09-19:
                # a real digest DID reach this subscriber (that's exactly
                # what "sent wins" means), so PUSH_DELIVERED here is just
                # as chargeable as the un-interrupted case. This is now
                # the SECOND exception handler with the same "sent wins"
                # pattern below it, both mirroring this one exactly --
                # any fourth "sent wins, record PUSH_DELIVERED" site
                # added later must charge here too, not just record.
                subscriber_ops.try_consume_push(chat_id)
            else:
                _record(chat_id, push_outcome_ops.PUSH_MODEL_ERROR,
                        f"model call failed with {detail}", now, detail=detail)
            if delivered is not None:
                subscriber_ops.mark_links_shown(chat_id, delivered, now)
                subscriber_ops.advance_last_push_at(chat_id, now)
            continue
        except Exception as exc:
            detail = repr(exc)
            # Same "sent wins" reasoning as the _ModelStageError branch above.
            if sent:
                outcome_detail = f"{sent} interest(s) sent" + (f", {blocked} blocked" if blocked else "") + \
                    f", then cycle failed with {detail}"
                _record(chat_id, push_outcome_ops.PUSH_DELIVERED,
                        f"sent {sent} message(s), then cycle failed with {detail}",
                        now, detail=outcome_detail)
                # Same charge, same reasoning as the _ModelStageError
                # branch above -- see its own comment.
                subscriber_ops.try_consume_push(chat_id)
            else:
                _record(chat_id, push_outcome_ops.PUSH_CYCLE_FAILED,
                        f"cycle failed with {detail}", now, detail=detail)
            if delivered is not None:
                subscriber_ops.mark_links_shown(chat_id, delivered, now)
                subscriber_ops.advance_last_push_at(chat_id, now)
            continue

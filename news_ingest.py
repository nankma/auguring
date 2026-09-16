"""
Periodic news ingestion for the local cache -- see
docs/plans/local-news-cache-plan.md. Pulls from every enabled source on its own
schedule, classifies each cycle's newly-fetched articles in one batched
call, writes them to news_cache.py, and sweeps expired entries -- folded
into the same cycle rather than a separate job, per the plan doc's
resolved "cleanup mechanism" question.

Deliberately separate from news_push.py: that module fetches per-
subscriber, per-interest, to build one subscriber's digest. This module
fetches once, generally, to keep the shared cache stocked -- one pull can
then satisfy every subscriber's queries against it for the next 2 days
(see the plan doc's "Refinement" section on why a scarce source's calls
are worth sharing, not spending per-query).

Query-capable sources (forum/api class -- hackernews, arxiv, newsapi,
gnews, perigon) fetch "everything since this source's last successful
pull" instead of a flat top-N: a flat 5 was a real bottleneck on an
active source, discarding genuinely new articles beyond the first 5
regardless of how much was actually new. RSS-class sources (16 of the 21
registered) keep a flat top-N cap instead -- a plain RSS/Atom feed has no
query or date-range parameter to ask for "since X" at all, it always
returns whatever the publisher currently has in the feed (see
news_sources.py's SOURCE_REGISTRY comment), so there's nothing to switch
to. That cap was raised from 5 to MAX_RESULTS_PER_SOURCE_RSS (200,
2026-08-16) -- 5 was arbitrary and cut real digests down to a handful of
items; 200 comfortably exceeds what any of these feeds actually carry
(most run 20-50 entries), so this is now "take everything the feed has",
not a real limit.

Deduplication against what's already cached (`_existing_cache_links`,
loaded once per cycle from news_cache.read_all()) skips re-classifying a
link this cycle already has a live cache entry for -- necessary once the
RSS cap stopped being small: at 200/feed, most of a cycle's fetch is
usually the *same* items as last cycle (feeds don't turn over that fast),
and without this check every one of them would get re-sent through a real
paid DeepSeek classification call every 4 hours for no reason --
news_cache.write_article's own overwrite-by-link-hash already makes a
redundant write harmless, but a redundant *classification call* isn't
free the way an overwrite is.

Two complementary mechanisms, not one, per source-specific findings from
live-testing this 2026-08-16 (see each source's own NewsSourceAdapter
class under news_adapters/ -- HackerNewsAdapter.pull/ArxivAdapter.pull/
GNewsAdapter.pull/NewsApiAdapter.pull -- for the per-source detail):
  - Server-side date filter (_SERVER_SIDE_SINCE_SOURCES) -- passed as the
    `since` kwarg, for the 3 sources confirmed live to support it
    correctly. Reduces payload size/wasted budget on rate-limited sources.
  - Client-side filter (applied here, after every query-capable source's
    fetch) -- articles with published_dt at or before the cutoff are
    dropped regardless of what the server did. This is the actually-
    authoritative "new" boundary, and the only mechanism at all for
    newsapi/perigon, whose server-side date param is either
    counterproductive (NewsAPI's free-tier delay) or unverified (Perigon,
    no key to test against).

**The cutoff is the newest article's own published_dt actually seen from
that source (`source_state_ops.get_source_last_article_dt`/
`set_source_last_article_dt`), not when the job last ran
(`last_pulled_at`).** A real design correction, 2026-08-16: using
`last_pulled_at` (wall-clock job time) for this meant a source that
indexes an article with a delay (confirmed live for NewsAPI's free tier --
up to ~36h, see docs/current/ai-news-sources.md) could have that article
silently skipped forever -- `last_pulled_at` keeps advancing every cycle
whether or not anything new was actually found, so a delayed article's
published_dt can fall behind a since-cutoff that already moved past it by
the time the source finally surfaces it. `last_article_dt` only advances
when a newer article is actually observed (the max published_dt among
that cycle's genuinely-new articles), so it can never outrun what's truly
been seen the way a wall-clock timestamp can. See
get_source_last_article_dt's docstring in source_state_ops.py for the full
reasoning.
"""

import functools
import time
import unicodedata
from datetime import datetime, timezone

from app_settings import get_settings
import news_cache
import news_classify
import news_embed
import news_keyness
import news_sources
import api_budget_ops
import category_ops
import interest_cache_ops
import source_state_ops
from opentelemetry import trace
from telemetry import EventLogger, get_event_logger
from telemetry_providers import Level

# A no-op when no tracer provider is configured -- which is every test and
# CI run -- because OpenTelemetry's default is a no-op tracer. Same
# reasoning as news_push._tracer.
_tracer = trace.get_tracer("argus.news_ingest")
_events: EventLogger = get_event_logger("argus.news_ingest")

# Raised from 5 to 200 on 2026-08-16 -- see module docstring. Still a real
# cap, not "unlimited": RSS feeds are the publisher's own choice of how
# much to include, but this comfortably exceeds it for every source
# currently registered (checked against docs/current/ai-news-sources.md's
# per-source notes, all well under 200 items/feed).
MAX_RESULTS_PER_SOURCE_RSS = 200
# Safety cap for query-capable sources once they're fetching "since last
# pull" rather than a flat top-N -- the real limit is now the since-filter,
# not this count, but an unbounded ask is still worth guarding against a
# pathological burst after a long outage. Generous on purpose (hackernews
# alone returned 45 hits in a 6h window for one query, live-tested
# 2026-08-16); sources with their own lower hard cap (e.g. GNews's
# 10/request free-tier limit) just clamp silently, no error.
MAX_RESULTS_PER_SOURCE_SINCE_LAST_PULL = 50
# The fallback used by _interval_hours() when a source has no
# news_source.<name>.interval_hours override in Settings.
DEFAULT_INTERVAL_HOURS = get_settings().resolved("news_source.default_interval_hours", default=4)
# The general delay between consecutive same-section calls to the SAME
# source, used as one shared value rather than a per-source table since
# other sources' limits aren't always documented, and slack here is cheap
# regardless (cycles run every 4h+). Raised from 1.1 to 3.0 on 2026-09-15:
# arXiv's own Terms of Use (info.arxiv.org/help/api/tou.html) says "make
# no more than one request every three seconds" for export.arxiv.org, and
# 1.1s across arxiv's 6 sequential per-cycle section calls was almost
# certainly what started producing 429s/timeouts around 2026-09-11 (see
# docs/current/ai-news-sources.md) -- 3.0s comfortably covers GNews's
# 1 req/sec free-tier limit too, so nothing else needed to change.
REQUEST_DELAY_SECONDS = get_settings().resolved("news_source.request_delay_seconds", default=3.0)

# Above this share of non-Latin letters, an article is dropped at ingestion
# and never cached. Measured on the 2026-08-21 snapshot: 66 of 2,706 titles
# (2.4%) are over it, 65 of them from newsapi -- Traditional Chinese market
# reports, a Japanese HuggingFace post, and so on.
#
# They were not reaching digests (newsapi is in RESTRICTED_SOURCES, which
# select_candidate_articles skips), but they were being cached, embedded and
# clustered, where a language outlier is maximally distant from an
# English-dominant corpus by construction. That is what put a Chinese fund
# prospectus at the top of "most novel article in Finance" in
# docs/analysis/cluster-measurements.md -- a measurement artifact rather
# than a finding.
MAX_NON_LATIN_RATIO = 0.30


def is_latin_script(text: str) -> bool:
    """Whether `text` is written predominantly in the Latin alphabet.

    A SCRIPT test, not a language test, and the distinction is the whole
    design. It drops Chinese, Japanese, Korean, Arabic, Cyrillic,
    Devanagari, Thai and Hebrew; it does NOT drop Spanish, French or
    German, and is not meant to -- that leakage is accepted rather than
    chased, because the alternatives are worse:

    - A language-detection library is a new dependency for a job that a
      character scan does. This repo has already been bitten once by a
      dependency's transitive imports (see CLAUDE.md on arize-phoenix).
    - The obvious zero-dependency substitute -- scoring English function
      words -- was measured against the same snapshot and flagged 7% of
      titles wrongly, because headlines drop articles and arxiv titles
      barely use them at all ("Fast high-dimensional mean testing via
      logistic regression" reads as non-English to that heuristic).

    Non-letters are ignored entirely, so punctuation, digits and emoji
    neither save nor condemn a title. Text with no letters at all is
    kept: there is nothing to judge, and this pipeline fails open
    everywhere else for the same reason.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    non_latin = sum(
        1 for c in letters
        if not unicodedata.name(c, "").startswith("LATIN")
    )
    return non_latin / len(letters) <= MAX_NON_LATIN_RATIO


_DEFAULT_QUERY = "technology"

_SOURCE_CLASS = {name: source_class for name, _fn, _env, source_class in news_sources.SOURCE_REGISTRY}

# "forum"/"api" classes are the query-capable sources per news_sources.py's
# SOURCE_REGISTRY comment -- "rss" sources have no query or date-range
# parameter at all, so there's nothing to switch to since-based fetching
# for them.
_TIME_FILTERABLE_CLASSES = {"forum", "api"}

# Of the 5 time-filterable sources, only these 3 get the `since` kwarg
# passed through to their fetch function (a server-side date filter) --
# see NewsApiAdapter.pull/PerigonAdapter.pull's own comments for why the
# other two (newsapi, perigon) deliberately don't: a free-tier delay and
# an unverified param, respectively. All 5 still get the client-side
# published_dt filter below regardless -- that's the mechanism actually
# doing the filtering for those two.
_SERVER_SIDE_SINCE_SOURCES = {"hackernews", "arxiv", "gnews"}


def _source_settings_entry(source_key: str) -> dict | None:
    """This source's own news_source.api OR news_source.rss entry (raw,
    not credential-resolved -- interval_hours/daily_cap are never
    trailsign-resolve nodes in practice, so there's nothing to resolve
    for them), or None if it has neither -- covers only the two always-on
    free sources (hackernews, arxiv), which aren't settings-driven at all
    (see news_sources._always_on_sources's own docstring).

    Named for both lists (not just "_api_entry", its 2026-09 name) since
    an RSS source can carry the same interval_hours override an api
    source always could -- added so venturebeat_ai could be throttled to
    a daily check-in without a code change once its feed started
    returning a site-wide Vercel bot-challenge 429, found 2026-09-14/15
    (see docs/current/ai-news-sources.md). daily_cap only means anything
    for api-class sources (RSS has no query budget to spend), but nothing
    stops setting it on an rss entry too -- it would just never be read.

    Reads news_sources._raw_api_entries/_raw_rss_entries fresh on every
    call, same "live, not cached at import" reasoning _interval_hours/
    _daily_cap have always had -- a deployer can change an override in
    settings.yml and see it reflected without a restart, even though
    which SOURCES exist at all is fixed at import time (see
    news_sources.SOURCE_REGISTRY).

    Checks the api list first, then rss -- first match wins. Nothing
    today enforces that a key can't appear in both lists (no real source
    does, and nothing would stop a deployer from misconfiguring one that
    did); if it ever happened, the api entry would win silently and the
    rss entry's own overrides would be ignored without any error. Same
    "first match in a list, no dedup" shape _raw_api_entries() already
    has within its own list -- flagged here rather than guarded against,
    since a real collision has never happened and guarding against a
    config mistake that can't currently occur would be speculative."""
    for entry in news_sources._raw_api_entries():
        if entry.get("key") == source_key:
            return entry
    for entry in news_sources._raw_rss_entries():
        if entry.get("key") == source_key:
            return entry
    return None


def _interval_hours(source_key: str) -> int:
    """news_source.api[].interval_hours or news_source.rss[].interval_hours
    for this source -- docs/plans/local-news-cache-plan.md's resolved
    "pull interval" question. Sources with no override (no matching
    entry at all, or an entry with no interval_hours field) use
    DEFAULT_INTERVAL_HOURS (unrestricted sources, and GNews -- its
    100/day budget comfortably covers 6 pulls/day at the default
    interval)."""
    entry = _source_settings_entry(source_key)
    return (entry or {}).get("interval_hours", DEFAULT_INTERVAL_HOURS)


def _daily_cap(source_key: str) -> int | None:
    """news_source.api[].daily_cap for this source --
    docs/plans/local-news-cache-plan.md's Perigon/NewsAPI worked
    examples. None (the default) means no cap."""
    entry = _source_settings_entry(source_key)
    return (entry or {}).get("daily_cap", None)


def _is_source_due(source_key: str, last_pulled_at: datetime | None, now: datetime) -> bool:
    if last_pulled_at is None:
        return True
    elapsed_hours = (now - last_pulled_at).total_seconds() / 3600
    return elapsed_hours >= _interval_hours(source_key)


def _cutoff_key(source_key: str, section: str | None) -> str:
    """The key `last_article_dt` is tracked under.

    Per SECTION, not per source, once a source has sections. They advance
    at wildly different rates -- cs.AI produces dozens of papers a day and
    physics.optics a handful -- so one shared cutoff lets the fast section
    drag it past the slow one's genuinely-new articles, which are then
    never offered again. That is the same class of bug this module's
    docstring already records fixing once (last_pulled_at vs
    last_article_dt), one level down.

    Latent before sections existed, because the multiple queries a source
    looped over were arbitrary interest phrasings with no systematic
    cadence difference. Sections are fixed and disjoint, so a persistent
    mismatch is now structural rather than occasional.

    The same table (`source_pull_state`) already stores plain source keys
    in its `source` column -- a composite `source:section` string just
    extends what that column holds, no separate schema needed."""
    return source_key if section is None else f"{source_key}:{section}"


def _sections_for_source(source_key: str, now: datetime) -> list[str | None]:
    """Which sections to pull this source by. `[None]` means "one call, no
    section" -- RSS feeds, which ignore anything passed to them.

    Replaces querying by subscriber interest, which was a sampling-bias bug
    hiding in plain sight: the corpus could only ever contain answers to
    questions subscribers had already asked, so nothing new could be
    discovered and the bias compounded every cycle. It was also dirty in
    practice -- "AOI" pulled Taiwanese optical-inspection news, Japanese
    anime, and half a page of Chinese, and "Bitcoin" pulled Spanish-language
    finance.

    Sections are unambiguous and they are the newsroom's own front page
    rather than an answer to a question we asked. agent.py's search_news
    still passes a real user question; that is a legitimate query and is
    untouched.

    Budget-capped sources get ONE section per pull, rotating
    deterministically so the whole list is covered over a few days rather
    than one section being pulled forever. Uncapped ones take every section
    each cycle."""
    source_class = _SOURCE_CLASS.get(source_key)
    if source_class == "rss":
        return [None]

    sections = news_sources.SOURCE_SECTIONS.get(source_key)
    if not sections:
        # A query-capable source with no section vocabulary -- currently
        # only Perigon, whose API has no top-headlines equivalent. It
        # therefore loses the per-tick rotation it used to get and falls
        # back to a single fixed query, which is a real reduction in
        # coverage, accepted rather than overlooked: Perigon has been out
        # of quota since 2026-08-15 (docs/plans/security-plan.md finding
        # 21) and is excluded from subscriber digests anyway. Revisit if it
        # is ever brought back.
        return [None]

    if _daily_cap(source_key) is not None:
        interval_seconds = _interval_hours(source_key) * 3600
        tick_number = int(now.timestamp() // interval_seconds)
        return [sections[tick_number % len(sections)]]

    return list(sections)


def _report_category_proposals(now: datetime) -> None:
    """Prints what the classifier has been asking for that the taxonomy
    doesn't have.

    Deliberately only a log line for now. The threshold that decides when
    this becomes an admin prompt (docs/plans/taxonomy-and-admin-plan.md A4)
    is currently a placeholder of 5-in-30-days picked from a single
    observation of "Education", and choosing it properly needs the real
    distribution first. Accumulating visibly and deciding later beats
    guessing a threshold now and then tuning it against alerts nobody
    trusts."""
    proposals = category_ops.count_recent_sightings(now)
    if not proposals:
        return
    ranked = sorted(proposals.items(), key=lambda kv: -kv[1])
    # Truncated because this prints every cycle and the tail is noise: a
    # label seen once is a model slip, and the decision this feeds is about
    # what keeps recurring. The full list is in the table.
    summary = ", ".join(f"{name} x{count}" for name, count in ranked[:8])
    print(f"[news_ingest] taxonomy gaps proposed by the classifier "
          f"(last {category_ops.CATEGORY_SIGHTING_RETENTION_DAYS}d): {summary}")


def _emit_heartbeat() -> None:
    """One span per cycle, whether or not anything new was found -- same
    shape as news_push._emit_heartbeat. Answers "is this job ticking at
    all" from Logfire's side, independent of whether any individual
    source happened to be due this cycle (see _pull_source's own
    ingest_source_pull span for that finer-grained question) -- a Logfire
    alert on this span's absence replaced healthcheck.py's in-process
    polling (2026-08-25/29, see docs/plans/observability-platform-plan.md).

    Called unconditionally at the very top of run_ingestion_cycle,
    deliberately BEFORE the `if not fetched: return` further down --
    news_push._emit_heartbeat's count attribute (subscriber count) is
    known before that function's own loop runs; the equivalent count
    here (new articles cached) is only known at the very end, past a
    return this cycle might never reach. Carrying no count is the
    correct tradeoff: a heartbeat that sometimes doesn't fire is a much
    worse bug than one with a less informative payload."""
    with _tracer.start_as_current_span("ingest_heartbeat") as span:
        span.set_attribute("heartbeat.job", "ingest_tick")


def _pull_source(
    source_key: str, fetch, now: datetime, existing_links: set[str],
) -> tuple[list[tuple[str, dict]], int, int]:
    """Pulls one source for one cycle, wrapped in a single `ingest_source_pull`
    span covering the whole attempt -- including the not-due/budget-capped
    skip cases, so every source gets exactly one span every cycle whether
    or not it was actually fetched. That's what makes "the cycle silently
    stopped partway through, never reaching source N" distinguishable in
    Logfire from "source N correctly wasn't due yet" -- the print-only
    output this replaces never made that distinction queryable.

    A source can have several sections (arxiv, hackernews, newsapi,
    gnews); the existing per-section `fetch_source` span
    (news_sources.traced_fetch) nests inside this one automatically via
    OTel's own context propagation, so section-level detail (including
    its own `error` attribute) isn't lost, just not promoted to this
    span's source-level `pull.outcome` -- `failed` only when EVERY
    attempted section errored, `success` if at least one didn't (a
    source that's basically alive shouldn't read as failed over one
    transient section error).

    Returns (fetched, duplicates_skipped, non_latin_skipped) for the
    caller to fold into its own cycle-wide totals -- `existing_links` is
    mutated in place (same set the caller keeps checking against for
    later sources this same cycle)."""
    with _tracer.start_as_current_span("ingest_source_pull") as pull_span:
        pull_span.set_attribute("pull.source", source_key)
        pull_span.set_attribute("pull.expected_interval_hours", _interval_hours(source_key))

        last_pulled_at = source_state_ops.get_source_last_pulled_at(source_key)
        if not _is_source_due(source_key, last_pulled_at, now):
            pull_span.set_attribute("pull.outcome", "not_due")
            print(f"[news_ingest] {source_key}: not due yet")
            return [], 0, 0

        daily_cap = _daily_cap(source_key)
        if daily_cap is not None and not api_budget_ops.try_consume_api_budget(
            source_key, daily_cap, now.date().isoformat()
        ):
            pull_span.set_attribute("pull.outcome", "budget_exhausted")
            print(f"[news_ingest] {source_key}: daily budget of {daily_cap} reached, skipping")
            return [], 0, 0

        time_filterable = _SOURCE_CLASS.get(source_key) in _TIME_FILTERABLE_CLASSES
        max_results = MAX_RESULTS_PER_SOURCE_SINCE_LAST_PULL if time_filterable else MAX_RESULTS_PER_SOURCE_RSS
        # The newest article's own published_dt actually seen from this
        # source so far -- NOT last_pulled_at (see module docstring for
        # why that was the wrong cutoff: it advances on wall-clock time
        # regardless of whether anything new was found, which can
        # permanently skip an article a source indexes with a delay).
        sections = _sections_for_source(source_key, now)
        source_articles = 0
        source_new = 0
        source_non_latin = 0
        sections_failed = 0
        fetched: list[tuple[str, dict]] = []
        duplicates_skipped = 0
        non_latin_skipped = 0
        for i, section in enumerate(sections):
            # Read per section, not once for the source -- see _cutoff_key.
            last_article_dt = source_state_ops.get_source_last_article_dt(
                _cutoff_key(source_key, section))
            newest_seen_this_cycle = last_article_dt
            # Server-side date filter for the 3 sources confirmed to support
            # it correctly (see _SERVER_SIDE_SINCE_SOURCES above) -- an
            # efficiency optimization only; the client-side filter below is
            # what's actually authoritative for "new", regardless of whether
            # this fires. None on the first-ever pull -- nothing to filter
            # against yet, so just take the unfiltered top-N up to the cap.
            if (time_filterable and source_key in _SERVER_SIDE_SINCE_SOURCES
                    and last_article_dt is not None):
                fetch_call = functools.partial(fetch, since=last_article_dt)
            else:
                fetch_call = fetch
            if i > 0:
                # Real incident, first deploy of this job: GNews's
                # documented 1-request/second limit (docs/current/ai-news-sources.md)
                # returned 429 on 5 of 7 back-to-back calls for the same
                # source in one cycle. A flat delay between consecutive
                # calls to the SAME source is cheap here (cycles run every
                # 4h+, a few extra seconds is nothing) and avoids needing a
                # per-source rate table for limits that aren't always
                # documented up front.
                time.sleep(REQUEST_DELAY_SECONDS)
            try:
                # The section, not a query: the fetchers keep their query
                # parameter for agent.py's search_news, and ignore it when
                # a section is given.
                call = (functools.partial(fetch_call, section=section)
                        if section is not None else fetch_call)
                articles = news_sources.traced_fetch(
                    source_key, call, _DEFAULT_QUERY, max_results, section=section)
            except Exception as exc:
                print(f"[news_ingest] {source_key}: fetch(section={section!r}) "
                      f"failed with {exc!r}")
                pull_span.record_exception(exc)
                sections_failed += 1
                continue
            if time_filterable and last_article_dt is not None:
                # The actually-authoritative "new" check -- applies
                # regardless of whether a server-side filter ran above, so
                # it's correct even for newsapi/perigon (no server-side
                # filter at all) and even if a server-side filter silently
                # didn't behave as expected. An article with no parseable
                # published_dt is kept rather than dropped -- can't tell if
                # it's new, and news_cache's dedup-by-link-hash makes
                # re-caching a harmless overwrite either way (see
                # news_cache.write_article).
                articles = [a for a in articles if a.get("published_dt") is None or a["published_dt"] > last_article_dt]
            source_articles += len(articles)
            for article in articles:
                link = article.get("link")
                if not link:
                    continue
                if link in existing_links:
                    duplicates_skipped += 1
                    continue
                # Checked on the title, not the summary: a summary can
                # quote a foreign-language statement inside an otherwise
                # English article, and the title is what every downstream
                # consumer -- classification, embedding, the digest
                # listing -- actually reads. Deliberately NOT added to
                # existing_links, so this costs one string scan per cycle
                # rather than being remembered as "seen".
                if not is_latin_script(article.get("title") or ""):
                    non_latin_skipped += 1
                    source_non_latin += 1
                    continue
                existing_links.add(link)
                fetched.append((source_key, article))
                source_new += 1
                published_dt = article.get("published_dt")
                if published_dt is not None and (newest_seen_this_cycle is None or published_dt > newest_seen_this_cycle):
                    newest_seen_this_cycle = published_dt

            if (time_filterable and newest_seen_this_cycle is not None
                    and newest_seen_this_cycle != last_article_dt):
                source_state_ops.set_source_last_article_dt(
                    _cutoff_key(source_key, section), newest_seen_this_cycle)

        # last_pulled_at stays per SOURCE: the pull interval and the daily
        # budget are properties of the source, not of one of its sections.
        source_state_ops.set_source_last_pulled_at(source_key, now)
        pull_span.set_attribute("pull.sections_attempted", len(sections))
        pull_span.set_attribute("pull.sections_failed", sections_failed)
        pull_span.set_attribute(
            "pull.outcome",
            "failed" if sections and sections_failed == len(sections) else "success")
        print(
            f"[news_ingest] {source_key}: fetched {source_articles} article(s) across "
            f"{len(sections)} section{'' if len(sections) == 1 else 's'} -- "
            f"{source_new} new, "
            f"{source_articles - source_new - source_non_latin} already cached, "
            f"{source_non_latin} dropped as non-Latin"
        )
        return fetched, duplicates_skipped, non_latin_skipped


def run_ingestion_cycle(model, now: datetime | None = None, embedder=None) -> None:
    """One scheduler tick. Every outcome is printed -- same reasoning as
    news_push.py's run_push_cycle: a silent per-source/per-cycle failure
    was a real incident there (docs/reference/observability-and-debugging.md),
    worth not repeating here.

    `embedder=None` (the default, and what every existing test and
    call site gets without changes) means every article is cached with
    embedding=None -- see news_embed's module docstring on why a missing
    embedder must degrade the pipeline, never break it. Pass one built
    by news_embed.build_embedder() to actually populate it."""
    now = now or datetime.now(timezone.utc)
    _emit_heartbeat()

    deleted = news_cache.cleanup_expired(now)
    print(f"[news_ingest] tick at {now.isoformat()}: cleaned up {deleted} expired cache entr{'y' if deleted == 1 else 'ies'}")
    # Sightings age out on the same tick as the cache, so the threshold
    # question stays "how often recently" rather than "how often ever" --
    # see category_ops.CATEGORY_SIGHTING_RETENTION_DAYS.
    category_ops.prune_category_sightings(now)
    # Reported here, next to the prune, rather than after classification:
    # both are about accumulated evidence and neither depends on whether
    # this cycle fetched anything. Placing it after the `if not fetched`
    # return meant a quiet cycle pruned the evidence but never showed it.
    _report_category_proposals(now)

    fetched: list[tuple[str, dict]] = []
    # Loaded once per cycle (after cleanup_expired, so already-expired
    # links are gone and eligible to be re-added) -- skips re-classifying
    # a link this cycle already has a live cache entry for. Matters most
    # for RSS-class sources now that their cap is 200, not 5 (see module
    # docstring): most of a 200-item RSS pull is typically unchanged from
    # last cycle, and without this check every one of them would cost a
    # real, paid DeepSeek classification call every cycle for no reason.
    existing_links = {a["link"] for a in news_cache.read_all() if a.get("link")}
    total_duplicates_skipped = 0
    non_latin_skipped = 0

    for source_key, fetch in news_sources.enabled_sources():
        fetched_here, dup, non_latin = _pull_source(
            source_key, fetch, now, existing_links)
        fetched.extend(fetched_here)
        total_duplicates_skipped += dup
        non_latin_skipped += non_latin

    if non_latin_skipped:
        print(f"[news_ingest] dropped {non_latin_skipped} non-Latin-script "
              f"article(s) at ingestion (see is_latin_script)")

    if not fetched:
        print(
            f"[news_ingest] tick at {now.isoformat()}: nothing new to classify "
            f"({total_duplicates_skipped} already-cached article(s) skipped)"
        )
        return

    # Loaded once per cycle, not per chunk: the taxonomy must not change
    # underneath a batch, or the prompt and the validation set would
    # disagree about what a valid answer is.
    taxonomy = news_classify.Taxonomy.from_rows(category_ops.get_active_categories())
    categories_by_index = news_classify.classify_articles(
        model, [a for _, a in fetched], taxonomy,
        on_unknown_label=lambda label, article: category_ops.record_category_sighting(
            label, now, article.get("link"), article.get("title")
        ),
    )
    # Title+summary, not title alone -- measured 2026-08-25
    # (docs/analysis/cluster-measurements.md, "Title+summary embedding")
    # to genuinely improve recall for articles whose headline is written
    # in a business-outcome style with no topic vocabulary in it (the
    # actual on-topic content lives only in the summary). No article body
    # exists anywhere in this system to embed instead -- RSS summaries are
    # capped at 300 chars in news_sources.py, and that's the richest text
    # available. `summary` is None for some sources (e.g. hackernews), so
    # this falls back to title alone for those rather than embedding "None".
    # One batch call for the whole cycle (never per-article -- see
    # news_embed.embed_texts's own docstring on why), aligned back to
    # `fetched` by plain list index: unlike classify_articles, embed_texts
    # never drops an item, so there's no _by_index dict to look up, just
    # a same-length list.
    embeddings = news_embed.embed_texts(
        embedder,
        [f"{a.get('title') or ''} {a.get('summary') or ''}".strip() for _, a in fetched],
    )
    unclassified = 0
    for i, (source_key, article) in enumerate(fetched):
        # Three distinct outcomes, three distinct records. `.get(i)` is None
        # when the chunk this article was in failed, which is NOT the same as
        # the model saying nothing applied -- writing [] for both is what let
        # a three-day classification outage look like normal operation.
        categories = categories_by_index.get(i)
        if categories is not None and not categories:
            categories = [category_ops.UNCLASSIFIABLE]
        if categories is None:
            unclassified += 1
        news_cache.write_article(source_key, article, categories, now, embedding=embeddings[i])
    if unclassified:
        print(f"[news_ingest] {unclassified} of {len(fetched)} article(s) cached "
              f"WITHOUT being classified -- the classifier failed for them")

    # The new/duplicate split here is logged specifically so the RSS cap
    # (MAX_RESULTS_PER_SOURCE_RSS) can be tuned later from real data rather
    # than guessed at -- see the module docstring for why 200 was picked.
    print(
        f"[news_ingest] tick at {now.isoformat()}: cached {len(fetched)} new article(s), "
        f"{total_duplicates_skipped} already-cached article(s) skipped"
    )

    _refresh_category_keyness(now)


def _refresh_category_keyness(now: datetime) -> None:
    """Recomputes news_keyness's per-category "how foreign is this word"
    scores over the WHOLE current cache (not just this cycle's new
    articles -- keyness needs the full category pool to mean anything)
    and persists them, once per active category, so news_push.py's read
    at push time is a cheap local DB lookup rather than ever running
    NLTK live in the push path. See news_keyness.py's own module
    docstring for the full design and docs/analysis/cluster-
    measurements.md's "Offbeat selection, take two" for the measurement
    behind it.

    Runs every ingestion cycle (every INGEST_TICK_SECONDS, same cadence
    as embedding computation), deliberately -- not on a separate, slower
    schedule. A periodic-but-separate job was considered and rejected:
    the gap between "an article is ingested" and "its category's keyness
    table reflects it" would otherwise be a real staleness window, and
    the articles most likely to fall in that window are exactly the
    newest, most novelty-relevant ones -- the opposite of what this
    feature is for.

    POS-tagging (build_noun_index) is the one expensive step and is
    shared across every active category -- computing keyness for 13
    categories costs one tagging pass, not 13. Measured on the Phoenix
    VM's own hardware (an OCI VM.Standard.E2.1.Micro, same shape as the
    bot VM): 12.6s wall time and 84.6 MB peak RSS for 2673 real articles.

    Fails open per category (a tagging exception or an empty pool for one
    category doesn't block the others) and as a whole (an exception here
    is a push-quality regression, never something that should take down
    an otherwise-successful ingestion cycle)."""
    try:
        articles = news_cache.read_all()
        if not articles:
            return
        doc_terms, global_df = news_keyness.build_noun_index(articles)
        categories = [name for name, _description in category_ops.get_active_categories()]
        for category in categories:
            scores = news_keyness.category_keyness(articles, doc_terms, global_df, category)
            interest_cache_ops.set_category_keyness(category, scores)
        print(f"[news_ingest] tick at {now.isoformat()}: refreshed keyness for {len(categories)} categor{'y' if len(categories) == 1 else 'ies'}")
    except Exception as exc:
        _events.log("keyness_refresh_failed",
                     "category keyness refresh failed, offbeat selection degrades to keyword-only/recency this cycle",
                     level=Level.WARN, exc=exc)

"""
Pluggable AI-industry news sources for the search_news tool in agent.py.

Each source is a function `fetch(query, max_results) -> list[dict]` returning
a normalized article shape: {"title", "link", "source", "summary",
"published", "published_dt"}. "published" is the raw, source-specific date
string (for display); "published_dt" is that same date parsed into a
timezone-aware datetime (UTC), or None if parsing failed -- added so
callers (search_news, and news_push.py's periodic-digest dedup) can reason
about recency without each doing their own per-source date parsing. See
the incident this responds to: search_news wasn't surfacing "published" to
the model at all, so it had no way to judge freshness or avoid repeating
itself across calls -- see docs/plans/bot-features-plan.md item 5.

SOURCE_REGISTRY lists sources with the env var (if any) required to enable
them; enabled_sources() skips key-gated sources whose key isn't set, so the
tool degrades gracefully instead of erroring. See docs/current/ai-news-sources.md
for what each source is and how to add a new one.

RSS sources are configured via Settings (news_source.rss in settings.yml),
not hardcoded here -- see _rss_sources_from_settings.

The two free, always-on sources (hackernews, arxiv) and the three
credential-gated sources (newsapi, gnews, perigon) are NewsSourceAdapter
classes under news_adapters/ -- see that package's __init__.py for the
interface (initialize()/pull()) and the discover_adapter_types()/
validate_configured_types() mechanism that turns a news_source.api entry
into a live adapter instance. hackernews/arxiv are wired in directly by
_always_on_sources() below (always on, no credential, no settings entry
needed); newsapi/gnews/perigon are read from news_source.api, one entry
per source -- see _api_sources_from_settings.
"""

from opentelemetry import trace
from trailsign import Settings, SettingsError

import feedparser
import requests

from app_settings import get_settings
from news_adapters import discover_adapter_types, validate_configured_types
from news_adapters._util import _redact, _REQUEST_HEADERS, _parse_rss_published
from news_adapters.arxiv import ArxivAdapter
from news_adapters.hackernews import HackerNewsAdapter

import re


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_summary(raw: str | None, max_len: int = 300) -> str | None:
    """feedparser's entry.summary carries the feed's own <description>,
    which several sources give as a real lede paragraph (confirmed live:
    Guardian gives ~1200 chars of actual editorial context, MarketWatch a
    full sentence explaining *why* a stock moved) -- previously discarded
    outright by _fetch_rss hardcoding summary=None, so the model only ever
    saw a bare title. Some feeds embed raw HTML (<p>, <a href>) in the
    description; stripped here since it's read by the model, not rendered.
    Returns None (not "") when the feed genuinely has nothing, so downstream
    code can still tell "no summary" from "empty after cleaning" apart."""
    if not raw:
        return None
    text = _HTML_TAG_RE.sub("", raw).replace("\n", " ").strip()
    return text[:max_len] if text else None


def _fetch_rss(url: str, source_name: str, max_results: int = 5) -> list[dict]:
    resp = requests.get(url, timeout=10, headers=_REQUEST_HEADERS)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    return [
        {
            "title": entry.get("title"),
            "link": entry.get("link"),
            "source": source_name,
            "summary": _clean_summary(entry.get("summary")),
            "published": entry.get("published"),
            "published_dt": _parse_rss_published(entry),
        }
        for entry in feed.entries[:max_results]
    ]


# RSS sources are pure data (a URL and a display name -- see _fetch_rss
# above, every one of them is query-less and returns the same latest-N
# regardless of what was asked), so they live in Settings
# (news_source.rss, a list of {key, display_name, url}) instead of one
# hardcoded function per feed. A deployer adds/removes/edits feeds by
# editing their own settings.yml, no code change -- see settings.yml's
# own news_source.rss section for the current list, grouped the same way
# this file used to group the individual fetch_ functions (AI/ML
# publications, mainstream press business/technology, enterprise IT
# trade press, consumer/gadget press). See docs/current/ai-news-sources.md for
# sources tested and rejected (CNN's feeds are abandoned -- lastBuildDate
# over a year stale; Fortune blocks with 403; Reuters discontinued public
# RSS in 2020).


def _make_rss_fetcher(url: str, display_name: str) -> callable:
    """One closure per configured feed, matching the (query, max_results)
    -> list[dict] shape every other source's fetch function has -- `query`
    is accepted and ignored, same as the old hardcoded fetch_ functions
    (RSS has no query parameter to filter by, see _fetch_rss)."""
    def fetch(query: str = None, max_results: int = 5) -> list[dict]:
        return _fetch_rss(url, display_name, max_results)
    return fetch


def _raw_rss_entries() -> list[dict]:
    """Raw news_source.rss entries. No credential to resolve here (RSS
    never carries one), so unlike _raw_api_entries this is just the plain
    resolved list -- exposed as its own function so news_ingest.py can
    look up one entry's optional interval_hours override (e.g.
    venturebeat_ai's, see settings.yml) the same way it already does for
    news_source.api entries, without needing its own copy of this read."""
    return get_settings().resolved("news_source.rss", default=[])


def _rss_sources_from_settings() -> list[tuple[str, callable, None, str]]:
    """Builds the RSS portion of SOURCE_REGISTRY from news_source.rss --
    default=[] rather than required=True, since a deployer running with
    zero configured RSS feeds is a legitimate (if sparse) state, not a
    misconfiguration -- same "fails open, doesn't crash" shape as every
    other optional subsystem in this project (e.g. news_embed's embedder).
    Each entry becomes a 4-tuple matching every other SOURCE_REGISTRY row:
    (key, fetch_fn, required_env=None -- RSS never needs a key, source_class="rss")."""
    entries = _raw_rss_entries()
    return [
        (entry["key"], _make_rss_fetcher(entry["url"], entry["display_name"]), None, "rss")
        for entry in entries
    ]


def _make_adapter_fetcher(adapter) -> callable:
    """Adapts one already-initialize()'d adapter instance's pull() method
    to the (query, max_results, since=None, section=None) -> list[dict]
    shape every other source's fetch function has, so news_ingest.py's
    existing functools.partial(fetch, since=...)/(fetch, section=...)
    wiring (see its _pull_source) works unchanged regardless of whether
    the callable underneath is a plain function or an adapter's bound
    method."""
    def fetch(query: str, max_results: int, since=None, section=None) -> list[dict]:
        return adapter.pull(query, max_results, since=since, section=section)
    return fetch


def _always_on_sources() -> list[tuple[str, callable, None, str]]:
    """hackernews/arxiv -- free, no credential, always registered
    regardless of settings (these were never optional, unlike
    newsapi/gnews/perigon below). Not read from news_source.api: there's
    nothing to configure for a source with no credential and no override,
    so routing them through the same settings-driven discovery/validation
    as the credentialed sources would only add ceremony with no behavior
    to justify it. Still implemented as NewsSourceAdapter classes (see
    news_adapters/hackernews.py, arxiv.py) for consistency with the
    credentialed sources -- just wired in directly here instead of via
    _api_sources_from_settings."""
    entries = [
        ("hackernews", HackerNewsAdapter(), "forum"),
        ("arxiv", ArxivAdapter(), "api"),
    ]
    for _key, adapter, _source_class in entries:
        adapter.initialize({})
    return [
        (key, _make_adapter_fetcher(adapter), None, source_class)
        for key, adapter, source_class in entries
    ]


def _raw_api_entries() -> list[dict]:
    """Raw (unresolved) news_source.api entries -- reading this list via
    Settings.resolved() directly would recursively resolve every entry's
    api-key too, and one entry with an unresolvable credential (e.g. an
    unset env var for a source that's intentionally not configured) would
    raise SettingsError for the WHOLE list, taking every OTHER configured
    source down with it. Confirmed live, not assumed: a Settings wrapping
    a news_source.api list with one resolvable entry and one entry whose
    api-key points at an unset env var raises the moment .resolved(
    "news_source.api") is called, even though only the second entry is
    actually broken.

    Reading the raw list here (each entry's own trailsign-resolve nodes
    still intact, not yet resolved) and resolving each entry's own
    api-key independently (see _resolved_api_key) is what preserves the
    "one bad optional credential degrades that ONE source, not the whole
    process" contract every other optional-source path in this file has
    always had. `key`/`type`/`interval_hours`/`daily_cap` are read
    straight off these raw dicts elsewhere (news_ingest.py's
    _interval_hours/_daily_cap) without going through Settings.resolved()
    at all -- they're always plain literals in practice, never
    trailsign-resolve nodes, so there's nothing to resolve for them."""
    return get_settings()._raw.get("news_source", {}).get("api", [])


def _resolved_api_key(entry: dict) -> str | None:
    """Resolves one entry's own api-key node in isolation -- see
    _raw_api_entries for why this can't just be part of a bulk
    Settings.resolved() call on the whole list. None if unresolvable (an
    unset env var, e.g.) -- matches this file's established "optional
    source silently degrades" contract, never raises."""
    try:
        return Settings(entry).resolved("api-key", default=None)
    except SettingsError:
        return None


def _api_sources_from_settings() -> list[tuple[str, callable, None, str]]:
    """Builds the credential-gated portion of SOURCE_REGISTRY from
    news_source.api -- one settings entry per source, each naming which
    news_adapters/ class (its `type`) handles it. Mirrors
    _rss_sources_from_settings's shape (default=[] via _raw_api_entries:
    an empty/absent list is a legitimate "no optional sources configured"
    state, not an error).

    Two things happen here that _rss_sources_from_settings doesn't need:
    (1) validate_configured_types -- fails the whole process at import
    time if a configured `type` has no matching class under
    news_adapters/, per the explicit "don't start the service" requirement
    for that case; (2) per-entry credential resolution via
    _resolved_api_key, which silently DROPS (not raises) an entry whose
    api-key can't be resolved -- same "optional source degrades quietly"
    contract this file has always had. Resolved once here, at
    SOURCE_REGISTRY-build (import) time, not lazily re-checked on every
    enabled_sources() call the way the old env-var-based gate was -- a
    credential added or removed after import needs a re-import (a
    process restart) to take effect, not just a settings edit. See
    docs/standaloneplan/01-settings-migration.md for this tradeoff."""
    configured = _raw_api_entries()
    discovered = discover_adapter_types()
    validate_configured_types(discovered, configured)

    entries = []
    for entry in configured:
        resolved_entry = dict(entry)
        if "api-key" in entry:
            api_key = _resolved_api_key(entry)
            if api_key is None:
                continue
            resolved_entry["api-key"] = api_key
        adapter = discovered[entry["type"]]()
        adapter.initialize(resolved_entry)
        # Every source configured via news_source.api is real query-based
        # search, JSON REST -- that's the "api" source_class by definition
        # (see the SOURCE_REGISTRY comment below for what each class means).
        entries.append((entry["key"], _make_adapter_fetcher(adapter), None, "api"))
    return entries


# The section vocabulary each query-capable source accepts, as an
# alternative to a search query. The values are dictated by each API and
# live here alongside the source registry; the reasoning for pulling by
# section at all is ingestion policy and lives with it, in
# news_ingest._sections_for_source.
#
# Used only by news_ingest's scheduled pulls. agent.py's search_news still
# passes a real user question, which is a legitimate query.
SOURCE_SECTIONS: dict[str, list[str]] = {
    # https://newsapi.org/docs/endpoints/top-headlines -- fixed vocabulary
    "newsapi": ["technology", "business", "science", "health"],
    # https://gnews.io/docs/v4 -- `topic` on /top-headlines
    "gnews": ["technology", "business", "science", "world"],
    # arXiv subject classes. Deliberately includes quant-ph and
    # physics.optics: subscribers follow quantum computing/sensing and
    # optical communications, and free-text search found 36 and 6 articles
    # for those -- too few to cluster -- while the subject classes are
    # exactly the right handle. These are papers, not industry news, so
    # they widen the corpus more than they feed digests.
    "arxiv": ["cs.AI", "cs.LG", "cs.RO", "cs.CR", "quant-ph", "physics.optics"],
    # Algolia's front_page tag is HN's own ranking, which is a better
    # relevance signal than anything a query could express here.
    "hackernews": ["front_page"],
}


# --- Registry -------------------------------------------------------------

# (name, fetch_fn, gate_or_None, source_class)
#
# source_class is descriptive, not behavioral -- enabled_sources() ignores
# it today. It exists because the registry stopped being uniform once
# mainstream press was added: most sources here are "forum" (community
# board) or "api" (real query-based search) are the exception, not the
# rule -- only hackernews/arxiv/newsapi/gnews/perigon actually filter by
# `query`; every "rss" source (news_source.rss in Settings, see
# _rss_sources_from_settings above) returns its latest N items regardless
# of what was asked (see _fetch_rss). That distinction matters for
# anything downstream that assumes a nonzero result means a real topic
# match -- see docs/plans/local-news-cache-plan.md.
#
#   forum -- community-curated discussion board, not edited articles
#   api   -- real query-based search, JSON REST
#   rss   -- standard RSS/Atom feed, query-less (latest N regardless)
#
# `gate` is always None now, for every source class -- credential gating
# for the api-class sources happens once, at _api_sources_from_settings's
# construction time (an unresolvable credential just means that entry
# never makes it into this list at all), not via a lazily-rechecked gate
# value the way the old env-var-based design worked. The slot is kept
# for tuple-shape stability (RESTRICTED_SOURCES filtering, tests, and
# news_ingest._SOURCE_CLASS all unpack 4-tuples) even though nothing
# reads it anymore.
SOURCE_REGISTRY = _always_on_sources() + _api_sources_from_settings() + _rss_sources_from_settings()


# Sources gated behind per-user access, on top of the credential gate
# above -- not because they're technically different (they're plain "api"-
# class sources like GNews), but because their real-world usage is
# constrained in ways that don't scale to every caller of search_news:
# NewsAPI's free tier is documented as development/testing only, not
# production (docs/current/ai-news-sources.md), and Perigon's 150/month
# budget is already fully spoken for by news_ingest.py's own scheduled
# pulls (3/day, see docs/plans/local-news-cache-plan.md) -- search_news
# calling them too, on every matching on-demand query from every user,
# would exhaust both almost immediately. GNews is deliberately not here:
# its 100/day budget has real headroom beyond what news_ingest.py alone
# uses.
RESTRICTED_SOURCES = {"newsapi", "perigon"}


def enabled_sources(include_restricted: bool = True) -> list[tuple[str, callable]]:
    """(name, fetch_fn) pairs usable right now: the always-on free
    sources, RSS sources, and any credential-gated source whose
    news_source.api[].api-key resolved at SOURCE_REGISTRY construction
    time (see _api_sources_from_settings). `include_restricted`
    additionally excludes RESTRICTED_SOURCES when False -- see agent.py's
    search_news, the only caller that ever passes False; every other
    caller (news_ingest.py) keeps the default so it's unaffected. Same
    2-tuple shape as before source_class was added -- callers and tests
    all unpack exactly (name, fn)."""
    return [
        (name, fn)
        for name, fn, _gate, _source_class in SOURCE_REGISTRY
        if include_restricted or name not in RESTRICTED_SOURCES
    ]


_tracer = trace.get_tracer(__name__)


def traced_fetch(source_key: str, fetch: callable, query: str, max_results: int,
                 section: str | None = None) -> list[dict]:
    """Wraps one source's fetch call in an OpenTelemetry span, so it shows
    up in the telemetry backend's unified trace view alongside LLM calls
    -- this is the layer openinference-instrumentation-langchain's
    auto-instrumentation can't reach on its own. A plain fetch() here is a
    bare requests.get() invoked from news_ingest.py's scheduled pull loop
    or agent.py's search_news, entirely outside LangChain's tool-calling
    loop; auto-instrumentation only sees LangChain-mediated calls (LLM
    invocations, @tool-decorated tool calls), so without this wrapper
    these fetches are invisible to telemetry regardless of whether it's
    enabled. See docs/plans/local-news-cache-plan.md's "API call
    visibility" section.

    trace.get_tracer() returns a real tracer once agent.py's
    setup_telemetry() has built the general TracerProvider (see
    telemetry.py -- this module's spans are general-category, so it's
    always that one, never the llm provider), or a safe no-op tracer if
    it hasn't (telemetry.providers empty, or setup_telemetry() never
    called at all, e.g. in tests) -- this call has no effect either way
    when tracing isn't configured, same "no-op by default" shape as
    everywhere else telemetry touches this project.

    Re-raises on fetch failure after recording it on the span -- callers
    already have their own per-source try/except (news_ingest.py,
    search_news), so error isolation is unchanged; this only adds
    visibility, it doesn't change control flow."""
    with _tracer.start_as_current_span("fetch_source") as span:
        span.set_attribute("source_key", source_key)
        # A section-based pull ignores the query, so recording it would put
        # the same placeholder on every ingestion span and hide which
        # section was actually fetched -- the one thing worth knowing when
        # diagnosing one of these.
        if section is not None:
            span.set_attribute("section", section)
        else:
            span.set_attribute("query", query)
        span.set_attribute("restricted", source_key in RESTRICTED_SOURCES)
        try:
            articles = fetch(query, max_results)
        except Exception as exc:
            # Redacted for the same reason as _raise_for_status: a fetch
            # error's text can carry the request URL, and this attribute is
            # shipped to the telemetry backend and retained there (Logfire's
            # own retention policy, not configured here -- see
            # docs/system-overview.md §C4). Belt-and-braces --
            # _raise_for_status already strips it at the source for the
            # key-gated adapters, but this is the last point before the
            # value leaves the process.
            span.set_attribute("error", _redact(exc))
            span.set_attribute("article_count", 0)
            raise
        span.set_attribute("article_count", len(articles))
        return articles

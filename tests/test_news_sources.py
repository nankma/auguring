import pytest
from trailsign import Settings, SettingsError

import news_sources
from news_adapters import discover_adapter_types, validate_configured_types
from news_adapters._util import FeedBlockedError
from tests.fixtures import HACKERNEWS_RESPONSE, RSS_RESPONSE


def _set_api_sources(monkeypatch, entries):
    """Replaces the old _set_news_source_key -- news_source.api is a LIST
    now ([{key, type, api-key, ...}, ...]), not separate
    news_source.newsapi/.gnews/.perigon dict keys -- e.g.
    _set_api_sources(monkeypatch, [{"key": "newsapi", "type": "newsapi",
    "api-key": "fake-key"}]). An empty list (or omitting api entries
    entirely) is the "nothing configured" state.

    Carries conftest.py's injected fake_rss_source RSS entry along --
    this replaces the WHOLE fake Settings object (get_settings is
    monkeypatched wholesale, there's no way to override just news_source.
    api on top of the real one), so a test using this helper and then
    calling _rebuild_registry must still see the same RSS source
    enabled_sources() tests everywhere else rely on."""
    fake_settings = Settings({"news_source": {
        "api": entries,
        "rss": [{"key": "fake_rss_source", "display_name": "Fake RSS Source", "url": "https://fake.invalid/feed.xml"}],
    }})
    monkeypatch.setattr(news_sources, "get_settings", lambda: fake_settings)


def _rebuild_registry(monkeypatch):
    """news_sources.SOURCE_REGISTRY is a module-level constant, computed
    once at import time -- credential resolution for news_source.api
    entries now happens at THAT construction point (see
    news_sources._api_sources_from_settings's own docstring), not
    lazily re-checked on every enabled_sources() call the way the old
    env-var gate was. A test that changes Settings mid-test and needs
    enabled_sources() to reflect it has to explicitly rebuild the
    registry, which is what this helper does."""
    registry = (
        news_sources._always_on_sources()
        + news_sources._api_sources_from_settings()
        + news_sources._rss_sources_from_settings()
    )
    monkeypatch.setattr(news_sources, "SOURCE_REGISTRY", registry)
    return registry


# --- RSS mechanism (unchanged by this refactor) ----------------------------


def test_parse_rss_published_handles_missing():
    assert news_sources._parse_rss_published({}) is None


def test_fetch_rss_generic(requests_mock):
    requests_mock.get("https://example.com/feed.xml", text=RSS_RESPONSE)

    articles = news_sources._fetch_rss("https://example.com/feed.xml", "Fake Blog", 5)

    assert len(articles) == 2
    assert articles[0]["title"] == "Fake Blog Post One"
    assert articles[0]["link"] == "https://example.com/blog/post-one"
    assert articles[0]["source"] == "Fake Blog"


def test_fetch_rss_generic_respects_max_results(requests_mock):
    requests_mock.get("https://example.com/feed.xml", text=RSS_RESPONSE)

    articles = news_sources._fetch_rss("https://example.com/feed.xml", "Fake Blog", 1)

    assert len(articles) == 1


def test_make_rss_fetcher_uses_the_given_url_and_display_name(requests_mock):
    requests_mock.get("https://example.com/feed.xml", text=RSS_RESPONSE)

    fetch = news_sources._make_rss_fetcher("https://example.com/feed.xml", "Fake Blog")
    articles = fetch()

    assert len(articles) == 2
    assert articles[0]["source"] == "Fake Blog"


def test_make_rss_fetcher_ignores_query_and_respects_max_results(requests_mock):
    requests_mock.get("https://example.com/feed.xml", text=RSS_RESPONSE)

    fetch = news_sources._make_rss_fetcher("https://example.com/feed.xml", "Fake Blog")
    articles = fetch("this query is ignored, RSS has no query parameter", max_results=1)

    assert len(articles) == 1


def test_rss_sources_from_settings_builds_one_entry_per_config_row(monkeypatch):
    """The mechanism, not any specific real feed -- what's actually in
    news_source.rss is settings data now (see settings.yml), not a code
    invariant to lock in a unit test."""
    fake_settings = Settings({"news_source": {"rss": [
        {"key": "fake_a", "display_name": "Fake A", "url": "https://a.example.com/feed.xml"},
        {"key": "fake_b", "display_name": "Fake B", "url": "https://b.example.com/feed.xml"},
    ]}})
    monkeypatch.setattr(news_sources, "get_settings", lambda: fake_settings)

    entries = news_sources._rss_sources_from_settings()

    assert [key for key, *_ in entries] == ["fake_a", "fake_b"]
    assert all(required_env is None for _key, _fn, required_env, _cls in entries)
    assert all(source_class == "rss" for *_rest, source_class in entries)


def test_rss_sources_from_settings_entries_are_independently_callable(requests_mock, monkeypatch):
    requests_mock.get("https://a.example.com/feed.xml", text=RSS_RESPONSE)
    fake_settings = Settings({"news_source": {"rss": [
        {"key": "fake_a", "display_name": "Fake A", "url": "https://a.example.com/feed.xml"},
    ]}})
    monkeypatch.setattr(news_sources, "get_settings", lambda: fake_settings)

    [(_key, fetch, _required_env, _source_class)] = news_sources._rss_sources_from_settings()
    articles = fetch()

    assert articles[0]["source"] == "Fake A"


def test_rss_sources_from_settings_defaults_to_empty_list(monkeypatch):
    """A deployer with zero configured RSS feeds is a legitimate, sparse
    state -- fails open, doesn't raise."""
    monkeypatch.setattr(news_sources, "get_settings", lambda: Settings({}))

    assert news_sources._rss_sources_from_settings() == []


def test_two_different_rss_config_entries_produce_two_different_fetchers(monkeypatch):
    """The registry-breadth guarantee, generically: two distinct
    news_source.rss rows never collapse into the same fetch function."""
    fake_settings = Settings({"news_source": {"rss": [
        {"key": "fake_general", "display_name": "Fake General", "url": "https://a.example.com/feed.xml"},
        {"key": "fake_ai_only", "display_name": "Fake AI Only", "url": "https://b.example.com/feed.xml"},
    ]}})
    monkeypatch.setattr(news_sources, "get_settings", lambda: fake_settings)

    entries = news_sources._rss_sources_from_settings()
    fns = [fn for _key, fn, _env, _cls in entries]

    assert fns[0] is not fns[1]


# --- always-on free sources (hackernews, arxiv) ----------------------------


def test_always_on_sources_includes_hackernews_and_arxiv():
    keys = [key for key, _fn, _gate, _cls in news_sources._always_on_sources()]
    assert keys == ["hackernews", "arxiv"]
    classes = {key: cls for key, _fn, _gate, cls in news_sources._always_on_sources()}
    assert classes == {"hackernews": "forum", "arxiv": "api"}


def test_always_on_sources_are_independently_callable(requests_mock):
    requests_mock.get("https://hn.algolia.com/api/v1/search_by_date", json=HACKERNEWS_RESPONSE)

    entries = {key: fn for key, fn, _gate, _cls in news_sources._always_on_sources()}
    articles = entries["hackernews"]("AI", 5)

    assert len(articles) == 2


# --- news_source.api mechanism (adapter discovery/validation) -------------


def test_discover_adapter_types_finds_the_real_adapter_classes():
    discovered = discover_adapter_types()

    assert set(discovered) >= {"hackernews", "arxiv", "newsapi", "gnews", "perigon"}
    assert discovered["newsapi"].__name__ == "NewsApiAdapter"
    assert discovered["gnews"].__name__ == "GNewsAdapter"
    assert discovered["perigon"].__name__ == "PerigonAdapter"


def test_validate_configured_types_passes_for_known_types():
    discovered = discover_adapter_types()
    # must not raise
    validate_configured_types(discovered, [{"key": "newsapi", "type": "newsapi"}])


def test_validate_configured_types_raises_for_an_unknown_type():
    """The single most important new behavior in this refactor: a
    news_source.api entry naming a `type` with no matching adapter class
    must fail loudly at startup, not be silently dropped or skipped."""
    discovered = {"newsapi": object}

    with pytest.raises(SettingsError, match="madeup_type"):
        validate_configured_types(discovered, [{"key": "x", "type": "madeup_type"}])


def test_validate_configured_types_names_what_was_found():
    discovered = {"newsapi": object, "gnews": object}

    try:
        validate_configured_types(discovered, [{"key": "x", "type": "madeup_type"}])
        assert False, "expected SettingsError"
    except SettingsError as exc:
        assert "gnews" in str(exc) and "newsapi" in str(exc)


def test_api_sources_from_settings_raises_for_an_unknown_configured_type(monkeypatch):
    _set_api_sources(monkeypatch, [{"key": "x", "type": "not_a_real_adapter", "api-key": "k"}])

    with pytest.raises(SettingsError, match="not_a_real_adapter"):
        news_sources._api_sources_from_settings()


def test_api_sources_from_settings_builds_one_entry_per_config_row(monkeypatch):
    _set_api_sources(monkeypatch, [
        {"key": "newsapi", "type": "newsapi", "api-key": "fake-key"},
        {"key": "gnews", "type": "gnews", "api-key": "fake-key"},
    ])

    entries = news_sources._api_sources_from_settings()

    assert [key for key, *_ in entries] == ["newsapi", "gnews"]
    assert all(gate is None for _key, _fn, gate, _cls in entries)
    assert all(cls == "api" for *_rest, cls in entries)


def test_api_sources_from_settings_defaults_to_empty_list(monkeypatch):
    monkeypatch.setattr(news_sources, "get_settings", lambda: Settings({}))

    assert news_sources._api_sources_from_settings() == []


def test_api_sources_from_settings_skips_an_entry_whose_credential_is_unresolvable(monkeypatch):
    """The finding this refactor had to design around: resolving
    news_source.api as ONE list blows up the WHOLE list the moment any
    one entry's api-key is unresolvable (confirmed live against
    trailsign). _raw_api_entries + per-entry resolution
    (_resolved_api_key) is what keeps a misconfigured/intentionally-
    absent optional source's credential from taking every OTHER
    configured source down with it."""
    monkeypatch.setenv("REAL_KEY_ENV_VAR_FOR_TEST", "abc123")
    fake_settings = Settings({"news_source": {"api": [
        {"key": "newsapi", "type": "newsapi",
         "api-key": {"trailsign-resolve": "environment-variable", "name": "REAL_KEY_ENV_VAR_FOR_TEST"}},
        {"key": "gnews", "type": "gnews",
         "api-key": {"trailsign-resolve": "environment-variable", "name": "DEFINITELY_UNSET_ENV_VAR"}},
    ]}})
    monkeypatch.setattr(news_sources, "get_settings", lambda: fake_settings)

    entries = news_sources._api_sources_from_settings()

    assert [key for key, *_ in entries] == ["newsapi"]


def test_resolved_api_key_returns_none_for_an_unresolvable_credential(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_ENV_VAR", raising=False)
    entry = {"key": "gnews", "type": "gnews",
             "api-key": {"trailsign-resolve": "environment-variable", "name": "DEFINITELY_UNSET_ENV_VAR"}}

    assert news_sources._resolved_api_key(entry) is None


# --- enabled_sources ---------------------------------------------------


def test_enabled_sources_always_includes_free_sources(monkeypatch):
    """hackernews/arxiv are the two always-on free sources; fake_rss_source
    is conftest.py's injected news_source.rss entry, standing in for
    whatever a deployment's own RSS list actually contains."""
    _set_api_sources(monkeypatch, [])
    _rebuild_registry(monkeypatch)

    names = [name for name, _ in news_sources.enabled_sources()]

    for free_source in ("hackernews", "arxiv", "fake_rss_source"):
        assert free_source in names
    for gated_source in ("newsapi", "gnews", "perigon"):
        assert gated_source not in names


def test_enabled_sources_reflects_a_configured_credential(monkeypatch):
    _set_api_sources(monkeypatch, [])
    _rebuild_registry(monkeypatch)
    names = [name for name, _ in news_sources.enabled_sources()]
    assert "newsapi" not in names

    _set_api_sources(monkeypatch, [{"key": "newsapi", "type": "newsapi", "api-key": "fake-key"}])
    _rebuild_registry(monkeypatch)
    names = [name for name, _ in news_sources.enabled_sources()]
    assert "newsapi" in names


def test_enabled_sources_include_restricted_true_by_default(monkeypatch):
    _set_api_sources(monkeypatch, [
        {"key": "newsapi", "type": "newsapi", "api-key": "fake-key"},
        {"key": "perigon", "type": "perigon", "api-key": "fake-key"},
    ])
    _rebuild_registry(monkeypatch)

    names = [name for name, _ in news_sources.enabled_sources()]
    assert "newsapi" in names
    assert "perigon" in names


def test_enabled_sources_excludes_restricted_when_false(monkeypatch):
    _set_api_sources(monkeypatch, [
        {"key": "newsapi", "type": "newsapi", "api-key": "fake-key"},
        {"key": "perigon", "type": "perigon", "api-key": "fake-key"},
        {"key": "gnews", "type": "gnews", "api-key": "fake-key"},
    ])
    _rebuild_registry(monkeypatch)

    names = [name for name, _ in news_sources.enabled_sources(include_restricted=False)]
    assert "newsapi" not in names
    assert "perigon" not in names
    # gnews is a real api-class source but not in RESTRICTED_SOURCES -- its
    # budget has real headroom beyond what news_ingest.py alone uses
    assert "gnews" in names
    # unrestricted, always-on sources are unaffected
    assert "hackernews" in names
    assert "fake_rss_source" in names


# --- traced_fetch (unchanged) ------------------------------------------


def test_traced_fetch_returns_the_underlying_fetch_result():
    def fake_fetch(query, max_results):
        return [{"title": "a"}, {"title": "b"}]

    result = news_sources.traced_fetch("hackernews", fake_fetch, "AI", 5)

    assert result == [{"title": "a"}, {"title": "b"}]


def test_traced_fetch_reraises_on_error():
    def failing_fetch(query, max_results):
        raise RuntimeError("boom")

    try:
        news_sources.traced_fetch("hackernews", failing_fetch, "AI", 5)
        assert False, "expected RuntimeError to propagate"
    except RuntimeError as exc:
        assert str(exc) == "boom"


def test_traced_fetch_passes_query_and_max_results_through():
    captured = {}

    def fake_fetch(query, max_results):
        captured["query"] = query
        captured["max_results"] = max_results
        return []

    news_sources.traced_fetch("hackernews", fake_fetch, "robotics", 7)

    assert captured == {"query": "robotics", "max_results": 7}


def test_redact_strips_api_keys_from_error_text():
    # Real incident, 2026-08-19: GNews's and Perigon's live keys were found in
    # plaintext in `docker logs`, because requests puts the full request URL
    # into an HTTPError message and news_ingest.py logs the exception. These
    # are the exact two shapes that leaked, with fake key values.
    gnews = ("400 Client Error: Bad Request for url: https://gnews.io/api/v4/search"
             "?q=Edge+AI&lang=en&max=50&apikey=DEADBEEFCAFE1234&from=2026-08-19T01%3A18%3A39Z")
    perigon = ("403 Client Error: Forbidden for url: https://api.perigon.io/v1/all"
               "?q=quantum&size=50&apiKey=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    assert "DEADBEEFCAFE1234" not in news_sources._redact(gnews)
    assert "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" not in news_sources._redact(perigon)
    # the non-secret parts stay, so the message is still diagnosable
    assert "q=Edge+AI" in news_sources._redact(gnews)
    assert "400 Client Error" in news_sources._redact(gnews)


def test_traced_fetch_redacts_the_key_before_it_reaches_telemetry(monkeypatch):
    # traced_fetch's span attribute is shipped to the telemetry backend
    # (Logfire), so it is the last point the value could escape the process.
    recorded = {}

    class FakeSpan:
        def set_attribute(self, k, v):
            recorded[k] = v
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(news_sources._tracer, "start_as_current_span", lambda name: FakeSpan())

    def failing(query, max_results):
        raise RuntimeError("401 for url: https://x/y?apiKey=LEAKYVALUE999")

    try:
        news_sources.traced_fetch("gnews", failing, "AI", 5)
    except RuntimeError:
        pass
    assert "LEAKYVALUE999" not in recorded.get("error", "")
    assert "<redacted>" in recorded.get("error", "")


def test_traced_fetch_records_the_section_rather_than_a_placeholder_query():
    """A section pull ignores the query, so recording it would stamp the
    same placeholder on every ingestion span and hide the one thing worth
    knowing when diagnosing one."""
    captured = {}

    class FakeSpan:
        def set_attribute(self, k, v):
            captured[k] = v
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    import contextlib
    from unittest.mock import patch

    @contextlib.contextmanager
    def fake_span(name):
        yield FakeSpan()

    with patch.object(news_sources._tracer, "start_as_current_span", fake_span):
        news_sources.traced_fetch("arxiv", lambda q, n: [], "technology", 5,
                                  section="quant-ph")
    assert captured.get("section") == "quant-ph"
    assert "query" not in captured

    captured.clear()
    with patch.object(news_sources._tracer, "start_as_current_span", fake_span):
        news_sources.traced_fetch("arxiv", lambda q, n: [], "user question", 5)
    assert captured.get("query") == "user question"
    assert "section" not in captured


CHALLENGE_PAGE = "<!DOCTYPE html>\n<html><head><script>async function solved(d){}</script></head></html>"


def test_fetch_rss_raises_when_a_200_response_is_an_html_challenge_page(requests_mock):
    """Regression: The Register answered this VM's datacenter IP with an
    HTTP 200 anti-bot page, which feedparser turned into zero entries --
    "fetched 0 article(s)" with no failure logged anywhere."""
    requests_mock.get("https://example.com/feed.xml", text=CHALLENGE_PAGE)

    with pytest.raises(FeedBlockedError, match="https://example.com/feed.xml.*anti-bot"):
        news_sources._fetch_rss("https://example.com/feed.xml", "Fake A")


def test_fetch_rss_returns_empty_for_a_valid_feed_with_no_entries(requests_mock):
    requests_mock.get(
        "https://example.com/feed.xml",
        text='<?xml version="1.0"?><rss version="2.0"><channel><title>t</title></channel></rss>')

    assert news_sources._fetch_rss("https://example.com/feed.xml", "Fake A") == []


def test_fetch_rss_accepts_a_valid_feed_mislabelled_as_text_html(requests_mock):
    requests_mock.get(
        "https://example.com/feed.xml", text=RSS_RESPONSE, headers={"Content-Type": "text/html"})

    assert news_sources._fetch_rss("https://example.com/feed.xml", "Fake A")


def test_without_disabled_drops_the_listed_sources(monkeypatch):
    monkeypatch.setattr(
        news_sources, "get_settings", lambda: Settings({"news_source": {"disabled": ["b"]}}))
    registry = [("a", None, None, "rss"), ("b", None, None, "rss")]

    assert news_sources._without_disabled(registry) == [("a", None, None, "rss")]


def test_without_disabled_keeps_everything_by_default(monkeypatch):
    monkeypatch.setattr(news_sources, "get_settings", lambda: Settings({}))
    registry = [("a", None, None, "rss"), ("b", None, None, "rss")]

    assert news_sources._without_disabled(registry) == registry


def test_without_disabled_warns_about_an_unknown_key_without_raising(monkeypatch, capsys):
    monkeypatch.setattr(
        news_sources, "get_settings", lambda: Settings({"news_source": {"disabled": ["typo"]}}))
    registry = [("a", None, None, "rss")]

    assert news_sources._without_disabled(registry) == registry
    assert "typo" in capsys.readouterr().out

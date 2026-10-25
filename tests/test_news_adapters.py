import pytest
from datetime import datetime, timezone

from news_adapters._util import FeedBlockedError, _parse_iso_published
from news_adapters.arxiv import ArxivAdapter
from news_adapters.gnews import GNewsAdapter
from news_adapters.hackernews import HackerNewsAdapter
from news_adapters.newsapi import NewsApiAdapter
from news_adapters.perigon import PerigonAdapter
from tests.fixtures import (
    ARXIV_RESPONSE,
    GNEWS_RESPONSE,
    HACKERNEWS_RESPONSE,
    NEWSAPI_RESPONSE,
    PERIGON_RESPONSE,
)


# --- HackerNewsAdapter -----------------------------------------------------

def test_hackernews_pull(requests_mock):
    requests_mock.get("https://hn.algolia.com/api/v1/search_by_date", json=HACKERNEWS_RESPONSE)
    adapter = HackerNewsAdapter()
    adapter.initialize({})

    articles = adapter.pull("AI", 5)

    assert len(articles) == 2
    assert articles[0]["title"] == "Show HN: A tool for X"
    assert articles[0]["link"] == "https://example.com/show-hn-x"
    assert articles[0]["source"] == "Hacker News"
    # second hit has url=None in the fixture -> falls back to the HN item link
    assert articles[1]["link"] == "https://news.ycombinator.com/item?id=12346"
    assert articles[0]["published_dt"] == datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc)


def test_hackernews_omits_numeric_filter_when_since_not_given(requests_mock):
    requests_mock.get("https://hn.algolia.com/api/v1/search_by_date", json=HACKERNEWS_RESPONSE)
    adapter = HackerNewsAdapter()
    adapter.initialize({})

    adapter.pull("AI", 5)

    assert "numericFilters" not in requests_mock.last_request.qs


def test_hackernews_adds_numeric_filter_when_since_given(requests_mock):
    requests_mock.get("https://hn.algolia.com/api/v1/search_by_date", json=HACKERNEWS_RESPONSE)
    since = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    adapter = HackerNewsAdapter()
    adapter.initialize({})

    adapter.pull("AI", 5, since=since)

    assert requests_mock.last_request.qs["numericfilters"] == [f"created_at_i>{int(since.timestamp())}"]


def test_hackernews_section_uses_the_ranking_endpoint(requests_mock):
    """front_page is a RANKING. search_by_date would re-sort it into
    chronological order and throw away the only thing it was for."""
    m = requests_mock.get("https://hn.algolia.com/api/v1/search", json=HACKERNEWS_RESPONSE)
    adapter = HackerNewsAdapter()
    adapter.initialize({})

    adapter.pull("IGNORED", 5, section="front_page")

    assert m.last_request.qs["tags"] == ["front_page"]
    assert "query" not in m.last_request.qs, "the query is not sent in section mode"


def test_hackernews_section_still_honours_since(requests_mock):
    m = requests_mock.get("https://hn.algolia.com/api/v1/search", json=HACKERNEWS_RESPONSE)
    since = datetime(2026, 8, 20, tzinfo=timezone.utc)
    adapter = HackerNewsAdapter()
    adapter.initialize({})

    adapter.pull("IGNORED", 5, since=since, section="front_page")

    assert m.last_request.qs["numericfilters"] == [f"created_at_i>{int(since.timestamp())}"]


def test_hackernews_query_mode_is_unchanged(requests_mock):
    """agent.search_news must keep hitting the chronological search with a
    real query -- its failure mode is silent, since it would still return
    articles, just the wrong ones."""
    m = requests_mock.get("https://hn.algolia.com/api/v1/search_by_date", json=HACKERNEWS_RESPONSE)
    adapter = HackerNewsAdapter()
    adapter.initialize({})

    adapter.pull("nvidia earnings", 5)

    assert m.last_request.qs["query"] == ["nvidia earnings"]
    assert m.last_request.qs["tags"] == ["story"]


# --- ArxivAdapter -----------------------------------------------------------

def test_arxiv_pull(requests_mock):
    requests_mock.get("http://export.arxiv.org/api/query", text=ARXIV_RESPONSE)
    adapter = ArxivAdapter()
    adapter.initialize({})

    articles = adapter.pull("cat:cs.AI", 5)

    assert len(articles) == 1
    assert articles[0]["title"] == "A Study of Fake Papers for Testing"
    assert articles[0]["link"] == "https://arxiv.org/abs/2608.00001v1"
    assert articles[0]["source"] == "arXiv"
    assert "fake abstract" in articles[0]["summary"]
    assert "\n" not in articles[0]["title"]
    assert articles[0]["published_dt"] == datetime(2026, 8, 4, 0, 0, 0, tzinfo=timezone.utc)


def test_arxiv_raises_when_a_200_response_is_an_html_challenge_page(requests_mock):
    requests_mock.get("http://export.arxiv.org/api/query", text="<html><body>Just a moment</body></html>")
    adapter = ArxivAdapter()
    adapter.initialize({})

    with pytest.raises(FeedBlockedError):
        adapter.pull("cat:cs.AI", 5)


def test_arxiv_returns_empty_for_a_valid_feed_with_no_entries(requests_mock):
    requests_mock.get(
        "http://export.arxiv.org/api/query",
        text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>q</title></feed>')
    adapter = ArxivAdapter()
    adapter.initialize({})

    assert adapter.pull("cat:cs.AI", 5) == []


def test_arxiv_omits_date_range_when_since_not_given(requests_mock):
    requests_mock.get("http://export.arxiv.org/api/query", text=ARXIV_RESPONSE)
    adapter = ArxivAdapter()
    adapter.initialize({})

    adapter.pull("cat:cs.AI", 5)

    assert requests_mock.last_request.qs["search_query"] == ["cat:cs.ai"]


def test_arxiv_adds_date_range_when_since_given(requests_mock):
    requests_mock.get("http://export.arxiv.org/api/query", text=ARXIV_RESPONSE)
    since = datetime(2026, 8, 15, 12, 30, 0, tzinfo=timezone.utc)
    adapter = ArxivAdapter()
    adapter.initialize({})

    adapter.pull("cat:cs.AI", 5, since=since)

    assert requests_mock.last_request.qs["search_query"] == ["cat:cs.ai and submitteddate:[202608151230 to 99991231235959]"]


def test_arxiv_section_becomes_a_subject_class(requests_mock):
    m = requests_mock.get("http://export.arxiv.org/api/query", text=ARXIV_RESPONSE)
    adapter = ArxivAdapter()
    adapter.initialize({})

    adapter.pull("IGNORED", 5, section="quant-ph")

    assert m.last_request.qs["search_query"] == ["cat:quant-ph"]


def test_arxiv_section_and_since_compose(requests_mock):
    """arxiv is uncapped AND server-side-since, so every production pull
    takes both."""
    m = requests_mock.get("http://export.arxiv.org/api/query", text=ARXIV_RESPONSE)
    since = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
    adapter = ArxivAdapter()
    adapter.initialize({})

    adapter.pull("IGNORED", 5, since=since, section="physics.optics")

    q = m.last_request.qs["search_query"][0]
    assert q.startswith("cat:physics.optics"), "the section, not the ignored query"
    assert "submitteddate:[202608200900" in q


def test_arxiv_query_mode_is_unchanged(requests_mock):
    m = requests_mock.get("http://export.arxiv.org/api/query", text=ARXIV_RESPONSE)
    adapter = ArxivAdapter()
    adapter.initialize({})

    adapter.pull("photonic computing", 5)

    assert m.last_request.qs["search_query"] == ["photonic computing"]


# --- NewsApiAdapter ----------------------------------------------------------

def test_newsapi_pull(requests_mock):
    requests_mock.get("https://newsapi.org/v2/everything", json=NEWSAPI_RESPONSE)
    adapter = NewsApiAdapter()
    adapter.initialize({"api-key": "fake-key"})

    articles = adapter.pull("AI", 5)

    assert len(articles) == 1
    assert articles[0]["title"] == "Fake NewsAPI Article"
    assert articles[0]["source"] == "Fake News Outlet"


def test_newsapi_section_uses_top_headlines(requests_mock):
    m = requests_mock.get("https://newsapi.org/v2/top-headlines", json=NEWSAPI_RESPONSE)
    adapter = NewsApiAdapter()
    adapter.initialize({"api-key": "k"})

    adapter.pull("IGNORED", 5, section="science")

    assert m.last_request.qs["category"] == ["science"]
    assert m.last_request.qs["language"] == ["en"]
    assert "q" not in m.last_request.qs


def test_newsapi_query_mode_keeps_the_language_pin(requests_mock):
    """The pin applies to search too -- an unconstrained query here returned
    65 of 65 Chinese articles."""
    m = requests_mock.get("https://newsapi.org/v2/everything", json=NEWSAPI_RESPONSE)
    adapter = NewsApiAdapter()
    adapter.initialize({"api-key": "k"})

    adapter.pull("AOI", 5)

    assert m.last_request.qs["q"] == ["aoi"]
    assert m.last_request.qs["language"] == ["en"]


def test_newsapi_ignores_since_even_though_the_protocol_accepts_it(requests_mock):
    """Deliberate -- see NewsApiAdapter.pull's own comment for why
    (NewsAPI's free-tier delay makes a server-side date filter
    counterproductive; news_ingest.py relies on client-side filtering for
    this source instead). The NewsSourceAdapter Protocol requires every
    adapter's pull() to ACCEPT `since` uniformly -- that doesn't mean
    every adapter has to DO anything with it."""
    m = requests_mock.get("https://newsapi.org/v2/everything", json=NEWSAPI_RESPONSE)
    adapter = NewsApiAdapter()
    adapter.initialize({"api-key": "k"})

    adapter.pull("AI", 5, since=datetime(2026, 8, 15, tzinfo=timezone.utc))

    assert "from" not in m.last_request.qs


# --- GNewsAdapter ------------------------------------------------------------

def test_gnews_pull(requests_mock):
    requests_mock.get("https://gnews.io/api/v4/search", json=GNEWS_RESPONSE)
    adapter = GNewsAdapter()
    adapter.initialize({"api-key": "fake-key"})

    articles = adapter.pull("AI", 5)

    assert len(articles) == 1
    assert articles[0]["title"] == "Fake GNews Article"
    assert articles[0]["source"] == "Fake GNews Outlet"


def test_gnews_omits_from_when_since_not_given(requests_mock):
    requests_mock.get("https://gnews.io/api/v4/search", json=GNEWS_RESPONSE)
    adapter = GNewsAdapter()
    adapter.initialize({"api-key": "fake-key"})

    adapter.pull("AI", 5)

    assert "from" not in requests_mock.last_request.qs


def test_gnews_adds_from_when_since_given(requests_mock):
    requests_mock.get("https://gnews.io/api/v4/search", json=GNEWS_RESPONSE)
    since = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    adapter = GNewsAdapter()
    adapter.initialize({"api-key": "fake-key"})

    adapter.pull("AI", 5, since=since)

    assert requests_mock.last_request.qs["from"] == ["2026-08-15t12:00:00z"]


def test_gnews_section_uses_top_headlines(requests_mock):
    m = requests_mock.get("https://gnews.io/api/v4/top-headlines", json=GNEWS_RESPONSE)
    adapter = GNewsAdapter()
    adapter.initialize({"api-key": "k"})

    adapter.pull("IGNORED", 5, section="technology")

    assert m.last_request.qs["topic"] == ["technology"]
    assert m.last_request.qs["lang"] == ["en"]
    assert "q" not in m.last_request.qs


def test_gnews_section_and_since_compose(requests_mock):
    m = requests_mock.get("https://gnews.io/api/v4/top-headlines", json=GNEWS_RESPONSE)
    since = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
    adapter = GNewsAdapter()
    adapter.initialize({"api-key": "k"})

    adapter.pull("IGNORED", 5, since=since, section="business")

    assert m.last_request.qs["topic"] == ["business"]
    assert m.last_request.qs["from"] == ["2026-08-20t09:00:00z"]


def test_gnews_query_mode_is_unchanged(requests_mock):
    m = requests_mock.get("https://gnews.io/api/v4/search", json=GNEWS_RESPONSE)
    adapter = GNewsAdapter()
    adapter.initialize({"api-key": "k"})

    adapter.pull("nvidia earnings", 5)

    assert m.last_request.qs["q"] == ["nvidia earnings"]


def test_gnews_raise_for_status_redacts_the_key(requests_mock):
    requests_mock.get("https://gnews.io/api/v4/search", status_code=403, json={})
    adapter = GNewsAdapter()
    adapter.initialize({"api-key": "SUPERSECRETKEY123"})

    try:
        adapter.pull("AI", 5)
        assert False, "expected the 403 to propagate"
    except Exception as exc:
        assert "SUPERSECRETKEY123" not in str(exc), f"key leaked into: {exc}"
        assert "403" in str(exc)


# --- PerigonAdapter ----------------------------------------------------------

def test_perigon_pull(requests_mock):
    # NOTE: Perigon's real response shape is unverified (no API key available
    # to test against the live service — see docs/current/ai-news-sources.md). This
    # test only locks in that PerigonAdapter.pull parses the shape it's coded
    # to expect; it is not a guarantee that shape matches Perigon's actual API.
    requests_mock.get("https://api.perigon.io/v1/all", json=PERIGON_RESPONSE)
    adapter = PerigonAdapter()
    adapter.initialize({"api-key": "fake-key"})

    articles = adapter.pull("AI", 5)

    assert len(articles) == 1
    assert articles[0]["title"] == "Fake Perigon Article"
    assert articles[0]["source"] == "example.com"


def test_perigon_ignores_since_and_section_even_though_the_protocol_accepts_them(requests_mock):
    """Deliberate -- unverified date filter, and no top-headlines
    equivalent to switch to -- see PerigonAdapter.pull's own comment."""
    m = requests_mock.get("https://api.perigon.io/v1/all", json=PERIGON_RESPONSE)
    adapter = PerigonAdapter()
    adapter.initialize({"api-key": "k"})

    adapter.pull("AI", 5, since=datetime(2026, 8, 15, tzinfo=timezone.utc), section="ignored")

    assert m.last_request.qs["q"] == ["ai"]
    assert "from" not in m.last_request.qs


# --- _util._parse_iso_published ---------------------------------------------
# Shared by HackerNewsAdapter/ArxivAdapter/NewsApiAdapter/GNewsAdapter/
# PerigonAdapter -- moved here (from news_sources.py) along with the rest of
# news_adapters/_util.py, but these two regression tests were dropped in
# that move and are restored here against their new home.

def test_parse_iso_published_handles_missing_and_malformed():
    assert _parse_iso_published(None) is None
    assert _parse_iso_published("") is None
    assert _parse_iso_published("not a date") is None


def test_parse_iso_published_assumes_utc_when_offset_is_missing():
    # Real incident, 2026-08-14: a source returning a timestamp with no
    # offset at all produced a naive datetime, which crashed every
    # news_push.py cycle when compared against an aware `since` value.
    result = _parse_iso_published("2026-08-13T22:00:00")
    assert result == datetime(2026, 8, 13, 22, 0, 0, tzinfo=timezone.utc)
    assert result.tzinfo is not None

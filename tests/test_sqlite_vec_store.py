"""
Direct tests of SqliteVecStore -- constructs real instances against a
real temp SQLite file with the real sqlite-vec extension loaded (no
fakes; confirmed working locally and on INT's real container, see this
module's own docstring). Mirrors how storage/sqlite/subscriber.py's own
implementation details get tested directly, separate from the DAL layer
(news_cache.py here) that calls through get_vector_store().
"""

from datetime import datetime, timedelta, timezone

import pytest
from trailsign import Settings

import app_settings
from vector_store.sqlite_vec_store import SqliteVecStore


@pytest.fixture
def store(tmp_path):
    app_settings.reset_settings_for_tests(Settings({
        "news_cache": {"backend": {"sqlite_vec": {"path": str(tmp_path / "vec.db")}}},
    }))
    yield SqliteVecStore()
    app_settings.reset_settings_for_tests(None)


def _article(link="https://example.com/a", title="Title", source="BBC Business",
             summary="Summary", published_dt=None):
    return {
        "title": title, "link": link, "source": source, "summary": summary,
        "published": "Thu, 13 Aug 2026 10:00:00 GMT",
        "published_dt": published_dt or datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc),
    }


def test_write_then_read_all_round_trips_the_article(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("bbc_business", _article(), ["Finance", "Stock"], now)

    articles = store.read_all()
    assert len(articles) == 1
    a = articles[0]
    assert a["title"] == "Title"
    assert a["link"] == "https://example.com/a"
    assert a["source_key"] == "bbc_business"
    assert a["categories"] == ["Finance", "Stock"]
    assert a["fetched_at"] == now
    assert a["published_dt"] == datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc)


def test_re_fetching_the_same_link_overwrites_not_duplicates(store):
    t1 = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 13, 16, 0, 0, tzinfo=timezone.utc)
    store.write_article("bbc_business", _article(), ["Finance"], t1)
    store.write_article("bbc_business", _article(), ["Finance", "Stock"], t2)

    articles = store.read_all()
    assert len(articles) == 1
    assert articles[0]["fetched_at"] == t2
    assert articles[0]["categories"] == ["Finance", "Stock"]


def test_re_fetching_without_an_embedding_clears_a_previous_one(store):
    """The newest write wins entirely, embedding included -- same
    contract as YamlFilesStore.write_article's own docstring."""
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("bbc_business", _article(), [], now, embedding=[0.1] * 256)
    store.write_article("bbc_business", _article(), [], now, embedding=None)

    assert store.read_all()[0]["embedding"] is None


def test_different_sources_same_link_are_kept_separate(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("bbc_business", _article(link="https://x.com/1"), ["Finance"], now)
    store.write_article("guardian_business", _article(link="https://x.com/1"), ["Finance"], now)

    assert len(store.read_all()) == 2


def test_read_all_on_empty_store_returns_empty_list(store):
    assert store.read_all() == []


def test_write_article_without_an_embedding_stores_none(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("hackernews", _article(), [], now)
    assert store.read_all()[0]["embedding"] is None


def test_write_article_stores_and_round_trips_an_embedding(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    vector = [round(0.001 * i, 3) for i in range(256)]
    store.write_article("hackernews", _article(), [], now, embedding=vector)
    assert store.read_all()[0]["embedding"] == vector


def test_write_article_preserves_none_categories(store):
    """None, not [], when the article was never classified -- see
    write_article's own docstring for why the two are different facts."""
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("hackernews", _article(), None, now)
    assert store.read_all()[0]["categories"] is None


# --- cleanup_expired ---------------------------------------------------


def test_cleanup_expired_archives_rows_past_ttl(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    old = now - timedelta(hours=49)
    fresh = now - timedelta(hours=1)
    store.write_article("bbc_business", _article(link="https://x.com/old"), [], old)
    store.write_article("bbc_business", _article(link="https://x.com/fresh"), [], fresh)

    archived = store.cleanup_expired(now, ttl_hours=48)

    assert archived == 1
    remaining = store.read_all()
    assert len(remaining) == 1
    assert remaining[0]["link"] == "https://x.com/fresh"


def test_cleanup_expired_keeps_everything_under_ttl(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("bbc_business", _article(), [], now - timedelta(hours=1))

    archived = store.cleanup_expired(now, ttl_hours=48)

    assert archived == 0
    assert len(store.read_all()) == 1


def test_archived_rows_keep_their_content_but_stay_out_of_read_all(store):
    """Unlike YamlFilesStore's directory-based archive (a physical move),
    an archived row stays in the same table with archived_at set --
    confirm both halves: still queryable directly, but invisible to
    read_all()."""
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    fetched = now - timedelta(hours=72)
    store.write_article("bbc_business", _article(title="Archived title"), ["Finance"], fetched,
                        embedding=[0.1] * 256)

    store.cleanup_expired(now, ttl_hours=48)

    assert store.read_all() == []
    with store._engine.begin() as conn:
        from sqlalchemy import text
        row = conn.execute(text("SELECT title, categories, archived_at FROM articles")).fetchone()
    assert row[0] == "Archived title"
    assert row[2] is not None  # archived_at is set, not deleted


def test_re_expiring_the_same_link_does_not_duplicate_the_archived_row(store):
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    fetched = now - timedelta(hours=72)
    store.write_article("bbc_business", _article(title="First"), [], fetched)
    store.cleanup_expired(now, ttl_hours=48)
    store.write_article("bbc_business", _article(title="Second"), [], fetched)
    store.cleanup_expired(now, ttl_hours=48)

    with store._engine.begin() as conn:
        from sqlalchemy import text
        rows = conn.execute(text("SELECT title FROM articles")).fetchall()
    assert len(rows) == 1, "same link must not accumulate rows"
    assert rows[0][0] == "Second"


# --- search_similar ------------------------------------------------------


def _embedding(dominant_index: int) -> list[float]:
    """A one-hot-ish vector so cosine similarity cleanly separates
    "close to index i" from everything else, without needing real
    model2vec output."""
    vector = [0.01] * 256
    vector[dominant_index] = 1.0
    return vector


def test_search_similar_ranks_by_distance(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("a", _article(link="https://x/close"), [], now, embedding=_embedding(0))
    store.write_article("a", _article(link="https://x/far"), [], now, embedding=_embedding(200))

    results = store.search_similar(_embedding(0), top_k=2)

    assert [a["link"] for a in results] == ["https://x/close", "https://x/far"]


def test_search_similar_orders_by_distance_not_insertion_order(store):
    """Regression guard for the two-query shape search_similar uses (a
    MATCH-only query for candidate rowids, then a separate plain query
    for full rows): the second query has no guaranteed row order of its
    own, so final ordering must come from iterating the first query's
    distance-ranked id list, not from whatever order SQLite happens to
    return the second query's rows in. Inserting the FAR article first
    (giving it the lower rowid -- the order a naive "just return the
    second query's rows" implementation would produce) and the CLOSE one
    second specifically defeats a regression to that naive shape."""
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("a", _article(link="https://x/far"), [], now, embedding=_embedding(200))
    store.write_article("a", _article(link="https://x/close"), [], now, embedding=_embedding(0))

    results = store.search_similar(_embedding(0), top_k=2)

    assert [a["link"] for a in results] == ["https://x/close", "https://x/far"]


def test_search_similar_excludes_given_links(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("a", _article(link="https://x/close"), [], now, embedding=_embedding(0))
    store.write_article("a", _article(link="https://x/second"), [], now, embedding=_embedding(1))

    results = store.search_similar(_embedding(0), top_k=2, exclude_links={"https://x/close"})

    assert [a["link"] for a in results] == ["https://x/second"]


def test_search_similar_excludes_archived_rows(store):
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("a", _article(link="https://x/old"), [], now - timedelta(hours=72),
                        embedding=_embedding(0))
    store.cleanup_expired(now, ttl_hours=48)

    assert store.search_similar(_embedding(0), top_k=5) == []


def test_search_similar_ignores_articles_with_no_embedding(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    store.write_article("a", _article(link="https://x/no-embedding"), [], now, embedding=None)

    assert store.search_similar(_embedding(0), top_k=5) == []


def test_search_similar_returns_at_most_top_k(store):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        store.write_article("a", _article(link=f"https://x/{i}"), [], now, embedding=_embedding(i))

    assert len(store.search_similar(_embedding(0), top_k=3)) == 3

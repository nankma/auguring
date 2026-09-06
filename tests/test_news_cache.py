from datetime import datetime, timedelta, timezone

import news_cache
import vector_store


def _article(link="https://example.com/a", title="Title", source="BBC Business", summary="Summary"):
    return {
        "title": title,
        "link": link,
        "source": source,
        "summary": summary,
        "published": "Thu, 13 Aug 2026 10:00:00 GMT",
        "published_dt": datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc),
    }


def test_write_article_creates_a_file_named_by_source_and_link_hash(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    path = news_cache.write_article("bbc_business", _article(), ["Finance"], now)
    assert path.exists()
    assert path.name.startswith("bbc_business-")
    assert path.suffix == ".yaml"


def test_write_then_read_all_round_trips_the_article(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("bbc_business", _article(), ["Finance", "Stock"], now)

    articles = news_cache.read_all()
    assert len(articles) == 1
    a = articles[0]
    assert a["title"] == "Title"
    assert a["link"] == "https://example.com/a"
    assert a["source_key"] == "bbc_business"
    assert a["categories"] == ["Finance", "Stock"]
    assert a["fetched_at"] == now
    assert a["published_dt"] == datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc)


def test_re_fetching_the_same_link_overwrites_not_duplicates(isolated_news_cache):
    t1 = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 13, 16, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("bbc_business", _article(), ["Finance"], t1)
    news_cache.write_article("bbc_business", _article(), ["Finance", "Stock"], t2)

    articles = news_cache.read_all()
    assert len(articles) == 1
    assert articles[0]["fetched_at"] == t2
    assert articles[0]["categories"] == ["Finance", "Stock"]


def test_different_sources_same_link_are_kept_separate(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("bbc_business", _article(link="https://x.com/1"), ["Finance"], now)
    news_cache.write_article("guardian_business", _article(link="https://x.com/1"), ["Finance"], now)

    assert len(news_cache.read_all()) == 2


def test_cleanup_expired_removes_files_past_ttl(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    old = now - timedelta(hours=49)
    fresh = now - timedelta(hours=1)
    news_cache.write_article("bbc_business", _article(link="https://x.com/old"), [], old)
    news_cache.write_article("bbc_business", _article(link="https://x.com/fresh"), [], fresh)

    deleted = news_cache.cleanup_expired(now, ttl_hours=48)

    assert deleted == 1
    remaining = news_cache.read_all()
    assert len(remaining) == 1
    assert remaining[0]["link"] == "https://x.com/fresh"


def test_cleanup_expired_keeps_everything_under_ttl(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("bbc_business", _article(), [], now - timedelta(hours=1))

    deleted = news_cache.cleanup_expired(now, ttl_hours=48)

    assert deleted == 0
    assert len(news_cache.read_all()) == 1


def test_read_all_on_empty_cache_returns_empty_list(isolated_news_cache):
    assert news_cache.read_all() == []


def test_write_article_preserves_none_summary(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("hackernews", _article(summary=None), [], now)
    assert news_cache.read_all()[0]["summary"] is None


def test_write_article_without_an_embedding_stores_none(isolated_news_cache):
    """The default -- an article cached before news_embed.py existed, or
    one whose embed call failed, must round-trip as None, not a missing
    key (every consumer checks `is not None`, not `"embedding" in article`)."""
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("hackernews", _article(), [], now)
    assert news_cache.read_all()[0]["embedding"] is None


def test_write_article_stores_and_round_trips_an_embedding(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    vector = [0.1, -0.2, 0.3]
    news_cache.write_article("hackernews", _article(), [], now, embedding=vector)
    assert news_cache.read_all()[0]["embedding"] == vector


def test_cleanup_archives_instead_of_deleting_when_an_archive_is_configured(
    isolated_news_cache, monkeypatch, tmp_path
):
    archive = tmp_path / "archive"
    monkeypatch.setattr(vector_store.get_vector_store(), "_archive_dir_path", str(archive))
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    fetched = now - timedelta(hours=72)
    path = news_cache.write_article("bbc_business", _article(), ["Finance"], fetched)

    assert news_cache.cleanup_expired(now, ttl_hours=48) == 1

    assert not path.exists(), "the file must leave the active cache"
    assert news_cache.read_all() == [], "and must not come back from read_all"
    # filed under the day it was fetched, not the day it expired
    archived = archive / "2026-08-12" / path.name
    assert archived.exists()


def test_archived_article_keeps_its_content(isolated_news_cache, monkeypatch, tmp_path):
    archive = tmp_path / "archive"
    monkeypatch.setattr(vector_store.get_vector_store(), "_archive_dir_path", str(archive))
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    fetched = now - timedelta(hours=72)
    news_cache.write_article("bbc_business", _article(title="Archived title"),
                             ["Finance"], fetched)
    news_cache.cleanup_expired(now, ttl_hours=48)

    # The whole point is building a corpus later, so the archived record has
    # to still parse into the same shape read_all produces.
    import yaml
    files = list((archive / "2026-08-12").glob("*.yaml"))
    assert len(files) == 1
    record = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    assert record["title"] == "Archived title"
    assert record["categories"] == ["Finance"]
    assert record["source_key"] == "bbc_business"


def test_re_expiring_the_same_link_replaces_the_archived_copy(
    isolated_news_cache, monkeypatch, tmp_path
):
    """The filename is a hash of the link, so a collision in the archive
    means the same article was fetched and expired twice. Keeping the newer
    copy dedupes the corpus by link for free."""
    archive = tmp_path / "archive"
    monkeypatch.setattr(vector_store.get_vector_store(), "_archive_dir_path", str(archive))
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    fetched = now - timedelta(hours=72)

    news_cache.write_article("bbc_business", _article(title="First"), [], fetched)
    news_cache.cleanup_expired(now, ttl_hours=48)
    news_cache.write_article("bbc_business", _article(title="Second"), [], fetched)
    news_cache.cleanup_expired(now, ttl_hours=48)

    import yaml
    files = list((archive / "2026-08-12").glob("*.yaml"))
    assert len(files) == 1, "same link must not accumulate copies"
    assert yaml.safe_load(files[0].read_text(encoding="utf-8"))["title"] == "Second"


def test_unarchivable_file_is_deleted_rather_than_left_in_the_cache(
    isolated_news_cache, monkeypatch, tmp_path
):
    """If the archive can't be written to, the file must still leave the
    active cache -- otherwise it stays an expiry candidate forever and the
    failure repeats on every single ingestion cycle."""
    monkeypatch.setattr(vector_store.get_vector_store(), "_archive_dir_path", str(tmp_path / "archive"))
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    fetched = now - timedelta(hours=72)
    path = news_cache.write_article("bbc_business", _article(), [], fetched)

    def boom(self, target):
        raise OSError("cross-device link")

    monkeypatch.setattr(vector_store.yaml_files.Path, "replace", boom)
    assert news_cache.cleanup_expired(now, ttl_hours=48) == 1
    assert not path.exists()


def test_cleanup_still_deletes_when_no_archive_is_configured(isolated_news_cache, monkeypatch):
    monkeypatch.setattr(vector_store.get_vector_store(), "_archive_dir_path", None)
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    path = news_cache.write_article("bbc_business", _article(), [], now - timedelta(hours=72))
    assert news_cache.cleanup_expired(now, ttl_hours=48) == 1
    assert not path.exists()


def test_corrupt_file_is_archived_into_the_epoch_bucket(isolated_news_cache, monkeypatch, tmp_path):
    """A file whose YAML won't parse has no usable fetched_at to file it
    under. It must still leave the active cache, and must NOT land in
    today's bucket -- whatever later reads the archive to build a corpus
    would otherwise see corrupt data dated as fresh."""
    archive = tmp_path / "archive"
    monkeypatch.setattr(vector_store.get_vector_store(), "_archive_dir_path", str(archive))
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    bad = isolated_news_cache / "bbc_business-deadbeef.yaml"
    isolated_news_cache.mkdir(parents=True, exist_ok=True)
    bad.write_text("{{{ not yaml at all", encoding="utf-8")

    assert news_cache.cleanup_expired(now, ttl_hours=48) == 1

    assert not bad.exists()
    assert (archive / "1970-01-01" / bad.name).exists()
    assert not (archive / "2026-08-15").exists()


def _one_hot(dominant_index: int, dim: int = 8) -> list[float]:
    """A simple vector so cosine similarity cleanly separates "close to
    index i" from everything else, without needing real model2vec
    output -- same convention as tests/test_sqlite_vec_store.py's own
    _embedding helper."""
    vector = [0.01] * dim
    vector[dominant_index] = 1.0
    return vector


def test_search_similar_ranks_by_distance(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("a", _article(link="https://x/close"), [], now, embedding=_one_hot(0))
    news_cache.write_article("a", _article(link="https://x/far"), [], now, embedding=_one_hot(6))

    results = news_cache.search_similar(_one_hot(0), top_k=2)

    assert [a["link"] for a in results] == ["https://x/close", "https://x/far"]


def test_search_similar_excludes_given_links(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("a", _article(link="https://x/close"), [], now, embedding=_one_hot(0))
    news_cache.write_article("a", _article(link="https://x/second"), [], now, embedding=_one_hot(1))

    results = news_cache.search_similar(_one_hot(0), top_k=2, exclude_links={"https://x/close"})

    assert [a["link"] for a in results] == ["https://x/second"]


def test_search_similar_ignores_articles_with_no_embedding(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("a", _article(link="https://x/no-embedding"), [], now, embedding=None)

    assert news_cache.search_similar(_one_hot(0), top_k=5) == []


def test_search_similar_returns_at_most_top_k(isolated_news_cache):
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        news_cache.write_article("a", _article(link=f"https://x/{i}"), [], now, embedding=_one_hot(i))

    assert len(news_cache.search_similar(_one_hot(0), top_k=3)) == 3


def test_archived_files_do_not_come_back_through_read_all(isolated_news_cache, monkeypatch, tmp_path):
    """The archive exists to accumulate a corpus, not to extend the bot's
    working set. read_all globs the cache directory only, so an archived
    article must stay invisible to ingestion and push selection -- the TTL
    still governs what counts as current news."""
    archive = tmp_path / "archive"
    monkeypatch.setattr(vector_store.get_vector_store(), "_archive_dir_path", str(archive))
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    news_cache.write_article("bbc_business", _article(link="https://example.com/old"),
                             [], now - timedelta(hours=72))
    news_cache.write_article("bbc_business", _article(link="https://example.com/fresh"),
                             [], now - timedelta(hours=1))

    news_cache.cleanup_expired(now, ttl_hours=48)

    assert [a["link"] for a in news_cache.read_all()] == ["https://example.com/fresh"]

"""
Local cache for news articles -- see docs/plans/local-news-cache-plan.md.

Thin delegator, as of 2026-09-05: the actual storage is pluggable (see
vector_store/__init__.py's get_vector_store(), selected by
news_cache.backend.type) -- YamlFilesStore (one YAML file per article,
today's default/unchanged behavior) or SqliteVecStore (a SQLite table +
the sqlite-vec extension, for fast bulk reads and indexed similarity
search at larger corpus sizes). Every function here just forwards to
whichever backend is active, so callers (news_ingest.py, news_push.py,
agent.py) never need to know which one -- same "callers never see
get_storage()" shape as subscriber_ops.py.

DEFAULT_TTL_HOURS stays here, not per-backend: retention policy (how long
an article counts as current news) is a news_cache-level decision that
applies regardless of which backend stores the data, unlike CACHE_DIR/
ARCHIVE_DIR (backend-specific paths, now resolved inside each backend's
own __init__ -- see vector_store/yaml_files.py).

Retention is judged by `fetched_at` (when THIS system pulled the article),
not `published_dt` (the source's own claimed publish time) -- some
sources' publish dates don't parse at all (see news_sources.py), but
fetched_at is always known and controlled by our own code, so cleanup can
always trust it.
"""

from datetime import datetime
from pathlib import Path

from app_settings import get_settings
from vector_store import get_vector_store

DEFAULT_TTL_HOURS = get_settings().resolved("storage.news_cache_dir.ttl_hours", default=48)


def write_article(source_key: str, article: dict, categories: list[str] | None,
                  fetched_at: datetime, embedding: list[float] | None = None) -> Path | None:
    """See vector_store.VectorStore.write_article -- every backend
    implements the same contract described there. Only YamlFilesStore
    returns a real Path (the file it wrote); other backends return None."""
    return get_vector_store().write_article(source_key, article, categories, fetched_at, embedding)


def read_all() -> list[dict]:
    """See vector_store.VectorStore.read_all."""
    return get_vector_store().read_all()


def cleanup_expired(now: datetime, ttl_hours: int = DEFAULT_TTL_HOURS) -> int:
    """See vector_store.VectorStore.cleanup_expired."""
    return get_vector_store().cleanup_expired(now, ttl_hours)


def search_similar(query_vector: list[float], top_k: int,
                   exclude_links: set[str] | None = None) -> list[dict]:
    """See vector_store.VectorStore.search_similar. Not called by any
    production code yet (agent.py's search_news still does its own
    read_all() + news_embed.filter_by_relevance pass) -- exposed here so
    a future caller doesn't need to know about vector_store directly,
    consistent with everything else in this module."""
    return get_vector_store().search_similar(query_vector, top_k, exclude_links)

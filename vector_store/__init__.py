"""
News-cache backend selection. get_vector_store() returns a process-wide
singleton -- YamlFilesStore or SqliteVecStore, picked by
news_cache.backend.type -- that news_cache.py (a thin delegator, see its
own module docstring) calls into. Same "one settings-selected
implementation, built once" shape as storage/__init__.py's get_storage(),
just for the news article cache instead of the subscribers database.

Unlike storage/ (SqliteStorage/PostgresStorage share code via mixin
inheritance -- PostgresStorage IS-A SqliteStorage with a few overrides),
the backends here share no implementation at all: a flat YAML-file-per-
article cache and a SQLite+sqlite-vec store have nothing in common to
inherit. VectorStore below is a typing.Protocol (structural typing) for
exactly this reason -- same pattern this codebase already uses for
telemetry.EventLogger/Logger, another case of genuinely-different
concrete implementations behind one interface.
"""

from datetime import datetime
from pathlib import Path
from typing import Protocol

from app_settings import get_settings

_vector_store = None


class VectorStore(Protocol):
    """Every backend implements this. `write_article`/`read_all`/
    `cleanup_expired` are today's exact news_cache.py public API,
    unchanged in signature -- news_cache.py's own functions just forward
    to whichever backend get_vector_store() returns. `search_similar` is
    new (2026-09-05): a top-K similarity lookup a backend can answer
    without the caller loading the whole corpus into Python first. No
    caller uses it yet -- YamlFilesStore implements it by falling back to
    read_all() + the same Python-side cosine-similarity loop
    news_embed.filter_by_relevance already does, so the interface is
    uniform across backends even though only SqliteVecStore benefits from
    real indexing so far.

    write_article's return type is `Path | None`, not just `Path` --
    YamlFilesStore returns the file it wrote (a pre-existing behavior,
    still relied on by its own tests), but a database-backed store has no
    equivalent single-file path to hand back, so SqliteVecStore returns
    None. No production caller reads this return value either way."""

    def write_article(self, source_key: str, article: dict, categories: list[str] | None,
                       fetched_at: datetime, embedding: list[float] | None = None) -> Path | None:
        ...

    def read_all(self) -> list[dict]:
        ...

    def cleanup_expired(self, now: datetime, ttl_hours: int) -> int:
        ...

    def search_similar(self, query_vector: list[float], top_k: int,
                        exclude_links: set[str] | None = None) -> list[dict]:
        ...


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = _build_vector_store()
    return _vector_store


def _build_vector_store() -> VectorStore:
    backend = get_settings().resolved("news_cache.backend.type", default="yaml_files")
    if backend == "yaml_files":
        from vector_store.yaml_files import YamlFilesStore
        return YamlFilesStore()
    if backend == "sqlite_vec":
        from vector_store.sqlite_vec_store import SqliteVecStore
        return SqliteVecStore()
    raise ValueError(f"news_cache.backend.type={backend!r} is not a recognized backend")


def reset_vector_store_for_tests(store: VectorStore | None = None) -> None:
    """Test-only. No-arg call forces the next get_vector_store() to
    rebuild from settings/env; pass a store instance to inject a fake for
    the duration of a test. Mirrors storage.reset_storage_for_tests."""
    global _vector_store
    _vector_store = store

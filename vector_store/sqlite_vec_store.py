"""
SQLite + sqlite-vec backend for the news article cache -- see
vector_store/__init__.py for how this gets selected. A single SQLite
file with two tables: `articles` (everything, including the embedding
stored redundantly as plain JSON -- see below) and `articles_vec` (a
vec0 virtual table, sqlite-vec's indexed vector column, sharing rowids
with `articles` for whichever rows have an embedding).

Chosen over the default YamlFilesStore backend specifically to fix a
measured bottleneck: reading ~1600-2000 individual YAML files took ~12s
(per-file open+parse overhead, not data volume -- the whole cache is only
~6.5MB), and that cost is paid by both news_push.py's every scheduled
tick and agent.py's every on-demand search. One SQLite query reads the
same data in a single pass instead of ~2000 separate filesystem
operations.

sqlite-vec (https://github.com/asg017/sqlite-vec) measured 2026-09-05 on
INT's real container: ~176KB installed, zero transitive dependencies,
~180KB real incremental memory once numpy is already loaded (which it
already is via model2vec/news_embed.py) -- negligible against this
project's memory-constrained deploy target. See TODO.md for the still-open
questions this doesn't answer (indexed-search behavior at ~30k real
vectors, a retention TTL for archived rows).

Why the embedding is ALSO stored as plain JSON on `articles`, not only in
`articles_vec`: reading a stored vec0 embedding back out through a plain
SELECT (as opposed to feeding a query vector INTO a MATCH clause, which
is the one path actually verified live on INT) depends on sqlite-vec's
internal binary packing, which this project hasn't independently verified
byte-for-byte. Rather than ship an unverified deserializer for read_all()
(load-bearing for every push cycle and every search), the plain JSON
column is the trusted source of truth read_all() actually uses -- exactly
the same format YamlFilesStore already round-trips today, so its
correctness is not a new risk. `articles_vec` exists solely to serve
search_similar()'s indexed top-K lookup, where only the MATCH-query path
(the one path verified live) is ever used.

Retirement here is "mark archived_at", not "move to another location" --
unlike YamlFilesStore's directory-based archive (which exists specifically
because one flat directory of tens of thousands of files is slow to
list), a database table has no such cost, so archived rows just stay in
the same table with archived_at set, excluded from read_all()/
search_similar() by a WHERE clause. See TODO.md for the still-open "does
archived data need its own TTL" question -- deliberately not decided
here, mirroring YamlFilesStore's own archive (also never expired).
"""

import json
from datetime import datetime, timedelta

import sqlite_vec
from sqlalchemy import bindparam, create_engine, event, text

from app_settings import get_settings

# model2vec's minishlab/potion-base-8M -- see news_embed.py's own module
# docstring for why this model/dimension was chosen. vec0 requires a
# fixed dimension declared upfront; every embedding this project produces
# already shares this one model, so there's no per-article variation to
# accommodate.
EMBEDDING_DIM = 256


class SqliteVecStore:
    def __init__(self):
        path = get_settings().resolved("news_cache.backend.sqlite_vec.path", required=True)
        self._engine = create_engine(f"sqlite:///{path}")

        # sqlite-vec is a loadable extension, not a Python-side wrapper
        # around sqlite3 -- it must be loaded into EVERY new DBAPI
        # connection before any vec0 table can be created or queried on
        # it. SQLAlchemy's own connection pool can open more than one
        # over the engine's lifetime, so this hooks the pool's "connect"
        # event rather than loading it once at __init__ time.
        @event.listens_for(self._engine, "connect")
        def _load_sqlite_vec(dbapi_conn, connection_record):
            dbapi_conn.enable_load_extension(True)
            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)

        self._create_schema()

    def _create_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY,
                    source TEXT,
                    source_key TEXT,
                    title TEXT,
                    link TEXT NOT NULL,
                    summary TEXT,
                    published TEXT,
                    published_dt TEXT,
                    fetched_at TEXT,
                    categories TEXT,
                    embedding TEXT,
                    archived_at TEXT,
                    UNIQUE(source_key, link)
                )
                """
            ))
            conn.execute(text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS articles_vec USING vec0(embedding float[{EMBEDDING_DIM}])"
            ))

    @staticmethod
    def _iso(dt: datetime | None) -> str | None:
        return dt.isoformat() if dt else None

    @staticmethod
    def _parse_iso(raw: str | None) -> datetime | None:
        return datetime.fromisoformat(raw) if raw else None

    def _row_to_dict(self, row) -> dict:
        (_id, source, source_key, title, link, summary, published,
         published_dt, fetched_at, categories, embedding) = row
        return {
            "source": source, "source_key": source_key, "title": title, "link": link,
            "summary": summary, "published": published,
            "published_dt": self._parse_iso(published_dt),
            "fetched_at": self._parse_iso(fetched_at),
            "categories": json.loads(categories) if categories is not None else None,
            "embedding": json.loads(embedding) if embedding is not None else None,
        }

    # Return type is `None` specifically, not the Protocol's general
    # `Path | None` -- this backend never returns a Path.
    def write_article(self, source_key: str, article: dict, categories: list[str] | None,
                       fetched_at: datetime, embedding: list[float] | None = None) -> None:
        """Same overwrite-by-link contract as YamlFilesStore.write_article
        -- see that docstring. Re-writing an existing link replaces its
        articles_vec row entirely (deleted, then re-inserted only if this
        write has an embedding), matching "the newest write wins,
        embedding included" -- a re-fetch with embedding=None genuinely
        clears a previously-stored vector, same as the YAML backend.

        Returns None, not a Path -- there's no single-file equivalent for
        a database-backed store to hand back. See VectorStore.write_article's
        own docstring; no production caller reads this return value."""
        link = article["link"]
        embedding_json = json.dumps(list(embedding)) if embedding is not None else None
        categories_json = json.dumps(categories) if categories is not None else None
        with self._engine.begin() as conn:
            # (source_key, link), not link alone -- matches
            # YamlFilesStore's own identity key (its filename is
            # "{source_key}-{hash(link)}.yaml"), so the same link
            # reported by two different sources stays two articles here
            # too, not a collision. One upsert statement (matching the
            # table's own UNIQUE(source_key, link) constraint) rather
            # than a manual "SELECT, then branch into UPDATE or INSERT"
            # -- the constraint already makes the branch unnecessary.
            article_id = conn.execute(text(
                """
                INSERT INTO articles (source, source_key, title, link, summary, published,
                                      published_dt, fetched_at, categories, embedding)
                VALUES (:source, :source_key, :title, :link, :summary, :published,
                        :published_dt, :fetched_at, :categories, :embedding)
                ON CONFLICT(source_key, link) DO UPDATE SET
                    source=excluded.source, title=excluded.title, summary=excluded.summary,
                    published=excluded.published, published_dt=excluded.published_dt,
                    fetched_at=excluded.fetched_at, categories=excluded.categories,
                    embedding=excluded.embedding, archived_at=NULL
                RETURNING id
                """
            ), {
                "source": article.get("source"), "source_key": source_key,
                "title": article.get("title"), "link": link,
                "summary": article.get("summary"), "published": article.get("published"),
                "published_dt": self._iso(article.get("published_dt")),
                "fetched_at": self._iso(fetched_at),
                "categories": categories_json,
                "embedding": embedding_json,
            }).scalar_one()
            # Dropped and re-inserted unconditionally, not just on
            # update -- simpler than tracking "was this an insert or an
            # update" separately, and correct either way: a fresh insert
            # has nothing to drop, a re-fetch with embedding=None must
            # clear any previously-stored vector (see this method's own
            # docstring), and a re-fetch with a real embedding needs the
            # old vec0 row gone before the new one goes in regardless.
            conn.execute(text("DELETE FROM articles_vec WHERE rowid = :id"), {"id": article_id})
            if embedding is not None:
                conn.execute(
                    text("INSERT INTO articles_vec (rowid, embedding) VALUES (:id, :embedding)"),
                    {"id": article_id, "embedding": sqlite_vec.serialize_float32([float(x) for x in embedding])},
                )

    def read_all(self) -> list[dict]:
        """Reads the plain `embedding` JSON column, not articles_vec --
        see this module's own docstring for why. One query, not one
        open+parse per article."""
        with self._engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT id, source, source_key, title, link, summary, published,
                       published_dt, fetched_at, categories, embedding
                FROM articles WHERE archived_at IS NULL
                """
            )).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def cleanup_expired(self, now: datetime, ttl_hours: int) -> int:
        """Marks archived_at rather than deleting or moving -- see this
        module's own docstring for why a database table doesn't need
        YamlFilesStore's directory-based archive. A row with no
        parseable fetched_at is treated as expired, same as
        YamlFilesStore -- shouldn't happen (write_article always sets
        it), but cheap to guard."""
        cutoff = (now - timedelta(hours=ttl_hours)).isoformat()
        with self._engine.begin() as conn:
            result = conn.execute(text(
                """
                UPDATE articles SET archived_at = :now
                WHERE archived_at IS NULL AND (fetched_at IS NULL OR fetched_at < :cutoff)
                """
            ), {"now": now.isoformat(), "cutoff": cutoff})
            return result.rowcount

    def search_similar(self, query_vector: list[float], top_k: int,
                        exclude_links: set[str] | None = None) -> list[dict]:
        """The one real payoff of this backend over YamlFilesStore's
        naive fallback: articles_vec's index finds the top-K nearest
        neighbors without loading every embedding into Python first.

        Over-fetches a fixed buffer beyond top_k (not scaled by
        len(exclude_links) -- a first-cut heuristic, not a measured one,
        since most excluded links won't also be top matches in practice)
        so that filtering out already-archived/excluded results afterward
        still usually leaves top_k -- fewer than that is possible but
        acceptable (same "return what's genuinely available" shape as
        every other fail-open path in this codebase), not treated as an
        error."""
        exclude_links = exclude_links or set()
        overfetch = top_k + 50
        serialized = sqlite_vec.serialize_float32([float(x) for x in query_vector])
        with self._engine.begin() as conn:
            neighbor_rows = conn.execute(text(
                "SELECT rowid FROM articles_vec WHERE embedding MATCH :qv AND k = :k ORDER BY distance"
            ), {"qv": serialized, "k": overfetch}).fetchall()
            ordered_ids = [row[0] for row in neighbor_rows]
            if not ordered_ids:
                return []
            rows = conn.execute(
                text(
                    """
                    SELECT id, source, source_key, title, link, summary, published,
                           published_dt, fetched_at, categories, embedding
                    FROM articles WHERE id IN :ids AND archived_at IS NULL
                    """
                ).bindparams(bindparam("ids", expanding=True)),
                {"ids": ordered_ids},
            ).fetchall()
        by_id = {row[0]: row for row in rows}
        results = []
        for article_id in ordered_ids:
            row = by_id.get(article_id)
            if row is None:  # archived, or deleted since the MATCH query ran
                continue
            article = self._row_to_dict(row)
            if article["link"] in exclude_links:
                continue
            results.append(article)
            if len(results) >= top_k:
                break
        return results

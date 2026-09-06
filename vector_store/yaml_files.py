"""
Default news_cache.py backend: one YAML file per article, named
`{source}-{id}.yaml` where `id` is a short hash of the article's link.

This is the ORIGINAL news_cache.py implementation, extracted verbatim
into this class 2026-09-05 when the backend became pluggable (see
vector_store/__init__.py) -- behavior, docstrings, and reasoning are
unchanged from before the split, only the module/class boundary moved.
Kept as the settings default (news_cache.backend.type unset or
"yaml_files") so nothing changes for a deployment that doesn't opt into
news_cache.backend.type: sqlite_vec.

Hashing the link (rather than an incrementing counter, or a per-source ID
field not every source provides) means re-fetching the same article in a
later ingestion cycle produces the same filename and just overwrites
harmlessly -- free deduplication, no separate tracking state needed.

CACHE_DIR is configurable via storage.news_cache_dir (settings.yml, see
app_settings.py), same reasoning as storage.database.sqlite.path
-- local dev and the deployed container need different paths, and a
container restart shouldn't lose the cache if news_cache_dir points at
the same mounted volume as subscribers.db.

Retention is judged by `fetched_at` (when THIS system pulled the article),
not `published_dt` (the source's own claimed publish time) -- some
sources' publish dates don't parse at all (see news_sources.py), but
fetched_at is always known and controlled by our own code, so cleanup can
always trust it.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import yaml

from app_settings import get_settings
import news_embed


class YamlFilesStore:
    def __init__(self):
        # required=True, no default -- this is a value the service always
        # needs a real one for; settings.yml not having it is a deployment
        # mistake and should fail loudly at startup, not silently fall
        # back to anything (including the old NEWS_CACHE_DIR env var). See
        # docs/standaloneplan/01-settings-migration.md's "Migration methodology".
        self._cache_dir_path = get_settings().resolved("storage.news_cache_dir.path", required=True)

        # Where expired articles go instead of being deleted. Unset (the
        # default) keeps the original behaviour -- cleanup_expired unlinks
        # them.
        #
        # The point of archiving is corpus size. Every clustering and
        # retrieval measurement in docs/analysis/cluster-measurements.md
        # ran against ~2,262 articles, which is a 48-hour window, and
        # several conclusions were limited by it: HDBSCAN needs 8 articles
        # to form a cluster at all, so topics real subscribers follow
        # (semiconductors: 13 articles, optical: 6, AAOI: 0) could never
        # produce one. A month of accumulation is roughly 33,000 articles
        # at the current rate -- about 130 MB, against 33 GB free on the
        # VM, so disk is not a consideration.
        #
        # IMPORTANT: both this and news_cache_dir must point INSIDE the
        # mounted volume (/data on the deployed container). They don't by
        # default, and that is not a theoretical risk: as of 2026-08-19
        # the live cache sat at /app/news_cache, on the container
        # filesystem, so every redeploy silently destroyed it. 2,202
        # articles existed only because the container happened not to
        # have restarted in three days.
        # default=None here is a real, intentional value (archiving off),
        # not a stand-in for "figure this out" -- unlike CACHE_DIR above,
        # an absent key here is a legitimate, expected state, so this one
        # stays optional.
        self._archive_dir_path = get_settings().resolved("storage.news_archive_dir", default=None)

    def _cache_dir(self) -> Path:
        path = Path(self._cache_dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _archive_dir(self, fetched_at: datetime | None) -> Path:
        """Archived articles land in a per-day subdirectory. A month at
        the current ingestion rate is ~33k files; one flat directory of
        those is slow to list and awkward to copy off the VM in pieces,
        and the day boundary is the natural unit for both."""
        day = (fetched_at or datetime(1970, 1, 1)).date().isoformat()
        path = Path(self._archive_dir_path) / day
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _article_id(link: str) -> str:
        return hashlib.sha256(link.encode("utf-8")).hexdigest()[:12]

    def _file_path(self, source_key: str, link: str) -> Path:
        return self._cache_dir() / f"{source_key}-{self._article_id(link)}.yaml"

    @staticmethod
    def _iso(dt: datetime | None) -> str | None:
        return dt.isoformat() if dt else None

    @staticmethod
    def _parse_iso(raw: str | None) -> datetime | None:
        return datetime.fromisoformat(raw) if raw else None

    def write_article(self, source_key: str, article: dict, categories: list[str] | None,
                       fetched_at: datetime, embedding: list[float] | None = None) -> Path:
        """`source_key` is the registry key (e.g. "bbc_business", matching
        SOURCE_REGISTRY in news_sources.py), not the article's own "source"
        field (a display name like "BBC Business") -- the filename needs the
        short, filesystem-clean key, not the human-readable name. `article` is
        the normalized shape news_sources.py's fetch_* functions return
        (title/link/source/summary/published/published_dt). Overwrites
        silently if this exact link was already cached (from an earlier cycle,
        or a different source pulling the same story) -- the newer fetch and
        classification simply replace the older one.

        `embedding` is optional and None by default -- an article cached
        before news_embed.py existed, or one whose embedding call failed
        (see news_embed's module docstring on why that must never block
        caching), simply has none. Every consumer (near-duplicate collapse,
        offbeat selection in news_push.select_candidate_articles) already
        treats a missing embedding as "skip this feature for this article",
        the same fail-open shape as a missing `categories`."""
        link = article["link"]
        record = {
            "source": article.get("source"),
            "source_key": source_key,
            "title": article.get("title"),
            "link": link,
            "summary": article.get("summary"),
            "published": article.get("published"),
            "published_dt": self._iso(article.get("published_dt")),
            "fetched_at": self._iso(fetched_at),
            # None, not [], when the article was never classified. The two are
            # different facts -- "nothing applied" versus "we don't know" -- and
            # writing [] for both is what made a three-day classification outage
            # indistinguishable from normal operation. news_ingest writes
            # category_ops.UNCLASSIFIABLE for the first case.
            "categories": list(categories) if categories is not None else None,
            "embedding": list(embedding) if embedding is not None else None,
        }
        path = self._file_path(source_key, link)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False)
        return path

    def read_all(self) -> list[dict]:
        """Every currently-cached article, parsed back with published_dt and
        fetched_at as real datetimes (not the raw ISO strings written to disk)
        -- mirrors news_sources.py's own published/published_dt shape so
        downstream filtering code doesn't need to know the difference between
        a freshly-fetched article and one read back from the cache."""
        articles = []
        for path in self._cache_dir().glob("*.yaml"):
            try:
                with open(path, encoding="utf-8") as f:
                    record = yaml.safe_load(f)
            except (OSError, yaml.YAMLError):
                continue
            if not record:
                continue
            record["published_dt"] = self._parse_iso(record.get("published_dt"))
            record["fetched_at"] = self._parse_iso(record.get("fetched_at"))
            articles.append(record)
        return articles

    def _retire(self, path: Path, fetched_at: datetime | None) -> None:
        """Removes `path` from the active cache -- by moving it to the archive
        if one is configured, otherwise by deleting it.

        A same-named file already in the archive is replaced. The filename is a
        hash of the article's link, so a collision means the same article was
        re-fetched and re-expired; keeping the newer record is right, and it
        also means the archive is deduplicated by link for free, the same
        property write_article relies on."""
        if self._archive_dir_path:
            target = self._archive_dir(fetched_at) / path.name
            try:
                path.replace(target)
                return
            except OSError:
                # Cross-device rename, a read-only archive path, a full disk.
                # Fall through to deleting rather than leaving the file in the
                # active cache forever, which would make it a permanent
                # candidate and re-attempt this on every single cycle.
                print(f"[news_cache] could not archive {path.name}, deleting instead")
        path.unlink(missing_ok=True)

    def cleanup_expired(self, now: datetime, ttl_hours: int) -> int:
        """Retires every cached file whose fetched_at is older than ttl_hours,
        and returns how many. A file with no parseable fetched_at (shouldn't
        happen -- we always write it -- but cheap to guard) is treated as
        expired rather than kept forever by accident.

        "Retires" rather than "deletes" because NEWS_ARCHIVE_DIR turns this
        into a move -- see the archive_dir_path reasoning in __init__. Either
        way the file leaves the active cache, so read_all() and everything
        downstream see no difference; the TTL still governs what the bot
        considers current news."""
        retired = 0
        for path in self._cache_dir().glob("*.yaml"):
            try:
                with open(path, encoding="utf-8") as f:
                    record = yaml.safe_load(f)
            except (OSError, yaml.YAMLError):
                # Unparseable: retire it, but with no usable fetched_at to file
                # it under. It lands in the archive's epoch bucket rather than
                # today's, so a corrupt file can't masquerade as fresh data in
                # whatever later reads the archive.
                self._retire(path, None)
                retired += 1
                continue
            fetched_at = self._parse_iso((record or {}).get("fetched_at"))
            if fetched_at is None or (now - fetched_at).total_seconds() > ttl_hours * 3600:
                self._retire(path, fetched_at)
                retired += 1
        return retired

    def search_similar(self, query_vector: list[float], top_k: int,
                        exclude_links: set[str] | None = None) -> list[dict]:
        """Naive fallback: read_all() then rank in Python -- the same
        cosine-similarity approach news_embed.filter_by_relevance already
        uses, just collapsed to a plain top-K instead of that function's
        fraction/min/max clamp. Exists so the VectorStore interface is
        uniform across backends; this backend has no index to make it
        fast, only SqliteVecStore does."""
        exclude_links = exclude_links or set()
        scored = []
        for article in self.read_all():
            link = article.get("link")
            embedding = article.get("embedding")
            if not link or link in exclude_links or embedding is None:
                continue
            similarity = news_embed.cosine_similarity(embedding, query_vector)
            scored.append((similarity, article))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [article for _similarity, article in scored[:top_k]]

"""ArxivAdapter -- free, no API key, always registered directly by
news_sources._always_on_sources (not read from news_source.api -- see
that function's docstring)."""

from datetime import datetime

import feedparser
import requests

from news_adapters._util import _REQUEST_HEADERS, _parse_rss_published, _raise_if_html


class ArxivAdapter:
    TYPE = "arxiv"

    def initialize(self, config: dict) -> None:
        pass

    def pull(self, query: str = "cat:cs.AI", max_results: int = 5, since: datetime | None = None,
             section: str | None = None) -> list[dict]:
        """`since`, when given, appends a `submittedDate:[X TO 9999...]` range
        to the query -- confirmed live 2026-08-16 the syntax works. Note:
        arXiv's own indexing has a real multi-day lag (a plain, unfiltered
        query on 2026-08-16 returned nothing newer than 2026-08-13), so a
        short since-last-pull window (news_ingest's default interval is 4h)
        will often legitimately return nothing -- that's arXiv's real update
        cadence, not a bug, and no worse than before (today's flat top-N cap
        mostly re-fetches the same few papers on a source this slow)."""
        # A section is an arXiv subject class, which is what this archive
        # actually indexes by -- far better than free-text search, which
        # found only 36 quantum and 6 optics articles for subscribers who
        # follow exactly those topics.
        search_query = f"cat:{section}" if section else query
        if since is not None:
            search_query = f"{search_query} AND submittedDate:[{since.strftime('%Y%m%d%H%M')} TO 99991231235959]"
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": search_query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": max_results,
            },
            timeout=10,
            headers=_REQUEST_HEADERS,
        )
        resp.raise_for_status()
        _raise_if_html(resp, resp.url)
        feed = feedparser.parse(resp.text)
        return [
            {
                "title": entry.get("title", "").replace("\n", " ").strip(),
                "link": entry.get("link"),
                "source": "arXiv",
                "summary": entry.get("summary", "").replace("\n", " ").strip()[:300],
                "published": entry.get("published"),
                "published_dt": _parse_rss_published(entry),
            }
            for entry in feed.entries
        ]

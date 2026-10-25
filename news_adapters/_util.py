"""
Shared helpers used by every NewsSourceAdapter's pull() implementation
(and by news_sources.py's own RSS/traced_fetch code) -- relocated
verbatim from news_sources.py when the fetch_* functions became adapter
classes, so the docstrings/comments below (and the incidents they
record) are unchanged.
"""

import re
from datetime import datetime, timezone

import calendar

import requests

# Self-identifying, not a fake browser -- some feeds (TechRadar, confirmed
# live) return 403 to the bare `python-requests/x.x` default User-Agent but
# accept a real one; the honest fix is to say who we are, not to impersonate
# a browser. Applied to every source for consistency, not just the one that
# needed it -- a future source hitting the same block shouldn't need its own
# special case.
_USER_AGENT = "Mozilla/5.0 (compatible; ArgusNewsBot/1.0; +https://github.com/nankma/argus)"
_REQUEST_HEADERS = {"User-Agent": _USER_AGENT}


# Query-string parameters whose value is a credential. requests puts the full
# request URL into an HTTPError's message, and news_ingest.py logs that
# exception straight to stdout -- i.e. into `docker logs`, unredacted, on
# every failed fetch.
#
# Real incident, 2026-08-19: a routine check of the ingestion logs surfaced
# GNews's and Perigon's live API keys in plaintext, from a 400 and a 403
# respectively. Not a one-off mistake -- systematic, and it had been
# happening on every error since these sources were added. Both keys were
# rotated. traced_fetch's OpenTelemetry span carried the same value into
# the telemetry backend (Phoenix at the time; Logfire now).
_SECRET_QUERY_PARAM_RE = re.compile(r"((?:api[-_]?key|apikey|token)=)[^&\s]+", re.IGNORECASE)


def _redact(text: object) -> str:
    """Strips credential values out of anything about to be logged."""
    return _SECRET_QUERY_PARAM_RE.sub(r"\1<redacted>", str(text))


def _raise_for_status(resp: requests.Response) -> None:
    """requests.raise_for_status() with the credential stripped from the
    error message. Use this instead of resp.raise_for_status() for any
    source whose auth travels in the query string."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(_redact(exc)) from None


class FeedBlockedError(Exception):
    """A feed URL answered 2xx with an HTML document instead of a feed."""


def _raise_if_html(resp: requests.Response, url: str) -> None:
    """Raises FeedBlockedError when a 2xx response body is an HTML document.

    An HTML page is never a valid RSS/Atom feed, but feedparser quietly
    parses one into zero entries, so an anti-bot challenge page (HTTP 200,
    e.g. The Register's proof-of-work page served to datacenter IPs) used
    to look like a feed with nothing new: "fetched 0 article(s)", no
    failure anywhere. Sniffs the body rather than Content-Type, since some
    servers label a perfectly valid feed text/html. A feed that is valid
    but empty is not HTML and still parses to [].
    """
    head = resp.content[:512].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        raise FeedBlockedError(
            f"{_redact(url)} returned HTML ({resp.status_code}, {len(resp.content)} bytes) "
            "instead of a feed -- likely an anti-bot challenge page")


def _parse_iso_published(raw: str | None) -> datetime | None:
    """For sources that give an ISO-8601-ish string (HN's created_at,
    NewsAPI/GNews/Perigon's publishedAt/pubDate).

    Real incident, 2026-08-14: a source returning a timestamp with no UTC
    offset at all (e.g. "2026-08-13T22:00:00", no "Z", no "+00:00") made
    `datetime.fromisoformat` return a naive datetime, silently breaking
    this function's documented "always timezone-aware" contract. That
    naive value then crashed every news_push.py cycle for two real
    subscribers with `TypeError: can't compare offset-naive and
    offset-aware datetimes` (published_dt <= since, where since is
    always aware) -- not an occasional glitch, a deterministic failure on
    every single tick once that source had a new article. Fixed by
    assuming UTC for any parse that comes back naive, same as this
    function already does explicitly for "Z"."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_rss_published(entry) -> datetime | None:
    """For feedparser entries (arXiv, RSS blogs) -- feedparser normalizes
    whatever date format the feed uses into published_parsed (a UTC
    struct_time), which is far more reliable than parsing the raw
    "published" string ourselves."""
    parsed = entry.get("published_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)

"""
Telegram's HTML subset: parsing/validating the small tag vocabulary this
bot actually asks for (see agent.HTML_FORMATTING_RULES) -- b, i, a href.

Split out of bot.py 2026-08-28 because news_push.py needs validate() too
(for the pre-send retry loop in run_push_cycle) and bot.py already
imports news_push, so news_push.py importing bot.py back would be
circular. Everything here was previously private to bot.py; behavior is
unchanged, only the import path moved.
"""

import re

_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s[^>]*)?)>")

_ALLOWED_TAGS = {"b", "i", "a"}

# A "&" not immediately followed by a valid named/numeric entity --
# amp/lt/gt/quot cover everything agent.HTML_FORMATTING_RULES asks the
# model to escape; #NNN and #xHHH cover numeric character references, in
# case the model uses one instead.
_BARE_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);)")
_HREF_RE = re.compile(r'^\s+href="[^"]*"\s*$')


def is_html_balanced(text: str) -> bool:
    """True if `text` has no unclosed HTML tag -- open/close tag counts
    match. Doesn't validate proper nesting, just depth -- good enough for
    split_for_telegram's job (never split inside a tag pair), which is
    the only caller that needs depth alone rather than full validate()."""
    depth = 0
    for m in _TAG_RE.finditer(text):
        depth += -1 if m.group(1) == "/" else 1
        if depth < 0:
            return False
    return depth == 0


def strip_html_tags(text: str) -> str:
    return _TAG_RE.sub("", text)


def validate(text: str) -> str | None:
    """None if `text` is valid Telegram HTML for the tag set this bot
    actually asks for (b, i, a href). A reason string otherwise, written
    to be readable both by a human (an admin alert) and by the model
    itself (fed back into a retry prompt in news_push.write_push_digest).

    A best-effort predictor of Telegram's own parser, not a byte-for-byte
    reimplementation -- checks run in a fixed order and stop at the first
    failure, so the reason is specific and single-cause rather than a
    dump of everything wrong. Callers that need a true last-resort net
    for whatever this misses should still catch telegram.error.BadRequest
    at send time (see bot.py's send_push_digest/handle_message)."""
    matches = list(_TAG_RE.finditer(text))

    for m in matches:
        name = m.group(2).lower()
        if name not in _ALLOWED_TAGS:
            return f"disallowed tag <{m.group(1)}{m.group(2)}>"

    stack = []
    for m in matches:
        name = m.group(2).lower()
        if m.group(1) == "/":
            if not stack or stack[-1] != name:
                unclosed = f"<{stack[-1]}>" if stack else "no open tag"
                return f"</{name}> does not match {unclosed} -- crossed or unmatched nesting"
            stack.pop()
        else:
            stack.append(name)
    if stack:
        return f"unclosed <{stack[-1]}>"

    for m in matches:
        if m.group(1) == "/":
            continue
        name = m.group(2).lower()
        attrs = m.group(3)
        if name == "a" and not _HREF_RE.match(attrs):
            return f'malformed <a> tag (needs exactly href="..."): {m.group(0)!r}'

    pos = 0
    for m in matches:
        reason = _check_escaping(text[pos:m.start()])
        if reason:
            return reason
        pos = m.end()
    reason = _check_escaping(text[pos:])
    if reason:
        return reason

    return None


def _check_escaping(segment: str) -> str | None:
    m = _BARE_AMP_RE.search(segment)
    if m:
        return f"unescaped & at {m.group(0)!r} -- use &amp;"
    if "<" in segment:
        return "unescaped < outside a recognized tag -- use &lt;"
    if ">" in segment:
        return "unescaped > outside a recognized tag -- use &gt;"
    return None


# Distinct from _HREF_RE above (which validates one already-extracted <a>
# attribute string) -- this one finds every href value in free-form text.
_HREF_VALUE_RE = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)


def links_actually_sent(digest: str, candidates: list[dict]) -> list[str]:
    """Which candidate articles genuinely appear in a model-written
    digest/report. Shared by news_push.write_push_digest and
    agent.search_news -- moved here (a zero-dependency leaf module) from
    news_push.py 2026-09-05 specifically so agent.py could reuse it
    without a circular import (news_push.py already imports agent.py).

    The digest is free-form prose the model writes from a candidate
    list, and both callers' prompts explicitly tell it to omit candidates
    that aren't relevant -- so "what was offered" and "what the reader
    actually saw" are different sets. Only the latter should count as
    seen: marking an omitted candidate as sent would retire an article
    nobody ever read.

    Recovered by matching the digest's own <a href> targets against the
    candidate links, since both callers' report-structure prompts require
    every cited item to carry its source link and forbid inventing URLs."""
    hrefs = {h.strip() for h in _HREF_VALUE_RE.findall(digest or "")}
    return [a["link"] for a in candidates if a.get("link") in hrefs]

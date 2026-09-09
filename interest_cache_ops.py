"""
Data Access Layer for global (not per-subscriber) classification caches --
an interest's category mapping, retrieval-query expansion, and a
category's keyness scores.
"""

import json

from storage import get_storage


def get_cached_interest_categories(interests: list[str]) -> dict[str, list[str]]:
    """{interest: categories} for whichever of `interests` already have a
    cached classification -- an interest with no cached mapping is absent
    from the result, not present with an empty list."""
    if not interests:
        return {}
    rows = get_storage().get_cached_interest_categories(interests)
    return {interest: json.loads(categories_json) for interest, categories_json in rows}


def set_interest_categories(interest: str, categories: list[str]) -> None:
    get_storage().set_interest_categories(interest, json.dumps(categories))


def get_interest_query_expansion(interest: str) -> str | None:
    """None means never generated -- callers fall back to the bare
    interest string in that case."""
    return get_storage().get_interest_query_expansion(interest)


def set_interest_query_expansion(interest: str, expansion: str) -> None:
    get_storage().set_interest_query_expansion(interest, expansion)


def get_subscriber_interest_definition(chat_id: int, interest: str) -> str | None:
    """This ONE subscriber's own override, if they've refined it via
    find_interests -- see docs/plans/interest-definition-plan.md. None
    means they haven't; callers use resolve_interest_definition below
    rather than calling this directly, so the shared-default fallback
    isn't reimplemented at every call site."""
    return get_storage().get_subscriber_interest_definition(chat_id, interest)


def set_subscriber_interest_definition(chat_id: int, interest: str, expansion: str) -> None:
    get_storage().set_subscriber_interest_definition(chat_id, interest, expansion)


def resolve_interest_definition(chat_id: int, interest: str) -> str | None:
    """The read path every retrieval call site should use instead of
    get_interest_query_expansion directly: this subscriber's own
    refinement first, the shared/global default second, None if neither
    exists (the caller's existing bare-topic-string fallback still
    applies then -- this function adds a tier, it doesn't remove the
    existing one).

    Deliberately two separate tables, not one merged lookup with a
    subscriber_id column that's NULL for the shared row: the shared
    table is written automatically and constantly (every add_one_interest
    call, every search_news cache miss) by code that has no idea
    per-subscriber overrides exist, and mixing the two would risk an
    automatic write silently clobbering a subscriber's deliberate
    refinement if the two ever collided on write order. Keeping them
    apart makes that impossible by construction."""
    own = get_subscriber_interest_definition(chat_id, interest)
    if own is not None:
        return own
    return get_interest_query_expansion(interest)


def set_category_keyness(category: str, scores: dict[str, float]) -> None:
    """Replaces `category`'s entire row set atomically -- news_keyness.py
    recomputes this fresh every ingest cycle."""
    get_storage().set_category_keyness(category, list(scores.items()))


def get_category_keyness(category: str) -> dict[str, float]:
    """{} when nothing has been computed for this category yet -- callers
    treat "no keyness signal" as a normal, expected case."""
    rows = get_storage().get_category_keyness(category)
    return {term: score for term, score in rows}

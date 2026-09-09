import interest_cache_ops


def test_get_cached_interest_categories_empty_when_nothing_cached(isolated_subscribers_db):
    assert interest_cache_ops.get_cached_interest_categories(["AI"]) == {}


def test_get_cached_interest_categories_empty_input_returns_empty(isolated_subscribers_db):
    assert interest_cache_ops.get_cached_interest_categories([]) == {}


def test_set_and_get_cached_interest_categories(isolated_subscribers_db):
    interest_cache_ops.set_interest_categories("AI", ["AI", "Research"])
    assert interest_cache_ops.get_cached_interest_categories(["AI"]) == {"AI": ["AI", "Research"]}


def test_get_cached_interest_categories_only_returns_known_interests(isolated_subscribers_db):
    interest_cache_ops.set_interest_categories("AI", ["AI"])
    result = interest_cache_ops.get_cached_interest_categories(["AI", "AAOI"])
    assert result == {"AI": ["AI"]}
    assert "AAOI" not in result


def test_set_interest_categories_can_store_empty_list(isolated_subscribers_db):
    # A classifier miss (interest doesn't map to any category) is a real,
    # cacheable result -- distinct from "not yet classified at all".
    interest_cache_ops.set_interest_categories("some obscure ticker", [])
    assert interest_cache_ops.get_cached_interest_categories(["some obscure ticker"]) == {"some obscure ticker": []}


def test_set_interest_categories_upserts(isolated_subscribers_db):
    interest_cache_ops.set_interest_categories("AI", ["AI"])
    interest_cache_ops.set_interest_categories("AI", ["AI", "Research"])
    assert interest_cache_ops.get_cached_interest_categories(["AI"]) == {"AI": ["AI", "Research"]}


def test_get_interest_query_expansion_none_when_never_generated(isolated_subscribers_db):
    assert interest_cache_ops.get_interest_query_expansion("AI coding") is None


def test_set_interest_query_expansion_round_trips(isolated_subscribers_db):
    interest_cache_ops.set_interest_query_expansion("AI coding", "AI systems that assist developers...")
    assert interest_cache_ops.get_interest_query_expansion("AI coding") == "AI systems that assist developers..."


def test_set_interest_query_expansion_upserts(isolated_subscribers_db):
    interest_cache_ops.set_interest_query_expansion("AI coding", "first version")
    interest_cache_ops.set_interest_query_expansion("AI coding", "second version")
    assert interest_cache_ops.get_interest_query_expansion("AI coding") == "second version"


def test_get_subscriber_interest_definition_none_when_never_set(isolated_subscribers_db):
    assert interest_cache_ops.get_subscriber_interest_definition(7, "AI") is None


def test_set_subscriber_interest_definition_round_trips(isolated_subscribers_db):
    interest_cache_ops.set_subscriber_interest_definition(7, "AI", "this subscriber's own definition")
    assert interest_cache_ops.get_subscriber_interest_definition(7, "AI") == "this subscriber's own definition"


def test_set_subscriber_interest_definition_upserts(isolated_subscribers_db):
    interest_cache_ops.set_subscriber_interest_definition(7, "AI", "first version")
    interest_cache_ops.set_subscriber_interest_definition(7, "AI", "second version")
    assert interest_cache_ops.get_subscriber_interest_definition(7, "AI") == "second version"


def test_subscriber_interest_definition_is_scoped_per_chat_id(isolated_subscribers_db):
    """The whole point of the split from the shared table: one
    subscriber's refinement must not leak into another's."""
    interest_cache_ops.set_subscriber_interest_definition(7, "AI", "chat 7's definition")
    interest_cache_ops.set_subscriber_interest_definition(8, "AI", "chat 8's definition")
    assert interest_cache_ops.get_subscriber_interest_definition(7, "AI") == "chat 7's definition"
    assert interest_cache_ops.get_subscriber_interest_definition(8, "AI") == "chat 8's definition"


def test_subscriber_interest_definition_is_scoped_per_interest(isolated_subscribers_db):
    interest_cache_ops.set_subscriber_interest_definition(7, "AI", "AI definition")
    interest_cache_ops.set_subscriber_interest_definition(7, "robotics", "robotics definition")
    assert interest_cache_ops.get_subscriber_interest_definition(7, "AI") == "AI definition"
    assert interest_cache_ops.get_subscriber_interest_definition(7, "robotics") == "robotics definition"


def test_resolve_interest_definition_none_when_neither_tier_has_it(isolated_subscribers_db):
    assert interest_cache_ops.resolve_interest_definition(7, "AI") is None


def test_resolve_interest_definition_falls_back_to_the_shared_default(isolated_subscribers_db):
    interest_cache_ops.set_interest_query_expansion("AI", "the shared default")
    assert interest_cache_ops.resolve_interest_definition(7, "AI") == "the shared default"


def test_resolve_interest_definition_prefers_the_subscribers_own_override(isolated_subscribers_db):
    """The core guarantee of docs/plans/interest-definition-plan.md: a
    personal refinement must win over the shared default, not be
    shadowed by it."""
    interest_cache_ops.set_interest_query_expansion("AI", "the shared default")
    interest_cache_ops.set_subscriber_interest_definition(7, "AI", "chat 7's own refinement")
    assert interest_cache_ops.resolve_interest_definition(7, "AI") == "chat 7's own refinement"


def test_resolve_interest_definition_does_not_affect_other_subscribers(isolated_subscribers_db):
    """One subscriber refining a definition must not change what a
    DIFFERENT subscriber following the same interest word receives."""
    interest_cache_ops.set_interest_query_expansion("AI", "the shared default")
    interest_cache_ops.set_subscriber_interest_definition(7, "AI", "chat 7's own refinement")
    assert interest_cache_ops.resolve_interest_definition(8, "AI") == "the shared default"


def test_get_category_keyness_empty_when_never_computed(isolated_subscribers_db):
    assert interest_cache_ops.get_category_keyness("AI") == {}


def test_set_category_keyness_round_trips(isolated_subscribers_db):
    interest_cache_ops.set_category_keyness("AI", {"openai": 286.95, "quantum": -31.45})
    assert interest_cache_ops.get_category_keyness("AI") == {"openai": 286.95, "quantum": -31.45}


def test_set_category_keyness_replaces_the_whole_category_not_an_upsert(isolated_subscribers_db):
    """Unlike interest_query_expansions' single-key upsert, a category's
    entire row set is meant to be replaced together each news_ingest.py
    cycle -- a term that scored last cycle but not this one (dropped
    below the df floor, or the category pool changed) must not linger."""
    interest_cache_ops.set_category_keyness("AI", {"openai": 286.95, "stale_term": -5.0})
    interest_cache_ops.set_category_keyness("AI", {"openai": 320.62, "quantum": -31.45})
    assert interest_cache_ops.get_category_keyness("AI") == {"openai": 320.62, "quantum": -31.45}


def test_set_category_keyness_is_scoped_per_category(isolated_subscribers_db):
    interest_cache_ops.set_category_keyness("AI", {"openai": 286.95})
    interest_cache_ops.set_category_keyness("Finance", {"nasdaq": 150.0})
    assert interest_cache_ops.get_category_keyness("AI") == {"openai": 286.95}
    assert interest_cache_ops.get_category_keyness("Finance") == {"nasdaq": 150.0}

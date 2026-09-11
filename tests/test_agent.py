import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent
import guardrails
import news_cache
import news_classify
import news_embed
import interest_cache_ops
import subscriber_ops
from tests.fakes import FakeEmbedder, FakeToolCallingModel, RecordingCallbackHandler


def _fake_request(context):
    """_compose_prompt only reads request.runtime.context -- a real
    LangChain ModelRequest is unnecessary machinery for testing the
    prompt-composition logic in isolation."""
    return SimpleNamespace(runtime=SimpleNamespace(context=context))


def _record_chat_openai(monkeypatch):
    """build_model_from_config calls the module-level ChatOpenAI name, so
    monkeypatching it on the agent module (not the real langchain_openai
    class) lets these tests assert what was requested without constructing
    a real client or making any network call."""
    calls = []
    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kw: calls.append(kw) or "fake-model")
    return calls


def test_build_model_from_config_reads_url_model_key(monkeypatch):
    calls = _record_chat_openai(monkeypatch)
    cfg = {"url": "https://api.together.xyz/v1", "model": "deepseek-ai/DeepSeek-V4-Flash-0731", "api-key": "tgp_test"}
    result = agent.build_model_from_config(cfg)
    assert calls[0]["base_url"] == cfg["url"]
    assert calls[0]["model"] == cfg["model"]
    assert calls[0]["api_key"] == cfg["api-key"]
    assert result == "fake-model"


def test_build_model_from_config_omits_reasoning_effort_by_default(monkeypatch):
    """Unlike the old env-var-based build_model, this does NOT default to
    "none" -- that value is a DeepSeek-specific workaround (see the
    docstring), and a deployment on a different provider sets it in its
    own settings.yml only if it's actually needed."""
    calls = _record_chat_openai(monkeypatch)
    agent.build_model_from_config({"url": "u", "model": "m", "api-key": "k"})
    assert "reasoning_effort" not in calls[0]


def test_build_model_from_config_passes_reasoning_effort_when_given(monkeypatch):
    calls = _record_chat_openai(monkeypatch)
    agent.build_model_from_config({"url": "u", "model": "m", "api-key": "k", "reasoning_effort": "high"})
    assert calls[0]["reasoning_effort"] == "high"


def test_build_model_from_config_uses_default_timeout_when_not_in_cfg(monkeypatch):
    calls = _record_chat_openai(monkeypatch)
    agent.build_model_from_config({"url": "u", "model": "m", "api-key": "k"}, default_timeout=42.0)
    assert calls[0]["request_timeout"] == 42.0


def test_build_model_from_config_cfg_timeout_overrides_default(monkeypatch):
    calls = _record_chat_openai(monkeypatch)
    agent.build_model_from_config(
        {"url": "u", "model": "m", "api-key": "k", "request_timeout_seconds": 5.0}, default_timeout=42.0
    )
    assert calls[0]["request_timeout"] == 5.0


def test_build_model_from_config_omits_timeout_when_falsy(monkeypatch):
    calls = _record_chat_openai(monkeypatch)
    agent.build_model_from_config({"url": "u", "model": "m", "api-key": "k", "request_timeout_seconds": 0})
    assert "request_timeout" not in calls[0]


def test_build_model_from_settings_resolves_the_given_path(monkeypatch):
    """The usual entry point -- resolves a dotted Settings path (the shape
    a deployment's settings.yml provides) into a cfg dict, then delegates
    to build_model_from_config."""
    from trailsign import Settings

    calls = _record_chat_openai(monkeypatch)
    settings = Settings({"models": {"main": {"url": "u", "model": "m", "api-key": "k"}}})
    result = agent.build_model_from_settings(settings, "models.main")
    assert calls[0]["base_url"] == "u"
    assert calls[0]["model"] == "m"
    assert result == "fake-model"


def test_build_model_from_settings_raises_when_path_missing(monkeypatch):
    """A deployment with no models.* in its settings.yml should fail loudly
    at startup, not construct a half-built model."""
    from trailsign import Settings, SettingsError

    _record_chat_openai(monkeypatch)
    settings = Settings({})
    with pytest.raises(SettingsError):
        agent.build_model_from_settings(settings, "models.main")


def test_compose_prompt_defaults_to_news_query_when_no_category():
    prompt = agent._compose_prompt(_fake_request({}))
    assert agent._NEWS_QUERY_INSTRUCTIONS in prompt
    assert agent.LAYER1_IDENTITY in prompt


def test_compose_prompt_defaults_to_news_query_when_context_is_none():
    prompt = agent._compose_prompt(_fake_request(None))
    assert agent._NEWS_QUERY_INSTRUCTIONS in prompt


def test_compose_prompt_always_uses_news_query_instructions():
    # Route B (start_push/stop_push) is dispatched directly by
    # agent.dispatch_settings; everything else (news_query, and now
    # set_interest/remove_interest/set_language/find_interests) runs
    # through the agent loop, so this prompt only ever needs the
    # news_query instructions. The `category` context key no longer
    # selects anything here; this just confirms that stays true
    # regardless of what's passed.
    for category in (None, "news_query", "set_interest", "start_push"):
        prompt = agent._compose_prompt(_fake_request({"category": category}))
        assert agent._NEWS_QUERY_INSTRUCTIONS in prompt


def test_compose_prompt_includes_interests_when_set(isolated_subscribers_db):
    subscriber_ops.set_interests(101, ["AI", "robotics"])
    prompt = agent._compose_prompt(_fake_request({"chat_id": 101, "category": "news_query"}))
    assert "AI, robotics" in prompt


def test_compose_prompt_omits_interests_when_unset(isolated_subscribers_db):
    prompt = agent._compose_prompt(_fake_request({"chat_id": 102, "category": "news_query"}))
    assert "stated interests" not in prompt


def test_compose_prompt_omits_interests_when_no_chat_id():
    prompt = agent._compose_prompt(_fake_request({"category": "news_query"}))
    assert "stated interests" not in prompt


def test_compose_prompt_includes_language_when_set(isolated_subscribers_db):
    subscriber_ops.set_language(103, "Spanish")
    prompt = agent._compose_prompt(_fake_request({"chat_id": 103, "category": "news_query"}))
    assert "Spanish" in prompt
    assert "preferred reply language" in prompt


def test_compose_prompt_omits_language_when_unset(isolated_subscribers_db):
    prompt = agent._compose_prompt(_fake_request({"chat_id": 104, "category": "news_query"}))
    assert "preferred reply language" not in prompt


def test_compose_prompt_language_applies_regardless_of_category(isolated_subscribers_db):
    # Real requirement: unlike interests (news_query-only), a language
    # preference must govern every reply, including subscription
    # confirmations -- see docs/plans/bot-features-plan.md item 2.
    subscriber_ops.set_language(105, "French")
    for category in ("news_query", "set_interest", "start_push", "set_language"):
        prompt = agent._compose_prompt(_fake_request({"chat_id": 105, "category": category}))
        assert "French" in prompt


def _classification(category, **kwargs):
    return guardrails.MessageClassification(on_topic=True, categories=[category], **kwargs)


def test_dispatch_settings_start_push_enables_and_sets_interval(isolated_subscribers_db):
    result = agent.dispatch_settings("start_push", 205, _classification("start_push", push_interval_hours=6))
    assert "every 6 hour(s)" in result
    assert subscriber_ops.get_push_enabled(205) is True
    assert subscriber_ops.get_push_interval_hours(205) == 6


def test_dispatch_settings_start_push_no_interval_leaves_existing(isolated_subscribers_db):
    subscriber_ops.set_push_interval_hours(206, 12)
    result = agent.dispatch_settings("start_push", 206, _classification("start_push"))
    assert "every 12 hour(s)" in result
    assert subscriber_ops.get_push_interval_hours(206) == 12


def test_dispatch_settings_start_push_invalid_interval_reports_error(isolated_subscribers_db):
    result = agent.dispatch_settings("start_push", 207, _classification("start_push", push_interval_hours=0))
    assert "couldn't set that interval" in result
    assert subscriber_ops.get_push_enabled(207) is True  # the enable itself still succeeded


def test_dispatch_settings_stop_push_disables(isolated_subscribers_db):
    subscriber_ops.set_push_enabled(208, True)
    result = agent.dispatch_settings("stop_push", 208, _classification("stop_push"))
    assert "Turned off" in result
    assert subscriber_ops.get_push_enabled(208) is False


def test_dispatch_settings_rejects_non_route_b_category():
    try:
        agent.dispatch_settings("news_query", 1, _classification("news_query"))
        assert False, "expected ValueError"
    except ValueError:
        pass


# agent.search_news is a plain function now (chat_id, query, history,
# model, guard_model, embedder), not a LangChain tool an agent loop
# decides whether/how often to call -- see that function's own module
# note (2026-09-05) for why. These tests call it directly.


def _cached_article(link, title, categories=None, embedding=None, published="2026-09-01"):
    return {
        "title": title, "link": link, "source": "TestSource", "source_key": "test",
        "categories": categories, "embedding": embedding,
        "published": published, "published_dt": None,
    }


class _RecordingModel(FakeToolCallingModel):
    """Captures the exact messages passed to invoke() -- the candidate
    listing agent.search_news builds -- same RecordingModel-subclasses-
    FakeToolCallingModel convention tests/test_news_push.py's own
    write_push_digest tests already use. `captured` is a declared
    pydantic field (like FakeToolCallingModel's own `responses`/`i`),
    not a plain instance attribute -- BaseChatModel is a pydantic model
    and rejects undeclared attributes."""
    captured: list = []

    def invoke(self, messages, *args, **kwargs):
        self.captured = messages
        return super().invoke(messages, *args, **kwargs)


def test_search_news_sends_only_relevant_candidates_to_the_model(monkeypatch, isolated_subscribers_db):
    """No live source fetch any more -- search_news reads whatever
    news_ingest.py already cached, the same corpus news_push.py's digest
    pipeline reads (news_cache.read_all), and only passes the
    embedding-relevant subset to the model -- never the whole cache."""
    embedder = FakeEmbedder()
    on_topic = _cached_article(
        "https://example.com/coding", "New AI coding assistant launches",
        embedding=news_embed.embed_one(embedder, "AI coding assistant launches"),
    )
    off_topic = _cached_article(
        "https://example.com/weather", "Storm hits coastal region",
        embedding=news_embed.embed_one(embedder, "Storm hits coastal region"),
    )
    monkeypatch.setattr(news_cache, "read_all", lambda: [on_topic, off_topic])
    # A pool this small (2) is smaller than the real SEARCH_RELEVANCE_KEEP_MIN
    # default (20), which would otherwise clamp to "keep everything" and
    # never actually exercise the relevance gate -- lower it so this test
    # can distinguish "excluded" from "pool too small to filter at all".
    monkeypatch.setattr(agent, "SEARCH_RELEVANCE_KEEP_MIN", 1)
    model = _RecordingModel(responses=[AIMessage(content="<b>Report</b>")])

    result = agent.search_news(1, "AI coding assistant", [], model, None, embedder)

    listing = model.captured[1]["content"]
    assert "New AI coding assistant launches" in listing
    assert "Storm hits coastal region" not in listing
    assert result == "<b>Report</b>"


def test_search_news_sends_only_the_newest_max_results_candidates(monkeypatch, isolated_subscribers_db):
    """SEARCH_MAX_RESULTS=5 (the user's own explicit call, after rejecting
    a proposed 20) must actually cap what reaches the model, and
    survivors must be newest-first -- recency governs order, relevance
    only gates inclusion, same principle as news_push.py's own digest
    cut."""
    embedder = FakeEmbedder()
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    articles = [
        _cached_article(
            f"https://example.com/{i}", f"AI coding update {i}",
            embedding=news_embed.embed_one(embedder, f"AI coding update {i}"),
        )
        for i in range(7)
    ]
    for i, a in enumerate(articles):
        a["published_dt"] = base + timedelta(hours=i)  # index 6 is newest
    monkeypatch.setattr(news_cache, "read_all", lambda: articles)
    model = _RecordingModel(responses=[AIMessage(content="<b>Report</b>")])

    agent.search_news(1, "AI coding", [], model, None, embedder)

    listing = model.captured[1]["content"]
    expected_newest_first = [f"https://example.com/{i}" for i in (6, 5, 4, 3, 2)]
    positions = [listing.index(link) for link in expected_newest_first]
    assert positions == sorted(positions)  # each appears, in this exact order
    assert "https://example.com/1" not in listing  # 6th/7th-newest, truncated
    assert "https://example.com/0" not in listing


def test_search_news_excludes_already_shown_links(monkeypatch, isolated_subscribers_db):
    """Shares subscriber_ops' pushed_links dedup memory with news_push.py --
    an article a push digest already delivered must not resurface here.
    An empty candidate pool short-circuits before the model is ever
    called -- see search_news's own "no is no" design."""
    embedder = FakeEmbedder()
    link = "https://example.com/coding"
    article = _cached_article(
        link, "New AI coding assistant launches",
        embedding=news_embed.embed_one(embedder, "New AI coding assistant launches"),
    )
    monkeypatch.setattr(news_cache, "read_all", lambda: [article])
    subscriber_ops.mark_links_shown(1, [link], datetime.now(timezone.utc))
    model = MagicMock()

    result = agent.search_news(1, "AI coding assistant", [], model, None, embedder)

    assert result == agent._no_results_message("AI coding assistant")
    model.invoke.assert_not_called()


def test_search_news_marks_only_actually_cited_links_shown_without_advancing_last_push_at(
    monkeypatch, isolated_subscribers_db
):
    """The other half of the shared-dedup contract: a search result must
    itself become "already shown" (so a later push doesn't re-send it),
    but must NOT touch last_push_at -- a manual search must never delay
    this subscriber's own scheduled push. See subscriber_ops.mark_links_shown
    and advance_last_push_at's docstrings for the 2026-09-04 split. Only
    what the model actually cites counts as shown -- same convention as
    telegram_html.links_actually_sent, shared with news_push.py."""
    embedder = FakeEmbedder()
    link = "https://example.com/coding"
    article = _cached_article(
        link, "New AI coding assistant launches",
        embedding=news_embed.embed_one(embedder, "New AI coding assistant launches"),
    )
    monkeypatch.setattr(news_cache, "read_all", lambda: [article])
    assert subscriber_ops.get_last_push_at(1) is None
    model = FakeToolCallingModel(
        responses=[AIMessage(content=f'<a href="{link}">New AI coding assistant launches</a>')]
    )

    agent.search_news(1, "AI coding assistant", [], model, None, embedder)

    assert link in subscriber_ops.get_pushed_links(1)
    assert subscriber_ops.get_last_push_at(1) is None


def test_search_news_enforces_daily_quota(monkeypatch, isolated_subscribers_db):
    monkeypatch.setattr(agent, "SEARCH_DAILY_LIMIT", 1)
    monkeypatch.setattr(news_cache, "read_all", lambda: [])
    model = MagicMock()

    first = agent.search_news(1, "AI", [], model, None, None)
    second = agent.search_news(1, "AI", [], model, None, None)

    assert first == agent._no_results_message("AI")  # cap not yet reached, just an empty cache
    assert "today's searches" in second
    model.invoke.assert_not_called()


def test_search_news_generates_and_caches_a_query_definition_when_uncached(monkeypatch, isolated_subscribers_db):
    monkeypatch.setattr(news_cache, "read_all", lambda: [])
    monkeypatch.setattr(agent, "_rewrite_search_query", lambda query, history, guard_model: query)
    expand_calls = []

    def fake_expand(model, interest):
        expand_calls.append(interest)
        return "a generated definition"

    monkeypatch.setattr(news_classify, "expand_interest_for_retrieval", fake_expand)

    result = agent.search_news(1, "AI coding", [], MagicMock(), "fake-guard-model", None)

    assert expand_calls == ["AI coding"]
    assert interest_cache_ops.get_interest_query_expansion("AI coding") == "a generated definition"
    assert result == agent._no_results_message("AI coding")


def test_search_news_reuses_a_cached_query_definition(monkeypatch, isolated_subscribers_db):
    interest_cache_ops.set_interest_query_expansion("AI coding", "already cached definition")
    monkeypatch.setattr(news_cache, "read_all", lambda: [])
    monkeypatch.setattr(agent, "_rewrite_search_query", lambda query, history, guard_model: query)

    def fail_if_called(model, interest):
        raise AssertionError("should not regenerate a cached definition")

    monkeypatch.setattr(news_classify, "expand_interest_for_retrieval", fail_if_called)

    result = agent.search_news(1, "AI coding", [], MagicMock(), "fake-guard-model", None)

    assert result == agent._no_results_message("AI coding")


def test_search_news_prefers_the_subscribers_own_definition_over_the_shared_one(
    monkeypatch, isolated_subscribers_db
):
    """docs/plans/interest-definition-plan.md: a subscriber who has
    refined a definition via find_interests must have search_news use
    it too, not just push -- retrieval should be consistent across both
    paths. Captures the actual query text handed to the relevance
    filter, rather than only checking the no-results message, so this
    would fail if the wrong tier's definition were ever picked."""
    interest_cache_ops.set_interest_query_expansion("AI coding", "the shared default")
    interest_cache_ops.set_subscriber_interest_definition(1, "AI coding", "chat 1's own refinement")
    monkeypatch.setattr(news_cache, "read_all", lambda: [{"link": "https://ex.invalid/a", "title": "x"}])
    monkeypatch.setattr(agent, "_rewrite_search_query", lambda query, history, guard_model: query)
    captured = {}
    def fake_filter(pool, embedder, query_text, **kw):
        captured["query_text"] = query_text
        return []
    monkeypatch.setattr(news_embed, "filter_by_relevance", fake_filter)

    agent.search_news(1, "AI coding", [], MagicMock(), "fake-guard-model", None)

    assert captured["query_text"] == "chat 1's own refinement"


def test_search_news_a_different_subscriber_still_gets_the_shared_definition(
    monkeypatch, isolated_subscribers_db
):
    """The flip side of the test above: chat 1's refinement must not leak
    into chat 2's retrieval."""
    interest_cache_ops.set_interest_query_expansion("AI coding", "the shared default")
    interest_cache_ops.set_subscriber_interest_definition(1, "AI coding", "chat 1's own refinement")
    monkeypatch.setattr(news_cache, "read_all", lambda: [{"link": "https://ex.invalid/a", "title": "x"}])
    monkeypatch.setattr(agent, "_rewrite_search_query", lambda query, history, guard_model: query)
    captured = {}
    def fake_filter(pool, embedder, query_text, **kw):
        captured["query_text"] = query_text
        return []
    monkeypatch.setattr(news_embed, "filter_by_relevance", fake_filter)

    agent.search_news(2, "AI coding", [], MagicMock(), "fake-guard-model", None)

    assert captured["query_text"] == "the shared default"


def test_search_news_no_results_message(monkeypatch, isolated_subscribers_db):
    monkeypatch.setattr(news_cache, "read_all", lambda: [])

    result = agent.search_news(1, "AI coding", [], MagicMock(), None, None)

    assert result == 'No related news found for "AI coding".'


# --- query-rewrite: resolving a context-dependent follow-up ----------------


def test_search_news_rewrites_a_follow_up_using_conversation_history(monkeypatch, isolated_subscribers_db):
    """"what about Nvidia?" only makes sense with the conversation above
    it -- the rewrite step must resolve it into a standalone topic before
    anything downstream (definition generation, embedding, the
    no-results message) ever sees the raw follow-up phrasing."""
    monkeypatch.setattr(news_cache, "read_all", lambda: [])
    guard_model = FakeToolCallingModel(responses=[AIMessage(content="Nvidia")])
    history = [HumanMessage(content="What's new in AI chips?"), AIMessage(content="...")]

    result = agent.search_news(1, "what about Nvidia?", history, MagicMock(), guard_model, None)

    assert result == 'No related news found for "Nvidia".'


def test_search_news_skips_rewrite_when_guard_model_is_none(monkeypatch, isolated_subscribers_db):
    monkeypatch.setattr(news_cache, "read_all", lambda: [])

    result = agent.search_news(1, "what about Nvidia?", [], MagicMock(), None, None)

    assert result == 'No related news found for "what about Nvidia?".'


def test_search_news_rewrite_failure_falls_back_to_the_raw_query(monkeypatch, isolated_subscribers_db):
    """Degrades to the unresolved raw query rather than crashing -- an
    unresolved follow-up is a worse search, not a broken one; same
    fail-open shape as the rest of this pipeline."""
    monkeypatch.setattr(news_cache, "read_all", lambda: [])
    # Isolated from definition generation, which isn't what this test is
    # about -- guard_model's generic MagicMock().invoke() would otherwise
    # also stand in for expand_interest_for_retrieval's own internal
    # with_structured_output(...).invoke() call, returning an
    # un-storable MagicMock instead of a real string.
    monkeypatch.setattr(news_classify, "expand_interest_for_retrieval", lambda model, interest: None)
    guard_model = MagicMock()
    guard_model.invoke.side_effect = RuntimeError("simulated model failure")

    result = agent.search_news(1, "what about Nvidia?", [], MagicMock(), guard_model, None)

    assert result == 'No related news found for "what about Nvidia?".'


def test_run_agent_no_tool_call_direct_answer():
    fake_model = FakeToolCallingModel(responses=[AIMessage(content="Hi there!")])
    built = agent.build_agent(fake_model)

    result = agent.run_agent(built, [{"role": "user", "content": "hello"}])

    assert result[-1].content == "Hi there!"
    assert not any(isinstance(m, ToolMessage) for m in result)


def test_run_agent_records_callback_events():
    fake_model = FakeToolCallingModel(responses=[AIMessage(content="Hi there!")])
    built = agent.build_agent(fake_model)
    recorder = RecordingCallbackHandler()

    agent.run_agent(built, [{"role": "user", "content": "hello"}], callbacks=[recorder])

    event_types = [e["type"] for e in recorder.events]
    assert "llm_start" in event_types
    assert "llm_end" in event_types


def _normalized(english, narrower=()):
    """What normalize_interest_detailed returns. Built as the real model so
    a field added to it shows up here rather than being silently absent."""
    return news_classify.NormalizedInterest(
        reasoning="", english=english, is_umbrella=bool(narrower),
        narrower_examples=list(narrower))


def test_add_one_interest_stores_the_english_form(isolated_subscribers_db, monkeypatch):
    """Interest text is a live search query, a BM25 match target and a
    classification input, and all three are English-facing -- gnews and
    newsapi both pin lang=en, so a Chinese interest returns nothing at
    all, and BM25 scored 0% recall for 光通訊 against an English corpus."""
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("Optical Communications"))

    agent.add_one_interest(7, "光通訊", "fake", [], "definition")

    assert subscriber_ops.get_interests(7) == ["Optical Communications"]


def test_add_one_interest_passes_existing_interests_as_context(isolated_subscribers_db, monkeypatch):
    seen = {}

    def fake(model, text, alongside=None):
        seen["alongside"] = alongside
        return _normalized("Automated Optical Inspection")

    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed", fake)
    subscriber_ops.add_interest(7, "AAOI")

    agent.add_one_interest(7, "AOI", "fake", ["AAOI"], "definition")

    assert seen["alongside"] == ["AAOI"]


def test_add_one_interest_falls_back_to_the_original_when_normalization_fails(
    isolated_subscribers_db, monkeypatch
):
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: None)

    agent.add_one_interest(7, "光通訊", "fake", [], "definition")

    assert subscriber_ops.get_interests(7) == ["光通訊"], "stored, just not translated"


def test_add_one_interest_without_a_model_stores_the_raw_topic(isolated_subscribers_db):
    """Stays usable without a model -- tests and the CLI both exercise it
    that way."""
    agent.add_one_interest(7, "robotics", None, [], "definition")

    assert subscriber_ops.get_interests(7) == ["robotics"]


def test_add_one_interest_confirmation_names_what_was_actually_stored(
    isolated_subscribers_db, monkeypatch
):
    """The confirmation said "Added 光通訊" while the database held
    "Optical Communications" -- the opposite of the reason for normalizing
    in the open, and the first place the subscriber would have seen how
    they were understood."""
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("Optical Communications"))

    reply = agent.add_one_interest(7, "光通訊", "fake", [], "definition")

    assert "Optical Communications" in reply
    assert "光通訊" not in reply
    assert subscriber_ops.get_interests(7) == ["Optical Communications"]


def test_duplicate_interest_also_names_the_stored_form(
    isolated_subscribers_db, monkeypatch
):
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("Optical Communications"))
    subscriber_ops.add_interest(7, "Optical Communications")

    reply = agent.add_one_interest(7, "光通訊", "fake", ["Optical Communications"], "definition")

    assert "Optical Communications" in reply
    assert "already have" in reply


# --- the confirmed definition is written to the subscriber's own tier ----
# (docs/plans/interest-finder-plan.md's front-door redesign, 2026-09-10):
# add_one_interest no longer generates a definition blindly -- the caller
# always has one the subscriber already saw and confirmed via
# propose_interest's baked-in preview.

def test_add_one_interest_stores_the_given_definition_in_the_subscribers_own_tier(
    isolated_subscribers_db
):
    agent.add_one_interest(7, "robotics", None, [], "hands-on robotics projects and demos")

    assert interest_cache_ops.get_subscriber_interest_definition(7, "robotics") == \
        "hands-on robotics projects and demos"
    # Interests are not shared (2026-09-10 direction) -- confirming one
    # must never write the shared/global default other subscribers fall
    # back to.
    assert interest_cache_ops.get_interest_query_expansion("robotics") is None


def test_add_one_interest_does_not_touch_the_definition_when_already_following(
    isolated_subscribers_db
):
    """An "already have it" reply means nothing was added -- overwriting
    an existing definition here would let a stray re-add clobber a
    deliberate prior refinement; that is execute_redefine's job."""
    subscriber_ops.add_interest(7, "robotics")
    interest_cache_ops.set_subscriber_interest_definition(7, "robotics", "the existing definition")

    reply = agent.add_one_interest(7, "robotics", None, ["robotics"], "a completely different definition")

    assert "already have" in reply
    assert interest_cache_ops.get_subscriber_interest_definition(7, "robotics") == "the existing definition"


def test_add_one_interest_does_not_store_a_definition_when_the_cap_refuses_it(
    isolated_subscribers_db
):
    topics = [f"topic {i}" for i in range(subscriber_ops.MAX_INTERESTS)]
    subscriber_ops.set_interests(7, topics)

    agent.add_one_interest(7, "one more", None, list(topics), "definition")

    assert interest_cache_ops.get_subscriber_interest_definition(7, "one more") is None


# --- breadth hint and the interest cap ----------------------------------

def test_a_broad_interest_is_stored_and_hinted_not_refused(
    isolated_subscribers_db, monkeypatch
):
    """A hint, never a question: asking would need "this subscriber owes me
    an answer" state that the next message would otherwise route straight
    past. The broad interest is still stored -- the subscriber asked for
    it."""
    monkeypatch.setattr(
        agent.news_classify, "normalize_interest_detailed",
        lambda model, text, alongside=None: _normalized(
            "AI", narrower=["AI Agent", "AI Coding", "Local LLM"]))

    reply = agent.add_one_interest(7, "AI", "fake", [], "definition")

    assert subscriber_ops.get_interests(7) == ["AI"]
    assert "Added AI to your interests." in reply
    assert "AI Agent" in reply and "Local LLM" in reply


def test_a_specific_interest_gets_no_hint(isolated_subscribers_db, monkeypatch):
    monkeypatch.setattr(
        agent.news_classify, "normalize_interest_detailed",
        lambda model, text, alongside=None: _normalized("Local LLM"))

    reply = agent.add_one_interest(7, "local llm", "fake", [], "definition")

    assert reply == "Added Local LLM to your interests."


def test_at_most_three_narrower_examples_are_offered(isolated_subscribers_db, monkeypatch):
    """The model is asked for 2-4 and could return more; a confirmation that
    lists eight alternatives stops reading as a hint."""
    monkeypatch.setattr(
        agent.news_classify, "normalize_interest_detailed",
        lambda model, text, alongside=None: _normalized(
            "AI", narrower=[f"Thing {i}" for i in range(8)]))

    reply = agent.add_one_interest(7, "AI", "fake", [], "definition")

    assert "Thing 2" in reply
    assert "Thing 3" not in reply


def test_adding_past_the_cap_is_refused_in_words(isolated_subscribers_db):
    topics = [f"topic {i}" for i in range(subscriber_ops.MAX_INTERESTS)]
    subscriber_ops.set_interests(7, topics)

    reply = agent.add_one_interest(7, "one more", None, list(topics), "definition")

    assert "one more" in reply
    assert str(subscriber_ops.MAX_INTERESTS) in reply
    assert "one more" not in subscriber_ops.get_interests(7)
    assert len(subscriber_ops.get_interests(7)) == subscriber_ops.MAX_INTERESTS


def test_re_adding_an_existing_interest_at_the_cap_is_not_an_error(isolated_subscribers_db):
    """Being at the cap must not turn a no-op into a failure message."""
    topics = [f"topic {i}" for i in range(subscriber_ops.MAX_INTERESTS)]
    subscriber_ops.set_interests(7, topics)

    reply = agent.add_one_interest(7, "topic 3", None, list(topics), "definition")

    assert "already have" in reply
    assert subscriber_ops.get_interests(7) == topics


def test_narrower_examples_without_the_umbrella_verdict_are_ignored(
    isolated_subscribers_db, monkeypatch
):
    """The measured failure mode: asked only for narrower readings, the live
    model produced them for "Local LLM", "AI Agent" and "Optical
    communications" too. The explicit verdict is what gates the hint, so a
    populated list on its own must not be enough."""
    monkeypatch.setattr(
        agent.news_classify, "normalize_interest_detailed",
        lambda model, text, alongside=None: news_classify.NormalizedInterest(
            reasoning="", english="Local LLM", is_umbrella=False,
            narrower_examples=["On-device AI models", "Edge inference LLM"]))

    reply = agent.add_one_interest(7, "local llm", "fake", [], "definition")

    assert reply == "Added Local LLM to your interests."


def test_normalize_interest_detailed_is_never_given_a_list_that_later_mutates(
    isolated_subscribers_db, monkeypatch
):
    """Regression for a bug caught while writing this fix: `known` used to
    be passed BY REFERENCE and then mutated in place after the call, so a
    caller holding onto `alongside` (a test double, or any future
    consumer) would see later topics leak into what should be a snapshot
    of state at call time."""
    captured = []
    monkeypatch.setattr(
        agent.news_classify, "normalize_interest_detailed",
        lambda model, text, alongside=None: (captured.append(alongside), _normalized(text))[1])
    known = ["AAOI"]

    agent.add_one_interest(7, "AOI", "fake", known, "definition")
    known.append("something added after the call")

    assert captured[0] == ["AAOI"], "must not see the later mutation of `known`"

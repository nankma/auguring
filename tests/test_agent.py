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
    # Route B (set_interest/remove_interest/start_push/stop_push/
    # set_language) is dispatched directly by agent.dispatch_settings now
    # -- the agent loop, and therefore this prompt, only ever runs for
    # news_query. The `category` context key no longer selects anything
    # here; this just confirms that stays true regardless of what's passed.
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


def test_dispatch_settings_set_interest_adds_new_topic(isolated_subscribers_db):
    result = agent.dispatch_settings("set_interest", 201, _classification("set_interest", topics=["robotics"]))
    assert "Added robotics" in result
    assert subscriber_ops.get_interests(201) == ["robotics"]


def test_dispatch_settings_set_interest_already_covered(isolated_subscribers_db):
    subscriber_ops.set_interests(202, ["robotics"])
    result = agent.dispatch_settings("set_interest", 202, _classification("set_interest", topics=["robotics"]))
    assert "already have robotics" in result
    assert subscriber_ops.get_interests(202) == ["robotics"]


def test_dispatch_settings_remove_interest_removes_existing(isolated_subscribers_db):
    subscriber_ops.set_interests(203, ["robotics", "AI"])
    result = agent.dispatch_settings("remove_interest", 203, _classification("remove_interest", topics=["robotics"]))
    assert "Removed robotics" in result
    assert subscriber_ops.get_interests(203) == ["AI"]


def test_dispatch_settings_remove_interest_not_present(isolated_subscribers_db):
    result = agent.dispatch_settings("remove_interest", 204, _classification("remove_interest", topics=["robotics"]))
    assert "wasn't in your interests" in result


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


def test_dispatch_settings_set_language_sets_new_language(isolated_subscribers_db):
    result = agent.dispatch_settings("set_language", 209, _classification("set_language", language="Spanish"))
    assert "Spanish" in result
    assert subscriber_ops.get_language(209) == "Spanish"


def test_dispatch_settings_set_language_reports_current_when_none_named(isolated_subscribers_db):
    subscriber_ops.set_language(210, "French")
    result = agent.dispatch_settings("set_language", 210, _classification("set_language"))
    assert "currently set to French" in result
    assert subscriber_ops.get_language(210) == "French"  # unchanged


def test_dispatch_settings_set_language_reports_unset_when_none_named_and_unset(isolated_subscribers_db):
    result = agent.dispatch_settings("set_language", 211, _classification("set_language"))
    assert "No reply language is set" in result


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


def test_set_interest_stores_the_english_form(isolated_subscribers_db, monkeypatch):
    """Interest text is a live search query, a BM25 match target and a
    classification input, and all three are English-facing -- gnews and
    newsapi both pin lang=en, so a Chinese interest returns nothing at
    all, and BM25 scored 0% recall for 光通訊 against an English corpus."""
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("Optical Communications"))
    classification = SimpleNamespace(topics=["光通訊"])

    agent.dispatch_settings("set_interest", 7, classification, model="fake")

    assert subscriber_ops.get_interests(7) == ["Optical Communications"]


def test_set_interest_passes_existing_interests_as_context(isolated_subscribers_db, monkeypatch):
    seen = {}

    def fake(model, text, alongside=None):
        seen["alongside"] = alongside
        return _normalized("Automated Optical Inspection")

    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed", fake)
    subscriber_ops.add_interest(7, "AAOI")

    agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["AOI"]), model="fake")

    assert seen["alongside"] == ["AAOI"]


def test_set_interest_falls_back_to_the_original_when_normalization_fails(
    isolated_subscribers_db, monkeypatch
):
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: None)

    agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["光通訊"]), model="fake")

    assert subscriber_ops.get_interests(7) == ["光通訊"], "stored, just not translated"


def test_set_interest_without_a_model_stores_the_raw_topic(isolated_subscribers_db):
    """The settings path stays usable without a model -- tests and the CLI
    both exercise it that way."""
    agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["robotics"]))

    assert subscriber_ops.get_interests(7) == ["robotics"]


def test_set_interest_confirmation_names_what_was_actually_stored(
    isolated_subscribers_db, monkeypatch
):
    """The confirmation said "Added 光通訊" while the database held
    "Optical Communications" -- the opposite of the reason for normalizing
    in the open, and the first place the subscriber would have seen how
    they were understood. The four earlier tests all asserted on
    get_interests and none on the reply, which is why it survived."""
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("Optical Communications"))

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["光通訊"]),
                                    model="fake")

    assert "Optical Communications" in reply
    assert "光通訊" not in reply
    assert subscriber_ops.get_interests(7) == ["Optical Communications"]


def test_duplicate_interest_message_also_names_the_stored_form(
    isolated_subscribers_db, monkeypatch
):
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("Optical Communications"))
    subscriber_ops.add_interest(7, "Optical Communications")

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["光通訊"]),
                                    model="fake")

    assert "Optical Communications" in reply
    assert "already have" in reply


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

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["AI"]),
                                    model="fake")

    assert subscriber_ops.get_interests(7) == ["AI"]
    assert "Added AI to your interests." in reply
    assert "AI Agent" in reply and "Local LLM" in reply


def test_a_specific_interest_gets_no_hint(isolated_subscribers_db, monkeypatch):
    monkeypatch.setattr(
        agent.news_classify, "normalize_interest_detailed",
        lambda model, text, alongside=None: _normalized("Local LLM"))

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["local llm"]),
                                    model="fake")

    assert reply == "Added Local LLM to your interests."


def test_at_most_three_narrower_examples_are_offered(isolated_subscribers_db, monkeypatch):
    """The model is asked for 2-4 and could return more; a confirmation that
    lists eight alternatives stops reading as a hint."""
    monkeypatch.setattr(
        agent.news_classify, "normalize_interest_detailed",
        lambda model, text, alongside=None: _normalized(
            "AI", narrower=[f"Thing {i}" for i in range(8)]))

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["AI"]),
                                    model="fake")

    assert "Thing 2" in reply
    assert "Thing 3" not in reply


def test_adding_past_the_cap_is_refused_in_words(isolated_subscribers_db):
    subscriber_ops.set_interests(7, [f"topic {i}" for i in range(subscriber_ops.MAX_INTERESTS)])

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["one more"]))

    assert "one more" in reply
    assert str(subscriber_ops.MAX_INTERESTS) in reply
    assert "one more" not in subscriber_ops.get_interests(7)
    assert len(subscriber_ops.get_interests(7)) == subscriber_ops.MAX_INTERESTS


def test_re_adding_an_existing_interest_at_the_cap_is_not_an_error(isolated_subscribers_db):
    """Being at the cap must not turn a no-op into a failure message."""
    topics = [f"topic {i}" for i in range(subscriber_ops.MAX_INTERESTS)]
    subscriber_ops.set_interests(7, topics)

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["topic 3"]))

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

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["local llm"]),
                                    model="fake")

    assert reply == "Added Local LLM to your interests."


# --- multi-topic set_interest / remove_interest --------------------------
# The 2026-08-25 bug: MessageClassification.topic used to be a single
# string, so "Add AI agent, AI coding, LLM" had no way to become three
# interests. Measured live: the router sometimes joined them into one
# garbled entry, sometimes silently dropped everything but one item, and
# normalize_interest_detailed sometimes compressed the whole request down
# to "AI" -- which then fuzzy-duplicate-matched an existing "AI" interest
# and reported nothing was added, exactly what the user hit.

def test_set_interest_with_multiple_topics_adds_each_one(
    isolated_subscribers_db, monkeypatch
):
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized(text))

    reply = agent.dispatch_settings(
        "set_interest", 7,
        SimpleNamespace(topics=["AI agent", "AI coding", "LLM"]), model="fake")

    assert subscriber_ops.get_interests(7) == ["AI agent", "AI coding", "LLM"]
    assert "Added AI agent" in reply
    assert "Added AI coding" in reply
    assert "Added LLM" in reply


def test_a_later_topic_in_the_same_message_sees_earlier_ones_as_context(
    isolated_subscribers_db, monkeypatch
):
    """`known` grows as topics resolve within one message, not just across
    messages -- the second item in "add AAOI, AOI" should get to
    disambiguate against the first even though neither was stored yet when
    the message arrived."""
    seen_alongside = []

    def fake(model, text, alongside=None):
        seen_alongside.append(list(alongside or []))
        return _normalized(text)

    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed", fake)

    agent.dispatch_settings(
        "set_interest", 7, SimpleNamespace(topics=["AAOI", "AOI"]), model="fake")

    assert seen_alongside[0] == []
    assert seen_alongside[1] == ["AAOI"]


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

    agent.dispatch_settings(
        "set_interest", 7, SimpleNamespace(topics=["AAOI", "AOI"]), model="fake")

    assert captured[0] == [], "must still be empty, not mutated by the second topic's add"


def test_one_topic_hitting_the_cap_does_not_stop_the_others(
    isolated_subscribers_db, monkeypatch
):
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized(text))
    subscriber_ops.set_interests(7, [f"x{i}" for i in range(subscriber_ops.MAX_INTERESTS - 1)])

    reply = agent.dispatch_settings(
        "set_interest", 7, SimpleNamespace(topics=["room for one", "no room for two"]),
        model="fake")

    assert "Added room for one" in reply
    assert "Couldn't add no room for two" in reply
    assert subscriber_ops.get_interests(7) == [f"x{i}" for i in range(subscriber_ops.MAX_INTERESTS - 1)] + ["room for one"]


def test_a_cap_refused_topic_is_not_treated_as_known_by_a_later_topic(
    isolated_subscribers_db, monkeypatch
):
    """A topic that failed to store must not appear in the `alongside`
    context handed to the next topic's normalization call -- it was never
    actually added, so it isn't a real interest to disambiguate against."""
    seen_alongside = []

    def fake(model, text, alongside=None):
        seen_alongside.append(list(alongside or []))
        return _normalized(text)

    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed", fake)
    subscriber_ops.set_interests(7, [f"x{i}" for i in range(subscriber_ops.MAX_INTERESTS)])

    agent.dispatch_settings(
        "set_interest", 7, SimpleNamespace(topics=["refused", "second"]), model="fake")

    assert "refused" not in seen_alongside[1]


def test_set_interest_with_no_topics_extracted_does_not_crash(isolated_subscribers_db):
    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=[]))
    assert "Didn't catch" in reply
    assert subscriber_ops.get_interests(7) == []


def test_remove_interest_with_multiple_topics_removes_each_one(isolated_subscribers_db):
    subscriber_ops.set_interests(7, ["AI agent", "AI coding", "LLM", "robotics"])

    reply = agent.dispatch_settings(
        "remove_interest", 7, SimpleNamespace(topics=["AI agent", "LLM"]))

    assert subscriber_ops.get_interests(7) == ["AI coding", "robotics"]
    assert "Removed AI agent" in reply
    assert "Removed LLM" in reply


def test_remove_interest_reports_a_topic_that_was_never_there(isolated_subscribers_db):
    subscriber_ops.set_interests(7, ["robotics"])

    reply = agent.dispatch_settings(
        "remove_interest", 7, SimpleNamespace(topics=["robotics", "nonexistent"]))

    assert "Removed robotics" in reply
    assert "nonexistent wasn't in your interests" in reply
    assert subscriber_ops.get_interests(7) == []


def test_remove_interest_with_no_topics_extracted_does_not_crash(isolated_subscribers_db):
    reply = agent.dispatch_settings("remove_interest", 7, SimpleNamespace(topics=[]))
    assert "Didn't catch" in reply

def test_two_topics_normalizing_to_the_same_label_report_a_duplicate_not_a_double_add(
    isolated_subscribers_db, monkeypatch
):
    """"machine learning" and "ML" both normalize to the same stored label
    within one message. add_interest's duplicate check reads the DB fresh
    each call, which stays in sync with `known` because every successful
    add updates both together -- the second one must report already-have,
    not silently add a second copy."""
    monkeypatch.setattr(
        agent.news_classify, "normalize_interest_detailed",
        lambda model, text, alongside=None: _normalized("Machine Learning"))

    reply = agent.dispatch_settings(
        "set_interest", 7, SimpleNamespace(topics=["machine learning", "ML"]),
        model="fake")

    assert subscriber_ops.get_interests(7) == ["Machine Learning"]
    assert "Added Machine Learning" in reply
    assert "already have Machine Learning" in reply


def test_removing_the_same_topic_twice_in_one_message_is_not_an_error(
    isolated_subscribers_db
):
    subscriber_ops.set_interests(7, ["AI"])

    reply = agent.dispatch_settings(
        "remove_interest", 7, SimpleNamespace(topics=["AI", "AI"]))

    assert subscriber_ops.get_interests(7) == []
    assert "Removed AI" in reply
    assert "AI wasn't in your interests" in reply


# --- retrieval query expansion, cached on first add (2026-08-25) ---------

def test_a_new_interest_gets_its_retrieval_expansion_generated_and_cached(
    isolated_subscribers_db, monkeypatch
):
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("AI coding"))
    expand = MagicMock(return_value="AI systems that assist developers writing code.")
    monkeypatch.setattr(agent.news_classify, "expand_interest_for_retrieval", expand)

    agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["AI coding"]), model="fake-model")

    expand.assert_called_once_with("fake-model", "AI coding")
    assert interest_cache_ops.get_interest_query_expansion("AI coding") == \
        "AI systems that assist developers writing code."


def test_an_already_cached_expansion_is_not_regenerated(isolated_subscribers_db, monkeypatch):
    """The cache is global (interest_cache_ops), not per
    subscriber -- a topic another subscriber already caused to be
    generated must not cost a second LLM call just because a different
    chat_id adds the same normalized topic."""
    interest_cache_ops.set_interest_query_expansion("AI coding", "already cached")
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("AI coding"))
    expand = MagicMock()
    monkeypatch.setattr(agent.news_classify, "expand_interest_for_retrieval", expand)

    agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["AI coding"]), model="fake-model")

    expand.assert_not_called()
    assert interest_cache_ops.get_interest_query_expansion("AI coding") == "already cached"


def test_no_model_means_no_expansion_attempt(isolated_subscribers_db, monkeypatch):
    """Settings path stays usable without a model -- same convention as
    normalize_interest_detailed being skipped when model is None."""
    expand = MagicMock()
    monkeypatch.setattr(agent.news_classify, "expand_interest_for_retrieval", expand)

    agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["AI coding"]))

    expand.assert_not_called()
    assert interest_cache_ops.get_interest_query_expansion("AI coding") is None


def test_expansion_generation_failure_does_not_block_adding_the_interest(
    isolated_subscribers_db, monkeypatch
):
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("AI coding"))
    monkeypatch.setattr(agent.news_classify, "expand_interest_for_retrieval",
                        MagicMock(return_value=None))

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["AI coding"]), model="fake-model")

    assert "Added AI coding" in reply
    assert subscriber_ops.get_interests(7) == ["AI coding"]
    assert interest_cache_ops.get_interest_query_expansion("AI coding") is None


def test_expansion_is_generated_once_even_when_the_add_itself_is_a_duplicate(
    isolated_subscribers_db, monkeypatch
):
    """The expansion cache exists for every future subscriber who adds this
    topic, not just this call -- so it's still worth populating even when
    THIS subscriber already has the interest and the add is a no-op."""
    subscriber_ops.set_interests(7, ["AI coding"])
    monkeypatch.setattr(agent.news_classify, "normalize_interest_detailed",
                        lambda model, text, alongside=None: _normalized("AI coding"))
    expand = MagicMock(return_value="a definition")
    monkeypatch.setattr(agent.news_classify, "expand_interest_for_retrieval", expand)

    reply = agent.dispatch_settings("set_interest", 7, SimpleNamespace(topics=["AI coding"]), model="fake-model")

    assert "already have AI coding" in reply
    expand.assert_called_once()
    assert interest_cache_ops.get_interest_query_expansion("AI coding") == "a definition"

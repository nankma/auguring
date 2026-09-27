from unittest.mock import MagicMock

import news_jev_filter


def _article(title, summary="", link=None):
    return {"title": title, "summary": summary, "link": link or f"https://example.com/{title}"}


def _mock_jev(monkeypatch, return_value=None, side_effect=None):
    mock = MagicMock(return_value=return_value, side_effect=side_effect)
    monkeypatch.setattr(news_jev_filter.jev_client, "ask", mock)
    return mock


def _answers(n: int, related: dict[int, float] | None = None, interesting: dict[int, float] | None = None) -> dict:
    """A full answers dict for n articles, defaulting unrelated/uninteresting
    (both 0) unless overridden per index."""
    related = related or {}
    interesting = interesting or {}
    answers = {}
    for i in range(n):
        answers[f"related_{i}"] = {"score": related.get(i, 0.0)}
        answers[f"interesting_{i}"] = {"score": interesting.get(i, 0.0)}
    return answers


def test_drops_articles_below_the_related_threshold(monkeypatch):
    articles = [_article("on-topic"), _article("borderline"), _article("off-topic")]
    _mock_jev(monkeypatch, return_value=_answers(
        3, related={0: 5, 1: 2, 2: 0}, interesting={0: 1, 1: 1, 2: 1}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert result == [articles[0]]


def test_keeps_articles_at_exactly_the_threshold(monkeypatch):
    articles = [_article("a")]
    _mock_jev(monkeypatch, return_value=_answers(1, related={0: 3}, interesting={0: 0}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert result == [articles[0]]


def test_ranks_by_related_then_interesting_descending(monkeypatch):
    articles = [_article("a"), _article("b"), _article("c")]
    _mock_jev(monkeypatch, return_value=_answers(
        3, related={0: 3, 1: 5, 2: 5}, interesting={0: 3, 1: 1, 2: 2}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    # b and c both beat a on related (primary key); between b/c, c wins on interesting
    assert result == [articles[2], articles[1], articles[0]]


def test_ties_keep_incoming_order(monkeypatch):
    articles = [_article("a"), _article("b"), _article("c")]
    _mock_jev(monkeypatch, return_value=_answers(
        3, related={0: 4, 1: 4, 2: 4}, interesting={0: 2, 1: 2, 2: 2}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert result == articles


def test_caps_to_max_articles_before_asking_jev(monkeypatch):
    articles = [_article(f"a{i}") for i in range(5)]
    mock = _mock_jev(monkeypatch, return_value=_answers(3, related={0: 4, 1: 4, 2: 4}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=3)

    questions = mock.call_args.args[1]
    assert len(questions) == 6  # 2 questions x 3 capped articles
    assert all(a["link"] != articles[3]["link"] and a["link"] != articles[4]["link"] for a in result)


def test_empty_input_returns_empty_without_calling_jev(monkeypatch):
    mock = _mock_jev(monkeypatch)

    result = news_jev_filter.score_and_rank([], "AI agents", "fake-key", max_articles=10)

    assert result == []
    mock.assert_not_called()


def test_fails_open_to_capped_unranked_input_on_jev_error(monkeypatch, capsys):
    articles = [_article(f"a{i}") for i in range(5)]
    _mock_jev(monkeypatch, side_effect=RuntimeError("boom"))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=3)

    assert result == articles[:3]
    assert "Jev article scoring FAILED" in capsys.readouterr().out


def test_each_question_embeds_the_articles_own_text_not_a_shared_indexed_list(monkeypatch):
    """The load-bearing design point: Jev's own documented pattern (a
    shared `state.articles` list, questions referencing "article #i")
    was verified live to have a severe positional bias -- the identical
    article scored 0.95 at position 0 and as low as 0.14 at any later
    position, even in a 10-item batch (see this module's docstring).
    The fix is that each question's own instructions carries that
    article's title directly, and `state` never carries an articles
    list at all. This test exists specifically so nobody "cleans this
    up" back into the shared-list pattern without re-reading why."""
    articles = [_article("Unique headline A"), _article("Unique headline B")]
    mock = _mock_jev(monkeypatch, return_value=_answers(2, related={0: 4, 1: 4}))

    news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    state, questions = mock.call_args.args[0], mock.call_args.args[1]
    assert "articles" not in state
    assert "Unique headline A" in questions["related_0"]["instructions"]
    assert "Unique headline B" in questions["related_1"]["instructions"]


def test_definition_included_in_state_when_given(monkeypatch):
    mock = _mock_jev(monkeypatch, return_value=_answers(1, related={0: 4}))

    news_jev_filter.score_and_rank(
        [_article("a")], "agents", "fake-key", max_articles=10,
        definition="Software agents, not real-estate agents.")

    state = mock.call_args.args[0]
    assert state == {"topic": "agents", "definition": "Software agents, not real-estate agents."}


def test_definition_omitted_from_state_when_not_given(monkeypatch):
    mock = _mock_jev(monkeypatch, return_value=_answers(1, related={0: 4}))

    news_jev_filter.score_and_rank([_article("a")], "agents", "fake-key", max_articles=10)

    state = mock.call_args.args[0]
    assert "definition" not in state


def test_eye_opener_included_when_top_tier_interesting(monkeypatch):
    """A rejected (below-threshold-related) article still gets ONE bonus
    slot if it's genuinely the top interesting tier -- an intentional,
    high-bar exception, not a second-class inclusion path."""
    articles = [_article("kept"), _article("eye-opener"), _article("boring-reject")]
    _mock_jev(monkeypatch, return_value=_answers(
        3, related={0: 5, 1: 1, 2: 0}, interesting={0: 1, 1: 3, 2: 0}))  # 3 == top tier ("Very interesting")

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert result == [articles[0], articles[1]]


def test_no_eye_opener_when_nothing_rejected_reaches_top_interesting_tier(monkeypatch):
    articles = [_article("kept"), _article("mildly-interesting-reject")]
    _mock_jev(monkeypatch, return_value=_answers(
        2, related={0: 5, 1: 1}, interesting={0: 1, 1: 2}))  # 2 < top tier (3)

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert result == [articles[0]]


def test_at_most_one_eye_opener_even_if_several_rejects_qualify(monkeypatch):
    articles = [_article("kept"), _article("reject-a"), _article("reject-b")]
    _mock_jev(monkeypatch, return_value=_answers(
        3, related={0: 5, 1: 1, 2: 1}, interesting={0: 1, 1: 3, 2: 3}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert len(result) == 2
    assert result[0] == articles[0]
    assert result[1] in (articles[1], articles[2])

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
    unless overridden per index."""
    related = related or {}
    interesting = interesting or {}
    answers = {}
    for i in range(n):
        answers[f"related_{i}"] = {"noul": related.get(i, 0.0)}
        answers[f"interesting_{i}"] = {"score": interesting.get(i, 0.0)}
    return answers


def test_drops_unrelated_articles(monkeypatch):
    articles = [_article("on-topic"), _article("off-topic")]
    _mock_jev(monkeypatch, return_value=_answers(2, related={0: 0.9}, interesting={0: 2.0}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert result == [articles[0]]


def test_ranks_by_interestingness_descending(monkeypatch):
    articles = [_article("a"), _article("b"), _article("c")]
    _mock_jev(monkeypatch, return_value=_answers(
        3, related={0: 0.9, 1: 0.9, 2: 0.9}, interesting={0: 1.0, 1: 3.0, 2: 2.0}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert result == [articles[1], articles[2], articles[0]]


def test_ties_keep_incoming_order(monkeypatch):
    articles = [_article("a"), _article("b"), _article("c")]
    _mock_jev(monkeypatch, return_value=_answers(
        3, related={0: 0.9, 1: 0.9, 2: 0.9}, interesting={0: 2.0, 1: 2.0, 2: 2.0}))

    result = news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    assert result == articles


def test_caps_to_max_articles_before_asking_jev(monkeypatch):
    articles = [_article(f"a{i}") for i in range(5)]
    mock = _mock_jev(monkeypatch, return_value=_answers(3, related={0: 0.9, 1: 0.9, 2: 0.9}))

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
    mock = _mock_jev(monkeypatch, return_value=_answers(2, related={0: 0.9, 1: 0.9}))

    news_jev_filter.score_and_rank(articles, "AI agents", "fake-key", max_articles=10)

    state, questions = mock.call_args.args[0], mock.call_args.args[1]
    assert "articles" not in state
    assert "Unique headline A" in questions["related_0"]["instructions"]
    assert "Unique headline B" in questions["related_1"]["instructions"]

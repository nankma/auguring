import pytest
import requests

import jev_client
from tests.fakes import FakeSpan


def test_ask_sends_the_right_request_shape(requests_mock):
    requests_mock.post(
        jev_client.ENDPOINT,
        json={"model": "jev-1.13.0", "answers": {"on_topic": {"type": "noul", "noul": 0.9}}, "usage": {}},
    )

    questions = {"on_topic": {"type": "noul", "instructions": "Is this on topic?"}}
    jev_client.ask({"message": "hello"}, questions, "fake-key")

    request = requests_mock.request_history[0]
    assert request.headers["Authorization"] == "Bearer fake-key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.json() == {
        "model": jev_client.MODEL,
        "state": {"message": "hello"},
        "questions": questions,
    }


def test_ask_returns_the_answers_dict(requests_mock):
    requests_mock.post(
        jev_client.ENDPOINT,
        json={
            "model": "jev-1.13.0",
            "answers": {
                "on_topic": {"type": "noul", "noul": 0.92},
                "is_find_interests": {"type": "noul", "noul": 0.96},
            },
            "usage": {"input_tokens": 371, "output_tokens": 59},
        },
    )

    answers = jev_client.ask({"message": "help me find something"}, {}, "fake-key")

    assert answers == {
        "on_topic": {"type": "noul", "noul": 0.92},
        "is_find_interests": {"type": "noul", "noul": 0.96},
    }


def test_ask_raises_on_non_2xx(requests_mock):
    requests_mock.post(jev_client.ENDPOINT, status_code=401, json={"error": "invalid key"})

    with pytest.raises(requests.HTTPError):
        jev_client.ask({"message": "hello"}, {}, "bad-key")


def test_ask_raises_on_malformed_response(requests_mock):
    """A 200 with no "answers" key -- the caller (guardrails.py) is
    responsible for its own fail-open handling around this, same as it
    already does around every other failure shape."""
    requests_mock.post(jev_client.ENDPOINT, json={"model": "jev-1.13.0"})

    with pytest.raises(KeyError):
        jev_client.ask({"message": "hello"}, {}, "fake-key")


def test_ask_logs_usage_once_per_successful_call(monkeypatch, requests_mock):
    """Centralized here (not per-caller) specifically so real Jev spend
    is queryable without reconstructing it by hand against the live
    endpoint after the fact -- see this module's own docstring."""
    span = FakeSpan()
    monkeypatch.setattr(jev_client._events._tracer, "start_as_current_span", lambda name: span)
    requests_mock.post(
        jev_client.ENDPOINT,
        json={
            "model": "jev-1.13.0",
            "answers": {"on_topic": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 371, "output_tokens": 59, "cost": 0.000123},
        },
    )
    questions = {"on_topic": {"type": "noul", "instructions": "Is this on topic?"}}

    jev_client.ask({"message": "hello"}, questions, "fake-key")

    assert span.attrs["sample_question_id"] == "on_topic"
    assert span.attrs["num_questions"] == 1
    assert span.attrs["input_tokens"] == 371
    assert span.attrs["output_tokens"] == 59
    assert span.attrs["cost"] == 0.000123


def test_ask_logs_a_sample_question_id_to_disambiguate_same_sized_calls(monkeypatch, requests_mock):
    """Regression test for a real gap QA found, 2026-09-26: num_questions
    alone is a coincidental, not reliable, proxy for which caller made
    the call -- e.g. guardrails.is_output_on_topic with no user_text (2
    questions) and news_jev_filter scoring exactly 1 article (also 2
    questions) are indistinguishable by count alone. sample_question_id
    resolves this since each caller's own question-id keys never
    collide."""
    span = FakeSpan()
    monkeypatch.setattr(jev_client._events._tracer, "start_as_current_span", lambda name: span)
    requests_mock.post(
        jev_client.ENDPOINT,
        json={"model": "jev-1.13.0", "answers": {}, "usage": {}},
    )
    layer4_shaped_questions = {
        "discusses_own_configuration": {"type": "noul", "instructions": "?"},
        "appropriate_bot_content": {"type": "noul", "instructions": "?"},
    }

    jev_client.ask({"bot_reply": "..."}, layer4_shaped_questions, "fake-key")

    assert span.attrs["num_questions"] == 2
    assert span.attrs["sample_question_id"] == "appropriate_bot_content"


def test_ask_logs_usage_gracefully_when_the_response_has_no_usage_field(monkeypatch, requests_mock):
    span = FakeSpan()
    monkeypatch.setattr(jev_client._events._tracer, "start_as_current_span", lambda name: span)
    requests_mock.post(
        jev_client.ENDPOINT,
        json={"model": "jev-1.13.0", "answers": {"on_topic": {"type": "noul", "noul": 0.9}}},
    )

    jev_client.ask({"message": "hello"}, {"on_topic": {"type": "noul", "instructions": "?"}}, "fake-key")

    assert span.attrs["input_tokens"] is None
    assert span.attrs["output_tokens"] is None
    assert span.attrs["cost"] is None


def test_ask_does_not_log_usage_when_the_call_fails(monkeypatch, requests_mock):
    span = FakeSpan()
    monkeypatch.setattr(jev_client._events._tracer, "start_as_current_span", lambda name: span)
    requests_mock.post(jev_client.ENDPOINT, status_code=401, json={"error": "invalid key"})

    with pytest.raises(requests.HTTPError):
        jev_client.ask({"message": "hello"}, {}, "bad-key")

    assert span.attrs == {}

import pytest
import requests

import jev_client


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

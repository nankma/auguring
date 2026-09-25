"""
Thin adapter for TypeSafe AI's Jev "System One" typed-decision model,
reached via OpenRouter's alpha decisions endpoint -- not OpenAI-wire-
compatible, so agent.build_model_from_config's ChatOpenAI path can't
reach it (see TODO.md). Used by guardrails.py's two fixed, bounded,
typed-decision layers (layer 2's on-topic/category gate, layer 4's output
check) and by news_jev_filter.py's per-article relatedness/interestingness
scoring (called from inside agent.search_news) -- never for the
conversational agent's own tool-calling loop itself, which needs real
tool-calling and free-text generation Jev doesn't do (see
docs/plans/front-door-agent-plan.md item 5). search_news is reachable AS
A TOOL from that same conversational agent, but the Jev call inside it
is still one fixed, bounded, typed-decision step, not the agent loop
itself making the call.

Verified live against the real endpoint 2026-09-24 before this shipped:
response shape matches docs.typesafe.ai/api.md's documented schema
exactly (`{"answers": {qid: {"type": "noul", "noul": 0.0-1.0}}, ...}`),
~0.17s round trip for a 3-question call.
"""

import requests

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "~typesafe/jev-latest"


def ask(state, questions: dict, api_key: str, timeout: float = 10.0) -> dict:
    """One Jev decision call. `questions` is {question_id: {"type": "noul"
    | "choice" | "score", "instructions": str, "criteria": ...}}; `state`
    is the JSON-serializable context the questions are asked about (a
    dict with descriptive field names, not a bare string, per Jev's own
    docs recommendation for anything with more than one part).

    Returns the raw `answers` dict, keyed the same as `questions` --
    callers read answers[qid]["noul"]/["choice"]/["score"] themselves;
    this function does no interpretation of what the answers mean.

    Raises on any failure (network, non-2xx, malformed response) --
    callers are responsible for their own fail-open handling, same as
    guardrails.classify_message/is_output_on_topic already do around
    their own model calls."""
    response = requests.post(
        ENDPOINT,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={"model": MODEL, "state": state, "questions": questions},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["answers"]

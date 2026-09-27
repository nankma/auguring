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

Every call's `usage` is logged here, once, centrally -- not by each
caller -- specifically so a real question like "how many tokens did
Jev spend today, and on what" has an actual log to answer it from
instead of needing to be reconstructed by hand against the real
endpoint after the fact (see docs/plans/front-door-agent-plan.md's
2026-09-25 cost-investigation note for the reconstruction this
replaces). `ask()`'s own return contract is unchanged (still just
`answers`) so this needed no change at any of its three call sites.
"""

import requests

from telemetry import EventLogger, get_event_logger

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "~typesafe/jev-latest"

_events: EventLogger = get_event_logger("argus.jev_client")


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
    their own model calls. A raised exception means this never reaches
    the usage-logging line below -- a failed call has no usage to log,
    and the caller's own fail-open path already logs the failure itself
    (router_failed/output_check_failed/news_jev_filter_failed)."""
    response = requests.post(
        ENDPOINT,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={"model": MODEL, "state": state, "questions": questions},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage") or {}
    _events.log("jev_call_usage", {
        "message": "Jev call usage",
        # A sample question id, not a caller-supplied label -- this
        # function has no other way to know which caller it's serving,
        # and num_questions alone is an unreliable proxy (guardrails.py's
        # fixed 8/2/3 can collide with news_jev_filter.py's 2*len(articles)
        # at specific article counts). Every caller's own question-id
        # keys are already distinct and stable ("on_topic"/"is_*" only
        # from layer 2, "discusses_own_configuration"/"all_asks_addressed"
        # only from layer 4, "related_*"/"interesting_*" only from
        # news_jev_filter) -- one of them, sorted for determinism,
        # identifies the caller with zero call-site changes.
        "sample_question_id": min(questions, default=None),
        "num_questions": len(questions),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost": usage.get("cost"),
    })
    return body["answers"]

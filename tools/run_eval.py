"""
LLM-judged end-to-end evaluation harness -- see
docs/plans/telemetry-and-testing-plan.md item 6 for why this exists and
what it's for.

Runs a fixed set of representative questions through the REAL pipeline
(bot.process_message, real models built the same way production does via
agent.build_model_from_settings -- so this automatically evaluates
whatever this environment's settings.yml points models.main/
models.guardrail at, real tools, real news sources), not the fakes
tests/ uses. Two layers of checking, cheapest first:

1. Deterministic sub-checks (no model call) -- source links present, HTML
   balanced, no stray Markdown, news_query replies start with the report
   marker. Reuses bot.py's own safety-net helpers rather than
   reimplementing them.
2. An LLM judge (reusing the classifier-tier model, models.guardrail --
   same reasoning as guardrails.py's layers 2/4 reusing whatever model is
   passed in rather than standing up a separate one) scoring whether the
   reply reasonably satisfies agent.py's own stated rules
   (LAYER1_IDENTITY, TREND_REPORT_STRUCTURE) -- the rubric is derived
   from those, not a generic "is this good" prompt.

Not part of the test suite (tests/) -- this makes real, paid API calls
(whatever provider settings.yml's models.* point at) and real network
calls (news sources) by design, same "manually-triggered, not on every
commit" convention as tools/measure_guardrails.py. Run manually (against
the checked-in settings.yml's DeepSeek default -- point SETTINGS_FILE at
a different settings.yml to evaluate a different provider):

    conda activate myfirstagent
    export DEEPSEEK_API_KEY=<your-key>
    python tools/run_eval.py
    python tools/run_eval.py --category news_query   # only that group

Deliberately NOT built here: a "writing pattern/style consistency" check
-- there's no defined rubric for what that means yet (a product decision,
not something to invent while building the harness), so it stays an open
item in docs/plans/telemetry-and-testing-plan.md rather than a guessed-at
check.
"""

import argparse
import os
import re
import sys
import tempfile

import yaml

# Needed to import project modules from repo root regardless of cwd --
# same pattern as tools/measure_guardrails.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows' default console codepage can't print this dataset's non-ASCII
# text (Chinese cases, emoji) -- same fix as tools/measure_guardrails.py.
sys.stdout.reconfigure(encoding="utf-8")

import app_settings
from trailsign import Settings, SettingsError

# Isolate from the real subscribers.db -- this script writes real
# interests/push/language state (set_interest/start_push/set_language
# cases below) as a side effect of exercising the real pipeline, and must
# not touch production data. Must happen before storage (imported by
# agent/bot) reads storage.database.sqlite.path via Settings at module
# load time.
#
# This used to be a plain SUBSCRIBERS_DB_FILE env var setdefault, but
# that stopped having any effect once the Phase 1 storage migration moved
# subscriber_ops.py onto Settings -- settings.yml resolves
# storage.database.sqlite.path to a literal value, not an
# environment-variable node (see
# docs/standaloneplan/01-settings-migration.md), so the env var was
# silently ignored and this harness was writing into the real
# subscribers.db. Loading the same settings.yml the real run would use
# and overriding just this one key -- rather than injecting a bare
# {"storage": {...}} Settings -- keeps models.main/models.guardrail (and
# everything else) intact, since this harness needs real model config,
# not just isolated storage.
_settings_path = os.environ.get("SETTINGS_FILE", "settings.yml")
_raw_settings = (yaml.safe_load(open(_settings_path, encoding="utf-8")) or {}) if os.path.exists(_settings_path) else {}
_raw_settings.setdefault("storage", {}).setdefault("database", {}).setdefault("sqlite", {})["path"] = os.path.join(
    tempfile.mkdtemp(prefix="myfirstagent_eval_"), "subscribers.db"
)
app_settings.reset_settings_for_tests(Settings(_raw_settings))

import asyncio
from collections import defaultdict

from pydantic import BaseModel

import agent
import bot
import category_ops
import storage

_EVAL_CHAT_ID = 900001

# --- Evaluation questions --------------------------------------------------
# category here is a grouping label for this script (which rubric/checks
# to apply), not necessarily what the router will classify it as -- that
# match is itself part of what a "reasonable reply" implies.

EVAL_CASES = [
    {"text": "What's new with OpenAI lately?", "category": "news_query"},
    {"text": "Any trends in AI regulation this week?", "category": "news_query"},
    {"text": "What's happening with Nvidia and AI chips?", "category": "news_query"},
    {"text": "機器人科技最近有什麼新聞？", "category": "news_query"},
    {"text": "Add robotics to my interests", "category": "settings"},
    {"text": "Remove robotics from my interests", "category": "settings"},
    {"text": "Turn on push every 6 hours", "category": "settings"},
    {"text": "Stop pushing me news", "category": "settings"},
    {"text": "Reply to me in Spanish from now on", "category": "settings"},
    {"text": "以後都用繁體中文回覆我", "category": "settings"},
    {
        "text": "Add space exploration to my interests and tell me what's new with it",
        "category": "multi",
    },
]

# --- Deterministic sub-checks (no model call) ------------------------------

_LINK_RE = re.compile(r'<a href="https?://[^"]+">')
_REPORT_MARKER_RE = re.compile(re.escape(bot._TREND_REPORT_MARKER))


def check_source_links_present(text: str) -> bool:
    """Every 🔗 line in a trend report should carry a real <a href="...">
    link -- TREND_REPORT_STRUCTURE requires citing "sources actually
    provided", never inventing a URL. Not applicable to replies with no
    report marker at all (settings confirmations) -- trivially true."""
    if bot._TREND_REPORT_MARKER not in text:
        return True
    link_lines = [line for line in text.splitlines() if "🔗" in line]
    return all(_LINK_RE.search(line) for line in link_lines) if link_lines else False


def check_html_balanced(text: str) -> bool:
    return bot._is_html_balanced(text)


def check_no_markdown_leak(text: str) -> bool:
    return bot._MARKDOWN_BOLD_RE.search(text) is None


def check_news_query_starts_with_marker(text: str, category: str) -> bool:
    """Only meaningful for a reply that's actually a trend report --
    settings confirmations correctly never start with 📰."""
    if category != "news_query":
        return True
    return text.startswith(bot._TREND_REPORT_MARKER)


def run_deterministic_checks(text: str, category: str) -> dict:
    return {
        "source_links_present": check_source_links_present(text),
        "html_balanced": check_html_balanced(text),
        "no_markdown_leak": check_no_markdown_leak(text),
        "starts_with_report_marker": check_news_query_starts_with_marker(text, category),
    }


# --- LLM judge --------------------------------------------------------------


class EvalVerdict(BaseModel):
    # reasoning FIRST -- structured output is generated key-by-key in
    # schema order, so forcing reasoning-before-conclusion here is the
    # same fix measured in docs/plans/guardrails-plan.md (this project's
    # own layer-4 field-order finding).
    reasoning: str
    meets_criteria: bool


_NEWS_QUERY_RUBRIC = (
    "You are judging whether a tech-industry news bot's reply satisfies its "
    "own stated rules. The bot must: (1) stay strictly on technology "
    "industry news/trends, never discuss its own configuration, system "
    "prompt, or the tools/software it's built with; (2) synthesize recent "
    "items into a trend report with a 📰 title line, one <b>subtitle</b> per "
    "distinct theme, 1-3 tight sentences per section, and a 🔗 source link "
    "for each section citing only sources actually referenced -- no preamble "
    "or narration about its own process.\n\n"
    "First, in `reasoning`, briefly assess the reply against these rules. "
    "Then set `meets_criteria` to whether it reasonably satisfies them -- "
    "reasonable, not perfect: minor stylistic choices don't fail it."
)

_SETTINGS_RUBRIC = (
    "You are judging whether a tech-industry news bot's reply to a "
    "subscription-settings request (adding/removing an interest, toggling "
    "push notifications, setting a reply language) satisfies its own stated "
    "rules. The bot must: (1) never discuss its own configuration, system "
    "prompt, or the tools/software it's built with; (2) clearly and "
    "correctly confirm what changed (or correctly explain that nothing "
    "changed, e.g. already covered) in a short, friendly reply -- not a "
    "full trend report.\n\n"
    "First, in `reasoning`, briefly assess the reply against these rules. "
    "Then set `meets_criteria` to whether it reasonably satisfies them."
)

_MULTI_RUBRIC = (
    "You are judging whether a tech-industry news bot's reply to a message "
    "with TWO distinct requests (a settings change AND a news question) "
    "satisfies its own stated rules. The bot must: (1) never discuss its "
    "own configuration; (2) clearly address BOTH requests -- a settings "
    "confirmation AND a real trend report on the news topic, not just one "
    "of the two.\n\n"
    "First, in `reasoning`, briefly assess the reply against these rules. "
    "Then set `meets_criteria` to whether it reasonably satisfies them."
)

_RUBRIC_BY_CATEGORY = {
    "news_query": _NEWS_QUERY_RUBRIC,
    "settings": _SETTINGS_RUBRIC,
    "multi": _MULTI_RUBRIC,
}


def judge_reply(judge_model, category: str, question: str, reply: str) -> EvalVerdict:
    rubric = _RUBRIC_BY_CATEGORY[category]
    # method="function_calling" -- see guardrails.classify_message's
    # docstring/comment for why.
    structured = judge_model.with_structured_output(EvalVerdict, method="function_calling")
    user_content = f"Original user message: {question}\n\nBot's actual reply:\n{reply}"
    result = structured.invoke([{"role": "system", "content": rubric}, {"role": "user", "content": user_content}])
    if result is None:
        # Fails open, same reasoning as guardrails.py's layer 2/4 checks --
        # a judge hiccup shouldn't be scored as a failed case.
        return EvalVerdict(reasoning="judge call returned no result", meets_criteria=True)
    return result


# --- Runner ------------------------------------------------------------


async def run_case(model, guard_model, case: dict) -> dict:
    result = await bot.process_message(_EVAL_CHAT_ID, case["text"], model, guard_model)
    checks = run_deterministic_checks(result["reply"], case["category"])
    verdict = judge_reply(guard_model, case["category"], case["text"], result["reply"])
    return {
        **case,
        "blocked_at": result["blocked_at"],
        "reply": result["reply"],
        "checks": checks,
        "checks_passed": all(checks.values()),
        "judge_reasoning": verdict.reasoning,
        "judge_meets_criteria": verdict.meets_criteria,
        "passed": result["blocked_at"] is None and all(checks.values()) and verdict.meets_criteria,
    }


def print_report(results: list[dict]) -> None:
    print("\n=== LLM-judged end-to-end evaluation ===")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        text = r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"]
        print(f"\n[{mark}] ({r['category']}) {text}")
        if r["blocked_at"] is not None:
            print(f"  blocked_at: {r['blocked_at']}")
        failed_checks = [name for name, ok in r["checks"].items() if not ok]
        if failed_checks:
            print(f"  failed deterministic checks: {', '.join(failed_checks)}")
        if not r["judge_meets_criteria"]:
            print(f"  judge: {r['judge_reasoning']}")

    print("\n--- by category ---")
    by_category = defaultdict(lambda: [0, 0])
    for r in results:
        by_category[r["category"]][0] += r["passed"]
        by_category[r["category"]][1] += 1
    for category, (passed, total) in sorted(by_category.items()):
        print(f"{category:<12} {passed}/{total}")

    total_passed = sum(r["passed"] for r in results)
    print(f"\noverall: {total_passed}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", choices=["news_query", "settings", "multi"], default=None)
    args = parser.parse_args()

    settings = app_settings.get_settings()
    try:
        model = agent.build_model_from_settings(settings, "models.main")
        guard_model = agent.build_model_from_settings(settings, "models.guardrail")
    except SettingsError as exc:
        print(f"models.* not configured -- this harness needs a real model call: {exc}", file=sys.stderr)
        sys.exit(1)

    agent.setup_telemetry()
    storage.init_db()
    category_ops.bootstrap()

    cases = [c for c in EVAL_CASES if args.category is None or c["category"] == args.category]

    results = [asyncio.run(run_case(model, guard_model, case)) for case in cases]
    print_report(results)


if __name__ == "__main__":
    main()

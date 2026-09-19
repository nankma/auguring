"""
The outcome vocabulary news_push.py records for one subscriber's turn in
one push cycle. No storage of its own -- retired 2026-09-04 along with the
push_outcomes table it used to back (see git history): the only thing that
still needed a per-cycle read was the "how many consecutive chat_not_found
cycles" check, replaced by subscriber_ops.record_push_failure/
reset_push_consecutive_failures (a per-subscriber counter, not a scannable
history). The delivered/generated ratio (docs/plans/incident-monitoring-plan.md
criterion 3) is read from the Logfire span news_push._record already
emits, not from here -- see that function's own docstring.
"""

PUSH_DELIVERED = "delivered"
PUSH_NOTHING_NEW = "nothing_new"
PUSH_NOT_RELEVANT = "not_relevant"
PUSH_BLOCKED = "blocked"
PUSH_NO_INTERESTS = "no_interests"
PUSH_CHAT_NOT_FOUND = "chat_not_found"
PUSH_MODEL_ERROR = "model_error"
PUSH_CYCLE_FAILED = "cycle_failed"
PUSH_DISABLED = "disabled"
# A subscriber's free-trial push allowance ran out (requested 2026-09-18).
# Excluded from PUSH_GENERATED_OUTCOMES below for the same reason
# no_interests is: news_push._stop_push_at_trial_limit returns before
# write_push_digest, so no digest was paid for this cycle.
PUSH_TRIAL_LIMIT_REACHED = "trial_limit_reached"

# Outcomes meaning an LLM was actually called to write a digest -- the
# denominator of the delivered/generated ratio (criterion 3,
# docs/plans/incident-monitoring-plan.md). nothing_new/no_interests are
# excluded: they return before write_push_digest, so no digest was paid
# for, and counting them would let idle subscribers hide a collapsed
# delivery rate. Still used by news_push._record to compute the
# `push.generated` span attribute the Logfire alert reads.
PUSH_GENERATED_OUTCOMES = frozenset({
    PUSH_DELIVERED,
    PUSH_NOT_RELEVANT,
    PUSH_BLOCKED,
    PUSH_CHAT_NOT_FOUND,
    PUSH_MODEL_ERROR,
})

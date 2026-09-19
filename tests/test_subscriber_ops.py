import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

import storage
import subscriber_ops


def test_get_status_unknown_chat_returns_none(isolated_subscribers_db):
    assert subscriber_ops.get_status(12345) is None


def test_request_access_sets_pending(isolated_subscribers_db):
    subscriber_ops.request_access(1, "alice", "Alice")
    assert subscriber_ops.get_status(1) == subscriber_ops.PENDING


def test_request_access_does_not_reset_a_decision(isolated_subscribers_db):
    subscriber_ops.request_access(1, "alice", "Alice")
    subscriber_ops.decide(1, approved=True)
    subscriber_ops.request_access(1, "alice", "Alice")  # re-message after decision
    assert subscriber_ops.get_status(1) == subscriber_ops.APPROVED


def test_decide_approve(isolated_subscribers_db):
    subscriber_ops.request_access(2, "bob", "Bob")
    subscriber_ops.decide(2, approved=True)
    assert subscriber_ops.get_status(2) == subscriber_ops.APPROVED


def test_decide_deny(isolated_subscribers_db):
    subscriber_ops.request_access(3, "carol", "Carol")
    subscriber_ops.decide(3, approved=False)
    assert subscriber_ops.get_status(3) == subscriber_ops.DENIED


def test_list_pending_only_returns_pending(isolated_subscribers_db):
    subscriber_ops.request_access(4, "dave", "Dave")
    subscriber_ops.request_access(5, "erin", "Erin")
    subscriber_ops.decide(5, approved=True)
    pending = subscriber_ops.list_pending()
    assert [row[0] for row in pending] == [4]


def test_get_interests_empty_for_unset_chat(isolated_subscribers_db):
    assert subscriber_ops.get_interests(6) == []


def test_set_and_get_interests(isolated_subscribers_db):
    subscriber_ops.request_access(7, "frank", "Frank")
    subscriber_ops.set_interests(7, ["AI", "robotics"])
    assert subscriber_ops.get_interests(7) == ["AI", "robotics"]


def test_set_interests_upserts_when_no_existing_row(isolated_subscribers_db):
    # e.g. the admin, who never goes through request_access()
    subscriber_ops.set_interests(999, ["semiconductors"])
    assert subscriber_ops.get_interests(999) == ["semiconductors"]
    assert subscriber_ops.get_status(999) == subscriber_ops.APPROVED


def test_set_interests_overwrites_previous_value(isolated_subscribers_db):
    subscriber_ops.request_access(8, "grace", "Grace")
    subscriber_ops.set_interests(8, ["AI"])
    subscriber_ops.set_interests(8, ["quantum computing"])
    assert subscriber_ops.get_interests(8) == ["quantum computing"]


def test_init_db_migrates_schema_missing_interests_column(isolated_subscribers_db):
    # Simulate a DB created before the interests column existed.
    with sqlite3.connect(isolated_subscribers_db) as conn:
        conn.execute("DROP TABLE subscribers")
        conn.execute(
            """
            CREATE TABLE subscribers (
                chat_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                status TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                decided_at TEXT
            )
            """
        )
    storage.init_db()  # should migrate without error
    subscriber_ops.request_access(9, "henry", "Henry")
    subscriber_ops.set_interests(9, ["AI"])
    assert subscriber_ops.get_interests(9) == ["AI"]


def test_add_interest_appends(isolated_subscribers_db):
    subscriber_ops.request_access(10, "iris", "Iris")
    subscriber_ops.set_interests(10, ["AI"])
    result = subscriber_ops.add_interest(10, "robotics")
    assert result == ["AI", "robotics"]
    assert subscriber_ops.get_interests(10) == ["AI", "robotics"]


def test_add_interest_is_idempotent_case_insensitive(isolated_subscribers_db):
    subscriber_ops.request_access(11, "jack", "Jack")
    subscriber_ops.set_interests(11, ["AI"])
    result = subscriber_ops.add_interest(11, "ai")  # different case, same topic
    assert result == ["AI"]  # not duplicated


def test_add_interest_catches_near_duplicate_llm_phrasing(isolated_subscribers_db):
    # Real incident, 2026-08-08: the agent phrased the same conceptual
    # interest two different ways on two calls; exact-match dedup missed
    # it, leaving both in the list.
    subscriber_ops.request_access(23, "ruth", "Ruth")
    subscriber_ops.set_interests(23, ["Edge AI development boards (Raspberry Pi, NVIDIA Jetson, etc.)"])
    result = subscriber_ops.add_interest(23, "Edge AI development boards (Raspberry Pi, NVIDIA Jetson)")
    assert result == ["Edge AI development boards (Raspberry Pi, NVIDIA Jetson, etc.)"]


def test_add_interest_does_not_dedupe_unrelated_topics_sharing_one_word(isolated_subscribers_db):
    subscriber_ops.request_access(24, "sam", "Sam")
    subscriber_ops.set_interests(24, ["AI"])
    result = subscriber_ops.add_interest(24, "AI regulation")
    assert result == ["AI", "AI regulation"]


def test_add_interest_creates_row_when_none_exists(isolated_subscribers_db):
    result = subscriber_ops.add_interest(999, "semiconductors")
    assert result == ["semiconductors"]
    assert subscriber_ops.get_status(999) == subscriber_ops.APPROVED


def test_remove_interest(isolated_subscribers_db):
    subscriber_ops.request_access(12, "kate", "Kate")
    subscriber_ops.set_interests(12, ["AI", "robotics"])
    result = subscriber_ops.remove_interest(12, "AI")
    assert result == ["robotics"]
    assert subscriber_ops.get_interests(12) == ["robotics"]


def test_remove_interest_not_present_is_a_noop(isolated_subscribers_db):
    subscriber_ops.request_access(13, "liam", "Liam")
    subscriber_ops.set_interests(13, ["AI"])
    result = subscriber_ops.remove_interest(13, "robotics")
    assert result == ["AI"]


def test_get_push_enabled_defaults_false(isolated_subscribers_db):
    assert subscriber_ops.get_push_enabled(14) is False


def test_set_and_get_push_enabled(isolated_subscribers_db):
    subscriber_ops.request_access(15, "mia", "Mia")
    subscriber_ops.set_push_enabled(15, True)
    assert subscriber_ops.get_push_enabled(15) is True
    subscriber_ops.set_push_enabled(15, False)
    assert subscriber_ops.get_push_enabled(15) is False


def test_set_push_enabled_upserts_when_no_existing_row(isolated_subscribers_db):
    subscriber_ops.set_push_enabled(999, True)
    assert subscriber_ops.get_push_enabled(999) is True
    assert subscriber_ops.get_status(999) == subscriber_ops.APPROVED


def test_get_push_interval_hours_defaults(isolated_subscribers_db):
    assert subscriber_ops.get_push_interval_hours(16) == subscriber_ops.DEFAULT_PUSH_INTERVAL_HOURS


def test_set_and_get_push_interval_hours(isolated_subscribers_db):
    subscriber_ops.request_access(16, "noah", "Noah")
    subscriber_ops.set_push_interval_hours(16, 6)
    assert subscriber_ops.get_push_interval_hours(16) == 6


def test_set_push_interval_hours_rejects_below_minimum(isolated_subscribers_db):
    with pytest.raises(ValueError):
        subscriber_ops.set_push_interval_hours(16, 0)


def test_get_pushed_links_empty_for_unset_chat(isolated_subscribers_db):
    assert subscriber_ops.get_pushed_links(17) == []


def test_get_last_push_at_none_for_unset_chat(isolated_subscribers_db):
    assert subscriber_ops.get_last_push_at(17) is None


def test_mark_links_shown_sets_links(isolated_subscribers_db):
    pushed_at = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
    subscriber_ops.mark_links_shown(18, ["https://example.com/a", "https://example.com/b"], pushed_at)
    assert set(subscriber_ops.get_pushed_links(18, now=pushed_at)) == {
        "https://example.com/a", "https://example.com/b"
    }


def test_advance_last_push_at_sets_last_push_at(isolated_subscribers_db):
    """mark_links_shown and advance_last_push_at are two independent
    methods (2026-09-04 split) -- a manual search must be able to call
    mark_links_shown without also advancing the subscriber's push clock,
    so this confirms advance_last_push_at alone does its one job."""
    pushed_at = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
    subscriber_ops.advance_last_push_at(18, pushed_at)
    assert subscriber_ops.get_last_push_at(18) == pushed_at
    assert subscriber_ops.get_pushed_links(18, now=pushed_at) == []


def test_mark_links_shown_merges_and_dedupes_links(isolated_subscribers_db):
    t1 = datetime(2026, 8, 8, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 8, 12, tzinfo=timezone.utc)
    subscriber_ops.mark_links_shown(19, ["https://example.com/old"], t1)
    subscriber_ops.mark_links_shown(19, ["https://example.com/new", "https://example.com/old"], t2)
    assert set(subscriber_ops.get_pushed_links(19, now=t2)) == {
        "https://example.com/old", "https://example.com/new"
    }


def test_pushed_links_are_pruned_by_age_not_by_count(isolated_subscribers_db):
    """The count cap this replaced was silently wrong at short push
    intervals -- see subscriber_ops.PUSHED_LINK_RETENTION_HOURS."""
    recent = datetime(2026, 8, 8, tzinfo=timezone.utc)
    # sent a day ago -- comfortably inside the retention window
    subscriber_ops.mark_links_shown(30, ["https://example.com/yesterday"], recent - timedelta(hours=24))

    # far more links than the old 200-entry cap, all in one later push
    subscriber_ops.mark_links_shown(30, [f"https://example.com/{i}" for i in range(500)], recent)

    links = subscriber_ops.get_pushed_links(30, now=recent)
    assert len(links) == 501, "nothing should be evicted just for being numerous"
    assert "https://example.com/yesterday" in links, (
        "under the old count cap this would have been evicted by the 500 newer links, "
        "and the article resent"
    )

    # ...but once it falls outside the retention window it goes
    much_later = recent + timedelta(hours=subscriber_ops.PUSHED_LINK_RETENTION_HOURS + 1)
    assert subscriber_ops.get_pushed_links(30, now=much_later) == []


def test_pushed_links_reads_the_legacy_plain_list_format(isolated_subscribers_db):
    """Live subscriber rows predate the {link: sent_at} format. They must
    still dedupe rather than being dropped, which would resend articles the
    subscriber has already seen."""
    import json as _json
    import sqlite3 as _sqlite3

    subscriber_ops.request_access(31, "legacy", "Legacy")
    conn = _sqlite3.connect(isolated_subscribers_db)
    conn.execute("UPDATE subscribers SET pushed_links = ? WHERE chat_id = 31",
                 (_json.dumps(["https://example.com/seen-before"]),))
    conn.commit()
    conn.close()

    assert subscriber_ops.get_pushed_links(31) == ["https://example.com/seen-before"]


def test_pushed_links_keeps_a_link_with_an_unparseable_timestamp(isolated_subscribers_db):
    """Defensive branch in _parse_pushed_links -- an unparseable sent_at
    value must not silently drop the link (which would risk re-sending an
    article the subscriber already saw); it's kept and re-stamped instead."""
    import json as _json
    import sqlite3 as _sqlite3

    subscriber_ops.request_access(60, "odd", "Odd")
    conn = _sqlite3.connect(isolated_subscribers_db)
    conn.execute("UPDATE subscribers SET pushed_links = ? WHERE chat_id = 60",
                 (_json.dumps({"https://example.com/x": "not-a-timestamp"}),))
    conn.commit()
    conn.close()

    assert subscriber_ops.get_pushed_links(60) == ["https://example.com/x"]


def test_interests_by_staleness_empty_list_returns_empty(isolated_subscribers_db):
    assert subscriber_ops.interests_by_staleness(1, []) == []


def test_interests_by_staleness_never_pushed_topics_lead(isolated_subscribers_db):
    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    subscriber_ops.mark_interest_pushed(1, "AI", now)
    assert subscriber_ops.interests_by_staleness(1, ["AI", "robotics"]) == ["robotics", "AI"]


def test_interests_by_staleness_orders_by_last_pushed_at(isolated_subscribers_db):
    older = datetime(2026, 8, 18, tzinfo=timezone.utc)
    newer = datetime(2026, 8, 20, tzinfo=timezone.utc)
    subscriber_ops.mark_interest_pushed(1, "AI", newer)
    subscriber_ops.mark_interest_pushed(1, "robotics", older)
    assert subscriber_ops.interests_by_staleness(1, ["AI", "robotics"]) == ["robotics", "AI"]


def test_list_push_enabled_subscribers_only_returns_approved_and_enabled(isolated_subscribers_db):
    subscriber_ops.request_access(20, "olivia", "Olivia")
    subscriber_ops.decide(20, approved=True)
    subscriber_ops.set_interests(20, ["AI"])
    subscriber_ops.set_push_enabled(20, True)

    subscriber_ops.request_access(21, "peter", "Peter")
    subscriber_ops.decide(21, approved=True)
    subscriber_ops.set_push_enabled(21, False)  # opted out -- shouldn't show up

    subscriber_ops.request_access(22, "quinn", "Quinn")  # still pending -- shouldn't show up
    subscriber_ops.set_push_enabled(22, True)

    subscribers = subscriber_ops.list_push_enabled_subscribers()
    assert [s["chat_id"] for s in subscribers] == [20]
    assert subscribers[0]["interests"] == ["AI"]
    assert subscribers[0]["push_interval_hours"] == subscriber_ops.DEFAULT_PUSH_INTERVAL_HOURS
    assert subscribers[0]["last_push_at"] is None
    assert subscribers[0]["pushed_links"] == []
    assert subscribers[0]["language"] is None


def test_list_push_enabled_subscribers_includes_restricted_sources_flag(isolated_subscribers_db):
    subscriber_ops.request_access(23, "rex", "Rex")
    subscriber_ops.decide(23, approved=True)
    subscriber_ops.set_push_enabled(23, True)
    subscriber_ops.set_restricted_sources_enabled(23, True)

    subscriber_ops.request_access(24, "sam", "Sam")
    subscriber_ops.decide(24, approved=True)
    subscriber_ops.set_push_enabled(24, True)  # restricted flag left at its default (False)

    subscribers = {s["chat_id"]: s for s in subscriber_ops.list_push_enabled_subscribers()}
    assert subscribers[23]["restricted_sources_enabled"] is True
    assert subscribers[24]["restricted_sources_enabled"] is False


def test_get_language_none_for_unset_chat(isolated_subscribers_db):
    assert subscriber_ops.get_language(25) is None


def test_set_and_get_language(isolated_subscribers_db):
    subscriber_ops.request_access(25, "quinn2", "Quinn2")
    subscriber_ops.set_language(25, "Spanish")
    assert subscriber_ops.get_language(25) == "Spanish"


def test_set_language_upserts_when_no_existing_row(isolated_subscribers_db):
    subscriber_ops.set_language(999, "Chinese")
    assert subscriber_ops.get_language(999) == "Chinese"
    assert subscriber_ops.get_status(999) == subscriber_ops.APPROVED


def test_set_language_none_clears_it(isolated_subscribers_db):
    subscriber_ops.request_access(26, "rita", "Rita")
    subscriber_ops.set_language(26, "Japanese")
    subscriber_ops.set_language(26, None)
    assert subscriber_ops.get_language(26) is None


def test_list_push_enabled_subscribers_includes_language(isolated_subscribers_db):
    subscriber_ops.request_access(27, "sam2", "Sam2")
    subscriber_ops.decide(27, approved=True)
    subscriber_ops.set_push_enabled(27, True)
    subscriber_ops.set_language(27, "French")

    subscribers = subscriber_ops.list_push_enabled_subscribers()
    assert subscribers[0]["language"] == "French"


def test_list_all_interests_empty_when_nobody_has_any(isolated_subscribers_db):
    assert subscriber_ops.list_all_interests() == []


def test_list_all_interests_deduplicates_across_subscribers(isolated_subscribers_db):
    subscriber_ops.set_interests(1, ["bitcoin", "AI"])
    subscriber_ops.set_interests(2, ["AI", "robotics"])
    assert subscriber_ops.list_all_interests() == ["bitcoin", "AI", "robotics"]


def test_get_restricted_sources_enabled_defaults_false(isolated_subscribers_db):
    assert subscriber_ops.get_restricted_sources_enabled(1) is False


def test_set_restricted_sources_enabled_true(isolated_subscribers_db):
    subscriber_ops.set_restricted_sources_enabled(1, True)
    assert subscriber_ops.get_restricted_sources_enabled(1) is True
    # unrelated chat_id is unaffected
    assert subscriber_ops.get_restricted_sources_enabled(2) is False


def test_set_restricted_sources_enabled_can_be_revoked(isolated_subscribers_db):
    subscriber_ops.set_restricted_sources_enabled(1, True)
    subscriber_ops.set_restricted_sources_enabled(1, False)
    assert subscriber_ops.get_restricted_sources_enabled(1) is False


def test_set_restricted_sources_enabled_upserts_for_unknown_chat(isolated_subscribers_db):
    """No prior row for this chat_id (e.g. the admin, who bypasses
    request_access() entirely -- see check_access())."""
    assert subscriber_ops.get_status(42) is None
    subscriber_ops.set_restricted_sources_enabled(42, True)
    assert subscriber_ops.get_restricted_sources_enabled(42) is True


# --- test accounts (2026-08-21) -------------------------------------------


def test_push_skips_accounts_created_by_the_test_api(isolated_subscribers_db):
    """54 abandoned smoke-test rows had accumulated against 5 real
    subscribers, 19 still push-enabled, each drawing a billed and
    undeliverable digest every 6 hours until the balance ran out and real
    subscribers stopped receiving anything."""
    subscriber_ops.request_access(111, "real", "Real Person")
    subscriber_ops.decide(111, True)
    subscriber_ops.set_push_enabled(111, True)
    subscriber_ops.mark_test_account(222)
    subscriber_ops.set_push_enabled(222, True)

    due = [s["chat_id"] for s in subscriber_ops.list_push_enabled_subscribers()]

    assert due == [111]


def test_marking_works_before_the_row_exists(isolated_subscribers_db):
    """test_api flags the id at the door, before the pipeline has had a
    chance to create the subscriber."""
    subscriber_ops.mark_test_account(333)

    assert subscriber_ops.get_status(333) == subscriber_ops.APPROVED
    with storage.get_storage()._engine.begin() as conn:
        assert conn.execute(
            text("SELECT is_test FROM subscribers WHERE chat_id = 333")).fetchone()[0] == 1


def test_marking_an_existing_row_keeps_its_other_fields(isolated_subscribers_db):
    subscriber_ops.request_access(444, "someone", "Someone")
    subscriber_ops.decide(444, True)
    subscriber_ops.add_interest(444, "robotics")

    subscriber_ops.mark_test_account(444)

    assert subscriber_ops.get_interests(444) == ["robotics"]


def test_a_real_subscriber_is_never_excluded_by_a_null_is_test(isolated_subscribers_db):
    """Rows predating the column have is_test NULL, not 0 -- the query has
    to treat those as real or every existing subscriber stops receiving
    pushes on deploy."""
    subscriber_ops.request_access(555, "old", "Old Subscriber")
    subscriber_ops.decide(555, True)
    subscriber_ops.set_push_enabled(555, True)
    with storage.get_storage()._engine.begin() as conn:
        conn.execute(text("UPDATE subscribers SET is_test = NULL WHERE chat_id = 555"))

    assert [s["chat_id"] for s in subscriber_ops.list_push_enabled_subscribers()] == [555]


def test_external_id_is_stable_and_hides_the_chat_id(isolated_subscribers_db):
    """Telemetry leaves this machine; chat_id is a real Telegram account
    id. Stored rather than derived so it survives a change of scheme and
    so the mapping stays a row someone can look up during an incident."""
    subscriber_ops.request_access(778899, "u", "U")

    first = subscriber_ops.external_id(778899)
    assert first == subscriber_ops.external_id(778899)      # stable across calls
    assert "778899" not in first
    assert subscriber_ops.external_id(778899) != subscriber_ops.external_id(112233)


def test_external_id_survives_without_a_subscriber_row(isolated_subscribers_db):
    """A span attribute is never worth raising over, so a chat with no row
    still gets something stable rather than an exception."""
    got = subscriber_ops.external_id(999000)
    assert got == subscriber_ops.external_id(999000)
    assert "999000" not in got


# --- push_consecutive_failures (2026-09-04) --------------------------------
#
# Replaced the old push_outcomes event-log table (see git history) -- the
# only real reader was "how many chat_not_found cycles in a row", which a
# bounded per-subscriber counter answers without an ever-growing table.


def test_get_push_consecutive_failures_defaults_zero(isolated_subscribers_db):
    assert subscriber_ops.get_push_consecutive_failures(70) == 0


def test_record_push_failure_increments_and_returns_the_new_count(isolated_subscribers_db):
    assert subscriber_ops.record_push_failure(70) == 1
    assert subscriber_ops.record_push_failure(70) == 2
    assert subscriber_ops.get_push_consecutive_failures(70) == 2


def test_record_push_failure_creates_a_row_when_none_exists(isolated_subscribers_db):
    assert subscriber_ops.get_status(71) is None
    assert subscriber_ops.record_push_failure(71) == 1
    assert subscriber_ops.get_status(71) == subscriber_ops.APPROVED


def test_reset_push_consecutive_failures_clears_it(isolated_subscribers_db):
    subscriber_ops.record_push_failure(72)
    subscriber_ops.record_push_failure(72)
    subscriber_ops.reset_push_consecutive_failures(72)
    assert subscriber_ops.get_push_consecutive_failures(72) == 0


def test_push_consecutive_failures_is_scoped_per_subscriber(isolated_subscribers_db):
    subscriber_ops.record_push_failure(73)
    subscriber_ops.record_push_failure(73)
    assert subscriber_ops.get_push_consecutive_failures(73) == 2
    assert subscriber_ops.get_push_consecutive_failures(74) == 0


# --- search_news's per-subscriber daily quota -------------------------------
# A UTC-day-scoped counter stored directly on the subscribers row -- see
# storage's try_consume_search_quota docstring for why this is one
# check-and-increment call, not a separate get/set (avoids a race between
# reading the count and writing the incremented one).


def test_try_consume_search_query_allows_up_to_the_cap(isolated_subscribers_db):
    today = "2026-09-04"
    for _ in range(3):
        assert subscriber_ops.try_consume_search_query(80, today, daily_cap=3) is True
    assert subscriber_ops.try_consume_search_query(80, today, daily_cap=3) is False


def test_try_consume_search_query_resets_on_a_new_day(isolated_subscribers_db):
    assert subscriber_ops.try_consume_search_query(81, "2026-09-04", daily_cap=1) is True
    assert subscriber_ops.try_consume_search_query(81, "2026-09-04", daily_cap=1) is False
    # A later day resets the count rather than carrying yesterday's usage
    # forward -- there's no scheduled reset job, this is what performs it.
    assert subscriber_ops.try_consume_search_query(81, "2026-09-05", daily_cap=1) is True


def test_try_consume_search_query_is_scoped_per_subscriber(isolated_subscribers_db):
    today = "2026-09-04"
    assert subscriber_ops.try_consume_search_query(82, today, daily_cap=1) is True
    assert subscriber_ops.try_consume_search_query(82, today, daily_cap=1) is False
    assert subscriber_ops.try_consume_search_query(83, today, daily_cap=1) is True


# --- Free-trial usage caps (agent interactions / news pushes) ------------
# Deliberately unlike search_query's daily quota above: no reset, and
# NULL/-1 means unlimited rather than "not consumed yet today". See
# storage.try_consume_agent_interaction/try_consume_push's own docstrings.


def test_decide_approve_assigns_the_trial_allowances(isolated_subscribers_db):
    subscriber_ops.request_access(90, "frank", "Frank")
    subscriber_ops.decide(90, approved=True)
    assert subscriber_ops.get_agent_interactions_remaining(90) == subscriber_ops.TRIAL_AGENT_INTERACTION_LIMIT
    assert subscriber_ops.get_pushes_remaining(90) == subscriber_ops.TRIAL_PUSH_LIMIT


def test_a_pre_existing_approved_subscriber_from_before_this_feature_has_no_limit(isolated_subscribers_db):
    """A real subscriber approved BEFORE agent_interactions_remaining/
    pushes_remaining existed -- a legacy table missing those two columns
    (and status already 'approved' from before the migration ever ran,
    not a row decide() created after this feature shipped), migrated via
    ADDITIVE_COLUMNS. Same "drop table, minimal legacy schema, re-run
    init_db()" pattern as
    test_init_db_migrates_schema_missing_interests_column above -- the
    scenario test_a_subscriber_with_no_row_has_no_limit (below) doesn't
    actually cover, since that one has no row at all rather than a real
    pre-migration approved one."""
    with sqlite3.connect(isolated_subscribers_db) as conn:
        conn.execute("DROP TABLE subscribers")
        conn.execute(
            """
            CREATE TABLE subscribers (
                chat_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                status TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                decided_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO subscribers (chat_id, username, first_name, status, requested_at, decided_at) "
            "VALUES (100, 'legacy', 'Legacy', 'approved', '2026-01-01T00:00:00', '2026-01-01T00:00:00')"
        )
    storage.init_db()  # ADDITIVE_COLUMNS lands the two new columns as NULL on this row

    assert subscriber_ops.get_agent_interactions_remaining(100) is None
    assert subscriber_ops.get_pushes_remaining(100) is None
    assert subscriber_ops.try_consume_agent_interaction(100) is True
    assert subscriber_ops.try_consume_push(100) is True


def test_decide_deny_leaves_the_allowances_unset(isolated_subscribers_db):
    subscriber_ops.request_access(91, "grace", "Grace")
    subscriber_ops.decide(91, approved=False)
    assert subscriber_ops.get_agent_interactions_remaining(91) is None
    assert subscriber_ops.get_pushes_remaining(91) is None


def test_a_subscriber_with_no_row_has_no_limit(isolated_subscribers_db):
    """A chat_id nobody has ever called decide() for -- covers both a
    genuinely unknown chat_id and, more importantly, every subscriber
    approved before this feature existed: the schema migration leaves
    their row's new columns NULL, and NULL must mean unlimited, not
    zero."""
    assert subscriber_ops.get_agent_interactions_remaining(999) is None
    assert subscriber_ops.try_consume_agent_interaction(999) is True
    assert subscriber_ops.try_consume_push(999) is True


def test_try_consume_agent_interaction_decrements_and_blocks_at_zero(isolated_subscribers_db):
    subscriber_ops.request_access(92, "heidi", "Heidi")
    subscriber_ops.decide(92, approved=True)
    subscriber_ops.set_agent_interactions_remaining(92, 2)

    assert subscriber_ops.try_consume_agent_interaction(92) is True
    assert subscriber_ops.get_agent_interactions_remaining(92) == 1
    assert subscriber_ops.try_consume_agent_interaction(92) is True
    assert subscriber_ops.get_agent_interactions_remaining(92) == 0
    # At zero: refused, and NOT decremented further into the negatives.
    assert subscriber_ops.try_consume_agent_interaction(92) is False
    assert subscriber_ops.get_agent_interactions_remaining(92) == 0


def test_try_consume_agent_interaction_minus_one_means_unlimited(isolated_subscribers_db):
    subscriber_ops.request_access(93, "ivan", "Ivan")
    subscriber_ops.decide(93, approved=True)
    subscriber_ops.set_agent_interactions_remaining(93, -1)

    for _ in range(5):
        assert subscriber_ops.try_consume_agent_interaction(93) is True
    # Never decremented -- still exactly -1, not drifting toward 0.
    assert subscriber_ops.get_agent_interactions_remaining(93) == -1


def test_try_consume_push_decrements_and_blocks_at_zero(isolated_subscribers_db):
    subscriber_ops.request_access(94, "judy", "Judy")
    subscriber_ops.decide(94, approved=True)
    subscriber_ops.set_pushes_remaining(94, 1)

    assert subscriber_ops.try_consume_push(94) is True
    assert subscriber_ops.get_pushes_remaining(94) == 0
    assert subscriber_ops.try_consume_push(94) is False


def test_reset_agent_interaction_limit_restores_the_current_setting(isolated_subscribers_db):
    subscriber_ops.request_access(95, "mallory", "Mallory")
    subscriber_ops.decide(95, approved=True)
    subscriber_ops.set_agent_interactions_remaining(95, 0)

    subscriber_ops.reset_agent_interaction_limit(95)

    assert subscriber_ops.get_agent_interactions_remaining(95) == subscriber_ops.TRIAL_AGENT_INTERACTION_LIMIT


def test_reset_push_limit_restores_the_allowance_and_re_enables_push(isolated_subscribers_db):
    subscriber_ops.request_access(96, "niaj", "Niaj")
    subscriber_ops.decide(96, approved=True)
    subscriber_ops.set_pushes_remaining(96, 0)
    subscriber_ops.set_push_enabled(96, False)

    subscriber_ops.reset_push_limit(96)

    assert subscriber_ops.get_pushes_remaining(96) == subscriber_ops.TRIAL_PUSH_LIMIT
    assert subscriber_ops.get_push_enabled(96) is True

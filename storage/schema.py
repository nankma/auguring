"""
Table/MetaData definitions shared by every backend (SqliteStorage,
PostgresStorage, ...). Declared once, here, so `metadata.create_all(engine)`
generates dialect-correct DDL for whichever backend is active -- this is
the one place autoincrement-PK syntax (SQLite's `INTEGER PRIMARY KEY
AUTOINCREMENT` vs. Postgres's `SERIAL`/`IDENTITY`) and column-type mapping
differences are handled, so no Storage class needs its own CREATE TABLE
strings.

Column shapes are ported byte-for-byte from the pre-refactor users_db.py
schema -- see git history for the original CREATE TABLE statements and
their own per-column reasoning comments.
"""

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Table,
    Text,
)

metadata = MetaData()

subscribers = Table(
    "subscribers",
    metadata,
    Column("chat_id", Integer, primary_key=True),
    Column("username", Text),
    Column("first_name", Text),
    Column("status", Text, nullable=False),
    Column("requested_at", Text, nullable=False),
    Column("decided_at", Text),
    Column("interests", Text),
    Column("push_enabled", Integer),
    Column("push_interval_hours", Integer),
    Column("last_push_at", Text),
    Column("pushed_links", Text),
    Column("language", Text),
    Column("restricted_sources_enabled", Integer),
    Column("is_test", Integer),
    Column("external_id", Text),
    # Consecutive undeliverable (chat_not_found) push cycles running --
    # NOT a history table (push_outcomes, retired -- see git history and
    # docs/plans/incident-monitoring-plan.md), just a per-subscriber
    # counter. Incremented on chat_not_found, reset to 0 on any delivered
    # digest or when the user themselves turns push off (stop_push) --
    # deliberately NOT reset by the automatic 3-strikes disable, so
    # re-enabling push while still unreachable strikes out again
    # immediately rather than granting a fresh allowance. See
    # subscriber_ops.record_push_failure/reset_push_consecutive_failures
    # and news_push._strike_unreachable_subscriber's own docstring.
    Column("push_consecutive_failures", Integer),
    # search_news's daily quota -- a UTC-day-scoped counter, per subscriber,
    # deliberately modeled on api_budget's date+count shape but stored
    # directly on the subscriber row (one query, not a join) since there's
    # no per-source dimension to it. search_count_date holds the UTC date
    # (YYYY-MM-DD) the count applies to; subscriber_ops resets the count
    # to 0 whenever the stored date != today rather than a scheduled job.
    Column("search_count_date", Text),
    Column("search_count_today", Integer),
)

api_budget = Table(
    "api_budget",
    metadata,
    Column("source", Text, primary_key=True),
    Column("date", Text, primary_key=True),
    Column("count", Integer, nullable=False),
)

source_pull_state = Table(
    "source_pull_state",
    metadata,
    Column("source", Text, primary_key=True),
    Column("last_pulled_at", Text, nullable=False),
    Column("last_article_dt", Text),
)

interest_categories = Table(
    "interest_categories",
    metadata,
    Column("interest", Text, primary_key=True),
    Column("categories", Text, nullable=False),
)

interest_query_expansions = Table(
    "interest_query_expansions",
    metadata,
    Column("interest", Text, primary_key=True),
    Column("expansion", Text, nullable=False),
)

# A subscriber's own override of the shared expansion above -- see
# docs/plans/interest-definition-plan.md. Deliberately a separate table,
# not a migration of interest_query_expansions to a (chat_id, interest)
# key: the existing table keeps serving as the automatic, shared default
# every subscriber gets for free (written by agent.add_one_interest and
# search_news, unchanged), while this one is written ONLY through an
# explicit refinement in the find_interests conversation
# (interest_finder.execute_redefine). Reading both and preferring this
# one is interest_cache_ops.resolve_interest_definition's job.
subscriber_interest_definitions = Table(
    "subscriber_interest_definitions",
    metadata,
    Column("chat_id", Integer, primary_key=True),
    Column("interest", Text, primary_key=True),
    Column("expansion", Text, nullable=False),
)

health_state = Table(
    "health_state",
    metadata,
    Column("key", Text, primary_key=True),
    Column("value", Text, nullable=False),
)

category_keyness = Table(
    "category_keyness",
    metadata,
    Column("category", Text, primary_key=True),
    Column("term", Text, primary_key=True),
    Column("score", Float, nullable=False),
)

categories = Table(
    "categories",
    metadata,
    Column("name", Text, primary_key=True),
    Column("description", Text),
    Column("status", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("created_by", Text, nullable=False),
    Column("decided_at", Text),
    Column("decided_by", Text),
    Column("merged_into", Text),
    Column("centroid", LargeBinary),
    Column("sort_order", Integer, nullable=False, server_default="0"),
    Column("alerted_at", Text),
)

category_sightings = Table(
    "category_sightings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", Text, nullable=False),
    Column("seen_at", Text, nullable=False),
    Column("article_link", Text),
    Column("article_title", Text),
    Index("category_sightings_name_at", "name", "seen_at"),
)

interest_push_state = Table(
    "interest_push_state",
    metadata,
    Column("chat_id", Integer, primary_key=True),
    Column("topic", Text, primary_key=True),
    Column("last_pushed_at", Text, nullable=False),
)

# (table, column, sql_type) for columns added additively after a table's
# first release -- ALTER TABLE ADD COLUMN isn't idempotent, so an existing
# SQLite file (dev/PROD, both long-lived) that predates one of these needs
# it added at startup. A no-op on a table just created by create_all()
# above (which already declares every column in its final shape) or one
# already migrated. See storage/sqlite/_primitives.py's ensure_columns().
ADDITIVE_COLUMNS: list[tuple[str, str, str]] = [
    ("subscribers", "interests", "TEXT"),
    ("subscribers", "push_enabled", "INTEGER"),
    ("subscribers", "push_interval_hours", "INTEGER"),
    ("subscribers", "last_push_at", "TEXT"),
    ("subscribers", "pushed_links", "TEXT"),
    ("subscribers", "language", "TEXT"),
    ("subscribers", "restricted_sources_enabled", "INTEGER"),
    ("subscribers", "is_test", "INTEGER"),
    ("subscribers", "external_id", "TEXT"),
    ("subscribers", "push_consecutive_failures", "INTEGER"),
    ("subscribers", "search_count_date", "TEXT"),
    ("subscribers", "search_count_today", "INTEGER"),
    ("source_pull_state", "last_article_dt", "TEXT"),
    ("categories", "sort_order", "INTEGER NOT NULL DEFAULT 0"),
    ("categories", "alerted_at", "TEXT"),
]

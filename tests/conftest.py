import pytest

import app_settings
from trailsign import Settings

# Required settings, injected before the imports below -- module-level
# constants like message_archive.ARCHIVE_DIR are computed once at import
# time, so this has to run before those imports, not inside a fixture
# (fixtures run per-test, long after collection-time imports already
# happened). Values are placeholders -- individual tests that care
# override them via monkeypatch.setattr on the module constant directly
# (see isolated_message_archive below), or, for news_cache's now-pluggable
# backend (2026-09-05), by injecting a fresh backend instance via
# vector_store.reset_vector_store_for_tests (see isolated_news_cache
# below) -- not by touching Settings again.
# See docs/standaloneplan/01-settings-migration.md's "Migration
# methodology" for why these are required=True with no code-level
# fallback in the modules themselves.
app_settings.reset_settings_for_tests(Settings({
    "storage": {
        "news_cache_dir": {"path": "news_cache", "ttl_hours": 48},
        "message_archive_dir": {"path": "message_archive", "ttl_days": 7},
        "database": {"type": "sqlite", "sqlite": {"path": "subscribers.db"}},
    },
    "models": {
        "main": {"url": "https://example.invalid", "model": "fake-main", "api-key": "fake-key"},
        "guardrail": {"url": "https://example.invalid", "model": "fake-guardrail", "api-key": "fake-key"},
    },
    "news_source": {
        "rss": [
            {"key": "fake_rss_source", "display_name": "Fake RSS Source", "url": "https://fake.invalid/feed.xml"},
        ],
    },
    # Real trailsign-resolve nodes (not literal placeholder strings), so
    # tests that monkeypatch.setenv these same env var names (test_bot.py,
    # test_admin_bot.py, test_combined_bot.py, test_telemetry.py,
    # test_test_api.py) see that value flow through get_settings()/
    # resolved_optional() -- a literal string here would make those tests
    # pass regardless of what they set the env var to.
    "delivery": {
        "telegram": {
            "bot-token": {"trailsign-resolve": "environment-variable", "name": "TELEGRAM_BOT_TOKEN"},
            "admin-bot-token": {"trailsign-resolve": "environment-variable", "name": "ADMIN_BOT_TOKEN"},
            "admin-chat-id": {"trailsign-resolve": "environment-variable", "name": "ADMIN_CHAT_ID"},
        },
    },
    # Empty by default -- telemetry.py's setup_telemetry() treats an
    # empty/absent telemetry.providers the same as news_source.rss/
    # news_source.api's empty-list default (off, not an error), which is
    # exactly what every test/CI run needs (no test wants to actually
    # export spans). tests/test_telemetry.py builds its own one-off
    # Settings per test for the cases that need a real provider entry,
    # rather than this shared fixture carrying one.
    "telemetry": {
        "providers": [],
    },
    "test_api": {
        "enabled": {"trailsign-resolve": "environment-variable", "name": "ENABLE_TEST_API"},
        "port": {"trailsign-resolve": "environment-variable", "name": "TEST_API_PORT"},
    },
}))

from sqlalchemy import create_engine

import category_ops
import message_archive
import news_cache
import news_keyness
import storage
import vector_store
from storage.sqlite import SqliteStorage
from vector_store.yaml_files import YamlFilesStore
from tests.fakes import fake_pos_tag, fake_word_tokenize


@pytest.fixture
def isolated_news_cache(tmp_path):
    """Points the active YamlFilesStore backend at a temp directory for
    the duration of a test, so cache tests never touch the real
    news_cache/ directory. Injects a fresh instance via
    vector_store.reset_vector_store_for_tests (a module-level singleton,
    like storage.reset_storage_for_tests) -- 2026-09-05's pluggable
    news-cache-backend split moved CACHE_DIR/ARCHIVE_DIR off news_cache.py
    itself and into YamlFilesStore's own __init__, so this can no longer
    be a plain monkeypatch.setattr on a module constant. A test that needs
    to change ARCHIVE_DIR mid-test reaches this same instance via
    vector_store.get_vector_store() and monkeypatches its
    _archive_dir_path attribute directly."""
    store = YamlFilesStore.__new__(YamlFilesStore)
    store._cache_dir_path = str(tmp_path / "news_cache")
    store._archive_dir_path = None
    vector_store.reset_vector_store_for_tests(store)
    yield tmp_path / "news_cache"
    vector_store.reset_vector_store_for_tests(None)


@pytest.fixture(autouse=True)
def isolated_message_archive(monkeypatch, tmp_path):
    """Point message_archive.ARCHIVE_DIR at a temp directory for every
    test, autouse -- unlike news_cache's NEWS_ARCHIVE_DIR (default off,
    an explicit fixture is enough), MESSAGE_ARCHIVE_DIR defaults to a
    real relative directory (archiving is meant to always be on, see
    that module's docstring), so ANY test exercising handle_message/
    send_push_digest/run_push_cycle end-to-end -- not just tests that
    know to ask for isolation -- would otherwise write real files into
    the repo's working directory. Confirmed happening 2026-08-28 before
    this was autouse: a full test run left dozens of real .json files in
    ./message_archive/."""
    path = tmp_path / "message_archive"
    monkeypatch.setattr(message_archive, "ARCHIVE_DIR", str(path))
    return path




@pytest.fixture
def isolated_subscribers_db(tmp_path):
    """Points storage at a temp SQLite file for the duration of a test, so
    access-control tests never touch the real subscribers.db. Explicit
    teardown (not monkeypatch) because storage.get_storage() is a
    module-level singleton, not an attribute monkeypatch would revert on
    its own."""
    path = tmp_path / "subscribers.db"
    storage.reset_storage_for_tests(SqliteStorage(create_engine(f"sqlite:///{path}")))
    storage.init_db()
    category_ops.bootstrap()
    yield path
    storage.reset_storage_for_tests(None)


@pytest.fixture
def fake_nltk(monkeypatch):
    """Swaps news_keyness's pos_tag/word_tokenize for deterministic fakes
    (tests/fakes.py) so keyness tests don't need the real NLTK data files
    downloaded -- same reasoning as FakeEmbedder standing in for
    model2vec: fast, no real model/data load, and predictable from the
    fixture's own text alone. Every alphanumeric token in a fixture's
    title/summary counts as a noun under this fake, not just the words a
    real tagger would classify that way."""
    monkeypatch.setattr(news_keyness, "pos_tag", fake_pos_tag)
    monkeypatch.setattr(news_keyness, "word_tokenize", fake_word_tokenize)

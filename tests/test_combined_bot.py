import pytest
from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler
from trailsign import SettingsError

import combined_bot
import subscriber_ops

FAKE_TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"


def test_build_info_app_wires_bot_data(monkeypatch, isolated_subscribers_db):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    app = combined_bot.build_info_app(
        model="fake-model", admin_chat_id=999, admin_bot_token="admin-token",
        guard_model="fake-guard-model", embedder="fake-embedder",
    )

    assert app.bot_data["model"] == "fake-model"
    assert app.bot_data["admin_chat_id"] == 999
    assert app.bot_data["admin_bot_token"] == "admin-token"
    assert app.bot_data["guard_model"] == "fake-guard-model"
    assert app.bot_data["embedder"] == "fake-embedder"
    # build_info_app grants the admin restricted-source access at startup
    # (news_sources.RESTRICTED_SOURCES: NewsAPI, Perigon) -- confirm it
    # actually happened, not just that it didn't crash.
    assert subscriber_ops.get_restricted_sources_enabled(999) is True
    handlers = [h for group in app.handlers.values() for h in group]
    assert any(isinstance(h, MessageHandler) for h in handlers)
    # the periodic-push scheduler (docs/plans/bot-features-plan.md item 5) and
    # the news-cache ingestion job (docs/plans/local-news-cache-plan.md) must
    # both be wired up in the combined process too, not just standalone
    # bot.py -- asserting the callback names, not just a count, so a
    # future job silently failing to register (or one accidentally
    # registered twice) fails loudly here.
    job_callback_names = {job.callback.__name__ for job in app.job_queue.jobs()}
    assert job_callback_names == {"_push_job", "_ingest_job"}
    # Real incident, 2026-08-09: /start went unhandled (only checking "any
    # MessageHandler" wouldn't have caught this -- it needs its own
    # CommandHandler, since the plain-text MessageHandler excludes all
    # commands). Assert the exact command set so a future missing command
    # fails loudly here instead of silently in production.
    #
    # Every command of every handler, not next(iter(h.commands)): one
    # CommandHandler can serve several names (/start and /help share a
    # reply), and taking only the first would silently drop one.
    commands = {c for h in handlers if isinstance(h, CommandHandler) for c in h.commands}
    assert commands == {"start", "help", "interests", "language"}
    # And a catch-all BEHIND them, so an unregistered command gets an
    # answer rather than silence -- see bot.handle_unknown_command.
    #
    # Order is the load-bearing half: python-telegram-bot dispatches only
    # the first matching handler in a group, and this catch-all matches
    # EVERY command. Registered before the CommandHandlers it would
    # swallow /interests and /language and tell a user that commands which
    # exist do not. Asserted on the real Application's handler list rather
    # than by reading source, which is possible here (unlike bot.main())
    # because build_info_app needs no live token.
    catch_all_at = next(i for i, h in enumerate(handlers)
                        if isinstance(h, MessageHandler)
                        and h.callback.__name__ == "handle_unknown_command")
    for i, h in enumerate(handlers):
        if isinstance(h, CommandHandler):
            assert i < catch_all_at, f"{h.commands} must be registered before the catch-all"
    # The global PTB backstop (bot.register_error_handler) -- catches
    # anything a specific handler/job doesn't (see bot.py's own
    # docstring on why). One registered handler is enough to prove it
    # was wired up; its own behavior is tested in tests/test_bot.py.
    assert len(app.error_handlers) == 1


def test_build_admin_app_wires_bot_data(monkeypatch):
    monkeypatch.setenv("ADMIN_BOT_TOKEN", FAKE_TOKEN)
    app = combined_bot.build_admin_app(admin_chat_id=999, info_bot_token="info-token")

    assert app.bot_data["admin_chat_id"] == 999
    assert app.bot_data["info_bot_token"] == "info-token"
    handlers = [h for group in app.handlers.values() for h in group]
    assert any(isinstance(h, CallbackQueryHandler) for h in handlers)
    assert any(isinstance(h, MessageHandler) for h in handlers)
    assert len(app.error_handlers) == 1


def test_build_info_app_raises_when_the_bot_token_is_missing(monkeypatch):
    """delivery.telegram.bot-token is required=True -- same bracket-access
    (os.environ["X"], KeyError-on-missing) semantics the old direct env
    read had. main()/bot.py/admin_bot.py all hand-copy this identical
    resolved(..., required=True) call rather than sharing a helper (see
    docs/standaloneplan/01-settings-migration.md), so this is worth its own
    regression test -- a typo in one copy wouldn't be caught by testing
    only the others."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(SettingsError):
        combined_bot.build_info_app(
            model="fake-model", admin_chat_id=999, admin_bot_token="admin-token",
            guard_model="fake-guard-model", embedder="fake-embedder",
        )


def test_build_admin_app_raises_when_the_admin_bot_token_is_missing(monkeypatch):
    monkeypatch.delenv("ADMIN_BOT_TOKEN", raising=False)

    with pytest.raises(SettingsError):
        combined_bot.build_admin_app(admin_chat_id=999, info_bot_token="info-token")



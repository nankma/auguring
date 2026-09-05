"""
Runs bot.py's and admin_bot.py's Telegram Applications together in one
process/event loop, instead of as two separate processes/containers.

Why: each Application independently loads LangChain, python-telegram-bot,
etc. into memory -- running two OS processes duplicates that. This matters
on a small VM (e.g. Oracle's Always Free VM.Standard.E2.1.Micro, 1GB RAM)
where that duplication is a real constraint. Merging into one process
halves it, without changing the two-bot-token security design (see
docs/plans/bot-features-plan.md item 1) -- @mnkInfo_bot and @mnkInfoAdmin_bot
stay two separate Telegram identities, just served by one Python process.

bot.py and admin_bot.py keep their own standalone main()s too -- this file
only adds a combined option, reusing their handler functions and bot_data
wiring unchanged.

Whether the ingest/push jobs themselves are still ticking, and whether
Logfire itself is reachable, are Logfire's own questions now (see
news_ingest._emit_heartbeat/_pull_source, news_push._emit_heartbeat, the
Logfire alerts they feed, and docs/plans/observability-platform-plan.md)
-- not something this process checks or alerts on directly (Phoenix's
own reachability-monitoring loop, telemetry_monitor.py, was retired
alongside Phoenix itself; healthcheck.py's in-process job-liveness
polling was retired 2026-08-29 for the same reason -- see
docs/plans/telemetry-and-testing-plan.md).

Run:
    conda activate myfirstagent
    export DEEPSEEK_API_KEY=<your-deepseek-key>
    export TELEGRAM_BOT_TOKEN=<info-bot-token>
    export ADMIN_BOT_TOKEN=<admin-bot-token>
    export ADMIN_CHAT_ID=<your-telegram-numeric-user-id>
    python combined_bot.py
"""

import asyncio
import signal
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters
from agent import build_model_from_settings, setup_telemetry
from app_settings import get_settings
from ptb_error_handler import register_error_handler
import bot as info_bot
import admin_bot
import news_embed
import test_api
import category_ops
import storage
import subscriber_ops


def build_info_app(model, admin_chat_id: int, admin_bot_token: str, guard_model=None, embedder=None) -> Application:
    app = Application.builder().token(get_settings().resolved("delivery.telegram.bot-token", required=True)).build()
    app.bot_data["model"] = model
    # guard_model is independently configurable from agent's own model via
    # LLM_MODEL_CLASSIFIER -- see agent.build_model and
    # docs/plans/model-portability-plan.md's Level 2 per-stage routing.
    app.bot_data["guard_model"] = guard_model
    # None on any failure -- an enhancement to push quality, never
    # something the push/ingest jobs (bot.py's _push_job/_ingest_job,
    # reused here via register_push_job/register_ingest_job) require to
    # run. See news_embed's module docstring.
    app.bot_data["embedder"] = embedder
    app.bot_data["admin_chat_id"] = admin_chat_id
    app.bot_data["admin_bot_token"] = admin_bot_token
    # Idempotent -- see bot.py's main() for the same call and its reasoning.
    subscriber_ops.set_restricted_sources_enabled(admin_chat_id, True)
    app.add_handler(CommandHandler(["start", "help"], info_bot.handle_start_command))
    app.add_handler(CommandHandler("interests", info_bot.handle_interests_command))
    app.add_handler(CommandHandler("language", info_bot.handle_language_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, info_bot.handle_message))
    # Last: every real command above gets first refusal.
    app.add_handler(MessageHandler(filters.COMMAND, info_bot.handle_unknown_command))
    info_bot.register_push_job(app)
    info_bot.register_ingest_job(app)
    register_error_handler(app, "argus.bot")
    return app


def build_admin_app(admin_chat_id: int, info_bot_token: str) -> Application:
    app = Application.builder().token(get_settings().resolved("delivery.telegram.admin-bot-token", required=True)).build()
    app.bot_data["admin_chat_id"] = admin_chat_id
    app.bot_data["info_bot_token"] = info_bot_token
    # Disjoint patterns so the category buttons and the approve/deny
    # buttons can't catch each other's callbacks.
    app.add_handler(CallbackQueryHandler(admin_bot.handle_category_decision, pattern=r"^cat:"))
    app.add_handler(CallbackQueryHandler(admin_bot.handle_decision, pattern=r"^(approve|deny):"))
    app.add_handler(MessageHandler(filters.ALL, admin_bot.reject_non_admin))
    register_error_handler(app, "argus.admin_bot")
    return app


async def run_both(
    info_app: Application, admin_app: Application, admin_bot_token: str, admin_chat_id: int
) -> None:
    async with info_app, admin_app:
        await info_app.start()
        await info_app.updater.start_polling()
        await admin_app.start()
        await admin_app.updater.start_polling()

        test_api_server = test_api.start(
            info_app.bot_data["model"], info_app.bot_data["guard_model"], info_app.bot_data.get("embedder")
        )

        print("Both bots ready (polling). Ctrl+C to stop.")
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass  # Windows: no add_signal_handler, Ctrl+C raises KeyboardInterrupt instead

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            test_api.stop(test_api_server)
            await admin_app.updater.stop()
            await admin_app.stop()
            await info_app.updater.stop()
            await info_app.stop()


def main():
    setup_telemetry()
    storage.init_db()
    category_ops.bootstrap()
    admin_chat_id = int(get_settings().resolved("delivery.telegram.admin-chat-id", required=True))
    admin_bot_token = get_settings().resolved("delivery.telegram.admin-bot-token", required=True)
    info_bot_token = get_settings().resolved("delivery.telegram.bot-token", required=True)

    # Two independently-configured models -- see docs/plans/model-portability-plan.md
    # Level 2. `models.main`/`models.guardrail` in settings.yml both point
    # at the same underlying model by default (no second provider is set
    # up in the checked-in example), so this is plumbing, not a behavior
    # change, until a deployment's own settings.yml actually points them
    # at different providers/models.
    settings = get_settings()
    model = build_model_from_settings(settings, "models.main")
    # A short default_timeout here, not build_model_from_settings' usual
    # 60s -- this model backs layer 2/4 guardrail calls on a live Telegram
    # user's own message, not a background batch job. See
    # build_model_from_config's own docstring.
    guard_model = build_model_from_settings(settings, "models.guardrail", default_timeout=20.0)
    embedder = news_embed.build_embedder()

    info_app = build_info_app(model, admin_chat_id, admin_bot_token, guard_model=guard_model, embedder=embedder)
    admin_app = build_admin_app(admin_chat_id, info_bot_token)

    asyncio.run(run_both(info_app, admin_app, admin_bot_token, admin_chat_id))


if __name__ == "__main__":
    main()

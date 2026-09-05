"""
Telegram bot entry point for the agent — the headless alternative to
agent.py's CLI REPL. Polling mode: no public endpoint or TLS needed, and
the same shape works locally and in a long-running Kubernetes Deployment
later. See docs/plans/deployment-plan.md.

Reuses setup_telemetry from agent.py unchanged — this file only adds the
Telegram-specific plumbing (per-chat history, handler registration, the
polling loop). news_query replies go through agent.search_news, a plain
deterministic function, not agent.py's build_agent/run_agent tool-calling
loop -- see search_news's own docstring for why (kept working but
currently unused by any live route, in case a future feature needs it).

Access is gated by an approval workflow (see docs/plans/bot-features-plan.md item
1): ADMIN_CHAT_ID is always allowed; anyone else's first message registers
a pending request in the shared subscribers DB (subscriber_ops.py) and notifies
the admin via admin_bot.py — a separate bot/token — with Approve/Deny
buttons attached to the message.

Run:
    conda activate myfirstagent
    export DEEPSEEK_API_KEY=<your-deepseek-key>
    export TELEGRAM_BOT_TOKEN=<your-bot-token>
    export ADMIN_CHAT_ID=<your-telegram-numeric-user-id>
    export ADMIN_BOT_TOKEN=<second-bot-token-for-admin_bot.py>
    python bot.py
"""

import asyncio
import re
from datetime import datetime, timedelta, timezone
from langchain_core.messages import AIMessage, HumanMessage
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from agent import ROUTE_B_CATEGORIES, build_model_from_settings, dispatch_settings, search_news, setup_telemetry
from app_settings import get_settings
import admin_bot
import guardrails
import message_archive
import news_classify
import news_embed
import news_ingest
import news_push
import telegram_html
import test_api
import category_ops
import storage
import subscriber_ops
from ptb_error_handler import register_error_handler
from telemetry import EventLogger, get_event_logger
from telemetry_providers import Level

_events: EventLogger = get_event_logger("argus.bot")

TELEGRAM_MESSAGE_LIMIT = 4096

# How often the periodic-push scheduler checks who's due (see
# register_push_job below) -- independent of any individual subscriber's
# push_interval_hours (subscriber_ops.MIN_PUSH_INTERVAL_HOURS floors that at 1h),
# just fine-grained enough that a due subscriber isn't kept waiting long
# past their actual interval.
PUSH_TICK_SECONDS = get_settings().resolved("push.tick_seconds", default=900)

# Same tick shape as PUSH_TICK_SECONDS -- check frequently, let each
# source's own interval (news_source.<name>.interval_hours in Settings,
# news_ingest._interval_hours -- 4h default, longer for budget-capped
# sources) decide whether this tick actually does anything. See
# docs/plans/local-news-cache-plan.md. Lives under news_source.* (not
# push.*) since this is about the news-fetching pipeline's own cadence,
# not push delivery.
INGEST_TICK_SECONDS = get_settings().resolved("news_source.tick_seconds", default=900)

# Per-chat conversation history. In-memory only — lost on restart, same as
# the CLI's messages list. Not persisted; fine for now, revisit if needed.
# Each value is (messages, timestamps) -- a parallel list of when each
# message was added, needed for the age-based trim below since LangChain
# message objects/dicts carry no wall-clock timestamp of their own.
chat_histories: dict[int, tuple[list, list[datetime]]] = {}

# Real question, 2026-08-09: does the layered system prompt (agent.py's
# _compose_prompt) or this conversation history risk a context-window
# overflow over a long-lived process? The prompt layers don't -- they're
# rebuilt fresh on every call, never accumulated. This history DOES:
# run_agent resends the full accumulated list on every turn, with no
# trimming anywhere, so an active chat on a long-uptime process would grow
# without bound -- rising cost every turn, and eventually exceeding
# DeepSeek's context window outright rather than degrading gracefully.
# Trimmed aggressively on purpose: this bot's replies are effectively
# stateless per-topic (a news summary from an hour ago has little bearing
# on a new question), so there's little value in keeping much around.
MAX_HISTORY_AGE = timedelta(hours=1)
MAX_HISTORY_MESSAGES = 20


def _trim_history(messages: list, timestamps: list[datetime], now: datetime) -> tuple[list, list[datetime]]:
    """Drops messages older than MAX_HISTORY_AGE, then caps to the most
    recent MAX_HISTORY_MESSAGES -- both constraints apply together, so
    the result is always within both. Also drops any leading ToolMessage(s)
    left with no preceding tool-calling AIMessage in the kept window.

    Real incident, 2026-08-16 (docs/plans/guardrails-plan.md): a pure position/
    age cut can land the trim boundary in the middle of a tool-call/
    tool-response pair -- a stored history like [..., AIMessage(tool_calls=
    [...]), ToolMessage, ToolMessage, AIMessage(final answer)] gets cut to
    just [ToolMessage, ToolMessage, AIMessage(final answer)] if the count
    cap falls right after the tool-calling AIMessage. DeepSeek's API
    rejects that outright (400: "Messages with role 'tool' must be a
    response to a preceding message with 'tool_calls'") -- not a rare
    edge case, since every news_query/set_interest/start_push/etc. turn
    involves at least one tool call, so any chat with enough such turns
    within the trim window is at risk."""
    kept = [(m, t) for m, t in zip(messages, timestamps) if now - t <= MAX_HISTORY_AGE]
    kept = kept[-MAX_HISTORY_MESSAGES:]
    while kept and getattr(kept[0][0], "type", None) == "tool":
        kept = kept[1:]
    if not kept:
        return [], []
    trimmed_messages, trimmed_timestamps = zip(*kept)
    return list(trimmed_messages), list(trimmed_timestamps)


def _get_trimmed_history(chat_id: int) -> tuple[list, list[datetime]]:
    """Reads this chat's history and trims it, storing the trimmed result
    back immediately -- stale entries get dropped every turn regardless
    of whether *this* turn's exchange ends up persisted (see
    handle_message), not just when a new message happens to push past
    the limit."""
    messages, timestamps = chat_histories.get(chat_id, ([], []))
    messages, timestamps = _trim_history(messages, timestamps, datetime.now(timezone.utc))
    chat_histories[chat_id] = (messages, timestamps)
    return messages, timestamps


_MARKDOWN_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _normalize_markdown_bold(text: str) -> str:
    """Safety net for when the model ignores agent.py's "no Markdown,
    HTML only" instruction and emits **bold** anyway (a real incident,
    2026-08-08 — see the smoke-test table in the
    build-locally-deploy-remotely skill). Prompt compliance alone isn't
    reliable enough (same lesson as guardrails.py's classifiers), so this
    converts stray **bold** into real Telegram HTML instead of relying
    purely on the model following the rule -- turns a literal-asterisk
    bug into a no-op when the model behaves, and into a fix when it
    doesn't."""
    return _MARKDOWN_BOLD_RE.sub(r"<b>\1</b>", text)


_TREND_REPORT_MARKER = "📰"


def _strip_report_preamble(text: str) -> str:
    """Safety net for when the model narrates its process before the
    actual trend report despite agent.TREND_REPORT_STRUCTURE explicitly
    forbidding it ("no preamble... start directly with the 📰 title
    line"). Real incident, 2026-08-09: verified live that this prompt-only
    instruction alone did not reliably stop the model writing things like
    "Let me compile these into a report" before the real content -- same
    lesson as _normalize_markdown_bold. If the report marker appears
    anywhere but the very start, strips everything before it; a no-op for
    replies that never use the marker (confirmations, etc.)."""
    idx = text.find(_TREND_REPORT_MARKER)
    if idx > 0:
        return text[idx:]
    return text


def split_for_telegram(text: str) -> list[str]:
    """Telegram rejects messages over 4096 characters. Split on that
    boundary, preferring to break at a newline, and never at a point that
    would leave an HTML tag open in one chunk with its closing tag in the
    next — Telegram would reject that chunk as unparseable entities."""
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        return [text]
    chunks = []
    while text:
        if len(text) <= TELEGRAM_MESSAGE_LIMIT:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
        if split_at <= 0:
            split_at = TELEGRAM_MESSAGE_LIMIT
        while split_at > 0 and not telegram_html.is_html_balanced(text[:split_at]):
            prev_newline = text.rfind("\n", 0, split_at)
            split_at = prev_newline if prev_newline > 0 else split_at - 1
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


async def notify_admin(admin_bot_token: str, admin_chat_id: int, chat_id: int, user) -> None:
    """Ping the admin with Approve/Deny buttons for a new access request.
    Sent via the admin bot's own token (not this bot's) so the resulting
    button tap's callback_query lands on admin_bot.py's update stream,
    not this process's."""
    label = f"@{user.username}" if user.username else (user.first_name or str(chat_id))
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Approve", callback_data=f"approve:{chat_id}"),
                InlineKeyboardButton("Deny", callback_data=f"deny:{chat_id}"),
            ]
        ]
    )
    await Bot(token=admin_bot_token).send_message(
        chat_id=admin_chat_id,
        text=f"New access request from {label} (chat_id={chat_id}).",
        reply_markup=keyboard,
    )


async def check_access(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Gate access per docs/plans/bot-features-plan.md item 1. Returns True if the
    sender may proceed; otherwise replies explaining why and returns
    False."""
    chat_id = update.effective_chat.id
    if chat_id == context.bot_data["admin_chat_id"]:
        return True

    status = subscriber_ops.get_status(chat_id)
    if status == subscriber_ops.APPROVED:
        return True
    if status == subscriber_ops.PENDING:
        await update.message.reply_text("Your access request is still pending approval.")
        return False
    if status == subscriber_ops.DENIED:
        await update.message.reply_text("Access denied.")
        return False

    user = update.effective_user
    subscriber_ops.request_access(chat_id, user.username, user.first_name)
    await update.message.reply_text(
        "This bot is private. Your access request was sent to the owner — "
        "you'll be notified once it's reviewed."
    )
    await notify_admin(
        context.bot_data["admin_bot_token"], context.bot_data["admin_chat_id"], chat_id, user
    )
    return False


async def handle_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start -- Telegram's standard bot-initiation command, sent
    automatically by the client's own "START" button on first contact
    with any bot. Real incident, 2026-08-09: a genuinely new user's very
    first interaction is /start, not free text -- and the plain-text
    MessageHandler explicitly excludes all commands (~filters.COMMAND),
    so without a dedicated handler here, that first message went
    completely unhandled: no reply, no pending-request DB row, no error
    anywhere. Every brand-new user hit this, not just one -- it's the
    literal first thing Telegram prompts someone to do.

    check_access() already handles the new/pending/denied cases fully
    (registers the request, notifies the admin, replies to the user) and
    returns False for all of them, so there's nothing more to do here in
    those cases. Only an *already-approved* user typing /start again
    falls through to the capabilities message below, since check_access
    returns True silently for them.

    Also serves /help, which asks the same question. One reply rather than
    two capability lists that drift apart -- guardrails.REDIRECT_MESSAGE is
    the single place that says what this bot can do, and the guardrail's
    own off-topic redirect uses it too."""
    if not await check_access(update, context):
        return
    await update.message.reply_text(guardrails.REDIRECT_MESSAGE, parse_mode=ParseMode.HTML)


async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Any /command no CommandHandler above claimed.

    Without this the reply is *silence*, which is worse than an error: the
    plain-text MessageHandler excludes commands (~filters.COMMAND), so an
    unregistered command matches no handler at all -- no reply, no log
    line, nothing. That is not hypothetical. It is exactly how /start
    behaved for every brand-new user until 2026-08-09, and how /help
    behaved until 2026-08-21, when a user reported that typing it did
    nothing.

    Fixing those one command at a time treats the symptom; this closes the
    class. Registered after the real commands, so they still match first,
    and it answers with the same capabilities message rather than a bare
    "unknown command" -- someone who guessed at a command wants to know
    what the right one is."""
    if not await check_access(update, context):
        return
    await update.message.reply_text(
        "🤔 I don't have that command.\n\n" + guardrails.REDIRECT_MESSAGE,
        parse_mode=ParseMode.HTML,
    )


async def handle_interests_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/interests -- show current interests. /interests <comma, separated,
    topics> -- set them. /interests clear -- clear them. Stored per-chat
    via subscriber_ops.py; injected into the agent's context on future messages
    (see handle_message) so it can prioritize a subscriber's own topics
    when their question is general -- see docs/plans/bot-features-plan.md."""
    if not await check_access(update, context):
        return

    chat_id = update.effective_chat.id
    text_after_command = update.message.text.partition(" ")[2].strip()

    if not text_after_command:
        interests = subscriber_ops.get_interests(chat_id)
        if interests:
            await update.message.reply_text("Your interests: " + ", ".join(interests))
        else:
            await update.message.reply_text(
                "You haven't set any interests yet. Use /interests topic1, topic2 to set them."
            )
        return

    if text_after_command.lower() == "clear":
        subscriber_ops.set_interests(chat_id, [])
        await update.message.reply_text("Interests cleared.")
        return

    # Normalized here too, not only on the conversational path: this
    # command bypasses the router entirely, so an interest typed as
    # "/interests 光通訊" would otherwise be stored untranslated and search
    # for nothing. Same reasoning as dispatch_settings.
    raw = [t.strip() for t in text_after_command.split(",") if t.strip()]
    # This command writes the list wholesale rather than going through
    # subscriber_ops.add_interest, so it has to enforce the cap itself -- without
    # this, "/interests a, b, c, ..." is a way straight past it. Refused
    # rather than truncated, so the subscriber picks which ones survive
    # instead of the parser picking for them.
    if len(raw) > subscriber_ops.MAX_INTERESTS:
        await update.message.reply_text(
            f"That's {len(raw)} interests, and the maximum is "
            f"{subscriber_ops.MAX_INTERESTS}. Send a shorter list."
        )
        return
    model = context.bot_data.get("guard_model")
    # Each interest is disambiguated against the others being set in the
    # same command -- "/interests AAOI, AOI, semiconductors" gives the
    # model the context it needs to tell those two apart.
    interests = []
    for i, t in enumerate(raw):
        peers = raw[:i] + raw[i + 1:]
        interests.append(
            (news_classify.normalize_interest(model, t, alongside=peers) or t)
            if model else t
        )
    subscriber_ops.set_interests(chat_id, interests)
    await update.message.reply_text("Interests updated: " + ", ".join(interests))


async def handle_language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/language -- show current reply-language preference. /language
    <name> -- set it (e.g. "/language Spanish"). /language clear --
    reset to matching whatever language the user writes in (the default).
    Mirrors handle_interests_command; also settable via natural language
    (the set_language tool, routed by the "set_language" category -- see
    guardrails.py) since this is the same command-or-conversation dual
    surface every other subscription feature in this bot has."""
    if not await check_access(update, context):
        return

    chat_id = update.effective_chat.id
    text_after_command = update.message.text.partition(" ")[2].strip()

    if not text_after_command:
        language = subscriber_ops.get_language(chat_id)
        if language:
            await update.message.reply_text(f"Your reply language is set to: {language}")
        else:
            await update.message.reply_text(
                "No reply language set -- I match whichever language you write in. "
                "Use /language <name> (e.g. /language Spanish) to set one."
            )
        return

    if text_after_command.lower() == "clear":
        subscriber_ops.set_language(chat_id, None)
        await update.message.reply_text("Reply language cleared -- back to matching your message's language.")
        return

    subscriber_ops.set_language(chat_id, text_after_command)
    await update.message.reply_text(f"Reply language set to: {text_after_command}")


def _translate_confirmation(model, text: str, language: str) -> str:
    """The one model call Route B ever makes -- only when the user has a
    reply-language preference set, translating a Route B template
    confirmation into it (see docs/plans/context-management-plan.md's
    settings-dispatch refactor). Plain text in/out; Route B's templates
    never contain HTML, so there's nothing to escape."""
    prompt = (
        f"Translate the following short message into {language}, keeping "
        "its friendly, concise tone. Reply with ONLY the translated text "
        "-- no quotes, no explanation, nothing else."
    )
    result = model.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": text}])
    return result.content


async def _route_b_reply(chat_id: int, classification, guard_model, category: str) -> tuple[str | None, str]:
    """Runs one Route B category (interests/push/language) and returns
    (blocked_at, reply) -- no history side effects, since a multi-category
    turn (see _process_multi_category) needs to combine several of these
    into one history entry, not one each. See
    docs/plans/context-management-plan.md's settings-dispatch refactor and
    agent.dispatch_settings's own docstring for why this is model-free
    except for the one translation call below."""
    reply = dispatch_settings(category, chat_id, classification, model=guard_model)

    # A language preference governs every reply, not just news_query --
    # checked *after* dispatch_settings, so a set_language turn's own
    # confirmation is written in the language it just set. Layer 4 only
    # runs on this path: a plain English template isn't model output, so
    # there's nothing to check when no translation happened.
    language = subscriber_ops.get_language(chat_id)
    if language:
        reply = await asyncio.to_thread(_translate_confirmation, guard_model, reply, language)
        # Same safety nets Route A applies to its own model-generated text
        # -- this is real LLM output too, not the fixed template above.
        reply = _strip_report_preamble(_normalize_markdown_bold(reply))
        output_on_topic = await asyncio.to_thread(guardrails.is_output_on_topic, guard_model, reply, category)
        if not output_on_topic:
            return "layer4_output_check", guardrails.REDIRECT_MESSAGE

    return None, reply


async def _route_a_reply(
    model, chat_id: int, user_text: str, history: list, category: str, guard_model, embedder=None
) -> tuple[str | None, str]:
    """Runs the news_query search (agent.search_news -- a plain,
    deterministic function, not a tool-calling agent loop, see its own
    docstring) and returns (blocked_at, reply). `history` is this chat's
    prior turns, used only by search_news's own query-rewrite step to
    resolve a context-dependent follow-up ("what about Nvidia?") --
    passed straight through, not otherwise touched here. guard_model/
    embedder are both optional -- search_news degrades gracefully without
    either, so a caller that hasn't built one yet (or a test) can pass
    embedder=None."""
    try:
        report = await asyncio.to_thread(search_news, chat_id, user_text, history, model, guard_model, embedder)
    except Exception as exc:
        return "agent_error", f"Something went wrong: {exc}"

    final_content = _strip_report_preamble(_normalize_markdown_bold(report))

    # Guardrail layer 4: re-checks the model's actual output before it's
    # sent -- the layer that catches drift layers 1-3 missed, since the
    # failure is only visible in what the model wrote, not the input.
    output_on_topic = await asyncio.to_thread(guardrails.is_output_on_topic, guard_model, final_content, category)
    if not output_on_topic:
        return "layer4_output_check", guardrails.REDIRECT_MESSAGE

    return None, final_content


def _persist_turn(chat_id: int, all_messages: list, history_timestamps: list[datetime], new_messages: list) -> None:
    """Stores `all_messages` (the full list to keep, including everything
    already in history) as this chat's new history, with a fresh shared
    timestamp for `new_messages` -- the portion actually added this turn.
    Only ever called once a turn is accepted -- a rejected exchange
    doesn't pollute the conversation history the next turn sees."""
    now = datetime.now(timezone.utc)
    chat_histories[chat_id] = (all_messages, history_timestamps + [now] * len(new_messages))


async def _process_single_category(
    chat_id: int, user_text: str, category: str, classification, model, guard_model, history: list,
    history_timestamps: list[datetime], embedder=None,
) -> dict:
    """The ordinary case -- classification.categories has exactly one
    entry, the overwhelmingly common shape. Kept as its own path (rather
    than always going through the general multi-category loop) for
    symmetry with _process_multi_category, even though Route A and
    Route B now persist history the same simple way -- see
    docs/plans/context-management-plan.md's multi-category routing
    section."""
    if category in ROUTE_B_CATEGORIES:
        blocked_at, reply = await _route_b_reply(chat_id, classification, guard_model, category)
    else:
        # Route A: news_query. chat_id feeds search_news's per-subscriber
        # daily quota/dedup memory; `history` feeds its query-rewrite step
        # (resolving a context-dependent follow-up like "what about
        # Nvidia?"), nothing else.
        blocked_at, reply = await _route_a_reply(
            model, chat_id, user_text, history, category, guard_model, embedder
        )
    if blocked_at is not None:
        return {"blocked_at": blocked_at, "category": category, "reply": reply}
    new_messages = [HumanMessage(content=user_text), AIMessage(content=reply)]
    _persist_turn(chat_id, history + new_messages, history_timestamps, new_messages)
    return {"blocked_at": None, "category": category, "reply": reply}


async def _process_multi_category(
    chat_id: int, user_text: str, classification, model, guard_model, history: list,
    history_timestamps: list[datetime], embedder=None,
) -> dict:
    """A message carrying more than one intent (e.g. "add robotics to my
    interests and tell me what's new with it") -- each category in
    classification.categories is dispatched in order and the replies are
    joined into one message. All-or-nothing: if any segment is blocked,
    the whole reply becomes REDIRECT_MESSAGE rather than sending a partial
    result -- simplest and safest, matching the single-category behavior
    this generalizes. See docs/plans/context-management-plan.md's
    multi-category routing section for the design and its open points.

    Each category sees the same original `history` and `user_text` --
    not each other's replies from this same turn -- and the whole turn is
    persisted as a single (Human, AI) pair with the joined final text, not
    one pair per category, so history stays one entry per real user turn."""
    reply_parts = []
    for category in classification.categories:
        if category in ROUTE_B_CATEGORIES:
            blocked_at, reply = await _route_b_reply(chat_id, classification, guard_model, category)
        else:
            blocked_at, reply = await _route_a_reply(
                model, chat_id, user_text, history, category, guard_model, embedder
            )
        if blocked_at is not None:
            return {"blocked_at": blocked_at, "category": category, "reply": guardrails.REDIRECT_MESSAGE}
        reply_parts.append(reply)

    final_content = "\n\n".join(reply_parts)
    new_messages = [HumanMessage(content=user_text), AIMessage(content=final_content)]
    _persist_turn(chat_id, history + new_messages, history_timestamps, new_messages)

    return {"blocked_at": None, "category": classification.categories[0], "reply": final_content}


async def process_message(chat_id: int, user_text: str, model, guard_model, embedder=None) -> dict:
    """The actual guardrail/agent/formatting pipeline, independent of
    Telegram's Update/Context objects -- extracted so test_api.py's local
    curl endpoint (docs/reference/local-testing-api-plan.md) exercises this exact
    logic, not a separate reimplementation that could silently drift from
    what real Telegram traffic runs. handle_message (below) is now a thin
    wrapper: Telegram-specific I/O only, no pipeline logic of its own.

    Returns {"blocked_at": str|None, "category": str|None, "reply": str}.
    blocked_at names which layer stopped the message (None if it went all
    the way through) -- useful for a test caller to assert on without
    parsing the reply text or cross-referencing docker logs/Logfire.
    `category` is the first (or only) category classified for this turn --
    see _process_multi_category for how more than one is handled.

    Logs and re-raises on an unhandled failure ANYWHERE in the pipeline
    below (a DeepSeek timeout, a tool call raising, guardrails.
    classify_message itself failing outside its own internal fail-open
    try/except) -- this is the ONE place both real Telegram traffic
    (handle_message, which has no try/except of its own around this
    call) and test_api.py's /test_message go through, so logging here
    once covers both instead of duplicating it in every caller. Before
    this, an unhandled failure here reached test_api.py's caller only as
    an HTTP response body (nothing durable if that wasn't captured), and
    reached handle_message's caller only via python-telegram-bot's own
    unstructured default error logging -- neither path was queryable in
    Logfire (PROD) or the file provider (INT), and neither showed up
    anywhere this project's own telemetry could see. Real incident,
    2026-09-03: a live INT deploy test hit exactly this, with no way to
    reconstruct what happened afterward."""
    try:
        # Guardrail layer 1: free, local, zero-LLM-call pre-filter. See
        # docs/plans/guardrails-plan.md for the incident this design responds to.
        if guardrails.fails_local_prefilter(user_text):
            return {"blocked_at": "layer1_prefilter", "category": None, "reply": guardrails.REDIRECT_MESSAGE}

        # Guardrail layer 2 -- the router (docs/plans/context-management-plan.md):
        # one structured-output call answers "is this on-topic", "what kind of
        # request(s) is this", and (for Route B categories) the arguments each
        # one needs -- gating the expensive agent call and, for settings
        # categories, replacing it entirely.
        classification = await asyncio.to_thread(guardrails.classify_message, guard_model, user_text)
        if not classification.on_topic:
            return {"blocked_at": "layer2_router", "category": classification.categories[0], "reply": guardrails.REDIRECT_MESSAGE}

        history, history_timestamps = _get_trimmed_history(chat_id)

        if len(classification.categories) == 1:
            return await _process_single_category(
                chat_id, user_text, classification.categories[0], classification, model, guard_model,
                history, history_timestamps, embedder,
            )
        return await _process_multi_category(
            chat_id, user_text, classification, model, guard_model, history, history_timestamps, embedder
        )
    except Exception as exc:
        _events.log("process_message_failed", {"message": "unhandled pipeline failure", "chat_id": chat_id},
                     level=Level.ERROR, exc=exc)
        raise


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_access(update, context):
        return

    result = await process_message(
        update.effective_chat.id,
        update.message.text,
        context.bot_data["model"],
        context.bot_data["guard_model"],
        context.bot_data.get("embedder"),
    )
    final_content = result["reply"]

    chunks = split_for_telegram(final_content)
    delivered_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            # Small gap between sequential messages -- avoids hitting
            # Telegram's rate limit on rapid-fire sends and reads as a
            # steady stream instead of a burst.
            await asyncio.sleep(1)
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
            delivered_chunks.append(chunk)
        except BadRequest as exc:
            # The model didn't produce valid HTML (unescaped &/</>, a tag
            # it wasn't asked to use, etc.) -- fall back to plain text so
            # the user still gets an answer instead of silence. Logged
            # (2026-08-27) because it was previously silent: a real user
            # report of a digest whose links had visibly vanished was the
            # only reason this fallback's existence was ever noticed, and
            # there was no way to find out afterward what actually broke
            # the HTML. strip_html_tags removes EVERY tag on failure, not
            # just the offending one, so this is also the only record of
            # what the user was supposed to see (links, bold, etc.) before
            # the fallback flattened it.
            print(f"[bot] HTML send failed, falling back to plain text: {exc!r} -- chunk: {chunk[:500]!r}")
            stripped = telegram_html.strip_html_tags(chunk)
            await update.message.reply_text(stripped)
            delivered_chunks.append(stripped)

    # Archives what was actually delivered (post any strip-to-plain
    # fallback above), not the raw pre-fallback reply -- a delivery
    # record, not a debug log (the BadRequest print above already covers
    # that). topic is the guardrail-classified category, the closest
    # analog to a push digest's interest since an interactive reply has
    # no fixed topic of its own.
    message_archive.archive_message(
        update.effective_chat.id, "chat_reply", "\n".join(delivered_chunks), topic=result["category"])


async def send_push_digest(bot: Bot, chat_id: int, text: str, topic: str | None = None) -> None:
    """The `send` callback news_push.run_push_cycle() calls per due
    subscriber. Push digests go through the same HTML-formatting/trend-
    report prompt (agent.HTML_FORMATTING_RULES/TREND_REPORT_STRUCTURE) as
    a normal chat reply and can fail the same ways, so this reuses
    handle_message's exact pipeline instead of a simpler one-off send: the
    Markdown-leak and report-preamble safety nets, chunking for messages
    over Telegram's limit, and the BadRequest-on-bad-HTML fallback to
    plain text.

    `topic` (added 2026-08-28, the subscriber's interest -- run_push_cycle
    already has it in scope per-loop-iteration) is only used to label the
    archived record below; delivery itself doesn't need it."""
    normalized = _strip_report_preamble(_normalize_markdown_bold(text))
    chunks = split_for_telegram(normalized)
    delivered_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            await asyncio.sleep(1)
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML)
            delivered_chunks.append(chunk)
        except BadRequest as exc:
            # See handle_message's identical except block for why this is
            # logged -- same failure mode, same previously-silent gap.
            print(f"[news_push] HTML send failed for chat_id={chat_id}, falling back to plain "
                  f"text: {exc!r} -- chunk: {chunk[:500]!r}")
            stripped = telegram_html.strip_html_tags(chunk)
            await bot.send_message(chat_id=chat_id, text=stripped)
            delivered_chunks.append(stripped)

    message_archive.archive_message(chat_id, "push_digest", "\n".join(delivered_chunks), topic=topic)


async def _push_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    model = context.bot_data["guard_model"]

    async def send(chat_id: int, text: str, topic: str | None = None) -> None:
        await send_push_digest(context.bot, chat_id, text, topic=topic)

    # .get(), not [] -- an embedder is an enhancement (near-duplicate
    # collapse, offbeat selection), never something run_push_cycle
    # requires to function. See news_embed's module docstring.
    #
    # No admin_bot_token/admin_chat_id here (removed 2026-08-28) -- the
    # retry loop no longer decides anything alert-worthy or sends
    # Telegram directly, see news_push._emit_html_validation_attempt.
    await news_push.run_push_cycle(model, send, embedder=context.bot_data.get("embedder"))


def register_push_job(app: Application) -> None:
    """Wires up the periodic-push scheduler (docs/plans/bot-features-plan.md item
    5) -- requires the apscheduler dependency (see environment.yml) for
    Application.job_queue to exist at all. Called by both bot.py's own
    main() and combined_bot.py's, so standalone and combined deployment
    both get push. `first=10` just avoids doing real work in the same
    instant as startup; the actual per-subscriber due-check is time-based
    (subscriber_ops's push_interval_hours / last_push_at), not this delay."""
    app.job_queue.run_repeating(_push_job, interval=PUSH_TICK_SECONDS, first=10)


async def _ingest_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    model = context.bot_data["guard_model"]
    # run_ingestion_cycle does synchronous network calls across every
    # enabled source -- offloaded the same way handle_message offloads
    # run_agent, so a slow cycle can't block the bot's event loop.
    await asyncio.to_thread(
        news_ingest.run_ingestion_cycle, model, embedder=context.bot_data.get("embedder")
    )
    await review_category_proposals(
        model, context.bot_data["admin_bot_token"], context.bot_data["admin_chat_id"]
    )


async def review_category_proposals(model, admin_bot_token: str, admin_chat_id: int,
                                    now: datetime | None = None) -> int:
    """Asks the admin about category proposals that have crossed the
    threshold. Returns how many were raised.

    Runs after ingestion because that is when new sightings appear, and
    from the ingest job rather than the admin bot because this is a push:
    nobody is going to open the admin bot to check whether the taxonomy has
    gaps.

    A proposal is marked alerted only AFTER its message is sent. Marking
    first would drop it permanently if the send failed -- `alerted_at IS
    NULL` is what makes it eligible, so a failed send would look exactly
    like a delivered one. The cost of this ordering is at worst a duplicate
    message when a retry succeeds, and a duplicate is visible while a lost
    proposal is not."""
    now = now or datetime.now(timezone.utc)
    ready = category_ops.categories_ready_for_review(now)
    if not ready:
        return 0

    rows = category_ops.get_active_categories()
    active = [name for name, _ in rows]
    taxonomy = news_classify.Taxonomy.from_rows(rows)
    bot = Bot(token=admin_bot_token)
    raised = 0
    for name, hits in ready:
        examples = category_ops.category_examples(name)
        draft = await asyncio.to_thread(
            news_classify.draft_category_description,
            model, name, [title for title, _ in examples], taxonomy,
        )
        text, keyboard = admin_bot.build_category_review(
            name, hits, examples, draft, active
        )
        try:
            await bot.send_message(chat_id=admin_chat_id, text=text,
                                   parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except Exception as exc:
            _events.log("category_admin_notify_failed",
                         {"message": f"could not raise category {name!r} with the admin",
                          "name": name},
                         level=Level.WARN, exc=exc)
            continue
        category_ops.mark_category_alerted(name, now, draft)
        raised += 1
    print(f"[news_ingest] raised {raised} category proposal(s) with the admin")
    return raised


def register_ingest_job(app: Application) -> None:
    """Wires up the periodic news-cache ingestion job -- see
    docs/plans/local-news-cache-plan.md. Same registration shape as
    register_push_job, called from the same places."""
    app.job_queue.run_repeating(_ingest_job, interval=INGEST_TICK_SECONDS, first=10)


async def _start_test_api(app: Application) -> None:
    """post_init hook -- run_polling() manages its own event loop
    internally, so this is the standalone-bot.py equivalent of
    combined_bot.py's run_both() starting test_api after the loop is
    already running (test_api.start() needs asyncio.get_running_loop())."""
    app.bot_data["test_api_server"] = test_api.start(
        app.bot_data["model"], app.bot_data["guard_model"], app.bot_data.get("embedder")
    )


async def _stop_test_api(app: Application) -> None:
    test_api.stop(app.bot_data.get("test_api_server"))


def main():
    setup_telemetry()
    storage.init_db()
    category_ops.bootstrap()
    token = get_settings().resolved("delivery.telegram.bot-token", required=True)
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
    # None on any failure (missing package, missing model files, out of
    # memory) rather than raising -- an embedder is an enhancement to
    # push quality, never something startup depends on. See news_embed's
    # module docstring.
    embedder = news_embed.build_embedder()

    app = Application.builder().token(token).post_init(_start_test_api).post_shutdown(_stop_test_api).build()
    app.bot_data["model"] = model
    app.bot_data["guard_model"] = guard_model
    app.bot_data["embedder"] = embedder
    app.bot_data["admin_chat_id"] = int(get_settings().resolved("delivery.telegram.admin-chat-id", required=True))
    app.bot_data["admin_bot_token"] = get_settings().resolved("delivery.telegram.admin-bot-token", required=True)
    # Idempotent -- safe to run every startup. Only the admin gets
    # search_news access to news_sources.RESTRICTED_SOURCES (NewsAPI,
    # Perigon) by default; granting it to anyone else is a plain DB update
    # (subscriber_ops.set_restricted_sources_enabled), not a new code path.
    subscriber_ops.set_restricted_sources_enabled(app.bot_data["admin_chat_id"], True)
    app.add_handler(CommandHandler(["start", "help"], handle_start_command))
    app.add_handler(CommandHandler("interests", handle_interests_command))
    app.add_handler(CommandHandler("language", handle_language_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Last: every real command above gets first refusal.
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))
    register_push_job(app)
    register_ingest_job(app)
    register_error_handler(app, "argus.bot")

    print("Telegram bot ready (polling). Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

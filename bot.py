"""
Telegram bot entry point for the agent — the headless alternative to
agent.py's CLI REPL. Polling mode: no public endpoint or TLS needed, and
the same shape works locally and in a long-running Kubernetes Deployment
later. See docs/plans/deployment-plan.md.

Reuses setup_telemetry from agent.py unchanged — this file only adds the
Telegram-specific plumbing (per-chat conversation state, handler
registration, the polling loop). Every on-topic message -- a news
question, adding/removing/redefining an interest, push settings, reply
language -- goes through the same always-on interest_finder conversational
agent now (see process_message, docs/plans/front-door-agent-plan.md);
agent.search_news is one of that agent's own tools, not a separate
deterministic dispatch path.

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
import time
from datetime import datetime, timedelta, timezone
from langchain_core.messages import AIMessage, HumanMessage
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from agent import build_model_from_settings, setup_telemetry
from app_settings import get_settings
import admin_bot
import guardrails
import interest_finder
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

# What a subscriber sees once subscriber_ops.try_consume_agent_interaction
# refuses them (see process_message's own comment on why the check sits
# where it does). Plain text, not guardrails.REDIRECT_MESSAGE -- this
# isn't a guardrail violation, it's a real, honest reason their message
# didn't go through.
TRIAL_AGENT_LIMIT_MESSAGE = (
    "You've used all of your trial's AI interactions. Contact the admin "
    "if you'd like more."
)

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

# Per-chat conversation state -- messages AND a pending confirmation
# offer, as ONE object with ONE lifetime (docs/plans/front-door-agent-plan.md).
# In-memory only — lost on restart, same as the CLI's messages list. Not
# persisted; fine for now, revisit if needed.
#
# conversations[chat_id] = {
#     "messages": list, "timestamps": list[datetime] (parallel to messages
#         -- needed for the age-based trim below since LangChain message
#         objects/dicts carry no wall-clock timestamp of their own),
#     "pending_offer": {"topic":..., "action":..., "definition":...,
#         "set_at": datetime} | None,
# }
#
# Before 2026-09-24 this was two separate structures (chat_histories +
# interest_sessions) with two separate lifetimes -- and that mismatch was
# a real defect (docs/plans/front-door-agent-plan.md's "actual defect"
# section): a session could close in the same turn a question was asked,
# while the message history survived, so a subscriber's later "yes" could
# outlive the very question it was answering and fall through to be
# misinterpreted elsewhere. Merging them into one object with one trim
# policy removes the mismatch instead of patching each way it could occur.
conversations: dict[int, dict] = {}

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


def _get_conversation(chat_id: int) -> dict:
    """Reads this chat's conversation and trims it, storing the trimmed
    result back immediately -- stale entries get dropped every turn
    regardless of whether *this* turn's exchange ends up persisted (see
    handle_message), not just when a new message happens to push past a
    limit.

    The pending offer ages out by the SAME MAX_HISTORY_AGE rule as the
    messages around it, via its own "set_at" -- it is part of this
    conversation's data now, not a separately-tracked session with its
    own lifetime (docs/plans/front-door-agent-plan.md). Two checks, not
    one, because MAX_HISTORY_AGE and MAX_HISTORY_MESSAGES are independent
    caps: a pure age check alone would miss the case where the COUNT cap
    trims away the very message that made the offer (its timestamp always
    matches or precedes "set_at") while the offer itself is still within
    the age window -- which would leave classify_confirmation's
    history[-1] anchor pointing at a later, unrelated reply. Code-review
    finding: caught before this could actually happen live."""
    conv = conversations.get(chat_id, {"messages": [], "timestamps": [], "pending_offer": None})
    now = datetime.now(timezone.utc)
    messages, timestamps = _trim_history(conv["messages"], conv["timestamps"], now)
    pending_offer = conv["pending_offer"]
    if pending_offer is not None:
        if now - pending_offer["set_at"] > MAX_HISTORY_AGE:
            pending_offer = None
        elif timestamps and pending_offer["set_at"] < timestamps[0]:
            pending_offer = None
    conv = {"messages": messages, "timestamps": timestamps, "pending_offer": pending_offer}
    conversations[chat_id] = conv
    return conv


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


async def _notify_admin_of_trial_limit(
    admin_bot_token: str, admin_chat_id: int, chat_id: int, label: str, reset_kind: str,
) -> None:
    """Ping the admin when a subscriber's free-trial allowance runs out --
    `label` is what shows in the message ("AI interaction" / "news push"),
    `reset_kind` is "reset_agent" or "reset_push", matching admin_bot.py's
    `handle_trial_reset` callback-data prefix.

    A deliberate, narrow reintroduction of direct-to-admin Telegram
    messaging for `news_push.py`'s caller specifically -- `_push_job`
    dropped admin_bot_token/admin_chat_id on 2026-08-28 because the push
    RETRY loop no longer decides anything alert-worthy (that moved to
    Logfire alerts). This is a different kind of event: a subscriber
    hitting a business-policy limit needs a human decision (reset or
    leave it), the same "admin stays in the loop" shape as `notify_admin`
    above for a new access request -- not an ops-health signal, so it
    does not belong on the Logfire-alerts side of that 2026-08-28 split.

    Fails open, deliberately (found in qa-engineer review, 2026-09-19): a
    best-effort side notification must never take down its caller's own
    primary flow if it breaks. Uncaught, this coroutine's own exception
    would have silenced `handle_message`'s reply to the subscriber
    entirely (it's awaited before their reply_text call) and, worse,
    would have aborted `run_push_cycle`'s WHOLE tick -- every other due
    subscriber not yet processed that cycle -- since the caller
    (`_stop_push_at_trial_limit`) runs outside `run_push_cycle`'s own
    per-subscriber try/except isolation. A failed admin ping should cost
    exactly one missed notification, nothing else."""
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Reset", callback_data=f"trial:{reset_kind}:{chat_id}")]]
    )
    try:
        await Bot(token=admin_bot_token).send_message(
            chat_id=admin_chat_id,
            text=f"Subscriber {chat_id} reached their {label} trial limit.",
            reply_markup=keyboard,
        )
    except Exception as exc:
        _events.log("trial_limit_admin_notify_failed",
                     {"message": "notifying admin of a trial-limit event failed", "chat_id": chat_id, "label": label},
                     level=Level.WARN, exc=exc)


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
    # for nothing.
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


async def _execute_pending_proposal(
    chat_id: int, user_text: str, pending: dict, guard_model, jev_api_key: str,
    history: list, history_timestamps: list[datetime],
) -> dict:
    """Deterministically completes a propose_interest or
    propose_definition that the subscriber just confirmed -- see
    interest_finder.propose_interest's docstring for why this exists (a
    real 2026-09-08 incident: a model claimed it saved an interest
    without ever calling the tool). No agent loop runs here;
    the write happens directly, in code, the moment classify_confirmation
    says "affirm" -- it cannot depend on the model remembering to act.

    An error or a layer-4 block here explicitly clears the conversation's
    pending_offer (rather than leaving it standing) -- a confirmed
    proposal that failed to execute must not silently bind to a later,
    unrelated "yes"."""
    topic, action = pending["topic"], pending["action"]
    try:
        if action == "add":
            reply = interest_finder.execute_save(chat_id, topic, pending["definition"], guard_model)
        elif action == "remove":
            reply = interest_finder.execute_drop(chat_id, topic)
        elif action == "redefine":
            reply = interest_finder.execute_redefine(chat_id, topic, pending["definition"])
        else:
            raise ValueError(f"unknown pending_proposal action: {action!r}")
    except Exception as exc:
        conversations[chat_id]["pending_offer"] = None
        _events.log("interest_exploration_failed",
                     {"message": "executing a confirmed proposal raised", "chat_id": chat_id, "topic": topic},
                     level=Level.ERROR, exc=exc)
        return {"blocked_at": "agent_error", "category": "find_interests",
                "reply": f"Something went wrong: {exc}"}

    language = subscriber_ops.get_language(chat_id)
    if language:
        reply = await asyncio.to_thread(_translate_confirmation, guard_model, reply, language)
        reply = _strip_report_preamble(_normalize_markdown_bold(reply))

    output_on_topic = await asyncio.to_thread(
        guardrails.is_output_on_topic, reply, jev_api_key, user_text)
    if not output_on_topic:
        conversations[chat_id]["pending_offer"] = None
        return {"blocked_at": "layer4_output_check", "category": "find_interests",
                "reply": guardrails.REDIRECT_MESSAGE}

    new_messages = [HumanMessage(content=user_text), AIMessage(content=reply)]
    _persist_turn(chat_id, history + new_messages, history_timestamps, new_messages, None)
    return {"blocked_at": None, "category": "find_interests", "reply": reply}


async def _lost_context_reply(chat_id: int, guard_model) -> str:
    """The honest reply for a message that only makes sense as answering
    a pending offer, when this conversation has no pending offer on
    record -- docs/plans/front-door-agent-plan.md's actual defect: history
    and a pending offer used to have different lifetimes, so a stray
    "yes" could survive the very question it was answering. A fixed,
    translated template, not a model-generated guess -- guessing what
    they meant is exactly what this replaces."""
    reply = ("I think that's answering something I asked earlier, but I don't have a "
             "record of it any more -- it's been a while, or the bot restarted in "
             "between. Could you say again what you'd like?")
    language = subscriber_ops.get_language(chat_id)
    if language:
        reply = await asyncio.to_thread(_translate_confirmation, guard_model, reply, language)
    return reply


async def _process_agent_turn(
    chat_id: int, user_text: str, model, guard_model, jev_api_key: str, embedder=None,
) -> dict:
    """The single front-door path for every on-topic message, regardless
    of what layer 2 would classify it as -- a news question, adding/
    removing/redefining an interest, push settings, reply language, all
    go through the same always-on interest_finder conversational agent
    now (docs/plans/front-door-agent-plan.md). Replaces the old Route A
    (news_query, dispatched straight to agent.search_news)/Route B
    (start_push/stop_push, dispatched straight to agent.enable_push/
    disable_push)/interest_finder-only split: Step A already gave this
    agent every tool those routes used, so there was nothing left for a
    separate deterministic dispatch to do that the agent's own
    tool-calling can't.

    A pending offer (an unanswered propose_interest/propose_remove/
    propose_definition) takes priority over everything below, INCLUDING
    layer 2 -- a bare "yes" carries no topical signal for the router to
    classify, so letting layer 2 see it first would route it somewhere
    unrelated. More generally, layer 2 only ever runs on the FIRST
    message of a fresh (empty-history) conversation -- see the comment at
    its call site below for why a narrower "only skip it when there's a
    pending offer" gate was tried first and measured to be a real
    regression.

    When there is neither a pending offer NOR any conversation history at
    all, a message that only makes sense as answering something gets the
    honest _lost_context_reply instead of being guessed at by the agent
    -- see interest_finder.reads_as_bare_confirmation's own docstring for
    the incident this exists to catch. The "no history either" condition
    matters: with real history the top-level agent can resolve an
    ordinary contextual follow-up ("the first one") itself, same as any
    other reference to earlier in the conversation -- this check is only
    for the case where there is nothing left to resolve it against at
    all."""
    conv = _get_conversation(chat_id)
    history, history_timestamps, pending_offer = conv["messages"], conv["timestamps"], conv["pending_offer"]

    if pending_offer is not None:
        # The subscriber's reply answers whatever the assistant said LAST
        # (history[-1], persisted at the end of the turn that set/refreshed
        # this offer) -- not the offer in isolation. See
        # classify_confirmation's own docstring for the incident this
        # anchoring fixes.
        last_assistant_reply = history[-1].content if history else ""
        verdict = await asyncio.to_thread(
            interest_finder.classify_confirmation, guard_model, user_text, last_assistant_reply)
        if verdict == "affirm":
            return await _execute_pending_proposal(
                chat_id, user_text, pending_offer, guard_model, jev_api_key, history, history_timestamps)
        if verdict == "decline":
            pending_offer = None
    elif not history and await asyncio.to_thread(
            interest_finder.reads_as_bare_confirmation, guard_model, user_text):
        reply = await _lost_context_reply(chat_id, guard_model)
        # Only real model output (a translation) needs layer 4 -- same
        # reasoning as _execute_pending_proposal: the untranslated English
        # template is our own fixed string, but a translated one is real,
        # unchecked LLM output like any other. No user_text here (skipping
        # the completeness question): this reply doesn't attempt to
        # address the subscriber's message, it says plainly that it
        # couldn't -- "did it address everything asked" isn't a
        # meaningful question for an apology.
        if subscriber_ops.get_language(chat_id):
            reply = _strip_report_preamble(_normalize_markdown_bold(reply))
            output_on_topic = await asyncio.to_thread(guardrails.is_output_on_topic, reply, jev_api_key)
            if not output_on_topic:
                return {"blocked_at": "layer4_output_check", "category": "context_lost",
                        "reply": guardrails.REDIRECT_MESSAGE}
        new_messages = [HumanMessage(content=user_text), AIMessage(content=reply)]
        _persist_turn(chat_id, history + new_messages, history_timestamps, new_messages, None)
        return {"blocked_at": None, "category": "context_lost", "reply": reply}

    category = "find_interests"
    if not history:
        # Guardrail layer 2 -- the router, on Jev since
        # docs/plans/front-door-agent-plan.md item 5: one Jev call answers
        # "is this on-topic" and "what
        # kind of request(s) is this". Only its on_topic gate drives
        # anything now; `categories` is kept purely as a label for the
        # return value/telemetry below, not to pick a dispatch path.
        #
        # Runs ONLY on the first message of a fresh (empty-history)
        # conversation -- ANY ongoing conversation skips it, not just one
        # with a live pending offer. Measured live (qa-engineer,
        # 2026-09-24): a topic-free but genuinely contextual reply
        # ("sure", "the first one") is misclassified as off-topic by this
        # same router a third to all of the time -- the old design's
        # blanket "any message in an open exploration skips layer 2"
        # protected against exactly this, and narrowing that to
        # "only when a pending offer exists" (this module's first attempt
        # at this) was a real regression, not a simplification. Layer 1
        # (local prefilter) and layer 4 (output check) still run
        # regardless, same as they always did for a mid-exploration
        # message under the old design -- layer 2 was never the sole
        # defense against a mid-conversation off-topic pivot.
        _t0 = time.monotonic()
        classification = await asyncio.to_thread(guardrails.classify_message, user_text, jev_api_key)
        _events.log("latency_layer2_classify", {"message": "router classified the message",
                     "duration_seconds": round(time.monotonic() - _t0, 3)})
        if not classification.on_topic:
            return {"blocked_at": "layer2_router", "category": classification.categories[0],
                    "reply": guardrails.REDIRECT_MESSAGE}
        category = classification.categories[0]

    session = {"pending_proposal": pending_offer}
    try:
        _t0 = time.monotonic()
        reply = await asyncio.to_thread(
            interest_finder.run_turn, chat_id, user_text, history, session, model, guard_model, embedder
        )
        _events.log("latency_agent_turn", {"message": "front-door agent turn returned",
                     "duration_seconds": round(time.monotonic() - _t0, 3)})
    except Exception as exc:
        _events.log("agent_turn_failed", {"message": "front-door agent turn raised", "chat_id": chat_id},
                     level=Level.ERROR, exc=exc)
        return {"blocked_at": "agent_error", "category": category, "reply": f"Something went wrong: {exc}"}

    final_content = _strip_report_preamble(_normalize_markdown_bold(reply))

    output_on_topic = await asyncio.to_thread(
        guardrails.is_output_on_topic, final_content, jev_api_key, user_text)
    if not output_on_topic:
        return {"blocked_at": "layer4_output_check", "category": category, "reply": guardrails.REDIRECT_MESSAGE}

    new_messages = [HumanMessage(content=user_text), AIMessage(content=final_content)]
    _persist_turn(chat_id, history + new_messages, history_timestamps, new_messages, session.get("pending_proposal"))
    return {"blocked_at": None, "category": category, "reply": final_content}


def _persist_turn(
    chat_id: int, all_messages: list, history_timestamps: list[datetime], new_messages: list,
    pending_offer: dict | None,
) -> None:
    """Stores `all_messages` (the full list to keep, including everything
    already in history) as this chat's new history, with a fresh shared
    timestamp for `new_messages` -- the portion actually added this turn
    -- and `pending_offer` as the conversation's new outstanding proposal
    (or None). Only ever called once a turn is accepted -- a rejected
    exchange doesn't pollute the conversation the next turn sees.

    Stamps a fresh "set_at" onto `pending_offer` only if it doesn't
    already have one -- a brand-new proposal (propose_interest et al.
    always build a fresh dict with no "set_at") gets timed from now, while
    one just carried over unresolved from a prior turn (verdict was
    "unclear") keeps aging from when it was FIRST offered, not from every
    turn it's merely still standing."""
    now = datetime.now(timezone.utc)
    if pending_offer is not None and "set_at" not in pending_offer:
        pending_offer = {**pending_offer, "set_at": now}
    conversations[chat_id] = {
        "messages": all_messages,
        "timestamps": history_timestamps + [now] * len(new_messages),
        "pending_offer": pending_offer,
    }


async def process_message(
    chat_id: int, user_text: str, model, guard_model, jev_api_key: str, embedder=None,
) -> dict:
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
    `category` reflects what layer 2 classified the message as when it
    ran (see _process_agent_turn) -- purely informational now, not a
    dispatch decision; every on-topic message goes to the same agent.

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

        # Free-trial usage cap (requested 2026-09-18, changed 2026-09-19
        # after live INT testing surfaced that checking it only for a
        # brand-new request read as "no limit" to a subscriber who kept a
        # conversation going past their allowance). Every turn now spends
        # one interaction if the subscriber has a finite allowance,
        # whether it opens a new request or continues one already in
        # progress. Checked before anything else that costs a model call.
        # See subscriber_ops.try_consume_agent_interaction's own docstring
        # for what "no limit" (NULL/-1) means.
        if not subscriber_ops.try_consume_agent_interaction(chat_id):
            return {"blocked_at": "trial_limit_reached", "category": None, "reply": TRIAL_AGENT_LIMIT_MESSAGE}

        return await _process_agent_turn(chat_id, user_text, model, guard_model, jev_api_key, embedder)
    except Exception as exc:
        _events.log("process_message_failed", {"message": "unhandled pipeline failure", "chat_id": chat_id},
                     level=Level.ERROR, exc=exc)
        raise


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_access(update, context):
        return

    chat_id = update.effective_chat.id
    result = await process_message(
        chat_id,
        update.message.text,
        context.bot_data["model"],
        context.bot_data["guard_model"],
        context.bot_data["jev_api_key"],
        context.bot_data.get("embedder"),
    )
    final_content = result["reply"]

    if result["blocked_at"] == "trial_limit_reached":
        await _notify_admin_of_trial_limit(
            context.bot_data["admin_bot_token"], context.bot_data["admin_chat_id"],
            chat_id, "AI interaction", "reset_agent",
        )

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

    async def notify_admin_of_push_limit(chat_id: int) -> None:
        await _notify_admin_of_trial_limit(
            context.bot_data["admin_bot_token"], context.bot_data["admin_chat_id"],
            chat_id, "news push", "reset_push",
        )

    # .get(), not [] -- an embedder is an enhancement (near-duplicate
    # collapse, offbeat selection), never something run_push_cycle
    # requires to function. See news_embed's module docstring.
    #
    # admin_bot_token/admin_chat_id were removed from here 2026-08-28
    # because the retry loop didn't decide anything alert-worthy (ops
    # health moved to Logfire alerts, see news_push._emit_html_validation_attempt).
    # `notify_admin_of_push_limit` above is a narrower, deliberate
    # reintroduction for one specific business-policy event (a
    # subscriber's free-trial push allowance running out) -- see
    # _notify_admin_of_trial_limit's own docstring for why that's a
    # different kind of event, not a reversal of the 2026-08-28 reasoning.
    # Named distinctly from the module-level `notify_admin` (the
    # new-access-request approval ping, different signature entirely) so
    # the two don't read as the same function at a glance.
    await news_push.run_push_cycle(
        model, send, embedder=context.bot_data.get("embedder"),
        notify_admin=notify_admin_of_push_limit)


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
        app.bot_data["model"], app.bot_data["guard_model"], app.bot_data["jev_api_key"],
        app.bot_data.get("embedder"),
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
    # Jev (TypeSafe AI), reached via OpenRouter -- backs guardrails.py's
    # layers 2 and 4 (docs/plans/front-door-agent-plan.md item 5), not a
    # models.* entry since it isn't OpenAI-wire-compatible (jev_client.py
    # calls OpenRouter's decisions endpoint directly). required=True: with
    # Route A/B retired, there's no fallback path left for either layer
    # once this is missing, same criticality as DEEPSEEK_API_KEY above.
    jev_api_key = get_settings().resolved("jev.api-key", required=True)
    # None on any failure (missing package, missing model files, out of
    # memory) rather than raising -- an embedder is an enhancement to
    # push quality, never something startup depends on. See news_embed's
    # module docstring.
    embedder = news_embed.build_embedder()

    app = Application.builder().token(token).post_init(_start_test_api).post_shutdown(_stop_test_api).build()
    app.bot_data["model"] = model
    app.bot_data["guard_model"] = guard_model
    app.bot_data["jev_api_key"] = jev_api_key
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

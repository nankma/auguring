# Stays on conda-forge via micromamba (same as local dev and CI) rather than
# introducing a separate pip requirements file — one dependency list
# (environment.yml), no drift risk between local/CI/container installs.
FROM mambaorg/micromamba:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Bakes the embedding model's weights (~30 MB) into this layer at build
# time, with real network access -- so the running container never needs
# to reach huggingface.co. news_embed.py sets HF_HUB_OFFLINE=1 at import
# time specifically so a 1-OCPU VM with no guaranteed outbound access to
# that host gets a loud ValueError on any unexpected download attempt at
# runtime, never a silent hang or a slow ingestion cycle. Placed before
# the app-code COPY below so an ordinary commit doesn't invalidate this
# layer -- same reasoning as the micromamba install layer above it.
RUN micromamba run -n base python -c \
    "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M')"

# Same reasoning, same placement, for news_keyness.py's NLTK data: baked
# in at build time (with real network access) so the running container
# never reaches nltk.org's download host. Only the POS tagger and
# tokenizer -- no wordnet/omw-1.4, which docs/analysis/cluster-
# measurements.md's "Offbeat selection, take two" section found made
# results WORSE (a concrete/abstract noun filter that dropped coverage
# from 551/999 to 156/999 scorable articles) and was dropped from the
# shipped approach entirely. Downloads to NLTK's own default search path
# (~/nltk_data under $MAMBA_USER, the same user this RUN executes as),
# so news_keyness.py needs no explicit nltk.data.path wiring at runtime --
# unlike HF_HUB_OFFLINE for model2vec, NLTK has no "loud error instead of
# a network attempt" flag, so news_keyness.py's own fail-open try/except
# around pos_tag()/word_tokenize() is what stands in for that guard.
RUN micromamba run -n base python -c \
    "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt_tab')"

WORKDIR /app
# settings.yml (the local-dev file, relative paths like "news_cache")
# is deliberately NOT copied in -- if it were, it would sit at the
# exact default path SETTINGS_FILE looks for (app_settings.py), and the
# container would silently pick up wrong, non-persistent relative
# paths instead of the required=True check failing loudly the way it's
# supposed to when SETTINGS_FILE isn't explicitly configured for
# production (see docs/standaloneplan/01-settings-migration.md). Don't
# add it back without re-reading that reasoning first.
COPY --chown=$MAMBA_USER:$MAMBA_USER agent.py app_settings.py news_sources.py news_cache.py news_classify.py news_embed.py news_keyness.py news_ingest.py news_push.py bot.py admin_bot.py combined_bot.py telemetry.py ptb_error_handler.py guardrails.py subscriber_ops.py category_ops.py push_outcome_ops.py api_budget_ops.py interest_cache_ops.py source_state_ops.py test_api.py telegram_html.py message_archive.py settings.oracle.yml settings.int.yml docker-entrypoint.sh ./
# news_sources.py/telemetry.py each import from their own package -- a
# directory can't sit in the flat file list above (COPY needs its own
# destination), so each needs its own line. Missed once already
# (2026-09-02, caught by deploy-engineer before it reached INT: the
# file-list COPY above silently skips anything not named in it, no
# error, so the image built "successfully" without news_adapters/ and
# every import of news_sources crash-looped the container at startup)
# -- confirmed as a repeat class of bug, not a one-off, when
# telemetry_providers/ was added the same way and initially missed here
# too (caught by code review before it ever reached a build). If a
# future package gets added the same way, it needs a line here too, not
# just a file entry above.
COPY --chown=$MAMBA_USER:$MAMBA_USER news_adapters ./news_adapters
COPY --chown=$MAMBA_USER:$MAMBA_USER telemetry_providers ./telemetry_providers
COPY --chown=$MAMBA_USER:$MAMBA_USER storage ./storage
COPY --chown=$MAMBA_USER:$MAMBA_USER vector_store ./vector_store
RUN chmod +x docker-entrypoint.sh

# Real incident, 2026-08-09: `docker logs` returned zero lines for this
# container's entire uptime -- not even the "Both bots ready (polling)"
# startup print(), despite the process clearly running. Python
# block-buffers stdout when it isn't a TTY (true for any `docker run -d`
# container), so print() output -- including news_push.py's per-cycle
# outcome logging, added specifically for production visibility into the
# push scheduler -- was silently never reaching the log stream at all.
# PYTHONUNBUFFERED forces unbuffered stdout/stderr regardless of TTY
# status; standard fix for exactly this class of Docker logging gap.
ENV PYTHONUNBUFFERED=1

# DEEPSEEK_API_KEY, TELEGRAM_BOT_TOKEN, ADMIN_CHAT_ID, and ADMIN_BOT_TOKEN
# are never baked into the image. On OCI, docker-entrypoint.sh fetches them
# from OCI Vault at startup via Instance Principal auth -- pass the
# *_SECRET_OCID vars (not secrets themselves, just resource identifiers)
# instead of the real values. Elsewhere (local/Docker Desktop testing),
# pass the plain env vars directly via `docker run -e` as before -- the
# entrypoint only touches Vault when a *_SECRET_OCID var is actually set.
# See docs/security-plan.md finding 2 and docs/deployment-plan.md.
# LOGFIRE_ENABLED / LOGFIRE_API_KEY / SUBSCRIBERS_DB_FILE are optional,
# same reasoning as always.
#
# CMD runs combined_bot.py -- both Telegram bots (info + admin) in one
# process, one container, so LangChain/python-telegram-bot are only loaded
# once. Matters on small-RAM Always Free shapes (e.g. Oracle's
# VM.Standard.E2.1.Micro, 1GB). The two-bot-token security design is
# unchanged -- see docs/bot-features-plan.md item 1. bot.py and
# admin_bot.py can still be run standalone/as separate containers if ever
# wanted (`docker run ... myfirstagent-bot python bot.py`), e.g. if this
# ever moves to a shape with enough RAM that splitting them back out is
# preferable for isolation.
# Which commit is this image? Until 2026-08-22 nothing could answer that:
# `docker inspect` showed only the base image's labels, so a running
# container could be compared to another image by id but never traced back
# to source. tools/deploy.sh sets this and refuses a rebuild when nothing
# in the COPY list above changed since the deployed commit -- a restart is
# not free, it re-runs the push job at first=10s.
#
# Last so a new commit does not invalidate the expensive micromamba layer.
ARG GIT_COMMIT=unknown
LABEL commit=$GIT_COMMIT

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "combined_bot.py"]

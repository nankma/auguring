#!/usr/bin/env bash
# Fetches the real secret values from OCI Vault at container startup, using
# Instance Principal auth (no credentials file, no long-lived key baked into
# the image or passed via `docker run -e` -- see docs/security-plan.md
# finding 2). Only works when the container is actually running on an OCI
# Compute instance whose Dynamic Group has been granted `read secret-family`
# -- see docs/deployment-plan.md for the Vault/Dynamic Group/Policy setup.
#
# The *_SECRET_OCID env vars are resource identifiers, not secrets
# themselves -- safe to pass via `docker run -e` same as the VCN/instance
# OCIDs elsewhere in this project's docs.
set -e

# Setting our own ENTRYPOINT replaces the mambaorg/micromamba base image's
# own entrypoint, which is what normally activates the conda env (without
# this, `python`/`oci` aren't on PATH at all) -- reuse the same activation
# script the base image's own entrypoint calls, so this composes with the
# base image instead of fighting it.
source /usr/local/bin/_activate_current_env.sh

# Only fetches from Vault when a *_SECRET_OCID var is actually set -- lets
# local/Docker Desktop testing keep working unchanged by passing plain
# DEEPSEEK_API_KEY etc. via `docker run -e` (no OCI instance, no metadata
# service, no Vault access there), while the real deployment passes the
# *_SECRET_OCID vars instead and everything gets fetched for real.
fetch_secret() {
    oci secrets secret-bundle get --secret-id "$1" --auth instance_principal \
        --query 'data."secret-bundle-content".content' --raw-output | base64 -d
}

if [ -n "$DEEPSEEK_API_KEY_SECRET_OCID" ]; then
    export DEEPSEEK_API_KEY="$(fetch_secret "$DEEPSEEK_API_KEY_SECRET_OCID")"
fi
if [ -n "$TELEGRAM_BOT_TOKEN_SECRET_OCID" ]; then
    export TELEGRAM_BOT_TOKEN="$(fetch_secret "$TELEGRAM_BOT_TOKEN_SECRET_OCID")"
fi
if [ -n "$ADMIN_BOT_TOKEN_SECRET_OCID" ]; then
    export ADMIN_BOT_TOKEN="$(fetch_secret "$ADMIN_BOT_TOKEN_SECRET_OCID")"
fi
if [ -n "$ADMIN_CHAT_ID_SECRET_OCID" ]; then
    export ADMIN_CHAT_ID="$(fetch_secret "$ADMIN_CHAT_ID_SECRET_OCID")"
fi
# Resolving this secret is what turns tracing on now -- there's no
# separate LOGFIRE_ENABLED gate any more (telemetry.py reads
# telemetry.providers instead; an empty/absent list is off, same as
# not setting this secret at all). See settings.oracle.yml's own
# telemetry: comment.
if [ -n "$LOGFIRE_API_KEY_SECRET_OCID" ]; then
    export LOGFIRE_API_KEY="$(fetch_secret "$LOGFIRE_API_KEY_SECRET_OCID")"
fi
if [ -n "$GNEWS_API_KEY_SECRET_OCID" ]; then
    export GNEWS_API_KEY="$(fetch_secret "$GNEWS_API_KEY_SECRET_OCID")"
fi
if [ -n "$PERIGON_API_KEY_SECRET_OCID" ]; then
    export PERIGON_API_KEY="$(fetch_secret "$PERIGON_API_KEY_SECRET_OCID")"
fi
if [ -n "$NEWSAPI_API_KEY_SECRET_OCID" ]; then
    export NEWSAPI_API_KEY="$(fetch_secret "$NEWSAPI_API_KEY_SECRET_OCID")"
fi
# Backs guardrails.py's layers 2 and 4 via Jev/OpenRouter
# (docs/plans/front-door-agent-plan.md item 5) -- required=True in
# settings.oracle.yml's jev.api-key, so the container will not start
# without this resolving to a real value.
if [ -n "$OPENROUTER_API_KEY_SECRET_OCID" ]; then
    export OPENROUTER_API_KEY="$(fetch_secret "$OPENROUTER_API_KEY_SECRET_OCID")"
fi

exec "$@"

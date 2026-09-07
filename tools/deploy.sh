#!/usr/bin/env bash
#
# Build, transfer, restart, verify. No LLM in the loop.
#
# Why this exists
# ---------------
# Deploying is deterministic: ten commands in a fixed order with fixed
# checks. Driving it through an agent cost 184k tokens the first time and
# 100k the second, and almost none of that was the commands. A model is
# stateless, so every tool call re-sends the whole conversation -- measured
# on this project, 3,212 turns carried 1.6 BILLION re-read tokens against
# 6,411 tokens of genuinely new input, averaging half a million tokens of
# context per call. Cost scales with (turns x conversation length), which
# grows quadratically. A script's context is zero, forever.
#
# So: the happy path runs here. An agent is for DIAGNOSIS when this script
# fails -- hand it the log path this prints, not the whole deploy.
#
# Every check below exists because something once went wrong without it.
# `docs/plans/deployment-plan.md` and the `build-locally-deploy-remotely`
# skill have the incident histories; the one-line reasons are inline here.
#
# Usage:
#   tools/deploy.sh              # deploy HEAD of main
#   tools/deploy.sh --dry-run    # preflight only, change nothing
#   tools/deploy.sh --skip-ci    # deploy without waiting on CI (say why)
#   tools/deploy.sh --force      # deploy even if the image would be identical
#
set -uo pipefail

# On a Windows dev machine (Git Bash/MSYS), any argv token that looks like a
# Unix absolute path -- including embedded after `KEY=`, e.g. `-e
# SETTINGS_FILE=/app/settings.oracle.yml` -- gets silently rewritten to a
# Windows path (e.g. `C:/Program Files/Git/app/settings.oracle.yml`) before
# reaching a native .exe like docker.exe. Found 2026-09-01 deploying 21144dd:
# the new import-check env vars (below) were passed straight through as
# mangled paths, so trailsign resolved SETTINGS_FILE to a nonexistent file
# and raised SettingsError -- indistinguishable at a glance from the image
# genuinely being broken. Disabling MSYS's path conversion for this whole
# script is safe: nothing here relies on it, and every other Windows path
# (SSH_KEY, INFRA) is already converted explicitly instead. A no-op on
# Linux/macOS, where this variable does nothing.
export MSYS_NO_PATHCONV=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

INFRA="local-infra/infrastructure.yaml"     # gitignored: real IPs and keys
IMAGE="myfirstagent-bot"
CONTAINER="myfirstagent-bot"
LOGDIR="${TMPDIR:-/tmp}/argus-deploy-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"

DRY_RUN=0
SKIP_CI=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --skip-ci) SKIP_CI=1 ;;
        --force)   FORCE=1 ;;
        *) echo "unknown argument: $arg"; exit 2 ;;
    esac
done

step()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()    { printf '    ok  %s\n' "$*"; }
warn()  { printf '    !!  %s\n' "$*"; }
die()   { printf '\n\033[31mFAILED: %s\033[0m\n' "$*"; printf 'logs: %s\n' "$LOGDIR"; exit 1; }

# Runs a command with its output in a log file rather than on screen. On
# failure it surfaces only the lines that look like the error, never the
# whole log: a build log is thousands of lines of progress noise around one
# real message, and every line printed here is re-sent as input on every
# later call. The full log stays on disk for a human or an agent.
run() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    if "$@" > "$log" 2>&1; then
        return 0
    fi
    printf '\n    error lines from %s (full log on disk):\n' "$name"
    grep -iE "^ *(error|fatal)|error:|failed to|cannot |no such |permission denied" "$log" \
        | grep -viE "0 error|warning" | tail -6 | sed 's/^/        /'
    printf '        ...full log: %s\n' "$log"
    return 1
}

# Runs one of tools/'s verification scripts from HERE, not inside the
# container: they open their own SSH connections, and tools/ is
# deliberately outside the Dockerfile COPY list so nothing under it exists
# in the image.
#
# Defined up here with the other helpers, not next to its callers. A
# refactor on 2026-08-23 left it defined AFTER the verify section, so bash
# reported `host_check: command not found` for three checks and the script
# still printed "Deployed and verified" in green -- a deploy that retired
# Phoenix while its replacement went unverified.
host_check() {
    local name="$1"; shift
    if "$PY" "$@" > "$LOGDIR/$name.log" 2>&1; then
        ok "$name"
    else
        warn "$name FAILED -- see $LOGDIR/$name.log"
        grep -iE "error|failed|refused|timed out" "$LOGDIR/$name.log" | tail -3 | sed 's/^/        /'
        FAILURES=$((FAILURES + 1))
    fi
}

# ---------------------------------------------------------------- preflight

step "Preflight"

[ -f "$INFRA" ] || die "$INFRA not found -- it is gitignored, so a fresh clone has to obtain it separately"

# Scoped to the vm_bot:/ssh: top-level blocks specifically, not just the
# first match in the whole file. Found live 2026-09-07 deploying 6059823:
# an unscoped `grep -m1 -E '^\s+user:'` matched
# local-dev-machine-postgres:'s `user: myfirstagent` (a Postgres DB user,
# sitting earlier in the file) instead of ssh:'s `user: ubuntu` -- same
# "unscoped first match" failure class DOCKER_RUN below was already fixed
# for on 2026-09-01, just not applied here yet. Caught before any harm:
# the wrong SSH_USER made every SSH call in this script auth-fail, so the
# Transfer step died cleanly rather than doing anything to the VM -- but
# it burned a full build+preflight cycle to find out. Anchoring to
# `/^vm_bot:/`/`/^ssh:/` (column 0, so only a real top-level key matches)
# closes this the same way DOCKER_RUN's own comment already explains.
VM_IP=$(awk '/^vm_bot:/{f=1} f && /^\s+public_ip:/{print $2; exit}' "$INFRA")
SSH_KEY=$(awk '/^ssh:/{f=1} f && /^\s+private_key:/{sub(/^[^:]*: */,""); print; exit}' "$INFRA" | tr -d '\r')
SSH_USER=$(awk '/^ssh:/{f=1} f && /^\s+user:/{print $2; exit}' "$INFRA")
[ -n "$VM_IP" ] && [ -n "$SSH_KEY" ] && [ -n "$SSH_USER" ] || die "could not read VM details from $INFRA"
# Windows path from the yaml -> a path ssh can use under Git Bash.
case "$SSH_KEY" in
    [A-Za-z]:\\*) SSH_KEY="/$(echo "${SSH_KEY:0:1}" | tr 'A-Z' 'a-z')/$(echo "${SSH_KEY:3}" | tr '\\' '/')" ;;
esac
[ -f "$SSH_KEY" ] || die "ssh key not found at $SSH_KEY"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$SSH_USER@$VM_IP")
ok "target $SSH_USER@$VM_IP"

# The project's interpreter, not whatever `python` resolves to. The tools
# under tools/ import project modules -- check_logfire.py pulls in agent.py
# for SERVICE_NAME, which needs langchain -- and a bare `python` on this
# machine has none of that. Found 2026-08-23 when the check finally ran and
# reported ModuleNotFoundError instead of a telemetry verdict.
PY=$(conda run -n myfirstagent python -c "import sys; print(sys.executable)" 2>/dev/null | tr -d '')
[ -x "$PY" ] || PY=python
"$PY" -c "import langchain" 2>/dev/null     || die "no interpreter with the project's dependencies (tried: $PY). Is the myfirstagent env present?"
ok "python: $PY"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "main" ] || warn "on branch '$BRANCH', not main -- deploying it anyway"

# The image is built from the WORKING TREE, not from the commit. A dirty
# tree therefore ships something no commit describes, which is how you get
# a container nobody can reproduce.
if [ -n "$(git status --porcelain -- '*.py' Dockerfile environment.yml docker-entrypoint.sh)" ]; then
    warn "uncommitted changes to files that go into the image:"
    git status --porcelain -- '*.py' Dockerfile environment.yml docker-entrypoint.sh | sed 's/^/        /'
    warn "the image will contain these, but no commit will describe them"
fi

# CRLF in a shebang breaks ENTRYPOINT with `env: 'bash\r': No such file`.
# .gitattributes should prevent it; verify rather than trust, because the
# working tree is what gets built and autocrlf rewrites on checkout.
if grep -qU $'\r' docker-entrypoint.sh 2>/dev/null; then
    die "docker-entrypoint.sh has CRLF line endings. Fix: sed -i 's/\\r\$//' docker-entrypoint.sh"
fi
ok "docker-entrypoint.sh is LF"

# Does this change reach the container at all? Files outside the Dockerfile
# COPY list (docs, tools/, .claude/) change nothing in the image, and
# rebuilding for them is pure cost -- see CLAUDE.md's deploy policy. Worse
# than pure cost, in fact: a restart re-runs the push job at first=10s and
# re-opens the window where a deploy can kill a push cycle between the send
# and record_push, so a pointless deploy is not a harmless one.
DEPLOYED_ID=$("${SSH[@]}" "sudo docker inspect -f '{{.Image}}' $CONTAINER" 2>/dev/null | tr -d '\r')
ok "container currently runs image ${DEPLOYED_ID:0:19}"

COPIED=$(sed -n 's/^COPY .*MAMBA_USER \(.*\) \.\/$/\1/p' Dockerfile)
COPIED="$COPIED environment.yml Dockerfile"
DEPLOYED_SHA=$("${SSH[@]}" "sudo docker inspect -f '{{index .Config.Labels \"commit\"}}' $CONTAINER" 2>/dev/null | tr -d '\r')
if [ -n "$DEPLOYED_SHA" ] && [ "$DEPLOYED_SHA" != "<no value>" ]; then
    CHANGED=$(git diff --name-only "$DEPLOYED_SHA"..HEAD -- $COPIED 2>/dev/null)
    if [ -z "$CHANGED" ]; then
        warn "nothing in the image changed since ${DEPLOYED_SHA:0:8}"
        [ "$FORCE" -eq 1 ] || die "refusing a no-op deploy. Pass --force if you want the restart itself."
    fi
fi

# Is a push about to fire? A deploy is not a neutral act for the push job.
# `docker stop` kills the container between send() and record_push() if it
# lands mid-cycle, which leaves last_push_at unadvanced and re-sends the
# same digest after restart; and `first=10` re-runs the whole cycle ten
# seconds after start, so anyone already due is pushed immediately rather
# than at the next quarter hour. Neither is fatal, both are avoidable by
# not deploying into the window.
DUE=$("${SSH[@]}" "sudo docker exec -e SUBSCRIBERS_DB_FILE=/data/subscribers.db $CONTAINER \
    /opt/conda/bin/python -c \"
import users_db
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
for s in users_db.list_push_enabled_subscribers():
    lp, iv = s['last_push_at'], s['push_interval_hours']
    left = 0 if lp is None else iv - (now - lp).total_seconds()/3600
    if left < 0.25:
        print(f\\\"{s['chat_id']} due in {max(left,0):.2f}h\\\")
\"" 2>/dev/null | tr -d '\r')
if [ -n "$DUE" ]; then
    warn "a push is due within 15 minutes:"
    echo "$DUE" | sed 's/^/        /'
    [ "$FORCE" -eq 1 ] || die "deploying now could re-send or bring forward that push. Wait, or pass --force."
else
    ok "no push due within 15 minutes"
fi

if [ "$SKIP_CI" -eq 0 ]; then
    HEAD_SHA=$(git rev-parse HEAD)
    CI_STATUS=$(gh run list --limit 20 --json headSha,conclusion,status \
        --jq "[.[] | select(.headSha==\"$HEAD_SHA\")] | .[0] | \"\(.status)/\(.conclusion)\"" 2>/dev/null)
    case "$CI_STATUS" in
        completed/success) ok "CI green for ${HEAD_SHA:0:8}" ;;
        ""|null*)          die "no CI run found for ${HEAD_SHA:0:8}. Push it, or pass --skip-ci and say why." ;;
        *)                 die "CI for ${HEAD_SHA:0:8} is '$CI_STATUS'. A green local pytest is not evidence about CI." ;;
    esac
else
    warn "--skip-ci: CI not checked"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    step "Dry run: stopping before the build. Nothing was changed."
    exit 0
fi

# -------------------------------------------------------------------- build

step "Build"
run build docker build --build-arg "GIT_COMMIT=$(git rev-parse HEAD)" -t "$IMAGE" . || die "docker build failed -- see $LOGDIR/build.log"
LOCAL_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")
ok "built ${LOCAL_ID:0:19}"

# Two separate checks. The import proves the app is installed; the
# entrypoint one proves the container can actually START, which the import
# does NOT -- it bypasses ENTRYPOINT entirely, and that is exactly how a
# CRLF entrypoint reached production undetected.
#
# SETTINGS_ENV below is required since the settings-migration work
# (docs/standaloneplan/01-settings-migration.md): users_db.py/news_cache.py/
# message_archive.py now call trailsign's resolved(..., required=True) for
# storage.subscribers_db_file/news_cache_dir/news_archive_dir/
# message_archive_dir at IMPORT time, via settings.oracle.yml's
# environment-variable bridge nodes. A bare `docker run --rm "$IMAGE" python
# -c "import combined_bot"` with no env vars therefore always raises
# SettingsError now, regardless of whether the image itself is fine -- found
# 2026-09-01 deploying 21144dd, the first commit built with this check no
# longer matching the code it's checking. The values here don't need to be
# the real deployment's values (this container is never started for real),
# just present and pointing at writable-looking paths so resolution succeeds.
SETTINGS_ENV=(-e SETTINGS_FILE=/app/settings.oracle.yml -e NEWS_CACHE_DIR=/data/news_cache \
    -e NEWS_ARCHIVE_DIR=/data/news_archive -e MESSAGE_ARCHIVE_DIR=/data/message_archive \
    -e SUBSCRIBERS_DB_FILE=/data/subscribers.db)
run import docker run --rm "${SETTINGS_ENV[@]}" "$IMAGE" python -c "import combined_bot" || die "the image cannot import combined_bot"
ok "imports"
run entrypoint docker run --rm --entrypoint ./docker-entrypoint.sh "$IMAGE" python -c "print('ok')" \
    || die "ENTRYPOINT is broken -- see $LOGDIR/entrypoint.log"
ok "entrypoint runs"

# ----------------------------------------------------------------- baseline

# `python` is not on PATH inside the container -- the entrypoint activates
# the conda env, and docker exec skips the entrypoint. The first run of
# this script reported an error string as if it were the baseline data,
# which is worse than not checking: it printed "ok".
#
# Uses raw sqlite3 against SUBSCRIBERS_DB_FILE directly, not a Python
# module call -- found broken 2026-09-07 deploying 6059823: this used to
# import users_db and call list_push_enabled_subscribers(), but PR #79
# (commit 8a1d379, already on main well before this) split users_db.py
# into storage/'s pluggable backend + subscriber_ops.py's thin wrapper,
# so plain `import users_db` has raised ModuleNotFoundError on every
# deploy since -- silently, since a baseline-read failure is a `warn`,
# not a `die`. Every deploy between 8a1d379 and this fix ran with no
# pre/post subscriber-count comparison at all and nothing said so loudly.
# Raw SQL against the sqlite file is used instead of subscriber_ops
# (rather than just swapping the import) because vm_bot is sqlite-only by
# design (settings.oracle.yml pins storage.database.type: sqlite, see its
# own comment) -- this avoids re-breaking again the next time the
# in-process storage API's shape changes for an unrelated reason.
remote_counts() {
    local out
    out=$("${SSH[@]}" "sudo docker exec -e SUBSCRIBERS_DB_FILE=/data/subscribers.db $CONTAINER \
        /opt/conda/bin/python -c \"
import os, sqlite3
conn = sqlite3.connect(os.environ['SUBSCRIBERS_DB_FILE'])
total = conn.execute('SELECT COUNT(*) FROM subscribers').fetchone()[0]
push_enabled = conn.execute('SELECT COUNT(*) FROM subscribers WHERE push_enabled = 1').fetchone()[0]
print(total, push_enabled,
      len(os.listdir(os.environ.get('NEWS_CACHE_DIR', '/data/news_cache'))))
\"" 2>/dev/null | tr -d '\r')
    # Only numbers are data. Anything else is an error message.
    case "$out" in
        *[!0-9\ ]*|"") return 1 ;;
        *) printf '%s' "$out" ;;
    esac
}

step "Baseline from the running container"
if BASE=$(remote_counts); then
    ok "total subscribers / push-enabled / cached articles: $BASE"
else
    BASE=""
    warn "could not read a baseline -- continuing, but the post-deploy comparison is lost"
fi

# ----------------------------------------------------------------- transfer

step "Transfer"
docker save "$IMAGE:latest" | "${SSH[@]}" "sudo docker load" > "$LOGDIR/transfer.log" 2>&1 \
    || die "transfer failed -- see $LOGDIR/transfer.log"
REMOTE_ID=$("${SSH[@]}" "sudo docker image inspect -f '{{.Id}}' $IMAGE:latest" | tr -d '\r')
[ "$REMOTE_ID" = "$LOCAL_ID" ] || die "remote image $REMOTE_ID != local $LOCAL_ID -- the transfer did not land what was built"
ok "remote image matches local byte-for-byte"

# ------------------------------------------------------------------ restart

step "Restart"
# The exact flags live in the gitignored infra file so that secrets and IPs
# stay out of the repo, and so there is exactly one place to keep current.
# Real incident, 2026-08-29: the boundary regex used to be `[a-z_]+:`,
# which silently failed to match a `last_verified_<date>_<commit>:` key
# (digits aren't in that class) -- an established renaming convention in
# this same file, used every time a new deploy record is archived. Awk
# just kept reading past the real end of the command block scalar into
# the next several history entries, producing a docker-run.txt with
# unrelated YAML prose appended after `myfirstagent-bot` and a `docker
# run` that failed with a shell syntax error. `[A-Za-z0-9_]+:` matches
# any plain YAML key regardless of what's in its name -- the actual
# YAML rule this is approximating is "indentation drops back to the
# command key's own level", which no *specific* character class can
# reliably encode; broadening it removes the one dimension (which
# characters appear in a sibling key's name) most likely to change.
# `invm` scopes the search to the vm_bot: top-level section specifically.
# Found 2026-09-01 deploying 21144dd: without this scoping, awk matched the
# FIRST `command: |` anywhere in the file -- and local-int-machine's own
# docker_run.command block (added earlier this session, for the local INT
# environment) now sits above vm_bot's in the file, so this extracted and
# ran the INT container's command (test Telegram tokens, no Vault secrets,
# wrong data volume) against the PRODUCTION VM, after the real
# myfirstagent-bot container had already been stopped/removed -- a real
# outage, caught and fixed live, not by inspection. `/^vm_bot:/` anchors to
# column 0, so it only matches the real top-level key, never an indented
# occurrence.
# Stops on INDENTATION dropping to <=4 spaces, not on "looks like a YAML
# key" -- the actual YAML rule for a `|` block scalar's extent, which no
# specific character class can approximate (see the paragraph above,
# which already predicted this). Found broken a THIRD time, live,
# 2026-09-07 deploying 6059823: the widened `[A-Za-z0-9_]+:` character
# class still didn't match a `STATUS 2026-09-04 (FAILED DEPLOY, ROLLED
# BACK -- PROD is currently commit` freeform history line (spaces and
# parens before the first colon, not a plain identifier) -- so awk read
# straight through it into several paragraphs of deploy-history prose,
# appended all of it to docker-run.txt, and executed the lot over SSH.
# The `docker run -d ...` line itself is one contiguous command (no stray
# newline inside it) so it completed and the container actually started
# correctly *before* the trailing prose hit a real shell syntax error
# (`syntax error near unexpected token '('`) and made the whole `${SSH[@]}`
# call return non-zero -- `die()` fired on a deploy that had, underneath,
# already succeeded. Caught by checking the live container by hand
# (commit label, docker logs) after this die(), not by anything in this
# script. Indentation is the one dimension of a `|` block scalar that
# cannot drift out from under a fixed-width regex the way key-naming
# conventions repeatedly have.
DOCKER_RUN=$(awk '
    /^vm_bot:/{invm=1}
    invm && /^    command: \|/{f=1;next}
    f && $0 !~ /^[[:space:]]*$/ && $0 !~ /^ {6,}/{exit}
    f
' "$INFRA" | sed 's/^      //')
[ -n "$DOCKER_RUN" ] || die "could not read docker_run.command from $INFRA"
echo "$DOCKER_RUN" > "$LOGDIR/docker-run.txt"

"${SSH[@]}" "sudo docker stop $CONTAINER >/dev/null 2>&1; sudo docker rm $CONTAINER >/dev/null 2>&1; true"
"${SSH[@]}" "$DOCKER_RUN" > "$LOGDIR/run.log" 2>&1 || die "docker run failed -- see $LOGDIR/run.log and $LOGDIR/docker-run.txt"
sleep 20

STATE=$("${SSH[@]}" "sudo docker inspect -f '{{.State.Status}} {{.RestartCount}} {{.Image}}' $CONTAINER" | tr -d '\r')
set -- $STATE
[ "$1" = "running" ] || die "container is '$1', not running"
[ "$3" = "$LOCAL_ID" ] || die "container runs $3, not the image just built ($LOCAL_ID)"
ok "running the new image, restarts=$2"

# ------------------------------------------------------------- verification

step "Verify"
FAILURES=0
# A helper that does not exist prints "command not found" and moves on, so
# every check using it silently passes. Refuse to report on a run that
# could not have checked anything.
for fn in host_check run; do
    declare -F "$fn" >/dev/null || die "internal: $fn is not defined -- verification cannot run"
done

# Wait for readiness rather than sleeping a fixed amount. A restart has to
# activate the conda env and fetch four secrets from OCI Vault before it
# prints anything, which took longer than a flat 20s and made the first
# run of this script report an empty log for a container that was
# perfectly healthy.
READY=0
for _ in $(seq 1 30); do
    if "${SSH[@]}" "sudo docker logs --tail 200 $CONTAINER 2>&1 | grep -q 'Both bots ready'"; then
        READY=1; break
    fi
    sleep 5
done
if [ "$READY" -eq 1 ]; then
    LOGLINES=$("${SSH[@]}" "sudo docker logs --tail 200 $CONTAINER 2>&1 | wc -l" | tr -d '\r')
    ok "bots ready, docker logs has $LOGLINES lines"
else
    warn "'Both bots ready' never appeared -- see docker logs"
    FAILURES=$((FAILURES + 1))
fi

# These run from HERE, not inside the container: they open their own
# SSH connections, and tools/ is deliberately not in the image's COPY list.
# Check whichever backends the deploy actually enabled, read from the same
# docker run command that enabled them -- so retiring one backend does not
# leave a check that fails forever. A check known to fail is worse than no
# check: it trains you to ignore the output.
case "$DOCKER_RUN" in
    *LOGFIRE_ENABLED=true*)
        if [ -n "${LOGFIRE_API_KEY:-}" ]; then
            host_check logfire-telemetry tools/check_logfire.py \
                --bot-vm "$SSH_USER@$VM_IP" --bot-key "$SSH_KEY" --timeout 120
        else
            warn "LOGFIRE_ENABLED is set but LOGFIRE_API_KEY is not in this shell -- cannot verify"
            FAILURES=$((FAILURES + 1))
        fi
        ;;
    *) ok "Logfire not enabled -- skipping its check" ;;
esac

host_check data-persistence tools/check_data_persistence.py \
    --bot-vm "$SSH_USER@$VM_IP" --bot-key "$SSH_KEY" \
    --dir-env NEWS_CACHE_DIR --dir-env NEWS_ARCHIVE_DIR \
    --allow-empty NEWS_ARCHIVE_DIR

AFTER=$(remote_counts)
if [ -n "$BASE" ] && [ -n "$AFTER" ]; then
    if [ "$BASE" = "$AFTER" ]; then ok "subscriber and cache counts unchanged ($AFTER)"
    else warn "counts changed: before [$BASE] after [$AFTER] -- expected for cache, NOT for subscribers"; fi
fi

step "Smoke tests"
# The end-to-end check: this is what caught the DeepSeek thinking-mode
# outage (3/10 passing, every settings command misrouted) when every other
# check above was green.
if "$PY" tools/run_smoke_tests.py --bot-vm "$SSH_USER@$VM_IP" --bot-key "$SSH_KEY" \
        > "$LOGDIR/smoke.log" 2>&1; then
    ok "$(grep -oE '[0-9]+/[0-9]+ .*(passed|pass)' "$LOGDIR/smoke.log" | tail -1)"
else
    warn "smoke tests FAILED -- see $LOGDIR/smoke.log"
    grep -iE "^(FAIL|case [0-9]+|[0-9]+/[0-9]+)" "$LOGDIR/smoke.log" | head -12 | sed 's/^/        /'
    FAILURES=$((FAILURES + 1))
fi

# Silence here is the point: these lines only appear when a guardrail layer
# fell back, which used to happen with no trace at all.
FAILOPEN=$("${SSH[@]}" "sudo docker logs --tail 500 $CONTAINER 2>&1 | grep -c 'layer [24] FAILED'" | tr -d '\r')
if [ "${FAILOPEN:-0}" -gt 0 ]; then
    warn "guardrail fail-open occurred $FAILOPEN time(s) since restart -- grep docker logs for 'layer 2 FAILED'"
    FAILURES=$((FAILURES + 1))
else
    ok "no guardrail fail-open since restart"
fi

# ---------------------------------------------------------------- reporting

step "Result"
printf 'logs: %s\n' "$LOGDIR"
if [ "$FAILURES" -eq 0 ]; then
    printf '\033[32mDeployed %s and verified.\033[0m\n' "${LOCAL_ID:7:12}"
    printf 'Update %s: last_verified, image id, and the docker run command if it changed.\n' "$INFRA"
    exit 0
fi
printf '\033[31m%d verification(s) failed.\033[0m The new image IS running.\n' "$FAILURES"
printf 'Roll back with the previous image id if needed, or hand %s to an agent to diagnose.\n' "$LOGDIR"
exit 1

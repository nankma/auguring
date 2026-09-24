"""
Local-only HTTP test API -- lets the smoke-test checklist
(build-locally-deploy-remotely skill) run via curl instead of a real
Telegram client. See docs/reference/local-testing-api-plan.md.

Calls bot.process_message directly -- the exact same guardrail/agent/
formatting pipeline real Telegram traffic runs, not a separate
reimplementation. handle_message and this module both call it; neither
duplicates the pipeline itself. That also means process_message's own
try/except is where an unhandled pipeline failure gets logged (once,
via _events.log) before it re-raises -- do_POST's except block below
only translates the re-raised exception into an HTTP response, it does
not record it a second time.

Security: the boundary is the `docker run -p 127.0.0.1:8765:8765` flag,
NOT the bind address in this file. Binds to 0.0.0.0 *inside* the
container deliberately -- that's the container's own isolated network
namespace, unreachable from outside except through whatever port Docker
is told to publish. Real incident during this feature's first deploy:
binding to 127.0.0.1 here made the service completely unreachable even
from the VM's own localhost, because Docker's port-publishing NAT
delivers external traffic to the container's bridge interface, not its
loopback -- a service bound to a container's own 127.0.0.1 is invisible
to `docker run -p`, full stop. The actual security control lives
entirely in the host-side half of that flag (127.0.0.1, restricting
which host interface Docker publishes to) -- get that half right and the
in-process bind address genuinely doesn't matter for reachability from
outside the VM. See docs/reference/local-testing-api-plan.md for the corrected
docker run flag and why. Reachable only via SSH port-forward from a
machine already holding the VM's SSH key (`ssh -L 8765:127.0.0.1:8765
ubuntu@<vm-ip>`), the same trust boundary as SSH access to the VM itself
-- this endpoint doesn't grant anything beyond what SSH access already
does (docker exec into the container would let you do the same and
more). Gated behind ENABLE_TEST_API, off by default -- a production
deploy that never sets it never starts this server at all, zero attack
surface added to the normal case.

Stdlib-only (http.server + threading), deliberately -- this is a debug
tool, not a production API; aiohttp/FastAPI would be real weight for
something meant to disappear when ENABLE_TEST_API isn't set. The HTTP
handler runs in its own thread (http.server is synchronous) and bridges
into the bot's asyncio event loop via run_coroutine_threadsafe, since
process_message is a coroutine.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app_settings import resolved_optional
import bot as info_bot
import subscriber_ops

HOST = "0.0.0.0"
PORT = int(resolved_optional("test_api.port", default=8765))
CALL_TIMEOUT_SECONDS = 60


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Route through print (PYTHONUNBUFFERED, same as the rest of this
        # project) instead of BaseHTTPRequestHandler's default stderr
        # logging, so requests show up in `docker logs` consistently.
        print(f"[test_api] {self.address_string()} - {fmt % args}")

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/test_message":
            self._json_response(404, {"error": "unknown path, use POST /test_message"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            chat_id = int(body["chat_id"])
            text = str(body["text"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json_response(400, {"error": f"expected JSON body {{'chat_id': int, 'text': str}}: {exc!r}"})
            return

        # Everything arriving here is test traffic by definition -- this
        # endpoint exists for nothing else -- so the subscriber row the
        # pipeline is about to create (or update) is flagged before it runs.
        # Marking at the door rather than asking each test to clean up:
        # smoke tests drive the real pipeline, so verifying "start pushing
        # me news every 6 hours" genuinely turns push on, and a cleanup step
        # is skipped precisely when a test fails early.
        subscriber_ops.mark_test_account(chat_id)

        server = self.server  # type: _Server
        future = asyncio.run_coroutine_threadsafe(
            info_bot.process_message(
                chat_id, text, server.model, server.guard_model, server.jev_api_key, server.embedder),
            server.loop,
        )
        try:
            result = future.result(timeout=CALL_TIMEOUT_SECONDS)
        except Exception as exc:
            # process_message already logged this (see bot.py) unless
            # it's a concurrent.futures.TimeoutError -- that one means
            # THIS call gave up waiting, not that process_message itself
            # failed; it may still finish (or fail, and log itself) in
            # the background after this response is sent.
            self._json_response(500, {"error": f"process_message failed: {exc!r}"})
            return

        self._json_response(200, result)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, loop, model, guard_model, jev_api_key, embedder=None):
        super().__init__((HOST, PORT), _Handler)
        self.loop = loop
        self.model = model
        self.guard_model = guard_model
        self.jev_api_key = jev_api_key
        self.embedder = embedder


def start(model, guard_model, jev_api_key, embedder=None) -> _Server | None:
    """Starts the test API in a background thread if ENABLE_TEST_API is
    set; returns None (does nothing) otherwise. Call stop() with the
    returned server on shutdown. Must be called from the thread running
    the bot's asyncio event loop, so it can capture the loop via
    asyncio.get_running_loop() for run_coroutine_threadsafe to target."""
    if not resolved_optional("test_api.enabled"):
        return None
    loop = asyncio.get_running_loop()
    server = _Server(loop, model, guard_model, jev_api_key, embedder)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="test-api")
    thread.start()
    print(f"[test_api] listening on {HOST}:{PORT} (ENABLE_TEST_API set) -- POST /test_message")
    return server


def stop(server: "_Server | None") -> None:
    if server is not None:
        server.shutdown()
        server.server_close()

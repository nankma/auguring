import asyncio
import json
import threading
import urllib.error
import urllib.request

import pytest

import bot
import test_api


@pytest.fixture
def running_server(monkeypatch):
    """A real _Server, serving on a real (ephemeral) loopback port, backed
    by a real asyncio event loop running in its own thread -- exercises the
    actual HTTP + threading + run_coroutine_threadsafe bridge, not a mock
    of it. bot.process_message is monkeypatched so no real model/DB is
    needed; individual tests can further monkeypatch it for specific
    behavior."""
    monkeypatch.setattr(test_api, "PORT", 0)  # ephemeral port, avoids clashing with a real deploy

    async def fake_process_message(chat_id, text, model, guard_model, jev_api_key, embedder=None):
        return {"blocked_at": None, "category": "news_query", "reply": f"echo:{text}"}

    monkeypatch.setattr(bot, "process_message", fake_process_message)

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    # _Server's constructor is plain sync code (binds a socket, stores a
    # reference to `loop` for the handler to target later via
    # run_coroutine_threadsafe) -- no need to run it "on" the loop itself.
    server = test_api._Server(loop, "fake-model", "fake-guard-model", "fake-jev-key")
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()

    yield server

    test_api.stop(server)
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=5)


def _post(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/test_message",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_test_message_returns_process_message_result(running_server, isolated_subscribers_db):
    port = running_server.server_address[1]
    status, body = _post(port, {"chat_id": 1, "text": "hello"})
    assert status == 200
    assert body == {"blocked_at": None, "category": "news_query", "reply": "echo:hello"}


def test_unknown_path_returns_404(running_server):
    port = running_server.server_address[1]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/wrong_path", data=b"{}", method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        assert False, "expected a 404"
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_malformed_body_returns_400(running_server):
    port = running_server.server_address[1]
    status, body = _post(port, {"chat_id": "not-an-int-or-missing-text"})
    assert status == 400
    assert "error" in body


def test_missing_fields_returns_400(running_server):
    port = running_server.server_address[1]
    status, body = _post(port, {})
    assert status == 400


def test_process_message_failure_returns_500(running_server, monkeypatch, isolated_subscribers_db):
    """This endpoint's own responsibility is just the HTTP translation --
    a real exception in the pipeline becomes a 500 with the error in the
    body. process_message itself is responsible for logging the failure
    durably (see tests/test_bot.py's own coverage of that -- it's the
    one place both this endpoint and real Telegram traffic go through,
    so that's where the logging test lives, not here). isolated_subscribers_db
    is required, not optional -- do_POST calls subscriber_ops.mark_test_account
    unconditionally before process_message even runs, so without it this
    only passed locally by test-ordering luck (an earlier test already
    having created the real subscribers.db table) -- failed for real on
    CI, a fresh environment with no such luck (sqlite3.OperationalError:
    no such table: subscribers)."""
    async def failing_process_message(chat_id, text, model, guard_model, jev_api_key, embedder=None):
        raise RuntimeError("simulated pipeline failure")
    monkeypatch.setattr(bot, "process_message", failing_process_message)

    port = running_server.server_address[1]
    status, body = _post(port, {"chat_id": 1, "text": "hello"})

    assert status == 500
    assert "simulated pipeline failure" in body["error"]


def test_server_binds_to_all_interfaces_inside_the_container(running_server):
    """Deliberately 0.0.0.0, not 127.0.0.1 -- see test_api.py's module
    docstring. Real incident: binding to 127.0.0.1 here made the service
    unreachable even from the VM's own localhost, because Docker's
    port-publishing NAT delivers external traffic to the container's
    bridge interface, not its loopback. The actual security boundary is
    the host-side half of `docker run -p 127.0.0.1:8765:8765` -- this
    process binding broadly inside its own isolated container namespace
    is safe and, in fact, required for that flag to work at all."""
    assert running_server.server_address[0] == test_api.HOST == "0.0.0.0"


def test_start_does_nothing_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("ENABLE_TEST_API", raising=False)
    assert test_api.start(model="fake", guard_model="fake", jev_api_key="fake-jev-key") is None


def test_start_returns_server_when_env_var_set(monkeypatch):
    monkeypatch.setenv("ENABLE_TEST_API", "true")
    monkeypatch.setattr(test_api, "PORT", 0)

    async def _run():
        server = test_api.start(model="fake", guard_model="fake", jev_api_key="fake-jev-key")
        try:
            assert server is not None
            assert server.server_address[0] == "0.0.0.0"
        finally:
            test_api.stop(server)

    asyncio.run(_run())

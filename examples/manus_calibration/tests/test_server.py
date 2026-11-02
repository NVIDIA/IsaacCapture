# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The page server over real sockets and a real fake-worker subprocess."""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import commands
from isaaccapture_examples.manus_calibration import server as server_module
from isaaccapture_examples.manus_calibration.worker import OutcomeKind
from web_helpers import (
    CLIENT_A,
    CLIENT_B,
    Sse,
    api,
    raw_request,
    request,
    run_flow,
    serving,
)

HOLD = {
    "steps": [
        {"title": "A", "instruction": "a", "time": 0.0},
        {"title": "B", "instruction": "b", "time": 0.0},
    ]
}
# One hold step: the worker then waits for a confirmation, which keeps an attempt open.
WAITING = {"steps": [{"title": "A", "instruction": "a", "time": 0.0}]}
# A long motion step keeps the worker sampling after one confirmation.
SAMPLING = {"steps": [{"title": "W", "instruction": "w", "time": 3.0}]}
HEADERS_ON_EVERY_RESPONSE = {
    "content-security-policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
}


def test_static_assets_are_served_with_their_content_types(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            expected = {
                "/": "text/html; charset=utf-8",
                "/app.js": "text/javascript; charset=utf-8",
                "/style.css": "text/css; charset=utf-8",
            }
            for path, content_type in expected.items():
                reply = await request(server.port, "GET", path)
                assert reply.status == 200, path
                assert reply.headers["content-type"] == content_type
                assert int(reply.headers["content-length"]) == len(reply.body) > 0
            page = (await request(server.port, "GET", "/")).body.decode()
            assert "Manus glove calibration" in page
            assert '<script src="/app.js">' in page

    asyncio.run(scenario())


def test_a_query_string_does_not_change_the_route(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            assert (await request(server.port, "GET", "/?x=1")).status == 200

    asyncio.run(scenario())


def test_security_headers_are_on_every_kind_of_response(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            port = server.port
            replies = [
                await request(port, "GET", "/"),
                await request(port, "GET", "/missing"),
                await request(port, "GET", "/api/start"),
                await request(port, "GET", "/", host="evil.example"),
                await api(port, "/api/start", None, {"sides": ["left"]}),
                await api(port, "/api/cancel", CLIENT_A),
                await request(
                    port,
                    "POST",
                    "/api/start",
                    headers={"Content-Length": "999999", "X-Client-Id": CLIENT_A},
                ),
            ]
            assert {r.status for r in replies} == {200, 400, 403, 404, 405, 409, 413}
            for reply in replies:
                for name, value in HEADERS_ON_EVERY_RESPONSE.items():
                    assert reply.headers[name] == value, (reply.status, name)

    asyncio.run(scenario())


def test_the_event_stream_has_the_same_security_headers(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            assert sse.status == 200
            assert sse.headers["content-type"] == "text/event-stream"
            for name, value in HEADERS_ON_EVERY_RESPONSE.items():
                assert sse.headers[name] == value
            sse.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "sides", [["left"], ["right"], ["left", "right"]], ids=["left", "right", "both"]
)
def test_a_full_calibration_runs_over_http_and_sse(make_worker, sides):
    async def scenario():
        async with serving(make_worker, HOLD) as (server, store):
            sse = await Sse.open(server.port, CLIENT_A)
            first = await sse.event()
            assert first["stage"] == "idle"
            assert first["out_dir"] == str(store.directory)
            assert first["default_sides"] == "both"
            final = await run_flow(server.port, sse, CLIENT_A, sides)
            outcome = final["outcome"]
            assert outcome["kind"] == "completed"
            assert {f["side"] for f in outcome["files"]} == set(sides)
            assert outcome["directory"] == str(store.directory)
            assert final["active"] is False
            assert final["owner"] is False
            published = store.list_published()
            assert len(published) == 1
            for entry in outcome["files"]:
                assert (store.directory / entry["file"]).is_file()
            sse.close()

    asyncio.run(scenario())


def test_a_motion_step_reports_progress_to_the_page(make_worker):
    async def scenario():
        steps = {"steps": [{"title": "W", "instruction": "w", "time": 0.4}]}
        async with serving(make_worker, steps) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            states = []
            task = asyncio.create_task(run_flow(server.port, sse, CLIENT_A, ["left"]))
            # run_flow consumes the stream, so observe through a second reader.
            watcher = await Sse.open(server.port, CLIENT_B)
            async with asyncio.timeout(20):
                while True:
                    state = await watcher.event()
                    states.append(state)
                    if state["stage"] == "finished":
                        break
            _ = await task
            elapsed = [s["elapsed_s"] for s in states if s["elapsed_s"] is not None]
            assert elapsed
            assert max(elapsed) <= 0.4 + 1e-6
            sampling = [s for s in states if s["stage"] == "sampling"]
            assert sampling and sampling[0]["prompt"]["mode"] == "motion"
            sse.close()
            watcher.close()

    asyncio.run(scenario())


def test_another_browser_sees_the_attempt_read_only_and_never_the_token(make_worker):
    async def scenario():
        async with serving(make_worker, HOLD) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            other = await Sse.open(port, CLIENT_B)
            await owner.event()
            await other.event()
            flow = asyncio.create_task(run_flow(port, owner, CLIENT_A, ["left"]))
            seen = []
            async with asyncio.timeout(30):
                while True:
                    state = await other.event()
                    seen.append(state)
                    if state["stage"] == "finished":
                        break
            _ = await flow
            active = [s for s in seen if s["active"]]
            assert active
            assert all(s["owner"] is False for s in seen)
            assert all(s["prompt"]["token"] is None for s in active if s["prompt"])
            assert any(s["prompt"] for s in active)
            # No token reaches the other browser even as raw text.
            assert not any("left-" in payload for payload in other.payloads)
            owner.close()
            other.close()

    asyncio.run(scenario())


def test_another_browser_cannot_confirm_cancel_or_start_over_a_running_attempt(
    make_worker, commands_log
):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            other = await Sse.open(port, CLIENT_B)
            await owner.event()
            await other.event()
            assert (
                await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            ).status == 200
            state = await owner.until(lambda s: s["stage"] == "awaiting_confirmation")
            token = state["prompt"]["token"]
            assert token
            confirm = await api(port, "/api/confirm", CLIENT_B, {"token": token})
            assert confirm.status == 403
            assert (await api(port, "/api/cancel", CLIENT_B)).status == 403
            takeover = await api(port, "/api/start", CLIENT_B, {"sides": ["right"]})
            assert takeover.status == 409
            assert "already active" in takeover.json()["error"]
            assert "confirm" not in commands(commands_log)
            assert "cancel" not in commands(commands_log)
            # The owner is unaffected and can still cancel.
            assert (await api(port, "/api/cancel", CLIENT_A)).status == 200
            final = await owner.until(lambda s: s["stage"] == "finished")
            assert final["outcome"]["kind"] == "cancelled"
            owner.close()
            other.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("token", ["wrong-token", "right-0", "x" * 500])
def test_a_wrong_token_is_refused_and_no_step_starts(make_worker, commands_log, token):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            await owner.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await owner.until(lambda s: s["stage"] == "awaiting_confirmation")
            reply = await api(port, "/api/confirm", CLIENT_A, {"token": token})
            assert reply.status == 409
            assert reply.json()["ok"] is False
            assert "confirm" not in commands(commands_log)
            owner.close()

    asyncio.run(scenario())


def test_two_simultaneous_confirmations_of_one_token_send_one_command(
    make_worker, commands_log
):
    async def scenario():
        async with serving(make_worker, SAMPLING) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            await owner.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            state = await owner.until(lambda s: s["stage"] == "awaiting_confirmation")
            token = state["prompt"]["token"]
            replies = await asyncio.gather(
                api(port, "/api/confirm", CLIENT_A, {"token": token}),
                api(port, "/api/confirm", CLIENT_A, {"token": token}),
            )
            assert [r.status for r in replies] == [200, 200]
            await owner.until(lambda s: s["stage"] == "sampling")
            assert commands(commands_log).count("confirm") == 1
            # Repeating after acceptance is also a no-op for the worker.
            again = await api(port, "/api/confirm", CLIENT_A, {"token": token})
            assert again.status in (200, 409)
            assert commands(commands_log).count("confirm") == 1
            await api(port, "/api/cancel", CLIENT_A)
            await owner.until(lambda s: s["stage"] == "finished")
            owner.close()

    asyncio.run(scenario())


def test_a_second_start_while_active_is_a_conflict(make_worker):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            await owner.event()
            assert (
                await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            ).status == 200
            again = await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert again.status == 409
            assert "already active" in again.json()["error"]
            await api(port, "/api/cancel", CLIENT_A)
            await owner.until(lambda s: s["stage"] == "finished")
            owner.close()

    asyncio.run(scenario())


def test_confirm_and_cancel_with_nothing_active_are_conflicts(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            port = server.port
            assert (await api(port, "/api/cancel", CLIENT_A)).status == 409
            confirm = await api(port, "/api/confirm", CLIENT_A, {"token": "left-0"})
            assert confirm.status == 409

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"sides": []},
        {"sides": ["up"]},
        {"sides": ["left", "left"]},
        {"sides": "left"},
        {"sides": [1]},
        {"sides": None},
        {"sides": ["LEFT"]},
    ],
    ids=["missing", "empty", "unknown", "repeated", "string", "number", "null", "case"],
)
def test_an_invalid_start_body_is_a_bad_request_and_starts_nothing(
    make_worker, commands_log, body
):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            reply = await api(server.port, "/api/start", CLIENT_A, body)
            assert reply.status == 400
            assert reply.json()["ok"] is False
            assert server.controller.active is False
            assert commands(commands_log) == []
            sse.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/start", b"not json"),
        ("/api/start", b"[1, 2]"),
        ("/api/start", b'"left"'),
        ("/api/confirm", b"{}"),
        ("/api/confirm", b'{"token": ""}'),
        ("/api/confirm", b'{"token": 5}'),
    ],
)
def test_malformed_action_bodies_are_bad_requests(make_worker, path, payload):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            reply = await request(
                server.port,
                "POST",
                path,
                headers={
                    "Content-Type": "application/json",
                    "X-Client-Id": CLIENT_A,
                },
                body=payload,
            )
            assert reply.status == 400

    asyncio.run(scenario())


def test_a_json_content_type_with_a_charset_is_accepted(make_worker):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            reply = await api(
                server.port,
                "/api/start",
                CLIENT_A,
                {"sides": ["left"]},
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            assert reply.status == 200
            await api(server.port, "/api/cancel", CLIENT_A)
            await sse.until(lambda s: s["stage"] == "finished")
            sse.close()

    asyncio.run(scenario())


def test_a_form_encoded_post_is_rejected(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            reply = await request(
                server.port,
                "POST",
                "/api/start",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "X-Client-Id": CLIENT_A,
                },
                body=b"sides=left",
            )
            assert reply.status == 400
            assert server.controller.active is False

    asyncio.run(scenario())


def test_a_post_without_a_content_type_is_rejected(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            reply = await request(
                server.port,
                "POST",
                "/api/start",
                headers={"X-Client-Id": CLIENT_A},
                body=b'{"sides": ["left"]}',
            )
            assert reply.status == 400

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "client", [None, "", "short", "has.dots.in.it.1234", "x" * 65, "bad id 12345678"]
)
def test_a_missing_or_invalid_client_id_is_rejected_on_every_action(
    make_worker, client
):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            for path in ("/api/start", "/api/confirm", "/api/cancel"):
                reply = await api(
                    server.port, path, client, {"sides": ["left"], "token": "t"}
                )
                assert reply.status == 400, path
            assert server.controller.active is False

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?client=short",
        "?client=has.dots.in.it.1234",
        "?client=a&client=b",
        "?other=1",
    ],
)
def test_the_event_stream_needs_one_valid_client_id(make_worker, query):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            reply = await request(server.port, "GET", "/events" + query)
            assert reply.status == 400

    asyncio.run(scenario())


def test_an_oversized_body_is_refused(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            reply = await request(
                server.port,
                "POST",
                "/api/start",
                headers={
                    "Content-Type": "application/json",
                    "X-Client-Id": CLIENT_A,
                    "Content-Length": str(4 * 1024 + 1),
                },
                body=b"x" * (4 * 1024 + 1),
            )
            assert reply.status == 413

    asyncio.run(scenario())


@pytest.mark.parametrize("length", ["abc", "-1", "1.5", ""])
def test_a_malformed_content_length_is_refused(make_worker, length):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            reply = await request(
                server.port,
                "POST",
                "/api/start",
                headers={
                    "Content-Type": "application/json",
                    "X-Client-Id": CLIENT_A,
                    "Content-Length": length,
                },
            )
            assert reply.status == 413

    asyncio.run(scenario())


def test_oversized_headers_are_refused(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            reply = await request(
                server.port, "GET", "/", headers={"X-Pad": "x" * (20 * 1024)}
            )
            assert reply.status == 413

    asyncio.run(scenario())


def test_a_huge_header_block_is_never_served(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            try:
                reply = await request(
                    server.port, "GET", "/", headers={"X-Pad": "x" * (200 * 1024)}
                )
            except (ConnectionError, OSError):
                # The server may reset the connection before the reply is read.
                return
            assert reply.status == 413

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "data",
    [
        b"GARBAGE\r\n\r\n",
        b"GET /\r\n\r\n",
        b"GET / HTTP/2.0\r\nHost: x\r\n\r\n",
        b"GET / HTTP/1.1 extra\r\nHost: x\r\n\r\n",
    ],
    ids=["no-version", "two-parts", "http2", "four-parts"],
)
def test_a_malformed_request_line_is_a_bad_request(make_worker, data):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            assert (await raw_request(server.port, data)).status == 400

    asyncio.run(scenario())


def test_a_header_without_a_colon_is_a_bad_request(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            port = server.port
            data = (
                f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nBroken\r\n\r\n".encode()
            )
            assert (await raw_request(port, data)).status == 400

    asyncio.run(scenario())


def test_a_connection_dropped_mid_request_does_not_disturb_the_server(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            _reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(b"GET / HTTP/1.1\r\nHost: 127.0.0")
            await writer.drain()
            writer.close()
            await asyncio.sleep(0.1)
            assert (await request(server.port, "GET", "/")).status == 200

    asyncio.run(scenario())


def test_a_request_that_never_finishes_is_dropped_without_a_reply(
    make_worker, monkeypatch
):
    monkeypatch.setattr(server_module, "_REQUEST_TIMEOUT_S", 0.3)

    async def scenario():
        async with serving(make_worker) as (server, _store):
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(b"GET / HTTP/1.1\r\nHost: 127.0.0.1")
            await writer.drain()
            assert await asyncio.wait_for(reader.read(), 3) == b""
            writer.close()

    asyncio.run(scenario())


def test_only_the_servers_own_host_names_are_accepted(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            port = server.port
            accepted = [f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"]
            refused = [
                "evil.example",
                f"evil.example:{port}",
                "127.0.0.1",
                f"127.0.0.1:{port + 1}",
                f"127.0.0.1.evil.example:{port}",
                "",
            ]
            for host in accepted:
                assert (await request(port, "GET", "/", host=host)).status == 200, host
            for host in refused:
                assert (await request(port, "GET", "/", host=host)).status == 403, host

    asyncio.run(scenario())


def test_a_foreign_origin_is_refused_on_pages_streams_and_actions(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            port = server.port
            for origin in (
                "http://evil.example",
                "null",
                f"https://127.0.0.1:{port}",
                f"http://localhost:{port}",
                "http://127.0.0.1",
            ):
                page = await request(port, "GET", "/", headers={"Origin": origin})
                assert page.status == 403, origin
                stream = await request(
                    port,
                    "GET",
                    f"/events?client={CLIENT_A}",
                    headers={"Origin": origin},
                )
                assert stream.status == 403, origin
                action = await api(
                    port,
                    "/api/start",
                    CLIENT_A,
                    {"sides": ["left"]},
                    headers={"Origin": origin},
                )
                assert action.status == 403, origin
            assert server.controller.active is False

    asyncio.run(scenario())


def test_the_pages_own_origin_is_accepted(make_worker):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            origin = {"Origin": f"http://127.0.0.1:{port}"}
            assert (await request(port, "GET", "/", headers=origin)).status == 200
            sse = await Sse.open(port, CLIENT_A)
            await sse.event()
            reply = await api(
                port, "/api/start", CLIENT_A, {"sides": ["left"]}, headers=origin
            )
            assert reply.status == 200
            await api(port, "/api/cancel", CLIENT_A, headers=origin)
            await sse.until(lambda s: s["stage"] == "finished")
            sse.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("GET", "/api/start", 405),
        ("GET", "/api/confirm", 405),
        ("GET", "/api/cancel", 405),
        ("POST", "/events", 405),
        ("POST", "/", 405),
        ("POST", "/app.js", 405),
        ("DELETE", "/", 405),
        ("PUT", "/api/start", 405),
        ("HEAD", "/", 405),
        ("GET", "/missing", 404),
        ("POST", "/missing", 404),
        ("GET", "/api", 404),
        ("GET", "/static/app.js", 404),
        ("GET", "/../app.js", 404),
        ("GET", "/%2e%2e/app.js", 404),
    ],
)
def test_wrong_methods_and_unknown_paths(make_worker, method, path, status):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            assert (await request(server.port, method, path)).status == status

    asyncio.run(scenario())


def test_the_stream_sends_the_current_state_first_and_each_change(make_worker):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            sse = await Sse.open(port, CLIENT_A)
            first = await sse.event()
            assert first["stage"] == "idle"
            assert first["active"] is False
            assert first["gloves"] == []
            await api(port, "/api/start", CLIENT_A, {"sides": ["right"]})
            state = await sse.until(lambda s: s["stage"] == "awaiting_confirmation")
            assert state["owner"] is True
            assert state["active"] is True
            assert state["prompt"]["side"] == "right"
            assert state["gloves"] == [{"side": "right", "glove_id": 22}]
            await api(port, "/api/cancel", CLIENT_A)
            await sse.until(lambda s: s["stage"] == "finished")
            sse.close()

    asyncio.run(scenario())


def test_a_late_joining_browser_gets_the_running_state_immediately(make_worker):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            await owner.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await owner.until(lambda s: s["stage"] == "awaiting_confirmation")
            late = await Sse.open(port, CLIENT_B)
            state = await late.event()
            assert state["stage"] == "awaiting_confirmation"
            assert state["owner"] is False
            assert state["prompt"]["token"] is None
            await api(port, "/api/cancel", CLIENT_A)
            await late.until(lambda s: s["stage"] == "finished")
            owner.close()
            late.close()

    asyncio.run(scenario())


def test_an_idle_stream_sends_keepalive_comments(make_worker, monkeypatch):
    monkeypatch.setattr(server_module, "_KEEPALIVE_S", 0.2)

    async def scenario():
        async with serving(make_worker) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            async with asyncio.timeout(3):
                while (await sse.line()) != ": keepalive":
                    pass
            sse.close()

    asyncio.run(scenario())


def test_a_burst_of_changes_is_coalesced_into_few_events(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            for _ in range(50):
                server._on_change(None)
            count = 0
            try:
                async with asyncio.timeout(0.5):
                    while True:
                        await sse.event()
                        count += 1
            except TimeoutError:
                pass
            assert 1 <= count <= 3
            sse.close()

    asyncio.run(scenario())


def test_the_owner_leaving_cancels_the_attempt_after_the_grace_window(
    make_worker, monkeypatch
):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 0.3)

    async def scenario():
        async with serving(make_worker, WAITING) as (server, store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            watcher = await Sse.open(port, CLIENT_B)
            await owner.event()
            await watcher.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await owner.until(lambda s: s["stage"] == "awaiting_confirmation")
            owner.close()
            final = await watcher.until(lambda s: s["stage"] == "finished")
            assert final["outcome"]["kind"] == "cancelled"
            assert final["outcome"]["files"] == []
            assert store.list_published() == []
            outcome = server.controller.view().outcome
            assert outcome.exit_code == 0
            assert outcome.forced_termination is False
            assert server.controller.active is False
            watcher.close()

    asyncio.run(scenario())


def test_a_reload_within_the_grace_window_keeps_the_attempt(make_worker, monkeypatch):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 1.0)

    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            await owner.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await owner.until(lambda s: s["stage"] == "awaiting_confirmation")
            owner.close()
            await asyncio.sleep(0.1)
            again = await Sse.open(port, CLIENT_A)
            state = await again.event()
            assert state["owner"] is True
            assert state["prompt"]["token"]
            # Wait out the original window: the attempt must still be running.
            await asyncio.sleep(1.5)
            assert server.controller.active is True
            assert server.controller.view().stage.value == "awaiting_confirmation"
            assert (await api(port, "/api/cancel", CLIENT_A)).status == 200
            await again.until(lambda s: s["stage"] == "finished")
            again.close()

    asyncio.run(scenario())


def test_a_second_tab_with_the_same_id_keeps_the_owner_alive(make_worker, monkeypatch):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 0.3)

    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            first = await Sse.open(port, CLIENT_A)
            second = await Sse.open(port, CLIENT_A)
            await first.event()
            await second.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await first.until(lambda s: s["stage"] == "awaiting_confirmation")
            first.close()
            await asyncio.sleep(0.8)
            assert server.controller.active is True
            await api(port, "/api/cancel", CLIENT_A)
            await second.until(lambda s: s["stage"] == "finished")
            second.close()

    asyncio.run(scenario())


def test_a_non_owner_leaving_does_not_cancel_anything(make_worker, monkeypatch):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 0.2)

    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            other = await Sse.open(port, CLIENT_B)
            await owner.event()
            await other.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await owner.until(lambda s: s["stage"] == "awaiting_confirmation")
            other.close()
            await asyncio.sleep(0.8)
            assert server.controller.active is True
            await api(port, "/api/cancel", CLIENT_A)
            await owner.until(lambda s: s["stage"] == "finished")
            owner.close()

    asyncio.run(scenario())


def test_completed_files_survive_the_browser_leaving(make_worker, monkeypatch):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 0.2)

    async def scenario():
        async with serving(make_worker, HOLD) as (server, store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            final = await run_flow(server.port, sse, CLIENT_A, ["left", "right"])
            files = [store.directory / f["file"] for f in final["outcome"]["files"]]
            sse.close()
            await asyncio.sleep(0.8)
            assert len(files) == 2
            assert all(path.is_file() for path in files)
            assert len(store.list_published()) == 1
            assert server.controller.view().outcome.kind is OutcomeKind.COMPLETED

    asyncio.run(scenario())


def test_a_new_attempt_can_follow_a_finished_one_from_another_browser(make_worker):
    async def scenario():
        async with serving(make_worker, HOLD) as (server, store):
            port = server.port
            first = await Sse.open(port, CLIENT_A)
            second = await Sse.open(port, CLIENT_B)
            await first.event()
            await second.event()
            await run_flow(port, first, CLIENT_A, ["left"])
            # The second stream buffered the first attempt's states; read past them.
            await second.until(lambda s: s["stage"] == "finished")
            final = await run_flow(port, second, CLIENT_B, ["right"])
            assert final["outcome"]["kind"] == "completed"
            assert len(store.list_published()) == 2
            first.close()
            second.close()

    asyncio.run(scenario())


def test_aclose_cancels_an_active_attempt_and_reaps_the_worker(make_worker):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, store):
            port = server.port
            sse = await Sse.open(port, CLIENT_A)
            await sse.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await sse.until(lambda s: s["stage"] == "awaiting_confirmation")
            await server.aclose()
            outcome = server.controller.view().outcome
            assert outcome.kind is OutcomeKind.CANCELLED
            assert outcome.exit_code == 0
            assert outcome.forced_termination is False
            assert store.list_published() == []
            sse.close()

    asyncio.run(scenario())


def test_aclose_ends_open_event_streams(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            await server.aclose()
            assert await sse.at_end(5) is True

    asyncio.run(scenario())


def test_a_closed_server_refuses_new_connections(make_worker):
    async def scenario():
        async with serving(make_worker) as (server, _store):
            port = server.port
            await server.aclose()
            with pytest.raises(OSError):
                await asyncio.open_connection("127.0.0.1", port)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("scenario_flags", "kind"),
    [({"fail_at_step": ["left", 0]}, "failed"), ({"no_result": True}, "uncertain")],
    ids=["failed", "uncertain"],
)
def test_a_failed_or_uncertain_run_shows_the_log_tail_and_lists_no_files(
    make_worker, scenario_flags, kind
):
    async def scenario():
        async with serving(make_worker, {**HOLD, **scenario_flags}) as (server, store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            final = await run_flow(server.port, sse, CLIENT_A, ["left"])
            outcome = final["outcome"]
            assert outcome["kind"] == kind
            assert outcome["files"] == []
            assert outcome["directory"] is None
            assert outcome["log_tail"]
            assert any("fake worker log line" in line for line in outcome["log_tail"])
            assert store.list_published() == []
            sse.close()

    asyncio.run(scenario())


def test_a_cancelled_run_has_no_log_tail_and_no_files(make_worker):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, store):
            port = server.port
            sse = await Sse.open(port, CLIENT_A)
            await sse.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await sse.until(lambda s: s["stage"] == "awaiting_confirmation")
            await api(port, "/api/cancel", CLIENT_A)
            final = await sse.until(lambda s: s["stage"] == "finished")
            assert final["outcome"]["kind"] == "cancelled"
            assert final["outcome"]["files"] == []
            assert final["outcome"]["log_tail"] == []
            assert store.list_published() == []
            sse.close()

    asyncio.run(scenario())


def test_page_text_from_the_worker_reaches_the_browser_only_as_json_strings(
    make_worker,
):
    async def scenario():
        hostile = '<img src=x onerror="alert(1)">'
        steps = {"steps": [{"title": hostile, "instruction": hostile, "time": 0.0}]}
        async with serving(make_worker, steps) as (server, _store):
            port = server.port
            sse = await Sse.open(port, CLIENT_A)
            await sse.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            state = await sse.until(lambda s: s["stage"] == "awaiting_confirmation")
            # Passed through verbatim as data; the page renders it with textContent.
            assert state["prompt"]["title"] == hostile
            assert json.loads(sse.payloads[-1])["prompt"]["instruction"] == hostile
            page = (await request(port, "GET", "/app.js")).body.decode()
            assert "innerHTML" not in page
            await api(port, "/api/cancel", CLIENT_A)
            await sse.until(lambda s: s["stage"] == "finished")
            sse.close()

    asyncio.run(scenario())


CONTINUOUS = {
    "steps": [
        {
            "title": "Slow circle",
            "instruction": "Circle your hand slowly.",
            "time": 0.0,
            "continuous": True,
            "estimated": 3.0,
        },
        {
            "title": "Spread",
            "instruction": "Spread your fingers.",
            "time": 0.0,
            "continuous": True,
            "estimated": 1.5,
        },
    ]
}


def test_continuous_steps_reach_the_owner_with_their_estimate_and_hide_the_token_from_others(
    make_worker,
):
    async def scenario():
        async with serving(make_worker, CONTINUOUS) as (server, store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            other = await Sse.open(port, CLIENT_B)
            await owner.event()
            await other.event()
            owner_seen: list = []
            other_seen: list = []
            flow = asyncio.create_task(
                run_flow(port, owner, CLIENT_A, ["left", "right"], seen=owner_seen)
            )
            async with asyncio.timeout(30):
                while True:
                    state = await other.event()
                    other_seen.append(state)
                    if state["stage"] == "finished":
                        break
            final = await flow
            assert final["outcome"]["kind"] == "completed"
            assert {f["side"] for f in final["outcome"]["files"]} == {"left", "right"}
            assert len(store.list_published()) == 1

            def prompts(states):
                return [s["prompt"] for s in states if s["prompt"]]

            owner_prompts = prompts(owner_seen)
            # Two steps for each of two gloves, each offered to the owner as continuous.
            offered = {(p["side"], p["index"]): p for p in owner_prompts if p["token"]}
            assert sorted(offered) == [
                ("left", 0),
                ("left", 1),
                ("right", 0),
                ("right", 1),
            ]
            for (side, index), prompt in offered.items():
                assert prompt["continuous"] is True
                assert prompt["estimated_s"] == (3.0 if index == 0 else 1.5)
                assert prompt["mode"] == "hold"
                assert prompt["duration_s"] == 0.0
                assert prompt["token"] == f"{side}-{index}"
            # The other browser sees the same continuous steps but never a token.
            other_prompts = prompts(other_seen)
            assert other_prompts
            for prompt in other_prompts:
                assert prompt["continuous"] is True
                assert prompt["estimated_s"] > 0
                assert prompt["token"] is None
            assert all(s["owner"] is False for s in other_seen)
            assert not any(
                token in payload
                for payload in other.payloads
                for token in ("left-0", "left-1", "right-0", "right-1")
            )
            owner.close()
            other.close()

    asyncio.run(scenario())


def test_a_continuous_step_is_confirmed_like_a_hold_and_starts_no_sampling_window(
    make_worker, commands_log
):
    async def scenario():
        steps = {"steps": [CONTINUOUS["steps"][0]]}
        async with serving(make_worker, steps) as (server, _store):
            port = server.port
            owner = await Sse.open(port, CLIENT_A)
            await owner.event()
            seen: list = []
            final = await run_flow(port, owner, CLIENT_A, ["left"], seen=seen)
            assert final["outcome"]["kind"] == "completed"
            # A continuous step is hold-paced: no progress bar value is ever reported.
            assert all(s["elapsed_s"] is None for s in seen)
            assert commands(commands_log).count("confirm") == 1
            owner.close()

    asyncio.run(scenario())


def test_a_start_without_a_connected_event_stream_is_refused(make_worker, commands_log):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            reply = await api(server.port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert reply.status == 409
            assert "event stream" in reply.json()["error"]
            assert server.controller.active is False
            assert commands(commands_log) == []

    asyncio.run(scenario())


def test_a_start_after_the_event_stream_closed_is_refused(make_worker, commands_log):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            sse.close()
            await asyncio.sleep(0.3)
            reply = await api(server.port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert reply.status == 409
            assert server.controller.active is False
            assert commands(commands_log) == []

    asyncio.run(scenario())


def test_another_browsers_stream_does_not_satisfy_the_requirement(
    make_worker, commands_log
):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_B)
            await sse.event()
            reply = await api(server.port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert reply.status == 409
            assert server.controller.active is False
            assert commands(commands_log) == []
            sse.close()

    asyncio.run(scenario())


def test_a_start_is_accepted_once_the_stream_is_connected(make_worker):
    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            reply = await api(server.port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert reply.status == 200
            state = await sse.until(lambda s: s["stage"] == "awaiting_confirmation")
            assert state["owner"] is True
            await api(server.port, "/api/cancel", CLIENT_A)
            await sse.until(lambda s: s["stage"] == "finished")
            sse.close()

    asyncio.run(scenario())


def test_a_late_timer_from_an_earlier_attempt_cannot_cancel_a_later_one_of_another_browser(
    make_worker, monkeypatch
):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 0.6)

    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            first = await Sse.open(port, CLIENT_A)
            other = await Sse.open(port, CLIENT_B)
            await first.event()
            await other.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await first.until(lambda s: s["stage"] == "awaiting_confirmation")
            # The owner's only stream closes: a timer is armed for attempt A.
            first.close()
            await asyncio.sleep(0.15)
            assert (await api(port, "/api/cancel", CLIENT_A)).status == 200
            await other.until(lambda s: s["stage"] == "finished")
            # Another browser starts attempt B before A's timer fires.
            assert (
                await api(port, "/api/start", CLIENT_B, {"sides": ["right"]})
            ).status == 200
            await other.until(lambda s: s["stage"] == "awaiting_confirmation")
            attempt_b = server.controller.owner_attempt(CLIENT_B)
            await asyncio.sleep(1.0)
            assert server.controller.active is True
            assert server.controller.owner_attempt(CLIENT_B) is attempt_b
            assert server.controller.view().stage.value == "awaiting_confirmation"
            assert (await api(port, "/api/cancel", CLIENT_B)).status == 200
            final = await other.until(lambda s: s["stage"] == "finished")
            assert final["outcome"]["kind"] == "cancelled"
            other.close()

    asyncio.run(scenario())


def test_a_late_timer_from_an_earlier_attempt_cannot_cancel_a_later_one_of_the_same_browser(
    make_worker, monkeypatch
):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 0.6)

    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            first = await Sse.open(port, CLIENT_A)
            await first.event()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await first.until(lambda s: s["stage"] == "awaiting_confirmation")
            first.close()
            await asyncio.sleep(0.15)
            # The same browser ends attempt A, reconnects and starts attempt B.
            assert (await api(port, "/api/cancel", CLIENT_A)).status == 200
            second = await Sse.open(port, CLIENT_A)
            await second.until(lambda s: s["stage"] == "finished")
            assert (
                await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            ).status == 200
            await second.until(lambda s: s["stage"] == "awaiting_confirmation")
            attempt_b = server.controller.owner_attempt(CLIENT_A)
            await asyncio.sleep(1.0)
            assert server.controller.active is True
            assert server.controller.owner_attempt(CLIENT_A) is attempt_b
            assert (await api(port, "/api/cancel", CLIENT_A)).status == 200
            await second.until(lambda s: s["stage"] == "finished")
            second.close()

    asyncio.run(scenario())


def test_a_stream_that_owned_nothing_cannot_cancel_anyone_when_its_timer_fires(
    make_worker, monkeypatch
):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 0.3)

    async def scenario():
        async with serving(make_worker, WAITING) as (server, _store):
            port = server.port
            idle = await Sse.open(port, CLIENT_B)
            owner = await Sse.open(port, CLIENT_A)
            await idle.event()
            await owner.event()
            idle.close()
            await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            await owner.until(lambda s: s["stage"] == "awaiting_confirmation")
            await asyncio.sleep(0.8)
            assert server.controller.active is True
            assert server.controller.view().stage.value == "awaiting_confirmation"
            await api(port, "/api/cancel", CLIENT_A)
            await owner.until(lambda s: s["stage"] == "finished")
            owner.close()

    asyncio.run(scenario())


def test_the_server_forgets_a_tab_once_its_grace_window_ends(make_worker, monkeypatch):
    monkeypatch.setattr(server_module, "_DISCONNECT_GRACE_S", 0.2)

    async def scenario():
        async with serving(make_worker) as (server, _store):
            sse = await Sse.open(server.port, CLIENT_A)
            await sse.event()
            assert CLIENT_A in server._tabs
            sse.close()
            await asyncio.sleep(0.8)
            assert CLIENT_A not in server._tabs

    asyncio.run(scenario())


# --- The parent-side watchdog, seen through the page ---------------------------------------------

# Short deadlines keep these fast; the supervisor polls its deadline every 0.25 s.
WATCHDOG = {
    "startup_timeout_s": 1.0,
    "confirm_timeout_s": 1.0,
    "finish_timeout_s": 1.0,
    "term_grace_s": 1,
}


def test_a_confirm_the_sdk_never_answers_ends_uncertain_without_a_browser_cancel(
    make_worker, commands_log
):
    async def scenario():
        flags = {**WAITING, "hang_on_confirm": True}
        async with serving(make_worker, flags, **WATCHDOG) as (server, store):
            port = server.port
            sse = await Sse.open(port, CLIENT_A)
            await sse.event()
            started = await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert started.status == 200, started.body
            state = await sse.until(
                lambda s: (
                    s["stage"] == "awaiting_confirmation"
                    and s["prompt"]
                    and s["prompt"]["token"]
                )
            )
            # The confirm request stays open until the worker answers or the attempt ends.
            confirm = asyncio.create_task(
                api(port, "/api/confirm", CLIENT_A, {"token": state["prompt"]["token"]})
            )
            final = await sse.until(lambda s: s["stage"] == "finished", timeout=20)
            outcome = final["outcome"]
            assert outcome["kind"] == "uncertain"
            assert "did not respond within 1 s" in outcome["message"]
            assert "while waiting for the SDK to start the step" in outcome["message"]
            assert outcome["files"] == [] and outcome["directory"] is None
            assert outcome["log_tail"]
            assert final["active"] is False
            # The browser's own request was released, not left hanging, and never cancelled anything.
            reply = await asyncio.wait_for(confirm, 10)
            assert reply.status == 200, reply.body
            assert "cancel" not in commands(commands_log)
            assert store.list_published() == []
            sse.close()

    asyncio.run(scenario())


def test_a_worker_silent_after_start_ends_uncertain_over_http(
    make_worker, commands_log
):
    async def scenario():
        flags = {**WAITING, "hang_after_start": True}
        async with serving(make_worker, flags, **WATCHDOG) as (server, store):
            port = server.port
            sse = await Sse.open(port, CLIENT_A)
            await sse.event()
            started = await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert started.status == 200, started.body
            final = await sse.until(lambda s: s["stage"] == "finished", timeout=20)
            outcome = final["outcome"]
            assert outcome["kind"] == "uncertain"
            assert (
                "while waiting for SDK initialisation, connection and discovery"
                in outcome["message"]
            )
            assert outcome["files"] == []
            assert commands(commands_log) == ["start"]
            assert store.list_published() == []
            # The station is free again: a new attempt is not refused as busy.
            again = await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert again.status == 200, again.body
            sse.close()

    asyncio.run(scenario())


def test_a_prompt_left_longer_than_every_deadline_can_still_be_answered_over_http(
    make_worker,
):
    async def scenario():
        async with serving(make_worker, WAITING, **WATCHDOG) as (server, store):
            port = server.port
            sse = await Sse.open(port, CLIENT_A)
            await sse.event()
            started = await api(port, "/api/start", CLIENT_A, {"sides": ["left"]})
            assert started.status == 200, started.body
            state = await sse.until(
                lambda s: (
                    s["stage"] == "awaiting_confirmation"
                    and s["prompt"]
                    and s["prompt"]["token"]
                )
            )
            # An operator reading the prompt for far longer than any watchdog deadline.
            await asyncio.sleep(2.5)
            assert server.controller.view().stage.value == "awaiting_confirmation"
            reply = await api(
                port, "/api/confirm", CLIENT_A, {"token": state["prompt"]["token"]}
            )
            assert reply.status == 200, reply.body
            final = await sse.until(lambda s: s["stage"] == "finished", timeout=20)
            assert final["outcome"]["kind"] == "completed"
            assert len(store.list_published()) == 1
            sse.close()

    asyncio.run(scenario())

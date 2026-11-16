# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""view/server.py and view/page.html: the live view's HTTP side.

Each test opens one connection to 127.0.0.1 on a port the system picked and
makes a handful of requests on it. No browser is started. The view token is
the secret a page is served with when it is loaded; only the page loaded last
has the valid one.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import logging
import re
import socket
import threading
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import ViewStatus
from isaaccapture_examples.agentic_remote.view import server as view_server
from isaaccapture_examples.agentic_remote.view.model import NO_PICTURE, ViewModel
from isaaccapture_examples.agentic_remote.view.server import make_server
from link_bench import robot_state

PAGE = Path(view_server.__file__).with_name("page.html")
JPEG = b"\xff\xd8 a picture \xff\xd9"
NOW = 500.0
JSON_TYPE = {"Content-Type": "application/json"}
# Where a served page holds its view token: 24 random bytes as text for a URL.
TOKEN_IN_PAGE = re.compile(r'const TOKEN = "([A-Za-z0-9_-]{32})";')
LOGGER = "isaaccapture.examples.agentic_remote"


def ack(seq: int, token: str, visible: bool = True) -> dict[str, Any]:
    """The body of the page's POST /shown."""
    return {"seq": seq, "visible": visible, "token": token}


class Served:
    """A started server on a model, and one HTTP connection to it.

    `now` is what the server reads as this machine's monotonic clock.
    """

    def __init__(self) -> None:
        self.model = ViewModel()
        self.now = NOW
        # The model's first ROBOT_STATE came a minute before the pictures of
        # these tests, on the same clocks.
        state = robot_state()
        first = dataclasses.replace(
            state,
            clock_wall_s=state.clock_wall_s - 60.0,
            clock_mono_s=state.clock_mono_s - 60.0,
        )
        self.model.on_message(first, NOW - 60.0)
        self.model.on_message(state, NOW)
        self.server = make_server(
            self.model,
            "127.0.0.1",
            0,
            monotonic=lambda: self.now,
            picture_period_s=0.125,
        )
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        self.http = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)

    def ask(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        self.http.request(method, path, body=body, headers=headers or {})
        reply = self.http.getresponse()
        return reply.status, dict(reply.getheaders()), reply.read()

    def put_picture(self, seq: int = 1) -> None:
        """A camera picture that was exposed and arrived 0.1 s before NOW."""
        wall_now = self.model.thor_wall_now(NOW)
        self.model.put_picture(seq, JPEG, wall_now - 0.1, NOW - 0.1, False)

    def load(self) -> str:
        """Load the page as a browser does; the view token it was served with."""
        status, _, body = self.ask("GET", "/")
        assert status == 200
        found = TOKEN_IN_PAGE.search(body.decode("utf-8"))
        assert found is not None, "the served page holds no view token"
        return found.group(1)

    def fetch(self, token: str | None) -> tuple[int, str | None, bytes]:
        """GET /frame.jpg with that view token: status, picture number, body."""
        headers = {} if token is None else {"X-View-Token": token}
        status, told, body = self.ask("GET", "/frame.jpg", headers=headers)
        return status, told.get("X-Picture-Seq"), body

    def shown(self, body: Any, **headers: str) -> int:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        return self.ask("POST", "/shown", data, {**JSON_TYPE, **headers})[0]

    def live(self) -> bool:
        return self.model.view_status(NOW, self.model.thor_wall_now(NOW)).live

    def close(self) -> None:
        self.http.close()
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(5.0)


@pytest.fixture
def served() -> Iterator[Served]:
    made = Served()
    yield made
    made.close()


def test_the_page_frame_status_and_shown_routes(served):
    status, headers, body = served.ask("GET", "/")
    # The page as the file holds it, with a token where the file marks its place.
    page = PAGE.read_bytes()
    assert page.count(view_server.TOKEN_PLACE) == 1
    token = TOKEN_IN_PAGE.search(body.decode("utf-8")).group(1)
    assert status == 200
    assert body == page.replace(b"__VIEW_TOKEN__", token.encode())
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"
    assert view_server.TOKEN_HEADER == "X-View-Token"

    # No picture yet.
    assert served.fetch(token) == (204, None, b"")
    status, headers, body = served.ask("GET", "/status.json")
    assert status == 200 and headers["Content-Type"] == "application/json"
    told = json.loads(body)
    assert told["picture"]["overlay"] == NO_PICTURE
    assert told["page"] == {"picture_period_s": 0.125}
    wall_now = served.model.thor_wall_now(NOW)
    assert {**served.model.status_json(NOW, wall_now), "page": told["page"]} == told

    # The newest picture, with its number in the header.
    served.model.put_picture(6, b"old", wall_now - 0.3, NOW - 0.3, False)
    served.model.put_picture(7, JPEG, wall_now - 0.1, NOW - 0.1, False)
    status, headers, body = served.ask(
        "GET", "/frame.jpg?x=1", headers={"X-View-Token": token}
    )
    assert (status, body) == (200, JPEG)
    assert headers["X-Picture-Seq"] == "7"
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["Cache-Control"] == "no-store"

    # The page's acknowledgement makes the view live; a hidden tab ends that.
    assert not served.live()
    assert served.shown(ack(7, token)) == 204
    assert served.live()
    served.model.note_sent(ViewStatus(live=True, age_s=0.1))
    told = json.loads(served.ask("GET", "/status.json")[2])
    assert told["view_status"] == {"live": True, "age_s": 0.1}
    assert told["picture"]["seq"] == 7 and told["picture"]["overlay"] == ""
    assert served.shown(ack(7, token, visible=False)) == 204
    assert not served.live()


def test_a_bad_acknowledgement_is_refused_and_changes_nothing(served):
    served.put_picture()
    token = served.load()
    assert served.fetch(token)[0] == 200
    good = ack(1, token)
    for body in (
        b"not json",
        b"[1, true]",
        {**good, "seq": "1"},
        {**good, "seq": True},
        {**good, "seq": 1.5},
        {**good, "visible": 1},
        {"seq": 1, "token": token},
        {**good, "token": 5},
        {**good, "token": None},
        {**good, "token": [token]},
        b"\xff\xfe",
    ):
        assert served.shown(body) == 400, body
    assert served.shown(b" " * (view_server.MAX_SHOWN_BYTES + 1)) == 413
    assert not served.live()
    # From a page of another origin, and under another host name.
    assert served.shown(good, Origin="http://evil.test") == 403
    assert served.shown(good, Host="evil.test") == 403
    assert not served.live()
    own = f"http://127.0.0.1:{served.port}"
    assert served.shown(good, Origin=own) == 204
    assert served.live()


def test_nothing_but_the_four_routes_is_served(served):
    token = served.load()
    assert served.ask("GET", "/nothing")[0] == 404
    assert served.ask("GET", "/../etc/passwd")[0] == 404
    assert served.ask("GET", "/page.html")[0] == 404
    assert served.ask("GET", "/", headers={"Host": "evil.test"})[0] == 403
    # /shown is the one thing that can be sent to the view.
    for path in ("/", "/status.json", "/frame.jpg", "/stop", "/command"):
        assert served.ask("POST", path, b"{}", JSON_TYPE)[0] == 404, path
    assert served.ask("GET", "/shown")[0] == 404
    # None of these loaded the page: its token is still the valid one.
    assert served.fetch(token)[0] == 204


def test_the_other_methods_are_not_served():
    made = Served()
    try:
        for method in ("PUT", "DELETE", "PATCH"):
            # The server closes the connection after an unknown method.
            http_ = http.client.HTTPConnection("127.0.0.1", made.port, timeout=5.0)
            http_.request(method, "/shown", body=b"{}")
            assert http_.getresponse().status == 501
            http_.close()
    finally:
        made.close()


def test_the_server_binds_loopback_only(served):
    assert served.server.server_address[0] == "127.0.0.1"
    assert served.server.socket.getsockname()[0] == "127.0.0.1"
    model = ViewModel()
    for address in ("0.0.0.0", "", "localhost", "::", "::1", "127.0.0.2", "10.0.0.1"):
        with pytest.raises(ValueError, match=r"only 127\.0\.0\.1"):
            make_server(model, address, 0)


# The view token


def test_an_acknowledgement_without_the_pages_token_is_refused_and_the_view_is_not_live(
    served,
):
    served.put_picture()
    # No page was loaded: a body without a token is refused.
    assert served.shown({"seq": 1, "visible": True}) == 403
    token = served.load()
    assert served.fetch(token) == (200, "1", JPEG)
    assert served.shown({"seq": 1, "visible": True}) == 403
    for other in ("", token[:-1], token + "x", "x" * 32, "é" * 32):
        assert served.shown(ack(1, other)) == 403, other
    # The header that goes with a picture request is not where /shown looks.
    assert served.shown({"seq": 1, "visible": True}, **{"X-View-Token": token}) == 403
    assert served.model._ack is None and not served.live()


def test_an_acknowledgement_with_the_token_makes_the_view_live(served):
    served.put_picture()
    token = served.load()
    assert served.fetch(token) == (200, "1", JPEG)
    assert not served.live()
    assert served.shown(ack(1, token)) == 204
    assert served.live()


def test_an_acknowledgement_a_timeout_after_the_fetch_is_refused(served):
    served.put_picture()
    token = served.load()
    assert served.fetch(token)[0] == 200
    # ack_timeout_s is 1 s; the server's clock says when each request came.
    served.now = NOW + 1.0
    assert served.shown(ack(1, token)) == 403
    assert served.model._ack is None
    assert served.fetch(token)[0] == 200
    served.now = NOW + 1.99
    assert served.shown(ack(1, token)) == 204
    assert served.model._ack == (1, True, NOW + 1.99)


def test_a_reload_replaces_the_token_and_the_old_page_is_refused(served):
    served.put_picture()
    old = served.load()
    assert served.fetch(old)[0] == 200 and served.shown(ack(1, old)) == 204
    new = served.load()
    assert new != old
    # The page loaded before gets no picture, and what it says counts no more.
    assert served.fetch(old) == (403, None, b"")
    assert served.shown(ack(1, old)) == 403
    assert served.shown(ack(1, old, visible=False)) == 403
    assert served.model._ack is None
    # The new page is served and counts.
    assert served.fetch(new) == (200, "1", JPEG)
    assert served.shown(ack(1, new)) == 204
    # The old page going on to ask takes nothing from the new one.
    assert served.fetch(old)[0] == 403 and served.shown(ack(1, old)) == 403
    assert served.model._ack == (1, True, NOW) and served.live()


def test_a_reload_makes_the_view_not_live_until_the_new_page_acknowledges(served):
    served.put_picture()
    old = served.load()
    assert served.fetch(old)[0] == 200 and served.shown(ack(1, old)) == 204
    assert served.live()
    # At once, while the old page's acknowledgement is not a second old.
    new = served.load()
    assert not served.live()
    # The new page did not fetch the picture yet: it cannot acknowledge it.
    assert served.shown(ack(1, new)) == 403
    assert not served.live()
    # A picture it fetched and did not acknowledge does not count either.
    assert served.fetch(new)[0] == 200
    assert not served.live()
    assert served.shown(ack(1, new)) == 204
    assert served.live()


def test_a_picture_is_not_served_without_the_token(served):
    # Without a picture too: a page that was replaced learns it from the 403.
    assert served.fetch(None) == (403, None, b"")
    token = served.load()
    assert served.fetch(token) == (204, None, b"")
    assert served.fetch(None) == (403, None, b"")
    served.put_picture()
    for other in (None, "", token[:-1], token + "x", "x" * 32, "é" * 32):
        assert served.fetch(other) == (403, None, b""), other
    # A refused request is no fetch: the page cannot acknowledge that picture.
    assert served.shown(ack(1, token)) == 403
    assert served.fetch(token) == (200, "1", JPEG)
    assert served.shown(ack(1, token)) == 204


def test_the_token_is_not_in_status_json_and_not_in_the_log(served, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    served.put_picture()
    token = served.load()
    assert served.fetch(token)[0] == 200 and served.shown(ack(1, token)) == 204
    assert served.fetch("x" * 32)[0] == 403
    assert served.shown(ack(1, "x" * 32)) == 403
    assert served.shown({**ack(1, token), "token": 5}) == 400
    served.model.note_sent(ViewStatus(live=True, age_s=0.1))
    status, _, body = served.ask("GET", "/status.json")
    assert status == 200 and json.loads(body)["view_status"]["live"] is True
    assert token.encode() not in body and b"token" not in body.lower()
    # Every request above was logged, and so was the page load.
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for line in (
        "the page was loaded; a page loaded before counts no longer",
        '"GET / HTTP/1.1" 200',
        '"GET /frame.jpg HTTP/1.1" 200',
        '"GET /frame.jpg HTTP/1.1" 403',
        '"POST /shown HTTP/1.1" 204',
        '"POST /shown HTTP/1.1" 403',
        '"POST /shown HTTP/1.1" 400',
        '"GET /status.json HTTP/1.1" 200',
    ):
        assert line in logged, line
    assert token not in logged and token not in caplog.text


def request_line(served: Served, line: str) -> bytes:
    """Send that request line as it is, on a connection of its own; the reply."""
    with socket.create_connection(("127.0.0.1", served.port), timeout=5.0) as sock:
        sock.sendall(f"{line}\r\nHost: 127.0.0.1:{served.port}\r\n\r\n".encode())
        return sock.recv(4096)


def test_a_token_in_a_request_line_is_not_logged(served, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    served.put_picture()
    token = served.load()
    # The page sends its token in a header and in a body. A client that puts
    # it into the address is not served by it.
    assert served.ask("GET", f"/frame.jpg?token={token}")[0] == 403
    assert served.ask("GET", f"/status.json?{token}")[0] == 200
    assert served.ask("GET", f"/{token}")[0] == 404
    body = json.dumps({"seq": 1, "visible": True}).encode()
    assert served.ask("POST", f"/shown?token={token}", body, JSON_TYPE)[0] == 403
    # Request lines the server refuses, with the token as each of their words.
    for line in (
        f"{token} / HTTP/1.1",
        f"{token} /",
        f"GET / {token}",
        f"GET /x {token} extra words",
        f"GET /x {token} extra HTTP/1.1",
    ):
        assert request_line(served, line), line
    # Each request was logged, by the words the server knows and its status.
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for told in (
        '"GET /frame.jpg HTTP/1.1" 403',
        '"GET /status.json HTTP/1.1" 200',
        '"POST /shown HTTP/1.1" 403',
        '"another request" 404',
        '"another request" 501',
        '"another request" 400',
    ):
        assert told in logged, told
    assert token not in logged and token not in caplog.text
    # None of them loaded the page: its token is still the valid one.
    assert served.fetch(token)[0] == 200


def test_status_json_needs_no_token_and_replaces_none(served):
    served.put_picture()
    token = served.load()
    assert served.fetch(token)[0] == 200 and served.shown(ack(1, token)) == 204
    for headers in ({}, {"X-View-Token": "x" * 32}, {"X-View-Token": token}):
        status, _, body = served.ask("GET", "/status.json", headers=headers)
        assert status == 200 and json.loads(body)["picture"]["seq"] == 1
    # The page's token is still the valid one, and its acknowledgement stands.
    assert served.live()
    assert served.fetch(token)[0] == 200 and served.shown(ack(1, token)) == 204


# The page itself


class _Tags(HTMLParser):
    """Every tag of a page with its attributes, and the text of script and style."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: dict[str, str] = {"script": "", "style": ""}
        self._in = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        self._in = tag

    def handle_endtag(self, tag: str) -> None:
        self._in = ""

    def handle_data(self, data: str) -> None:
        if self._in in self.text:
            self.text[self._in] += data


def page_tags() -> _Tags:
    tags = _Tags()
    tags.feed(PAGE.read_text(encoding="utf-8"))
    return tags


def test_the_page_loads_nothing_from_another_host(served):
    text = PAGE.read_text(encoding="utf-8")
    page = page_tags()
    names = [tag for tag, _ in page.tags]
    # One inline script, one inline style, and no tag that loads anything.
    assert names.count("script") == names.count("style") == 1
    loaders = {"link", "img", "iframe", "object", "embed", "video", "audio", "source"}
    assert not loaders & set(names)
    for tag, attrs in page.tags:
        assert not {"src", "href", "srcset", "data", "action"} & set(attrs), tag
        assert not [name for name in attrs if name.startswith("on")], tag
    assert "://" not in text and "//" not in page.text["style"]
    assert "url(" not in text and "@import" not in text
    # The script asks only this server, for exactly three paths.
    script = page.text["script"]
    asked = re.findall(r"fetch\(\s*([^,)]+)", script)
    assert sorted(asked) == ['"/frame.jpg"', '"/shown"', '"/status.json"']
    for word in (
        "XMLHttpRequest",
        "WebSocket",
        "EventSource",
        "sendBeacon",
        "import(",
        "importScripts",
        "window.open",
        "location",
        "innerHTML",
        "outerHTML",
        "document.write",
        "eval(",
    ):
        assert word not in script, word
    # The server says the same to the browser.
    policy = served.ask("GET", "/")[1]["Content-Security-Policy"]
    assert policy == (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src 'self' blob:; connect-src 'self'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    )


def test_the_page_can_send_nothing_but_the_acknowledgement():
    page = page_tags()
    names = {tag for tag, _ in page.tags}
    assert not names & {"button", "input", "form", "select", "textarea", "a"}
    script = page.text["script"]
    # One POST in the whole script, to /shown, with a picture number, whether
    # the tab is visible and the page's view token.
    assert script.count("POST") == 1 and script.count("method:") == 1
    post = re.search(r'fetch\("/shown", \{method: "POST"', script)
    assert post is not None
    assert (
        'seq: seq, visible: document.visibilityState === "visible", token: TOKEN});'
        in script
    )
    assert "addEventListener" in script and script.count("addEventListener") == 1
    assert 'addEventListener("visibilitychange"' in script


def test_the_page_sends_its_token_with_picture_requests_and_acknowledgements_only():
    text = PAGE.read_text(encoding="utf-8")
    script = page_tags().text["script"]
    # One place in the file for the token the server puts in, in the script.
    assert text.count("__VIEW_TOKEN__") == 1
    assert 'const TOKEN = "__VIEW_TOKEN__";' in script
    assert re.search(
        r'fetch\("/frame\.jpg", \{cache: "no-store",\s+'
        r'headers: \{"X-View-Token": TOKEN\}\}\)',
        script,
    )
    # Beside its definition: the picture request and the acknowledgement.
    assert len(re.findall(r"\bTOKEN\b", script)) == 3
    assert 'fetch("/status.json", {cache: "no-store"})' in script
    assert "console." not in script and "Storage" not in script


def test_a_page_whose_token_is_refused_says_so_and_stops_asking_for_pictures():
    script = page_tags().text["script"]
    assert 'const REPLACED = "THIS PAGE WAS REPLACED: reload it";' in script
    # A 403 for a picture: the words go over the picture, and the function
    # returns before it asks for its next turn.
    refused = script[script.index("if (reply.status === 403) {") :]
    refused = refused[: refused.index("}")]
    assert "replaced = true;" in refused
    assert "overlay.textContent = REPLACED;" in refused
    assert refused.rstrip().endswith("return;")
    assert script.count("setTimeout(nextPicture, wait);") == 1
    assert script.index("return;") < script.index("setTimeout(nextPicture, wait);")
    # Only a refused picture request says so. The reply to an acknowledgement
    # is not read: the server refuses a late one of the valid page as well.
    assert script.count("reply.status === 403") == 1
    assert script.count("replaced = true;") == 1
    assert script.count("await fetch(") == 2 and ".then(" not in script
    assert '{"Content-Type": "application/json"}}).catch(() => {});' in script
    # The status poll does not write over the words, and a replaced page
    # acknowledges nothing when its tab is shown or hidden.
    assert "overlay.textContent = replaced ? REPLACED : picture.overlay;" in script
    assert "if (drawnSeq !== null && !replaced) { acknowledge(drawnSeq); }" in script


def test_the_page_writes_the_banner_over_the_picture(served):
    page = page_tags()
    # The banner lies in the stage, between the picture and the overlay.
    ids = [attrs.get("id") for _, attrs in page.tags]
    stage = ids.index("stage")
    assert ids[stage : stage + 4] == ["stage", "picture", "banner", "overlay"]
    assert 'banner.textContent = status.banner || "";' in page.text["script"]
    # The words come with the status while the Thor says a person is driving.
    assert json.loads(served.ask("GET", "/status.json")[2])["banner"] == ""
    driving = dataclasses.replace(robot_state(), health={"human": {"driving": True}})
    served.model.on_message(driving, NOW)
    told = json.loads(served.ask("GET", "/status.json")[2])
    assert told["banner"] == "HUMAN DRIVING"


def test_the_page_names_its_license():
    text = PAGE.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>\n<!--\nSPDX-FileCopyrightText: ")
    assert "SPDX-License-Identifier: Apache-2.0" in text

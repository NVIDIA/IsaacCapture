# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""view/server.py and view/page.html: the live view's HTTP side.

Each test opens one connection to 127.0.0.1 on a port the system picked and
makes a handful of requests on it. No browser is started.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import re
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


class Served:
    """A started server on a model, and one HTTP connection to it."""

    def __init__(self) -> None:
        self.model = ViewModel()
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
            self.model, "127.0.0.1", 0, monotonic=lambda: NOW, picture_period_s=0.125
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
    assert status == 200 and body == PAGE.read_bytes()
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"

    # No picture yet.
    status, headers, body = served.ask("GET", "/frame.jpg")
    assert (status, body) == (204, b"") and "X-Picture-Seq" not in headers
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
    status, headers, body = served.ask("GET", "/frame.jpg?x=1")
    assert (status, body) == (200, JPEG)
    assert headers["X-Picture-Seq"] == "7"
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["Cache-Control"] == "no-store"

    # The page's acknowledgement makes the view live; a hidden tab ends that.
    assert not served.live()
    assert served.shown({"seq": 7, "visible": True}) == 204
    assert served.live()
    served.model.note_sent(ViewStatus(live=True, age_s=0.1))
    told = json.loads(served.ask("GET", "/status.json")[2])
    assert told["view_status"] == {"live": True, "age_s": 0.1}
    assert told["picture"]["seq"] == 7 and told["picture"]["overlay"] == ""
    assert served.shown({"seq": 7, "visible": False}) == 204
    assert not served.live()


def test_a_bad_acknowledgement_is_refused_and_changes_nothing(served):
    wall_now = served.model.thor_wall_now(NOW)
    served.model.put_picture(1, JPEG, wall_now - 0.1, NOW - 0.1, False)
    for body in (
        b"not json",
        b"[1, true]",
        {"seq": "1", "visible": True},
        {"seq": True, "visible": True},
        {"seq": 1.5, "visible": True},
        {"seq": 1, "visible": 1},
        {"seq": 1},
        b"\xff\xfe",
    ):
        assert served.shown(body) == 400, body
    assert served.shown(b" " * (view_server.MAX_SHOWN_BYTES + 1)) == 413
    assert not served.live()
    # From a page of another origin, and under another host name.
    assert served.shown({"seq": 1, "visible": True}, Origin="http://evil.test") == 403
    assert served.shown({"seq": 1, "visible": True}, Host="evil.test") == 403
    assert not served.live()
    own = f"http://127.0.0.1:{served.port}"
    assert served.shown({"seq": 1, "visible": True}, Origin=own) == 204
    assert served.live()


def test_nothing_but_the_four_routes_is_served(served):
    assert served.ask("GET", "/nothing")[0] == 404
    assert served.ask("GET", "/../etc/passwd")[0] == 404
    assert served.ask("GET", "/page.html")[0] == 404
    assert served.ask("GET", "/", headers={"Host": "evil.test"})[0] == 403
    # /shown is the one thing that can be sent to the view.
    for path in ("/", "/status.json", "/frame.jpg", "/stop", "/command"):
        assert served.ask("POST", path, b"{}", JSON_TYPE)[0] == 404, path
    assert served.ask("GET", "/shown")[0] == 404


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
    # One POST in the whole script, to /shown, with a picture number and whether
    # the tab is visible.
    assert script.count("POST") == 1 and script.count("method:") == 1
    post = re.search(r'fetch\("/shown", \{method: "POST"', script)
    assert post is not None
    assert 'visible: document.visibilityState === "visible"' in script
    assert "addEventListener" in script and script.count("addEventListener") == 1
    assert 'addEventListener("visibilitychange"' in script


def test_the_page_names_its_license():
    text = PAGE.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>\n<!--\nSPDX-FileCopyrightText: ")
    assert "SPDX-License-Identifier: Apache-2.0" in text

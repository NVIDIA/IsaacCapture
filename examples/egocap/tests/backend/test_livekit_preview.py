# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The LiveKit viewer token (decoded independently of the signer) and GET /api/livekit-token."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from capture_server import protocol as p
from egocap_web.app import app
from egocap_web.livekit_preview import (
    API_KEY_ENV,
    API_SECRET_ENV,
    subscriber_token,
)

KEY = "APIkeyForTests"
SECRET = "super-secret-signing-value-0123456789"


def _unb64(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _decode(token: str) -> tuple[dict, dict, bytes, bytes]:
    """header, claims, raw signature, signing input -- parsed with nothing but base64 and json."""
    header_b64, claims_b64, sig_b64 = token.split(".")
    assert "=" not in token
    return (
        json.loads(_unb64(header_b64)),
        json.loads(_unb64(claims_b64)),
        _unb64(sig_b64),
        f"{header_b64}.{claims_b64}".encode("ascii"),
    )


def _hs256(secret: str, signing_input: bytes) -> bytes:
    return hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()


def test_token_header_is_hs256_jwt():
    header, _, _, _ = _decode(subscriber_token(api_key=KEY, api_secret=SECRET, room="r", now=1000))
    assert header == {"alg": "HS256", "typ": "JWT"}


def test_token_claims_name_issuer_validity_window_and_subscribe_only_grant():
    _, claims, _, _ = _decode(
        subscriber_token(api_key=KEY, api_secret=SECRET, room="sensing-preview", now=1000, ttl_s=60)
    )
    assert claims["iss"] == KEY
    assert claims["nbf"] == 1000
    assert claims["exp"] == 1060
    assert claims["sub"].startswith("egocap-viewer-")
    assert claims["video"] == {
        "room": "sensing-preview",
        "roomJoin": True,
        "canSubscribe": True,
        "canPublish": False,
        "canPublishData": False,
    }


def test_token_default_ttl_is_one_hour():
    _, claims, _, _ = _decode(subscriber_token(api_key=KEY, api_secret=SECRET, room="r", now=5))
    assert claims["exp"] - claims["nbf"] == 3600


def test_every_token_gets_a_unique_viewer_identity():
    subs = {
        _decode(subscriber_token(api_key=KEY, api_secret=SECRET, room="r", now=1))[1]["sub"] for _ in range(50)
    }
    assert len(subs) == 50


def test_signature_verifies_with_the_secret_and_not_with_another():
    _, _, signature, signing_input = _decode(subscriber_token(api_key=KEY, api_secret=SECRET, room="r", now=1))
    assert hmac.compare_digest(signature, _hs256(SECRET, signing_input))
    assert not hmac.compare_digest(signature, _hs256("another-secret", signing_input))


def test_token_never_contains_the_secret():
    token = subscriber_token(api_key=KEY, api_secret=SECRET, room="r", now=1)
    assert SECRET not in token
    assert SECRET.encode() not in b"".join(_unb64(s) for s in token.split(".")[:2])


def _offer(room: str = "sensing-preview") -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream(
        stream_id=16,
        name="Head left",
        url="wss://lk.example:7881",
        room=room,
        publisher="orin-sensing",
        track="sensing-left",
    )


@pytest.fixture
def client_factory():
    """A TestClient over the real app whose monitor reports a session offering the given streams."""
    previous = getattr(app.state, "monitor", None)

    def make(streams: tuple | None) -> TestClient:
        started = None if streams is None else SimpleNamespace(streams=streams)
        app.state.monitor = SimpleNamespace(native=SimpleNamespace(started=started))
        return TestClient(app)

    yield make
    app.state.monitor = previous


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv(API_SECRET_ENV, SECRET)


URL = "/api/livekit-token?room=sensing-preview"


def test_no_session_yet_gets_404(client_factory, credentials):
    response = client_factory(None).get(URL)
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"


def test_session_offering_nothing_gets_404(client_factory, credentials):
    assert client_factory(()).get(URL).status_code == 404


def test_session_with_only_non_livekit_streams_gets_404(client_factory, credentials):
    tracking = p.tracking_stream_descriptor((p.SourceType.HEAD,), 60)
    assert client_factory((tracking,)).get(URL).status_code == 404


def test_room_not_offered_gets_404_even_with_credentials(client_factory, credentials):
    response = client_factory((_offer("other-room"),)).get(URL)
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"


def test_missing_or_empty_room_query_gets_404(client_factory, credentials):
    client = client_factory((_offer(),))
    assert client.get("/api/livekit-token").status_code == 404
    assert client.get("/api/livekit-token?room=").status_code == 404


def test_old_route_is_gone(client_factory, credentials):
    assert client_factory((_offer(),)).get("/api/livekit-preview").status_code == 404


@pytest.mark.parametrize("missing", [(API_KEY_ENV,), (API_SECRET_ENV,), (API_KEY_ENV, API_SECRET_ENV)])
def test_missing_credentials_give_503_naming_variables_but_no_values(client_factory, monkeypatch, missing):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv(API_SECRET_ENV, SECRET)
    for name in missing:
        monkeypatch.delenv(name)
    response = client_factory((_offer(),)).get(URL)
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    detail = response.json()["detail"]
    assert API_KEY_ENV in detail and API_SECRET_ENV in detail
    assert SECRET not in response.text
    assert KEY not in response.text


def test_empty_credentials_count_as_unset(client_factory, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "")
    monkeypatch.setenv(API_SECRET_ENV, SECRET)
    assert client_factory((_offer(),)).get(URL).status_code == 503


def test_offered_room_gets_a_verifiable_subscribe_only_token(client_factory, credentials):
    response = client_factory((_offer(),)).get(URL)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert set(body) == {"token"}
    _, claims, signature, signing_input = _decode(body["token"])
    assert claims["video"]["room"] == "sensing-preview"
    assert claims["video"]["canPublish"] is False
    assert claims["iss"] == KEY
    assert hmac.compare_digest(signature, _hs256(SECRET, signing_input))


def test_room_is_matched_among_several_offered_streams(client_factory, credentials):
    offers = (_offer("room-a"), replace(_offer("sensing-preview"), stream_id=17, name="b", track="u"))
    assert client_factory(offers).get(URL).status_code == 200


def test_secret_is_never_in_the_response_body_or_headers(client_factory, credentials):
    response = client_factory((_offer(),)).get(URL)
    assert SECRET not in response.text
    assert SECRET not in str(dict(response.headers))


def test_each_request_gets_a_fresh_viewer_identity(client_factory, credentials):
    client = client_factory((_offer(),))
    subs = {_decode(client.get(URL).json()["token"])[1]["sub"] for _ in range(5)}
    assert len(subs) == 5

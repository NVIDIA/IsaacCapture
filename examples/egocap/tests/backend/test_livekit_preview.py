# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The LiveKit viewer token (decoded independently of the signer) and GET /api/livekit-preview."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from egocap_web.app import app
from egocap_web.livekit_preview import (
    API_KEY_ENV,
    API_SECRET_ENV,
    LiveKitPreview,
    LiveKitTrack,
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


@pytest.fixture
def client_factory():
    """A TestClient over the real app with app.state.livekit_preview set directly (no lifespan needed)."""
    previous = getattr(app.state, "livekit_preview", None)

    def make(preview: LiveKitPreview | None) -> TestClient:
        app.state.livekit_preview = preview
        return TestClient(app)

    yield make
    app.state.livekit_preview = previous


PREVIEW = LiveKitPreview(
    url="wss://lk.example:7881",
    room="sensing-preview",
    publisher="orin-sensing",
    tracks=(LiveKitTrack("Head left", "sensing-left"), LiveKitTrack("Head right", "sensing-right")),
)


def test_unconfigured_station_gets_404_with_no_store(client_factory, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv(API_SECRET_ENV, SECRET)
    response = client_factory(None).get("/api/livekit-preview")
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("missing", [(API_KEY_ENV,), (API_SECRET_ENV,), (API_KEY_ENV, API_SECRET_ENV)])
def test_missing_credentials_give_503_naming_variables_but_no_values(client_factory, monkeypatch, missing):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv(API_SECRET_ENV, SECRET)
    for name in missing:
        monkeypatch.delenv(name)
    response = client_factory(PREVIEW).get("/api/livekit-preview")
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    detail = response.json()["detail"]
    assert API_KEY_ENV in detail and API_SECRET_ENV in detail
    assert SECRET not in response.text
    assert KEY not in response.text


def test_empty_credentials_count_as_unset(client_factory, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "")
    monkeypatch.setenv(API_SECRET_ENV, SECRET)
    assert client_factory(PREVIEW).get("/api/livekit-preview").status_code == 503


def test_configured_station_gets_config_and_verifiable_token(client_factory, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv(API_SECRET_ENV, SECRET)
    response = client_factory(PREVIEW).get("/api/livekit-preview")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert set(body) == {"url", "room", "publisher", "token", "tracks"}
    assert body["url"] == "wss://lk.example:7881"
    assert body["room"] == "sensing-preview"
    assert body["publisher"] == "orin-sensing"
    assert body["tracks"] == [
        {"name": "Head left", "track": "sensing-left"},
        {"name": "Head right", "track": "sensing-right"},
    ]
    _, claims, signature, signing_input = _decode(body["token"])
    assert claims["video"]["room"] == "sensing-preview"
    assert claims["iss"] == KEY
    assert hmac.compare_digest(signature, _hs256(SECRET, signing_input))


def test_secret_is_never_in_the_response_body_or_headers(client_factory, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv(API_SECRET_ENV, SECRET)
    response = client_factory(PREVIEW).get("/api/livekit-preview")
    assert SECRET not in response.text
    assert SECRET not in str(dict(response.headers))


def test_each_request_gets_a_fresh_viewer_identity(client_factory, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv(API_SECRET_ENV, SECRET)
    client = client_factory(PREVIEW)
    subs = {_decode(client.get("/api/livekit-preview").json()["token"])[1]["sub"] for _ in range(5)}
    assert len(subs) == 5

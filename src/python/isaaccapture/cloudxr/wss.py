#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CloudXR WSS Proxy — terminates TLS and forwards WebSocket traffic to a CloudXR Runtime backend."""

import asyncio
import errno
import json
import logging
import mimetypes
import os
from http import HTTPStatus
from urllib.parse import unquote, urlparse
import ssl
import sys
from collections.abc import Callable
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path

from ..logging_config._core import DATE_FORMAT, LINE_FORMAT, logging_enabled
from .env_config import get_env_config
from .oob_teleop_env import (
    client_ui_fields_from_env,
    default_initial_stream_config,
    redact_control_token,
    wss_proxy_port,
)
from .oob_teleop_hub import OOB_WS_PATH

try:
    import websockets
    from websockets.asyncio.client import connect as ws_connect
    from websockets.asyncio.server import serve as ws_serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response
except ImportError:
    sys.exit(
        "Missing dependency: websockets >= 14\n"
        "Install with: uv pip install --find-links=install/wheels 'isaaccapture[cloudxr]'"
    )


def _patch_request_parser_for_cors():
    """Allow HTTP OPTIONS through the websockets request parser.

    ``websockets >= 14`` rejects non-GET methods in ``Request.parse`` before
    ``process_request`` fires.  This wraps the parser so that OPTIONS requests
    (CORS preflight) are surfaced as a normal ``Request`` — the existing
    ``process_request`` hook in ``_make_http_handler`` already returns the
    correct 200 + CORS-headers response for them.
    """
    from websockets.http11 import Request, parse_headers

    _orig_parse = Request.parse.__func__

    @classmethod
    def _cors_aware_parse(cls, read_line):
        """Monkey-patched ``Request.parse`` that maps OPTIONS requests to a synthetic CORS path."""
        try:
            return (yield from _orig_parse(cls, read_line))
        except ValueError as exc:
            if "got OPTIONS" not in str(exc):
                raise
            headers = yield from parse_headers(read_line)
            return cls("/__cors_preflight__", headers)

    Request.parse = _cors_aware_parse


_patch_request_parser_for_cors()

log = logging.getLogger("isaaccapture.cloudxr.wss")


class _WssSessionFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_control_token(super().format(record))


@dataclass(frozen=True)
class CertPaths:
    """Resolved paths for the WSS TLS certificate and private key."""

    cert_dir: Path
    cert_file: Path
    key_file: Path


def cert_paths_from_dir(cert_dir: Path) -> CertPaths:
    """Return a :class:`CertPaths` with ``server.crt`` / ``server.key`` under *cert_dir*."""
    cert_dir = cert_dir.resolve()
    return CertPaths(
        cert_dir=cert_dir,
        cert_file=cert_dir / "server.crt",
        key_file=cert_dir / "server.key",
    )


def _generate_self_signed_cert(cert_paths: CertPaths) -> None:
    """Generate a self-signed RSA certificate and private key using the cryptography package."""
    import datetime
    import ipaddress
    import stat

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    # Write the private key with mode 0o600 from the start to avoid a world-readable
    # window between write_bytes() and a subsequent chmod().
    key_fd = os.open(
        cert_paths.key_file,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    with os.fdopen(key_fd, "wb") as f:
        f.write(key_pem)
    cert_paths.cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def ensure_certificate(cert_paths: CertPaths) -> None:
    """Generate a self-signed certificate if one does not already exist."""
    cert_exists = cert_paths.cert_file.exists()
    key_exists = cert_paths.key_file.exists()
    if cert_exists != key_exists:
        missing_file = cert_paths.key_file if cert_exists else cert_paths.cert_file
        raise RuntimeError(
            f"Found partial TLS cert pair in {cert_paths.cert_dir}; missing {missing_file.name}. "
            "Restore both files or remove both and retry."
        )

    if cert_exists and key_exists:
        log.info("Using existing SSL certificate from %s", cert_paths.cert_file)
        return

    log.info("Generating self-signed SSL certificate ...")
    cert_paths.cert_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(cert_paths.cert_dir, os.W_OK | os.X_OK):
        raise RuntimeError(
            f"Certificate directory {cert_paths.cert_dir} is not writable. "
            f"Create it manually and retry: mkdir -p {cert_paths.cert_dir}"
        )
    try:
        _generate_self_signed_cert(cert_paths)
    except PermissionError as exc:
        raise RuntimeError(
            f"Permission denied writing TLS certificate to {cert_paths.cert_dir}. "
            f"Create it manually and retry: mkdir -p {cert_paths.cert_dir}"
        ) from exc
    log.info("SSL certificate generated at %s", cert_paths.cert_file)


def build_ssl_context(cert_paths: CertPaths) -> ssl.SSLContext:
    """Build a TLS 1.2+ server :class:`ssl.SSLContext` from *cert_paths*."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(
        certfile=str(cert_paths.cert_file), keyfile=str(cert_paths.key_file)
    )
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Expose-Headers": "*",
}


def _cert_html() -> bytes:
    """Return minimal HTML confirming the self-signed certificate was accepted."""
    return (
        b"<!doctype html><html><head><meta charset=utf-8>"
        b"<style>body{font-family:system-ui,sans-serif;display:flex;"
        b"align-items:center;justify-content:center;height:100vh;margin:0;"
        b"background:#f5f5f5;color:#222}div{text-align:center}"
        b"h1{font-weight:600;font-size:1.5rem;margin-bottom:.5rem}"
        b"p{color:#555;font-size:1rem}</style></head>"
        b"<body><div><h1>Certificate Accepted</h1>"
        b"<p>You can close this tab and return to the web client.</p>"
        b"</div></body></html>"
    )


def _normalize_request_path(raw_path: str) -> str:
    """Normalize HTTP request-target: query stripped, absolute-URL form, ``%``-decoding, ``//``, ``.`` / ``..``."""
    path = (raw_path or "/").split("?")[0] or "/"
    if path.startswith(("http://", "https://")):
        path = urlparse(path).path or "/"
    path = unquote(path, errors="replace")
    segments = [p for p in path.split("/") if p and p != "."]
    stack: list[str] = []
    for seg in segments:
        if seg == "..":
            if stack:
                stack.pop()
        else:
            stack.append(seg)
    if not stack:
        return "/"
    return "/" + "/".join(stack)


def _is_oob_hub_http_path(path: str) -> bool:
    """True for OOB HTTP API paths on the WSS proxy."""
    return path in (
        "/api/oob/v1/state",
        "/api/oob/v1/config",
    )


def _parse_query_params(raw_path: str) -> dict[str, str]:
    """First occurrence wins; keys and values are URL-decoded."""
    if "?" not in raw_path:
        return {}
    qs = raw_path.split("?", 1)[1]
    out: dict[str, str] = {}
    for part in qs.split("&"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            k = unquote(k)
            if k not in out:
                out[k] = unquote(v, errors="replace")
        else:
            k = unquote(part)
            if k not in out:
                out[k] = ""
    return out


def _stream_config_from_query(q: dict[str, str]) -> tuple[dict | None, str | None]:
    """Build ``StreamConfig`` patch from query string (``serverIP=`` / ``port=`` / …)."""
    cfg: dict[str, object] = {}
    if "serverIP" in q:
        cfg["serverIP"] = q["serverIP"]
    if "port" in q and q["port"] != "":
        try:
            cfg["port"] = int(q["port"], 10)
        except ValueError:
            return None, "port must be an integer"
    if "panelHiddenAtStart" in q and q["panelHiddenAtStart"] != "":
        s = q["panelHiddenAtStart"].strip().lower()
        if s in ("1", "true", "yes", "on"):
            cfg["panelHiddenAtStart"] = True
        elif s in ("0", "false", "no", "off"):
            cfg["panelHiddenAtStart"] = False
        else:
            return None, "panelHiddenAtStart must be true or false"
    if "codec" in q and q["codec"] != "":
        cfg["codec"] = q["codec"]
    return cfg, None


def _oob_token(request, q: dict[str, str]) -> str | None:
    """Extract the OOB control token from the ``X-Control-Token`` header or ``token`` query param."""
    h = request.headers.get("X-Control-Token")
    if h:
        return h
    t = q.get("token")
    return t if t else None


def _json_response(status: int, phrase: str, body: dict) -> Response:
    """Build a JSON :class:`Response` with CORS headers."""
    return Response(
        status,
        phrase,
        Headers({"Content-Type": "application/json", **CORS_HEADERS}),
        json.dumps(body).encode(),
    )


def _make_http_handler(backend_host, backend_port, hub=None, static_dir=None):
    """Return the WSS HTTP request handler (OOB hub API, static ``/client/``, cert page)."""

    async def handle_http_request(connection, request):
        """Dispatch non-WebSocket HTTP: CORS preflight, OOB APIs, static client, or cert HTML."""
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None

        if request.headers.get("Access-Control-Request-Method"):
            return Response(
                200,
                "OK",
                Headers({"Content-Type": "text/plain", **CORS_HEADERS}),
                b"OK",
            )

        path = _normalize_request_path(request.path or "/")
        raw_path = request.path or "/"
        q = _parse_query_params(raw_path)

        if hub is not None and _is_oob_hub_http_path(path):
            token = _oob_token(request, q)
            if path == "/api/oob/v1/state":
                if not hub.check_token(token):
                    return _json_response(
                        401, "Unauthorized", {"error": "Unauthorized"}
                    )
                snapshot = await hub.get_snapshot()
                return Response(
                    200,
                    "OK",
                    Headers({"Content-Type": "application/json", **CORS_HEADERS}),
                    json.dumps(snapshot).encode(),
                )

            if path == "/api/oob/v1/config":
                if not hub.check_token(token):
                    return _json_response(
                        401, "Unauthorized", {"error": "Unauthorized"}
                    )
                cfg, err = _stream_config_from_query(q)
                if err:
                    return _json_response(400, "Bad Request", {"error": err})
                payload = {
                    "config": cfg,
                    "targetClientId": q.get("targetClientId"),
                    "token": token,
                }
                status, body = await hub.http_oob_set_config(payload)
                phrase = {
                    200: "OK",
                    400: "Bad Request",
                    401: "Unauthorized",
                    404: "Not Found",
                }.get(status, "Error")
                return _json_response(status, phrase, body)

        if hub is None and _is_oob_hub_http_path(path):
            return Response(
                404,
                "Not Found",
                Headers({"Content-Type": "text/plain", **CORS_HEADERS}),
                b"Not found",
            )

        # Static web client (--host-client).
        if static_dir is not None and (
            path == "/client" or path.startswith("/client/")
        ):
            # index.html asks for ``bundle.js`` relatively, which the browser
            # resolves against ``/`` unless the directory URL ends in a slash.
            # ``path`` is normalized, so the raw target decides slash presence.
            target, _, query = raw_path.partition("?")
            if path == "/client" and not target.endswith("/"):
                location = "/client/" + (f"?{query}" if query else "")
                return Response(
                    HTTPStatus.FOUND,
                    HTTPStatus.FOUND.phrase,
                    Headers({"Location": location, **CORS_HEADERS}),
                    b"",
                )
            _MIME = {
                "index.html": "text/html; charset=utf-8",
                "bundle.js": "application/javascript; charset=utf-8",
                "bundle.emulator.js": "application/javascript; charset=utf-8",
            }
            tail = path[len("/client") :].lstrip("/") or "index.html"
            if tail not in _MIME and not tail.startswith(
                "npm/@webxr-input-profiles/assets@"
            ):
                return Response(
                    404,
                    "Not Found",
                    Headers({"Content-Type": "text/plain", **CORS_HEADERS}),
                    b"Not found",
                )
            try:
                body = (static_dir / tail).read_bytes()
            except OSError:
                return Response(
                    404,
                    "Not Found",
                    Headers({"Content-Type": "text/plain", **CORS_HEADERS}),
                    b"Not found",
                )
            content_type = _MIME.get(tail) or mimetypes.guess_type(tail)[0]
            return Response(
                200,
                "OK",
                # Preserve the existing route and cache semantics; only frame
                # the static response body explicitly for headset browsers.
                Headers(
                    {
                        "Content-Type": content_type or "application/octet-stream",
                        "Content-Length": str(len(body)),
                        "Cache-Control": "no-store",
                        **CORS_HEADERS,
                    }
                ),
                body,
            )

        return Response(
            200,
            "OK",
            Headers({"Content-Type": "text/html; charset=utf-8", **CORS_HEADERS}),
            _cert_html(),
        )

    return handle_http_request


def add_cors_headers(connection, request, response):
    """``process_response`` hook: inject CORS headers into every WebSocket upgrade response."""
    response.headers.update(CORS_HEADERS)


_SKIP_HEADERS = {
    "host",
    "upgrade",
    "connection",
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-accept",
    "sec-websocket-extensions",
    "sec-websocket-protocol",
}


def _is_backend_connection_refused(exc: BaseException) -> bool:
    """True when ``ws_connect`` failed because nothing is listening (runtime not running)."""
    if isinstance(exc, ConnectionRefusedError):
        return True
    if isinstance(exc, OSError) and exc.errno in (
        errno.ECONNREFUSED,
        getattr(errno, "WSAECONNREFUSED", -1),
    ):
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if "errno 61" in msg or "errno 111" in msg:
            return True
        if "connection refused" in msg:
            return True
    return False


async def _pipe(src, dst, label: str):
    """Forward every message from *src* to *dst*, propagating the close frame."""
    try:
        async for msg in src:
            if isinstance(msg, str):
                log.debug("%s text (%d chars): %s", label, len(msg), msg[:200])
            else:
                log.debug("%s binary (%d bytes)", label, len(msg))
            await dst.send(msg)
    except websockets.ConnectionClosed as exc:
        rcvd = exc.rcvd
        log.debug(
            "%s closed: code=%s reason=%s",
            label,
            rcvd.code if rcvd else None,
            rcvd.reason if rcvd else "",
        )
        try:
            if exc.rcvd:
                await dst.close(exc.rcvd.code, exc.rcvd.reason)
            else:
                await dst.close()
        except websockets.ConnectionClosed:
            pass


async def proxy_handler(client, backend_host: str, backend_port: int):
    """Bidirectionally proxy a WebSocket *client* connection to *backend_host:backend_port*."""
    path = client.request.path or "/"
    backend_uri = f"ws://{backend_host}:{backend_port}{path}"

    headers_to_forward = {
        k: v
        for k, v in client.request.headers.raw_items()
        if k.lower() not in _SKIP_HEADERS
    }

    subprotocols = client.request.headers.get_all("Sec-WebSocket-Protocol")

    try:
        backend = await ws_connect(
            backend_uri,
            additional_headers=headers_to_forward,
            subprotocols=subprotocols or None,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=10,
        )
    except OSError as exc:
        if _is_backend_connection_refused(exc):
            log.warning(
                "No CloudXR runtime at ws://%s:%s (connection refused) for path %s — "
                "expected when running WSS+hub without the runtime; teleop signaling uses %s.",
                backend_host,
                backend_port,
                path,
                OOB_WS_PATH,
            )
            return
        log.exception("Failed to connect to backend %s", backend_uri)
        return
    except Exception:
        log.exception("Failed to connect to backend %s", backend_uri)
        return

    log.info("Proxying %s -> %s", client.remote_address, backend_uri)

    try:
        client_to_backend = asyncio.create_task(
            _pipe(client, backend, f"client->backend [{path}]")
        )
        backend_to_client = asyncio.create_task(
            _pipe(backend, client, f"backend->client [{path}]")
        )

        _done, pending = await asyncio.wait(
            [client_to_backend, backend_to_client],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except Exception:
        log.exception("Proxy error on %s", path)
    finally:
        await backend.close()
        log.info("Connection closed: %s", path)


def default_cert_paths() -> CertPaths:
    """Return cert paths under the default location (~/.cloudxr/certs)."""
    return cert_paths_from_dir(Path(get_env_config().openxr_run_dir()).parent / "certs")


async def run(
    log_file_path: str | Path | None,
    stop_future: asyncio.Future,
    backend_host: str = "localhost",
    backend_port: int = 49100,
    proxy_port: int | None = None,
    setup_oob: bool = False,
    usb_local: bool = False,
    host_client: bool = False,
    on_listening: Callable[[], None] | None = None,
    recovery_config=None,
    on_oob_status: Callable[[dict], object] | None = None,
    on_oob_fatal: Callable[[Exception], object] | None = None,
) -> None:
    """Start the WSS proxy server and run until *stop_future* is resolved.

    *on_listening* is called once the socket is bound and accepting, which is
    the only point a caller in another thread can distinguish a proxy that is
    serving from one still generating certificates or about to fail on a
    taken port.
    """
    # Console output comes from propagation to the root `isaaccapture` logger's
    # own handler (isaaccapture.logging_config); this only adds an optional,
    # additional per-session file, on top of that, when the caller wants one.
    # With logging off, this module owns its console or file output instead.
    _handler = None
    _handler_loggers: list[logging.Logger] = []
    if log_file_path is not None:
        _handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    elif not logging_enabled():
        _handler = logging.StreamHandler(sys.stderr)
    failure_log = log
    if _handler is not None:
        _handler.setFormatter(_WssSessionFormatter(LINE_FORMAT, datefmt=DATE_FORMAT))
        if log_file_path is not None:
            failure_log = logging.getLogger(f"{__name__}.session_failure")
            failure_log.setLevel(logging.ERROR)
            failure_log.propagate = False
        # Tracked so the finally below can detach it from every logger it was
        # attached to, not just this module's: a logger still holding a closed
        # FileHandler reopens the file on its next record, and a second run()
        # would stack another handler on top and duplicate every line.
        _handler_loggers = [
            log,
            # Keep OOB transitions in the per-session log.
            logging.getLogger("isaaccapture.cloudxr.oob_teleop_adb"),
            logging.getLogger("isaaccapture.cloudxr.oob_teleop_env"),
            logging.getLogger("isaaccapture.cloudxr.oob_teleop_hub"),
            logging.getLogger("isaaccapture.cloudxr.oob_teleop_lifecycle"),
        ]
        if failure_log is not log:
            _handler_loggers.append(failure_log)
        for _attached_log in _handler_loggers:
            if not logging_enabled():
                _attached_log.setLevel(logging.INFO)
                _attached_log.propagate = False
            _attached_log.addHandler(_handler)

    was_listening = False
    failure_logged = False
    try:
        resolved_port = wss_proxy_port() if proxy_port is None else proxy_port

        logging.getLogger("websockets").setLevel(logging.WARNING)
        cert_paths = default_cert_paths()

        ensure_certificate(cert_paths)
        ssl_ctx = build_ssl_context(cert_paths)

        hub = None
        if setup_oob:
            from .oob_teleop_hub import OOBControlHub  # noqa: PLC0415

            control_token = os.environ.get("CONTROL_TOKEN") or None
            initial = {
                **default_initial_stream_config(resolved_port),
                **client_ui_fields_from_env(),
            }
            hub = OOBControlHub(control_token=control_token, initial_config=initial)
            log.info(
                "Teleop control hub enabled (token=%s) OOB_WS=%s initial_stream=%s",
                "set" if control_token else "none",
                OOB_WS_PATH,
                initial,
            )

        def handler(ws):
            """Route an incoming WebSocket to the OOB hub or the backend proxy."""
            if hub is not None:
                path = _normalize_request_path(ws.request.path or "/")
                if path == OOB_WS_PATH:
                    return hub.handle_connection(ws)
            return proxy_handler(ws, backend_host, backend_port)

        # /client/ on this WSS port for --host-client and --usb-local (same
        # files; USB-local reaches them via adb reverse of PROXY_PORT).
        _host_client_static_dir = None
        if host_client or usb_local:
            from .oob_teleop_env import require_web_client_static_dir  # noqa: PLC0415

            _host_client_static_dir = require_web_client_static_dir(
                require_health_probe=setup_oob and not os.getenv("TELEOP_OOB_HUB_ONLY")
            )

        http_handler = _make_http_handler(
            backend_host, backend_port, hub=hub, static_dir=_host_client_static_dir
        )

        async with ws_serve(
            handler,
            host="",
            port=resolved_port,
            ssl=ssl_ctx,
            process_request=http_handler,
            process_response=add_cors_headers,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=10,
        ):
            from .oob_teleop_env import (
                usb_turn_port,
            )
            from .oob_teleop_lifecycle import OobLifecycle

            lifecycle_task = None
            lifecycle = None
            try:
                log.info("WSS proxy listening on port %d", resolved_port)
                if on_listening is not None:
                    on_listening()
                was_listening = True
                if setup_oob and not os.getenv("TELEOP_OOB_HUB_ONLY"):
                    from .oob_teleop_env import resolve_oob_recovery_config

                    lifecycle = OobLifecycle(
                        hub=hub,
                        resolved_port=resolved_port,
                        usb_local=usb_local,
                        host_client=host_client,
                        turn_port=usb_turn_port() if usb_local else None,
                        config=recovery_config or resolve_oob_recovery_config(),
                        on_status=on_oob_status,
                    )
                    lifecycle_task = asyncio.create_task(
                        lifecycle.run(), name="cloudxr-oob-lifecycle"
                    )
                    done, _ = await asyncio.wait(
                        (stop_future, lifecycle_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if lifecycle_task in done and stop_future not in done:
                        try:
                            await lifecycle_task
                        except asyncio.CancelledError as exc:
                            error = RuntimeError(
                                "OOB lifecycle worker was cancelled unexpectedly"
                            )
                            failure_log.exception("OOB lifecycle worker failed")
                            failure_logged = True
                            if on_oob_fatal is not None:
                                try:
                                    reported = on_oob_fatal(error)
                                    if isawaitable(reported):
                                        await reported
                                except Exception:
                                    log.exception(
                                        "OOB fatal callback failed while reporting lifecycle cancellation"
                                    )
                            raise error from exc
                        except Exception as exc:
                            # Keep the traceback in the per-session file, not the console.
                            failure_log.exception("OOB lifecycle worker failed")
                            failure_logged = True
                            if on_oob_fatal is not None:
                                try:
                                    reported = on_oob_fatal(exc)
                                    if isawaitable(reported):
                                        await reported
                                except Exception:
                                    log.exception(
                                        "OOB fatal callback failed while reporting lifecycle failure"
                                    )
                            raise
                        else:
                            exc = RuntimeError(
                                "OOB lifecycle worker exited unexpectedly"
                            )
                            failure_log.error("%s", exc)
                            failure_logged = True
                            if on_oob_fatal is not None:
                                try:
                                    reported = on_oob_fatal(exc)
                                    if isawaitable(reported):
                                        await reported
                                except Exception:
                                    log.exception(
                                        "OOB fatal callback failed while reporting lifecycle exit"
                                    )
                            raise exc
                else:
                    await stop_future
            finally:
                if lifecycle_task is not None:
                    lifecycle_task.cancel()
                    try:
                        await lifecycle_task
                    except (asyncio.CancelledError, Exception):
                        pass

            log.info("Shutting down ...")
    except OSError as e:
        if was_listening and not failure_logged:
            failure_log.exception("WSS proxy failed after listening")
        if e.errno == errno.EADDRINUSE:
            raise RuntimeError(
                f"WSS proxy port {resolved_port} is already in use. "
                f"Set PROXY_PORT to a different port or stop the process using {resolved_port}."
            ) from e
        raise
    except Exception:
        if was_listening and not failure_logged:
            failure_log.exception("WSS proxy failed after listening")
        raise
    finally:
        for _attached_log in _handler_loggers:
            _attached_log.removeHandler(_handler)
        if _handler is not None:
            _handler.close()

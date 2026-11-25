# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate separate credentials and configs for a TLS-protected IPv4 LAN/VPN test."""

import copy
import ipaddress
import os
from pathlib import Path
import secrets
import shlex
import subprocess
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory

import yaml


def write_private(path, text):
    with open(path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
        handle.write(text)


def _generate_bundle(
    output, host_ip, session_template, camera_template, *, replace=True, **kwargs
):
    output = Path(output).absolute()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Output must be a directory, not a file or symlink")
    if output.exists() and not replace:
        raise FileExistsError(f"{output} already exists and replacement was disabled")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build alongside the destination so installation uses same-filesystem renames.
    with TemporaryDirectory(
        prefix=f".{output.name}.staging-", dir=output.parent
    ) as staging:
        fresh = _generate(
            Path(staging) / "bundle",
            host_ip,
            session_template,
            camera_template,
            **kwargs,
        )
        backup = None
        if output.exists():
            backup = output.with_name(
                f"{output.name}.backup.{datetime.now():%Y%m%d-%H%M%S-%f}-{secrets.token_hex(4)}"
            )
            output.rename(backup)
        try:
            fresh.rename(output)
        except OSError:
            if backup is not None:
                backup.rename(output)
            raise
    if backup is not None:
        print(f"Previous bundle preserved at {backup}")
    return output


def _generate(
    output,
    host_ip,
    session_template,
    camera_template,
    *,
    hours=24,
    room=None,
    signal_port=8443,
    http_port=7880,
    tcp_port=7881,
    udp_port=7882,
    tls_not_before=None,
    write_camera_profiles=None,
):
    from livekit import api

    address = ipaddress.IPv4Address(host_ip)
    if (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
    ):
        raise ValueError("Use the Distributed Cloud LAN/VPN IPv4 address, not loopback")
    if not 1 <= hours <= 168:
        raise ValueError("Token lifetime must be between 1 and 168 hours")
    if any(
        not 1024 <= port <= 65535
        for port in (signal_port, http_port, tcp_port, udp_port)
    ):
        raise ValueError("Ports must be between 1024 and 65535")
    if len({signal_port, http_port, tcp_port}) != 3:
        raise ValueError("TLS, HTTP and RTC TCP ports must be distinct")
    session = yaml.safe_load(Path(session_template).read_text())
    if not isinstance(session, dict) or not all(
        isinstance(session.get(site), dict) for site in ("host", "client")
    ):
        raise ValueError(
            "Session template needs host/client sections; update it before generating bundles"
        )
    camera = yaml.safe_load(Path(camera_template).read_text())
    cameras = [c for c in camera["cameras"] if c.get("enabled", True)]
    if len(cameras) != 1:
        raise ValueError("Network setup requires exactly one enabled camera")
    publisher = camera["livekit"].get("publisher", "client-camera")
    viewer = (
        camera["livekit"].get("viewer_identity", "host-video")
        + "-"
        + cameras[0]["name"]
    )
    probe = camera["livekit"].get("probe_identity", "host-probe")
    identities = [session[r]["identity"] for r in ("host", "client")] + [
        publisher,
        viewer,
        probe,
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("All component identities must be distinct")
    key, secret = "ra-" + secrets.token_hex(8), secrets.token_urlsafe(48)
    room = room or "remote-assistance-" + secrets.token_hex(6)
    url = f"wss://{address}:{signal_port}"

    def token(identity, publish=False, subscribe=False, data=False):
        return (
            api.AccessToken(key, secret)
            .with_identity(identity)
            .with_ttl(timedelta(hours=hours))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room,
                    can_publish=publish,
                    can_subscribe=subscribe,
                    can_publish_data=data,
                )
            )
            .to_jwt()
        )

    def settings(original, env):
        return {
            **original,
            "url": url,
            "room": room,
            "development": False,
            "token_env": env,
        }

    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Both bundles contain credentials. A local ignore also protects custom paths.
    write_private(output / ".gitignore", "*\n")
    for role in ("host", "client"):
        site = output / role
        site.mkdir(mode=0o700)
        cfg = copy.deepcopy(session)
        cfg["livekit"] = settings(cfg["livekit"], "RA_SESSION_TOKEN")
        for peer in ("host", "client"):
            cfg[peer]["livekit"] = settings(
                cfg[peer].get("livekit", {}), "RA_SESSION_TOKEN"
            )
        write_private(site / "session.yaml", yaml.safe_dump(cfg, sort_keys=False))
        video = copy.deepcopy(camera)
        video["livekit"] = settings(video["livekit"], "RA_CAMERA_TOKEN")
        write_private(site / "camera.yaml", yaml.safe_dump(video, sort_keys=False))
        if write_camera_profiles is not None:
            write_camera_profiles(site, video, site_role=role)
        credentials = {
            "RA_SESSION_TOKEN": token(session[role]["identity"], data=True),
            "RA_CAMERA_TOKEN": token(publisher, publish=True)
            if role == "client"
            else token(viewer, subscribe=True),
        }
        if role == "host":
            probe_cfg = copy.deepcopy(video)
            probe_cfg["livekit"]["token_env"] = "RA_PROBE_TOKEN"
            write_private(
                site / "probe.yaml", yaml.safe_dump(probe_cfg, sort_keys=False)
            )
            credentials["RA_PROBE_TOKEN"] = token(probe, subscribe=True)
        write_private(
            site / "credentials.env",
            "".join(
                f"export {name}={shlex.quote(value)}\n"
                for name, value in credentials.items()
            )
            + 'export SSL_CERT_FILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/ca.crt"\n',
        )
    server = {
        "development": False,
        "port": http_port,
        "bind_addresses": ["127.0.0.1"],
        "keys": {key: secret},
        "rtc": {
            "node_ip": str(address),
            "use_external_ip": False,
            "tcp_port": tcp_port,
            "udp_port": udp_port,
            "ips": {"includes": [f"{address}/32"]},
            "data_channel_max_buffered_amount": 1048576,
        },
        "logging": {"level": "info"},
    }
    write_private(
        output / "host" / "livekit.yaml", yaml.safe_dump(server, sort_keys=False)
    )
    compose = {
        "name": "isaac-remote-assistance-network",
        "services": {
            "livekit": {
                "image": "livekit/livekit-server:v1.13.7",
                "network_mode": "host",
                "command": ["--config", "/etc/livekit.yaml"],
                "volumes": ["./livekit.yaml:/etc/livekit.yaml:ro"],
            }
        },
    }
    write_private(
        output / "host" / "compose.yaml", yaml.safe_dump(compose, sort_keys=False)
    )
    tls = output / "host" / "tls"
    tls.mkdir(mode=0o700)
    if tls_not_before is None:
        make_certificates(tls, str(address))
    else:
        from .network_certificates import make_backdated_certificates

        make_backdated_certificates(tls, str(address), tls_not_before)
    for role in ("host", "client"):
        write_private(output / role / "ca.crt", (tls / "ca.crt").read_text())
    nginx = f"""events {{}}
http {{
    access_log off;
    map $http_upgrade $connection_upgrade {{ default upgrade; '' close; }}
    server {{
        listen {address}:{signal_port} ssl;
        ssl_certificate /etc/nginx/tls/server.crt;
        ssl_certificate_key /etc/nginx/tls/server.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        location / {{
            proxy_pass http://127.0.0.1:{http_port};
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_set_header Host $host;
            proxy_read_timeout 3600s;
            proxy_buffering off;
        }}
    }}
}}
"""
    write_private(output / "host/nginx.conf", nginx)
    compose["services"]["signaling"] = {
        "image": "nginx:1.28.0-alpine",
        "network_mode": "host",
        "depends_on": ["livekit"],
        "volumes": ["./nginx.conf:/etc/nginx/nginx.conf:ro", "./tls:/etc/nginx/tls:ro"],
    }
    # Replace only this newly generated file, before handing the bundle to callers.
    (output / "host/compose.yaml").write_text(yaml.safe_dump(compose, sort_keys=False))
    return output


def make_certificates(tls, address):
    def openssl(*args):
        subprocess.run(["openssl", *args], cwd=tls, check=True, capture_output=True)

    openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "7",
        "-keyout",
        "ca.key",
        "-out",
        "ca.crt",
        "-subj",
        "/CN=Remote Assistance Test CA",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
    )
    openssl(
        "req",
        "-new",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "server.key",
        "-out",
        "server.csr",
        "-subj",
        f"/CN={address}",
    )
    write_private(
        tls / "server.ext",
        f"subjectAltName=IP:{address}\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n",
    )
    openssl(
        "x509",
        "-req",
        "-in",
        "server.csr",
        "-CA",
        "ca.crt",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "server.crt",
        "-days",
        "7",
        "-extfile",
        "server.ext",
    )
    for path in tls.iterdir():
        path.chmod(0o600)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate separate credentials and configs for a TLS-protected IPv4 LAN/VPN test."""

import argparse
import copy
import ipaddress
import os
from pathlib import Path
import secrets
import shlex
import subprocess
from datetime import timedelta

from livekit import api
import yaml


def write_private(path, text):
    with open(path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
        handle.write(text)


def generate(
    output,
    edge_ip,
    session_template,
    camera_template,
    *,
    hours=24,
    room=None,
    signal_port=8443,
    http_port=7880,
    tcp_port=7881,
    udp_port=7882,
):
    address = ipaddress.IPv4Address(edge_ip)
    if (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
    ):
        raise ValueError("Use the Edge Compute LAN/VPN IPv4 address, not loopback")
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
    camera = yaml.safe_load(Path(camera_template).read_text())
    cameras = [c for c in camera["cameras"] if c.get("enabled", True)]
    if len(cameras) != 1:
        raise ValueError("Network setup requires exactly one enabled camera")
    publisher = camera["livekit"].get("publisher", "robot-camera")
    viewer = (
        camera["livekit"].get("viewer_identity", "edge-video")
        + "-"
        + cameras[0]["name"]
    )
    probe = camera["livekit"].get("probe_identity", "edge-probe")
    identities = [session[r]["identity"] for r in ("edge", "robot")] + [
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
    for role in ("edge", "robot"):
        site = output / role
        site.mkdir(mode=0o700)
        cfg = copy.deepcopy(session)
        cfg["livekit"] = settings(cfg["livekit"], "RA_SESSION_TOKEN")
        for peer in ("edge", "robot"):
            cfg[peer]["livekit"] = settings(
                cfg[peer].get("livekit", {}), "RA_SESSION_TOKEN"
            )
        write_private(site / "session.yaml", yaml.safe_dump(cfg, sort_keys=False))
        video = copy.deepcopy(camera)
        video["livekit"] = settings(video["livekit"], "RA_CAMERA_TOKEN")
        write_private(site / "camera.yaml", yaml.safe_dump(video, sort_keys=False))
        credentials = {
            "RA_SESSION_TOKEN": token(session[role]["identity"], data=True),
            "RA_CAMERA_TOKEN": token(publisher, publish=True)
            if role == "robot"
            else token(viewer, subscribe=True),
        }
        if role == "edge":
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
        output / "edge" / "livekit.yaml", yaml.safe_dump(server, sort_keys=False)
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
        output / "edge" / "compose.yaml", yaml.safe_dump(compose, sort_keys=False)
    )
    tls = output / "edge" / "tls"
    tls.mkdir(mode=0o700)
    make_certificates(tls, str(address))
    for role in ("edge", "robot"):
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
    write_private(output / "edge/nginx.conf", nginx)
    compose["services"]["signaling"] = {
        "image": "nginx:1.28.0-alpine",
        "network_mode": "host",
        "depends_on": ["livekit"],
        "volumes": ["./nginx.conf:/etc/nginx/nginx.conf:ro", "./tls:/etc/nginx/tls:ro"],
    }
    # Replace only this newly generated file, before handing the bundle to callers.
    (output / "edge/compose.yaml").write_text(yaml.safe_dump(compose, sort_keys=False))
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


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edge-ip",
        required=True,
        help="Edge LAN/VPN IPv4 reachable from both computers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New private directory; never overwritten",
    )
    parser.add_argument("--session-config", type=Path, default=root / "session.yaml")
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=root.parent / "camera_viz/configs/livekit-webcam.yaml",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Token lifetime, 1–168 hours (default: 24)",
    )
    parser.add_argument("--room", help="Room name (default: generated unique room)")
    parser.add_argument("--signal-port", type=int, default=8443)
    parser.add_argument("--http-port", type=int, default=7880)
    parser.add_argument("--tcp-port", type=int, default=7881)
    parser.add_argument("--udp-port", type=int, default=7882)
    args = parser.parse_args()
    try:
        output = generate(
            args.output,
            args.edge_ip,
            args.session_config,
            args.camera_config,
            hours=args.hours,
            room=args.room,
            signal_port=args.signal_port,
            http_port=args.http_port,
            tcp_port=args.tcp_port,
            udp_port=args.udp_port,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(
        f"Created {output}/edge and {output}/robot; tokens expire in {args.hours} hours."
    )
    print(
        "Keep the edge bundle on Edge Compute; transfer only the robot bundle to Robot Site."
    )
    print(
        "Signaling uses verified TLS; source each site credentials.env in its process terminals. Certificates expire in 7 days."
    )
    print("Direct LAN/VPN reachability required. No services were started or changed.")


if __name__ == "__main__":
    main()

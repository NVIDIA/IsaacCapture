# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate host/client bundles with the selected installed transport plugin."""

import argparse
from pathlib import Path
import os
import re
import uuid
import yaml
from isaac_media_transport.transport import get_transport
from isaac_media_transport.transport_plugins import available_transports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport", choices=available_transports(), default="livekit"
    )
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--session-config", type=Path, required=True)
    parser.add_argument(
        "--latency-config",
        type=Path,
        default=Path("examples/remote_assistance/latency.yaml"),
        help="Shared instrumentation policy; copied into both site bundles",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--signal-port", type=int, default=8443)
    parser.add_argument("--http-port", type=int, default=7880)
    parser.add_argument("--tcp-port", type=int, default=7881)
    parser.add_argument("--udp-port", type=int, default=7882)
    args = parser.parse_args()
    try:
        latency = yaml.safe_load(args.latency_config.read_text())
        if not isinstance(latency, dict):
            raise ValueError("Latency config must be a YAML mapping")
        latency["run_id"] = latency.get("run_id") or uuid.uuid4().hex
        from isaac_media_transport.latency import trace_log_path

        trace_log_path(latency["run_id"], "viewer")
        if latency.get("enabled", False):
            clock = latency.setdefault("clock", {})
            clock["client"] = os.path.expandvars(clock.get("client", ""))
            if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+", clock["client"]):
                raise ValueError(
                    "Set CLIENT_USER and CLIENT_IP, or clock.client in the latency YAML"
                )
        output = (
            get_transport({"transport": args.transport})
            .provisioner()
            .generate(
                args.output,
                args.host_ip,
                args.session_config,
                args.camera_config,
                hours=args.hours,
                signal_port=args.signal_port,
                http_port=args.http_port,
                tcp_port=args.tcp_port,
                udp_port=args.udp_port,
            )
        )
        for role in ("host", "client"):
            site = Path(output) / role
            target = site / "latency.yaml"
            target.write_text(yaml.safe_dump(latency, sort_keys=False))
            target.chmod(0o600)
            for name in ("camera.yaml", "session.yaml"):
                target = site / name
                cfg = yaml.safe_load(target.read_text())
                cfg.get("livekit", {}).pop("latency", None)
                cfg["latency_config"] = "latency.yaml"
                target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Created {output}/host and {output}/client using {args.transport}.")
    print(
        "Transfer client/ to Client Site, restart the network server, and source each site's credentials.env."
    )
    print(
        "This generation did not install packages or start services. Keep generated bundles private."
    )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate host/client bundles with the selected installed transport plugin."""

import argparse
from pathlib import Path
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
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--signal-port", type=int, default=8443)
    parser.add_argument("--http-port", type=int, default=7880)
    parser.add_argument("--tcp-port", type=int, default=7881)
    parser.add_argument("--udp-port", type=int, default=7882)
    args = parser.parse_args()
    try:
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

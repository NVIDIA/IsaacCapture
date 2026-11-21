#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in ZED sender to receiver check for a machine without a CUDA Python stack.

Runs the real zed_sender and the real receiver in discovery mode over loopback,
with NumPy stand-ins for CuPy and the decoder. It verifies what the receiver
learns from the wire and tells consumers: the stream layout, frame delivery per
stream, the latency CSV, and a layout change when the sender restarts with a
second config. Decoded pixels are not checked.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import types

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "camera_receiver"))
sys.path.insert(0, str(ROOT / "camera_receiver" / "tests"))
# The sibling receiver is a standalone source tree, so its path must be set before importing.
import camera_receiver  # noqa: E402
from fake_gpu import FakeDecoder, fake_cupy  # noqa: E402


def start_sender(args, config, log):
    return subprocess.Popen(
        [
            str(args.sender.resolve()),
            "--config",
            str(config.resolve()),
            "--port",
            str(args.port),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def stop_sender(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class Consumer:
    def __init__(self, path):
        deadline = time.monotonic() + 10
        while True:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(path)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        self.sock.settimeout(0.5)
        self.buffer = b""
        self.layouts = []
        self.sock.sendall(b'{"type":"subscribe","streams":[]}\n')

    def collect(self, seconds):
        """Frame counts per stream over the window; layout messages are recorded."""
        frames, first, last = Counter(), {}, {}
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise RuntimeError("receiver closed the consumer connection")
            self.buffer += chunk
            while b"\n" in self.buffer:
                line, _, self.buffer = self.buffer.partition(b"\n")
                message = json.loads(line)
                if message["type"] in ("subscribed", "streams"):
                    layout = [
                        f"{s['name']} {s['width']}x{s['height']}"
                        for s in message["streams"]
                    ]
                    if not self.layouts or self.layouts[-1] != layout:
                        self.layouts.append(layout)
                elif message["type"] == "frame":
                    now = time.monotonic()
                    name = message["stream"]
                    frames[name] += 1
                    first.setdefault(name, now)
                    last[name] = now
        return {
            name: dict(
                frames=count,
                fps=(count - 1) / (last[name] - first[name]) if count > 1 else 0.0,
            )
            for name, count in frames.items()
        }


def csv_summary(path):
    rows = list(csv.DictReader(path.open()))
    summary = {}
    for name in sorted({row["stream"] for row in rows}):
        deltas = sorted(
            float(row["receive_minus_send_ms"]) for row in rows if row["stream"] == name
        )
        summary[name] = dict(
            rows=len(deltas),
            receive_minus_send_ms=dict(
                mean=sum(deltas) / len(deltas),
                p95=deltas[int((len(deltas) - 1) * 0.95)],
                max=deltas[-1],
            ),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sender", type=Path, default=ROOT / "zed_sender/build/zed_sender"
    )
    parser.add_argument("--config", type=Path, required=True, help="sender config")
    parser.add_argument(
        "--second-config",
        type=Path,
        help="restart the sender with this config afterwards",
    )
    parser.add_argument("--port", type=int, default=55040)
    parser.add_argument("--seconds", type=float, default=8.0, help="per sender run")
    parser.add_argument(
        "--require", default="", help="streams that must deliver in the first run"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--socket",
        help="IPC socket path; Unix socket paths are limited to about 100 characters",
    )
    args = parser.parse_args()
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="zed-receiver-"))
    directory.mkdir(parents=True, exist_ok=True)

    sys.modules["cupy"] = fake_cupy()
    camera_receiver.NvVideoDecoder = FakeDecoder
    config = yaml.safe_load(
        (ROOT / "camera_receiver/configs/receiver_auto.yaml").read_text()
    )
    receiver = camera_receiver._build_receiver(
        config,
        types.SimpleNamespace(
            host="127.0.0.1",
            port=args.port,
            gpu_id=None,
            socket=args.socket
            or str(Path(tempfile.mkdtemp(prefix="zr-")) / "ipc.sock"),
            stats_log_interval_s=None,
            latency_csv=str(directory / "latency.csv"),
        ),
    )
    summary = dict(ok=False, output_dir=str(directory), runs=[])
    receiver.start()
    try:
        consumer = Consumer(receiver.socket_path)
        for index, sender_config in enumerate(
            filter(None, [args.config, args.second_config])
        ):
            with (directory / f"sender{index}.log").open("w") as log:
                process = start_sender(args, sender_config, log)
                try:
                    delivered = consumer.collect(args.seconds)
                finally:
                    stop_sender(process)
            summary["runs"].append(
                dict(
                    config=str(sender_config),
                    layout=consumer.layouts[-1],
                    delivered=delivered,
                )
            )
        summary["layouts_seen"] = consumer.layouts
        required = [name for name in args.require.split(",") if name]
        missing = [
            name for name in required if name not in summary["runs"][0]["delivered"]
        ]
        assert not missing, f"required streams did not deliver: {missing}"
        summary["ok"] = True
    except Exception as error:
        summary["error"] = repr(error)
    finally:
        receiver.stop()
    summary["latency_csv"] = csv_summary(directory / "latency.csv")
    rendered = json.dumps(summary, indent=2)
    (directory / "summary.json").write_text(rendered + "\n")
    print(rendered)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge DAYANGlove UDP frames into the local SchemaPusher process."""

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from receiver import DEFAULT_PROTOCOL, UdpReceiver, load_frame_type, positive_float

MAX_LINE_BYTES = 16384


def encode_line(packet):
    line = (json.dumps(packet, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(line) > MAX_LINE_BYTES:
        raise ValueError("DAYANGlove IPC frame exceeds size limit")
    return line


def repeated_field(field):
    # Protobuf 5.29 lacks is_repeated; Protobuf 7 removes label.
    repeated = getattr(field, "is_repeated", None)
    return field.label == field.LABEL_REPEATED if repeated is None else repeated


def encode_sample(sample):
    frame = sample.frame
    packet = {
        field.name: list(getattr(frame, field.name))
        if repeated_field(field)
        else getattr(frame, field.name)
        for field in frame.DESCRIPTOR.fields
    }
    packet.update(available=True, received_monotonic_ns=sample.received_monotonic_ns)
    return encode_line(packet)


class PublisherPipe:
    """Bound writes so a stalled publisher cannot freeze UDP reception indefinitely."""

    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)
        os.set_blocking(self.process.stdin.fileno(), False)

    def check(self):
        if self.process.poll() is not None:
            raise RuntimeError(
                f"DAYANGlove publisher exited: {self.process.returncode}"
            )

    def send(self, line):
        self.check()
        pending = memoryview(line)
        deadline = time.monotonic() + 0.5
        while pending:
            self.check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DAYANGlove publisher input stalled")
            _, writable, _ = select.select([], [self.process.stdin], [], remaining)
            if writable:
                try:
                    count = os.write(self.process.stdin.fileno(), pending)
                    pending = pending[count:]
                except BlockingIOError:
                    continue

    def close(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            raise RuntimeError("DAYANGlove publisher failed to shut down") from None
        if code:
            raise RuntimeError(f"DAYANGlove publisher exited: {code}")


class PublisherBridge:
    def __init__(self, sink):
        self.sink = sink
        self.active_hands = set()
        self.session = None

    def remove(self, side):
        self.sink.send(
            encode_line(
                dict(
                    hand=side,
                    available=False,
                    received_monotonic_ns=time.monotonic_ns(),
                )
            )
        )
        self.active_hands.discard(side)

    def update(self, state, sample):
        state.expire()
        removed = self.active_hands - set(state.latest)
        if self.session != state.active:
            removed = set(self.active_hands)
        for side in sorted(removed):
            self.remove(side)
        self.session = state.active
        if sample is not None and state.latest.get(sample.frame.hand) is sample:
            self.sink.send(encode_sample(sample))
            self.active_hands.add(sample.frame.hand)

    def close(self):
        for side in sorted(self.active_hands):
            self.remove(side)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publisher",
        type=Path,
        default=Path(__file__).resolve().with_name("dayanglove_plugin"),
    )
    parser.add_argument("--protocol-python", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--collection-prefix", default="dayanglove")
    parser.add_argument("--stale-after", type=positive_float, default=1.0)
    parser.add_argument("--duration", type=positive_float)
    args = parser.parse_args(argv)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        with UdpReceiver(
            load_frame_type(args.protocol_python),
            args.host,
            args.port,
            args.stale_after,
        ) as receiver:
            pipe = PublisherPipe(
                [
                    str(args.publisher.resolve()),
                    f"--collection-prefix={args.collection_prefix}",
                    f"--stale-after={args.stale_after}",
                ]
            )
            bridge = PublisherBridge(pipe)
            try:
                started = time.monotonic()
                print(f"DAYANGlove UDP ready: {receiver.address}", flush=True)
                while not stopped and (
                    args.duration is None or time.monotonic() - started < args.duration
                ):
                    pipe.check()
                    bridge.update(receiver.state, receiver.poll())
                bridge.close()
            finally:
                pipe.close()
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"DAYANGlove source: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receive DAYANGlove v1 datagrams without importing its application or models."""

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import importlib.util
import ipaddress
import json
import math
from pathlib import Path
import socket
import sys
import time

from google.protobuf.message import DecodeError, Message

MAX_PACKET_BYTES = 1400
DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "protocol/glove_frame_pb2.py"


def load_frame_type(protocol_python=DEFAULT_PROTOCOL):
    """Load the trusted, upstream-generated glove_frame_pb2.py supplied by the operator."""
    path = Path(protocol_python).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("_dayanglove_wire", path)
    if spec is None or spec.loader is None:
        raise ValueError("Expected a generated Python protocol file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame_type = getattr(module, "GloveFrame", None)
    if (
        frame_type is None
        or frame_type.DESCRIPTOR.full_name != "dayanglove.v1.GloveFrame"
    ):
        raise ValueError("Expected dayanglove.v1.GloveFrame")
    return frame_type


def decode_frame(payload, frame_type):
    if not 0 < len(payload) <= MAX_PACKET_BYTES:
        raise ValueError("Datagram must contain 1..1400 bytes")
    try:
        frame = frame_type.FromString(payload)
    except DecodeError as exc:
        raise ValueError("Malformed Protobuf datagram") from exc
    if frame.schema_version != 1:
        raise ValueError("Unsupported schema_version")
    if frame.hand not in (1, 2) or frame.mirror_x != (frame.hand == 1):
        raise ValueError("Invalid hand or mirror_x")
    if not frame.glove_id or not frame.session_id:
        raise ValueError("Missing glove_id or session_id")
    if frame.output_sequence == 0 or frame.frame_number == 0:
        raise ValueError("Missing output_sequence or frame_number")
    if frame.status > 255 or frame.device_sequence > 255:
        raise ValueError("Status and device sequence must fit uint8")
    if len(frame.palm_adc) != 210 or any(value > 4095 for value in frame.palm_adc):
        raise ValueError("Expected 210 raw 12-bit palm ADC values")
    if frame.pose_valid:
        if len(frame.mano_pose45) != 45 or len(frame.joints21_xyz_m) != 63:
            raise ValueError("Expected 45 pose components and 63 joint coordinates")
    elif frame.mano_pose45 or frame.joints21_xyz_m:
        raise ValueError("Invalid pose must have empty pose and joint arrays")
    if not all(math.isfinite(value) for value in frame.mano_pose45):
        raise ValueError("Non-finite pose")
    if not all(math.isfinite(value) for value in frame.joints21_xyz_m):
        raise ValueError("Non-finite joints")
    if not math.isfinite(frame.source_timestamp):
        raise ValueError("Non-finite source_timestamp")
    return frame


@dataclass(frozen=True)
class ReceivedFrame:
    frame: Message
    peer: tuple
    received_monotonic_ns: int


class StreamState:
    """Single-sender stream, with independent hand freshness and ordering."""

    def __init__(self, frame_type, stale_after=1.0):
        if not math.isfinite(stale_after) or stale_after <= 0:
            raise ValueError("stale_after must be finite and positive")
        self.frame_type = frame_type
        self.stale_ns = round(stale_after * 1e9)
        self.latest = {}
        self.counts = Counter()
        self.hand_counts = Counter()
        self.last_error = ""
        self.active = None
        self.retired = deque(maxlen=64)
        self.last_receive_ns = 0
        self.hand_sequences = {}
        self.high_sequence = 0

    def expire(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        expired = [
            side
            for side, sample in self.latest.items()
            if now_ns - sample.received_monotonic_ns >= self.stale_ns
        ]
        for side in expired:
            del self.latest[side]
            self.counts["expired_hands"] += 1
        return expired

    def feed(self, payload, peer, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        self.expire(now_ns)
        self.counts["datagrams"] += 1
        try:
            frame = decode_frame(payload, self.frame_type)
        except ValueError as exc:
            self.counts["invalid"] += 1
            self.last_error = str(exc)
            return None

        identity = (tuple(peer), frame.session_id)
        if identity != self.active:
            if identity in self.retired:
                self.counts["retired_session"] += 1
                return None
            if self.active is not None:
                # A second sender cannot replace a live stream. Restarts wait
                # for local silence; sender clocks are not comparable across hosts.
                if now_ns - self.last_receive_ns < self.stale_ns:
                    self.counts["other_stream"] += 1
                    return None
                self.retired.append(self.active)
                self.counts["session_changes"] += 1
            self.active = identity
            self.latest.clear()
            self.hand_sequences.clear()
            self.high_sequence = 0

        sequence = frame.output_sequence
        if sequence <= self.hand_sequences.get(frame.hand, 0):
            self.counts["out_of_order_or_duplicate"] += 1
            return None
        # This is the global sender sequence, not a per-hand counter. Jumps
        # include reordering and sender drops; they are not proven network loss.
        if self.high_sequence and sequence > self.high_sequence + 1:
            self.counts["sequence_jumps"] += sequence - self.high_sequence - 1
        self.high_sequence = max(self.high_sequence, sequence)
        self.hand_sequences[frame.hand] = sequence
        self.last_receive_ns = now_ns
        sample = ReceivedFrame(frame, tuple(peer), now_ns)
        self.latest[frame.hand] = sample
        self.counts["accepted"] += 1
        self.hand_counts[frame.hand] += 1
        return sample

    def summary(self):
        self.expire()
        return {
            "counts": dict(self.counts),
            "accepted_by_hand": dict(self.hand_counts),
            "last_error": self.last_error,
            "session_id": self.active[1] if self.active else None,
            "hands": {
                str(side): {
                    "glove_id": sample.frame.glove_id,
                    "frame_number": sample.frame.frame_number,
                    "output_sequence": sample.frame.output_sequence,
                    "simulation": sample.frame.simulation,
                    "pose_valid": sample.frame.pose_valid,
                    "pose_source": sample.frame.pose_source,
                    "pose_components": len(sample.frame.mano_pose45),
                    "joint_coordinates": len(sample.frame.joints21_xyz_m),
                    "palm_channels": len(sample.frame.palm_adc),
                }
                for side, sample in self.latest.items()
            },
        }


class UdpReceiver:
    def __init__(self, frame_type, host="127.0.0.1", port=50051, stale_after=1.0):
        address = ipaddress.ip_address(host)
        if not 0 <= port <= 65535:
            raise ValueError("port must be 0..65535")
        self.state = StreamState(frame_type, stale_after)
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        self.socket = socket.socket(family, socket.SOCK_DGRAM)
        try:
            self.socket.bind((str(address), port))
        except OSError:
            self.socket.close()
            raise

    @property
    def address(self):
        return self.socket.getsockname()

    def poll(self, timeout=0.1):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.socket.settimeout(timeout)
        try:
            payload, peer = self.socket.recvfrom(MAX_PACKET_BYTES + 1)
        except socket.timeout:
            self.state.expire()
            return None
        return self.state.feed(payload, peer)

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-python", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--stale-after", type=positive_float, default=1.0)
    parser.add_argument("--duration", type=positive_float, help="Stop after N seconds")
    parser.add_argument("--summary-file", type=Path)
    args = parser.parse_args(argv)
    try:
        frame_type = load_frame_type(args.protocol_python)
        with UdpReceiver(
            frame_type, args.host, args.port, args.stale_after
        ) as receiver:
            print(f"Listening on {receiver.address}; Ctrl+C to stop", flush=True)
            started = time.monotonic()
            next_report = started + 1
            try:
                while (
                    args.duration is None or time.monotonic() - started < args.duration
                ):
                    receiver.poll()
                    if time.monotonic() >= next_report:
                        print(json.dumps(receiver.state.summary()), flush=True)
                        next_report = time.monotonic() + 1
            except KeyboardInterrupt:
                pass
            summary = receiver.state.summary()
        print(json.dumps(summary), flush=True)
        if args.summary_file:
            args.summary_file.parent.mkdir(parents=True, exist_ok=True)
            args.summary_file.write_text(json.dumps(summary, indent=2) + "\n")
        return 0 if summary["counts"].get("accepted", 0) else 2
    except (OSError, ValueError, ImportError) as exc:
        print(f"DAYANGlove receiver: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

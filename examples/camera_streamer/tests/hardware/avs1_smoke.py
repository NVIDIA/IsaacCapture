#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded AVS1 stereo hardware smoke test."""

from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


HEADER = struct.Struct("!4sHHIHHQQQQIIQ")
STREAM = struct.Struct("!HHBBHIIIIQHHI")
SAMPLE = struct.Struct("!HBBHHIIQQQQQQII")


def recv_exact(sock: socket.socket, count: int) -> bytes:
    data = bytearray(count)
    view = memoryview(data)
    received = 0
    while received < count:
        size = sock.recv_into(view[received:])
        if size == 0:
            raise EOFError("sender closed before the requested AVS1 bundles arrived")
        received += size
    return bytes(data)


def nal_types(data: bytes) -> list[int]:
    result: list[int] = []
    index = 0
    while index + 4 < len(data):
        if data[index : index + 3] == b"\x00\x00\x01":
            header = index + 3
            index = header + 2
        elif data[index : index + 4] == b"\x00\x00\x00\x01":
            header = index + 4
            index = header + 2
        else:
            index += 1
            continue
        if header < len(data):
            result.append((data[header] >> 1) & 0x3F)
    return result


def parse_first_stream(packet: bytes):
    if len(packet) < HEADER.size:
        raise ValueError("short AVS1 packet")
    fields = HEADER.unpack_from(packet)
    if fields[0] != b"AVS1" or fields[1] != 1 or fields[3] != HEADER.size:
        raise ValueError("bad AVS1 header")
    stream_count, sample_count = fields[4], fields[5]
    descriptor_bytes, payload_bytes = fields[10], fields[11]
    if stream_count < 1 or sample_count < 2:
        raise ValueError("smoke packet has no stereo stream")
    stream = STREAM.unpack_from(packet, HEADER.size)
    if stream[0] != 0 or not stream[1] & 1 or not stream[1] & 2 or stream[2] != 1:
        raise ValueError("stream zero is not present stereo HEVC")
    first_sample, count = stream[10], stream[11]
    if first_sample != 0 or count != 2:
        raise ValueError("stream zero does not contain two ordered samples")
    sample_base = HEADER.size + stream_count * STREAM.size
    samples = [
        SAMPLE.unpack_from(packet, sample_base + index * SAMPLE.size)
        for index in range(2)
    ]
    payload_start = HEADER.size + descriptor_bytes
    payloads = []
    tsc = []
    for camera, sample in enumerate(samples):
        if sample[0] != 0 or sample[1] != camera or not sample[3] & 1:
            raise ValueError("invalid stereo sample descriptor")
        offset, size = sample[13], sample[14]
        if offset + size > payload_bytes or size == 0:
            raise ValueError("invalid stereo sample payload range")
        payloads.append(packet[payload_start + offset : payload_start + offset + size])
        tsc.append(sample[8] if sample[3] & 2 else None)
    return fields[6], payloads, tsc


def connect_with_retry(
    port: int, process: subprocess.Popen, timeout: float
) -> socket.socket:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "argus_sender exited before accepting the smoke receiver"
            )
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            sock.settimeout(10.0)
            return sock
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("timed out connecting to argus_sender")


def configured_port(config_path: str) -> int:
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    port = config.get("network", {}).get("port", 5000)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("network.port must be an integer in 1..65535")
    return port


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    port = configured_port(args.config)

    process = subprocess.Popen(
        [args.sender, "--config", args.config],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    try:
        sock = connect_with_retry(port, process, 15.0)
        first_nals: list[list[int]] | None = None
        last_tsc: list[int | None] = [None, None]
        max_delta_us = 0.0
        with sock:
            for expected_sequence in range(1, args.frames + 1):
                (length,) = struct.unpack("!I", recv_exact(sock, 4))
                if length <= 0 or length > 256 * 1024 * 1024:
                    raise ValueError(f"invalid AVS1 record length {length}")
                sequence, payloads, tsc = parse_first_stream(recv_exact(sock, length))
                if sequence != expected_sequence:
                    raise ValueError(
                        f"AVS1 sequence {sequence}, expected {expected_sequence}"
                    )
                if first_nals is None:
                    first_nals = [nal_types(payload) for payload in payloads]
                if all(value is not None for value in tsc):
                    max_delta_us = max(
                        max_delta_us, abs(int(tsc[1]) - int(tsc[0])) / 1000.0
                    )
                for eye in range(2):
                    if (
                        tsc[eye] is not None
                        and last_tsc[eye] is not None
                        and int(tsc[eye]) <= int(last_tsc[eye])
                    ):
                        raise ValueError(f"eye {eye} TSC did not increase")
                    last_tsc[eye] = tsc[eye]
        output, _ = process.communicate(timeout=20.0)
        if process.returncode != 0:
            raise RuntimeError(f"argus_sender exited {process.returncode}\n{output}")
        assert first_nals is not None
        for eye, types in enumerate(first_nals):
            if not {32, 33, 34}.issubset(types) or not any(
                16 <= value <= 23 for value in types
            ):
                raise ValueError(f"eye {eye} first AU lacks VPS/SPS/PPS/IRAP: {types}")
        print(
            f"PASS: AVS1 stereo hardware smoke received {args.frames} bundles; "
            f"max_abs_tsc_delta_us={max_delta_us:.3f}; first_nals={first_nals}"
        )
        return 0
    except Exception as exc:
        if process.poll() is None:
            process.terminate()
        try:
            remaining, _ = process.communicate(timeout=5.0)
            output += remaining
        except subprocess.TimeoutExpired:
            process.kill()
            remaining, _ = process.communicate()
            output += remaining
        print(f"FAIL: {exc}", file=sys.stderr)
        if output:
            print("--- argus_sender output ---", file=sys.stderr)
            print(output, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

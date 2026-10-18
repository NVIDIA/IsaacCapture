# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain worker exit status even when Kit terminates its Python process."""

import json
import os
import signal
import subprocess
import sys
import time
import uuid


def write_snapshot(root, value):
    temporary = root / "sim-state.tmp"
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(root / "sim-state.json")


def completion_snapshot(root, run_id, returncode, interrupted=False):
    path = root / "sim-state.json"
    value = json.loads(path.read_text()) if path.exists() else {"run_id": run_id}
    errors = []
    if value.get("run_id") != run_id:
        errors.append("Worker observation identity mismatch")
    if returncode != 0 or interrupted:
        errors.append(
            f"Worker exited with status {returncode}; interrupted={interrupted}"
        )
    if not value.get("native_environment_closed"):
        errors.append("Native environment did not confirm closure")
    if value.get("completion_error"):
        errors.append(value["completion_error"])
    value.update(
        phase="completed",
        timestamp=time.time(),
        worker_returncode=returncode,
        completion_error="; ".join(errors) or None,
    )
    return value


def supervise(root, arguments, timeout):
    root.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    command = [
        sys.executable,
        "-u",
        "-m",
        "isaaccapture_examples.synthetic_xr.franka.observe_record_demos",
        "--worker-run-id",
        run_id,
        *arguments,
    ]
    (root / "supervisor.json").write_text(
        json.dumps({"run_id": run_id, "command": command}, indent=2) + "\n"
    )
    process = subprocess.Popen(command, start_new_session=True)
    interrupted = False

    def forward(signum, frame):
        nonlocal interrupted
        interrupted = True
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)

    previous = {
        sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    forced = []
    try:
        deadline = time.monotonic() + timeout
        while (
            process.poll() is None and not interrupted and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        if process.poll() is None:
            interrupted = True
            for sig, grace in (
                (signal.SIGINT, 20),
                (signal.SIGTERM, 10),
                (signal.SIGKILL, 5),
            ):
                forced.append(signal.Signals(sig).name)
                os.killpg(process.pid, sig)
                try:
                    process.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
        code = process.wait(timeout=5)
        value = completion_snapshot(root, run_id, code, interrupted)
        value["shutdown_signals"] = forced
        write_snapshot(root, value)
        with (root / "sim-feedback.jsonl").open("a") as trace:
            trace.write(json.dumps(value, allow_nan=False) + "\n")
        (root / "worker-exit.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "returncode": code,
                    "interrupted": interrupted,
                    "shutdown_signals": forced,
                },
                indent=2,
            )
            + "\n"
        )
        return 0 if value["completion_error"] is None else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the optional Linux LiveKit encoded-receive extension outside the Python SDK."""

import argparse
from pathlib import Path
import shutil
import subprocess

REVISION = "42c106050e8a16c1509a5751d3c038cd9d17c78c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--toolchain", default="stable")
    parser.add_argument(
        "--timing", action="store_true", help="Add optional native receive timestamps"
    )
    parser.add_argument(
        "--playout-control",
        action="store_true",
        help="Expose opt-in zero playout delay; the runtime default is unchanged",
    )
    args = parser.parse_args()
    root = args.build_dir.expanduser().resolve()
    if not root.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/livekit/rust-sdks.git",
                str(root),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "checkout", "--detach", REVISION], check=True
        )
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REVISION:
        parser.error(f"Build checkout must be at {REVISION}; use a new --build-dir")
    # libyuv and codec submodules are required even when libwebrtc is prebuilt.
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--depth",
            "1",
        ],
        check=True,
    )
    patch = Path(__file__).parent / "native/livekit_encoded_receive.patch"
    timing = patch.with_name("livekit_receive_timing.patch").resolve()
    timing_applied = (
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "apply",
                "--unidiff-zero",
                "--reverse",
                "--check",
                str(timing),
            ],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if timing_applied and not args.timing:
        parser.error(
            "This build tree contains timing instrumentation; pass --timing or use a clean build tree"
        )
    reverse = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "apply",
            "--unidiff-zero",
            "--reverse",
            "--check",
            str(patch.resolve()),
        ],
        capture_output=True,
        check=False,
    )
    if reverse.returncode and not timing_applied:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "apply",
                "--unidiff-zero",
                "--check",
                str(patch.resolve()),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "apply", "--unidiff-zero", str(patch.resolve())],
            check=True,
        )
    if args.timing and not timing_applied:
        applied = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "apply",
                "--unidiff-zero",
                "--reverse",
                "--check",
                str(timing),
            ],
            capture_output=True,
            check=False,
        )
        if applied.returncode:
            subprocess.run(
                ["git", "-C", str(root), "apply", "--unidiff-zero", str(timing)],
                check=True,
            )
    playout = patch.with_name("livekit_playout_control.patch").resolve()
    playout_applied = (
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "apply",
                "--unidiff-zero",
                "--reverse",
                "--check",
                str(playout),
            ],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if playout_applied and not args.playout_control:
        parser.error(
            "This build tree contains playout control; pass --playout-control or use a clean tree"
        )
    if args.playout_control and not playout_applied:
        subprocess.run(
            ["git", "-C", str(root), "apply", "--unidiff-zero", str(playout)],
            check=True,
        )
    subprocess.run(
        [
            args.cargo,
            "+" + args.toolchain,
            "build",
            "--release",
            "-p",
            "livekit-ffi",
            "--manifest-path",
            str(root / "Cargo.toml"),
        ],
        check=True,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "target/release/liblivekit_ffi.so", output)
    print(f"Set livekit_cuda_library: {output}")


if __name__ == "__main__":
    main()

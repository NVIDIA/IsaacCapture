# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select a CUDA toolkit and exercise CuPy before declaring setup complete."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


def detect_toolkit() -> tuple[Path, tuple[int, int]]:
    # Select a system toolkit; check_kernel verifies what CuPy actually loads.
    explicit = os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")
    root = (
        Path(explicit)
        if explicit
        else (Path(nvcc).resolve().parent.parent if nvcc else Path("/usr/local/cuda"))
    )
    root = root.resolve()
    version = ""
    if (root / "version.json").is_file():
        version = json.loads((root / "version.json").read_text())["cuda"]["version"]
    elif (root / "version.txt").is_file():
        version = (root / "version.txt").read_text()
    elif root.is_dir():
        version = root.name if root.name.startswith("cuda-") else ""
    match = re.search(r"(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(
            f"Cannot determine the CUDA toolkit version at {root}. "
            "Install the toolkit matching your JetPack/driver and set CUDA_PATH "
            "to its directory, then rerun setup."
        )
    major, minor = map(int, match.groups())
    if major not in (12, 13):
        raise RuntimeError(
            f"Unsupported CUDA toolkit {major}.{minor}; use CUDA 12 or 13."
        )
    return root, (major, minor)


def check_kernel(expected_major: int | None = None) -> str:
    import cupy as cp

    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    nvrtc = cp.cuda.nvrtc.getVersion()
    if expected_major is not None and nvrtc[0] != expected_major:
        raise RuntimeError(
            f"Selected CUDA {expected_major}, but CuPy loaded NVRTC {nvrtc[0]}.{nvrtc[1]}. "
            "Check CUDA_PATH, LD_LIBRARY_PATH, and CUDA packages in the virtual environment."
        )
    # Allocation/import alone does not compile or execute a kernel.
    values = cp.arange(4, dtype=cp.float32) * 2 + 1
    cp.cuda.Stream.null.synchronize()
    if values.get().tolist() != [1.0, 3.0, 5.0, 7.0]:
        raise RuntimeError("CuPy kernel returned incorrect results")
    return f"CuPy kernel check passed: {name}, CuPy {cp.__version__}, NVRTC {nvrtc[0]}.{nvrtc[1]}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detect", action="store_true", help="print toolkit path, major, minor"
    )
    parser.add_argument("--cuda-major", type=int, help="expected NVRTC major version")
    args = parser.parse_args(argv)
    try:
        if args.detect:
            root, (major, minor) = detect_toolkit()
            print(f"{root}\t{major}\t{minor}")
        else:
            # A warm cache must not hide missing headers or a broken compiler.
            previous_cache = os.environ.get("CUPY_CACHE_DIR")
            try:
                with tempfile.TemporaryDirectory(prefix="camera-viz-cupy-") as cache:
                    os.environ["CUPY_CACHE_DIR"] = cache
                    print(check_kernel(args.cuda_major))
            finally:
                if previous_cache is None:
                    os.environ.pop("CUPY_CACHE_DIR", None)
                else:
                    os.environ["CUPY_CACHE_DIR"] = previous_cache
    except Exception as exc:
        print(f"CUDA check failed: {exc}", file=sys.stderr)
        if not args.detect:
            print(
                "Install matching CUDA runtime, NVRTC, headers, and CuPy. On Jetson, use "
                "your JetPack's CUDA packages and rerun camera_viz.sh setup --jetson. "
                "nvidia-smi reports driver support, not the installed toolkit.",
                file=sys.stderr,
            )
            if "CUDA_ERROR_NO_BINARY_FOR_GPU" in str(exc):
                print(
                    "For this kernel-binary mismatch, CUPY_COMPILE_WITH_PTX=1 may "
                    "work around an older NVRTC compiler; prefer a toolkit/CuPy "
                    "combination that supports your GPU.",
                    file=sys.stderr,
                )
            try:
                import cupy as cp

                cp.show_config()
            except Exception:
                pass  # Preserve the original failure even if diagnostics fail.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

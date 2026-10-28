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
import subprocess
import sys
import tempfile
from typing import NamedTuple

# Setup runs this with the system python3, before the sample is installed.
sys.path.append(str(Path(__file__).resolve().parents[1] / "python"))
from isaaccapture_examples.camera_viz import cuda_env  # noqa: E402

DEFAULT_ROOT = Path("/usr/local/cuda")
SUPPORTED_MAJORS = (12, 13)


class Toolkit(NamedTuple):
    root: Path
    version: tuple[int, int]
    # No version file, as with the distro nvidia-cuda-toolkit; see cuda_env.
    via_nvcc: bool


def _parse_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", text)
    return (int(match[1]), int(match[2])) if match else None


def _file_version(root: Path) -> tuple[int, int] | None:
    if (root / "version.json").is_file():
        return _parse_version(
            json.loads((root / "version.json").read_text())["cuda"]["version"]
        )
    if (root / "version.txt").is_file():
        return _parse_version((root / "version.txt").read_text())
    if root.is_dir() and root.name.startswith("cuda-"):
        return _parse_version(root.name)
    return None


def _nvcc_version(root: Path) -> tuple[int, int] | None:
    # The query CMake's FindCUDAToolkit makes. A root without headers is of no
    # use to CuPy, so it is not selected.
    nvcc = root / "bin" / "nvcc"
    if not (root / "include" / "cuda_runtime.h").is_file() or not nvcc.is_file():
        return None
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release (\d+\.\d+)", result.stdout)
    return _parse_version(match[1]) if match else None


def _candidate_roots() -> list[Path]:
    explicit = os.environ.get("CUDA_PATH")
    if explicit:
        return [Path(explicit).resolve()]
    roots = []
    search = os.environ.get("PATH", os.defpath)
    for entry in search.split(os.pathsep) if search else []:
        # shutil.which reads an empty entry as the current directory.
        nvcc = shutil.which("nvcc", path=entry or os.curdir)
        if nvcc:
            roots.append(Path(nvcc).resolve().parent.parent)
    # realpath, not resolve(): before Python 3.13, resolve() raises on a
    # symlink loop, which must not hide a toolkit found earlier on PATH.
    roots.append(Path(os.path.realpath(DEFAULT_ROOT)))
    return list(dict.fromkeys(roots))


def _versions(roots: list[Path]):
    # Version files first, so an installer's toolkit wins over a distro nvcc
    # earlier on PATH, which is often too old for current GPUs.
    nvcc_only = []
    for root in roots:
        version = _file_version(root)
        if version is None:
            nvcc_only.append(root)
        else:
            yield root, version, False
    for root in nvcc_only:
        yield root, _nvcc_version(root), True


def detect_toolkit() -> Toolkit:
    # Select a system toolkit; check_kernel verifies what CuPy actually loads.
    roots = _candidate_roots()
    rejected = []
    for root, version, via_nvcc in _versions(roots):
        if version is None:
            continue
        if version[0] in SUPPORTED_MAJORS:
            return Toolkit(root, version, via_nvcc)
        rejected.append(f"{version[0]}.{version[1]} at {root}")
    if rejected:
        raise RuntimeError(
            f"Unsupported CUDA toolkit {', '.join(rejected)}; use CUDA 12 or 13 "
            "and set CUDA_PATH to its root."
        )
    raise RuntimeError(
        f"Cannot determine the CUDA toolkit version at {', '.join(map(str, roots))}. "
        "Setup reads version.json, version.txt, a cuda-X.Y directory name, or "
        "nvcc --version where include/cuda_runtime.h is present. Install CUDA 12 "
        "or 13 (on Jetson, your JetPack's) and set CUDA_PATH to its root, then "
        "rerun setup."
    )


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
        "--detect",
        action="store_true",
        help="print toolkit path, major, minor, and the CUDA_PATH CuPy needs (or empty)",
    )
    parser.add_argument("--cuda-major", type=int, help="expected NVRTC major version")
    args = parser.parse_args(argv)
    try:
        if args.detect:
            toolkit = detect_toolkit()
            major, minor = toolkit.version
            cuda_path = toolkit.root if toolkit.via_nvcc else ""
            print(f"{toolkit.root}\t{major}\t{minor}\t{cuda_path}")
        else:
            cuda_env.apply_recorded_cuda_path()
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

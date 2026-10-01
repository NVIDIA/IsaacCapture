# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line for the wheel SBOM and license evidence collector."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import TOOL_NAME, TOOL_VERSION
from . import build as build_module
from . import discovery
from . import wheelfile
from . import evidence as evidence_module
from . import licensing
from . import validate as validate_module


def _repo_root(value: str | None) -> Path:
    """The checkout to inventory: given, else the one the caller is standing in.

    Not derived from this file's location -- the collector is installed into an
    environment of its own, so its own path says nothing about the checkout.
    """
    if value:
        return Path(value).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return Path.cwd()
    return (
        Path(result.stdout.strip()).resolve() if result.returncode == 0 else Path.cwd()
    )


def _report(failures: list[str], subject: str) -> int:
    if failures:
        print(f"{subject}: {len(failures)} problem(s)", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"{subject}: OK")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args.repo_root)
    build_dir = Path(args.build_dir)
    out_dir = Path(args.out_dir)

    license_data = (
        Path(args.license_data)
        if args.license_data
        else build_module.default_license_data(build_dir)
    )

    manifests: list[dict] = []
    for wheel in sorted(Path(item) for item in args.wheel):
        manifests.append(
            build_module.build(repo_root, build_dir, wheel, out_dir, license_data)
        )
        print(f"packaged evidence into {wheel.name}")

    # Through the same merge as a release: inheriting the first wheel's manifest
    # dated the set before the last wheel it binds existed, and skipped the
    # duplicate-filename check.
    merged = build_module.merge_records(
        [item for manifest in manifests for item in manifest["wheels"]]
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"manifest: {manifest_path}")

    for item in merged["wheels"]:
        # Both gap lists. They sit in the same record and mean the same kind of
        # thing to a reader; surfacing one and not the other hid the stricter one.
        for key in (
            "components_without_license_evidence",
            "components_with_unidentified_license",
        ):
            for entry in item[key]:
                print(
                    f"::warning::{item['filename']}: {entry['component']}: "
                    f"{entry['reason']}"
                )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    status = 0
    for wheel in sorted(Path(item) for item in args.wheel):
        failures = validate_module.check(
            wheel,
            Path(args.build_evidence) if args.build_evidence else None,
            Path(args.manifest) if args.manifest else None,
            Path(args.evidence_dir) if args.evidence_dir else None,
        )
        status |= _report(failures, wheel.name)
    return status


def _cmd_check_set(args: argparse.Namespace) -> int:
    failures = validate_module.check_set(
        Path(args.manifest), Path(args.wheel_dir), Path(args.evidence_dir)
    )
    return _report(failures, Path(args.manifest).name)


def _cmd_merge_manifests(args: argparse.Namespace) -> int:
    merged = build_module.merge_manifests([Path(item) for item in args.manifest])
    Path(args.output).write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"merged {len(merged['wheels'])} wheel record(s) -> {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}"
    )
    parser.add_argument(
        "--repo-root",
        help="IsaacCapture checkout (default: the git checkout you are standing in)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="discover what a build put in its wheels, then package the evidence into them",
    )
    build.add_argument("--build-dir", required=True, help="configured CMake build tree")
    build.add_argument("--wheel", required=True, action="append")
    build.add_argument("--out-dir", required=True)
    build.add_argument(
        "--license-data",
        help=(
            "SPDX license list json/ directory fetched by deps/third_party "
            "(default: <build-dir>/_deps/license-list-data-src/json)"
        ),
    )
    build.set_defaults(handler=_cmd_build)

    check = subparsers.add_parser(
        "check",
        help="check a wheel against the SBOM it carries, plus any evidence you supply",
    )
    check.add_argument("--wheel", required=True, action="append")
    check.add_argument(
        "--build-evidence", help="also check that the build accounts for every member"
    )
    check.add_argument(
        "--manifest", help="also check the wheel's bound digest and sidecars"
    )
    check.add_argument(
        "--evidence-dir", help="where the sidecars live (default: beside the manifest)"
    )
    check.set_defaults(handler=_cmd_check)

    check_set = subparsers.add_parser(
        "check-set", help="run check over every wheel a merged manifest advertises"
    )
    check_set.add_argument("--manifest", required=True)
    check_set.add_argument("--wheel-dir", required=True)
    check_set.add_argument("--evidence-dir", required=True)
    check_set.set_defaults(handler=_cmd_check_set)

    merge = subparsers.add_parser(
        "merge-manifests", help="fold per-variant manifests into one"
    )
    merge.add_argument("--output", required=True)
    merge.add_argument("manifest", nargs="+")
    merge.set_defaults(handler=_cmd_merge_manifests)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (
        evidence_module.EvidenceError,
        build_module.BuildError,
        licensing.CorpusError,
        discovery.ArchiveError,
        wheelfile.WheelError,
        validate_module.ManifestError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

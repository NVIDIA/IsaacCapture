# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line for the wheel SBOM and license evidence collector."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from . import TOOL_NAME, TOOL_VERSION
from . import build as build_module
from . import discovery
from . import wheelfile
from . import evidence as evidence_module
from . import licensing
from . import report as report_module
from . import validate as validate_module


def _repo_root(value: str | None) -> Path:
    """The checkout to inventory: given, else the one the caller is standing in.

    Not derived from this file's location -- the collector is installed into an
    environment of its own, so its own path says nothing about the checkout.
    """
    if value:
        return Path(value).resolve()
    top = discovery.git(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(top).resolve() if top else Path.cwd()


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

    licensing.load_corpus(license_data)
    discovery = evidence_module.discover(repo_root, build_dir)

    manifests: list[dict] = []
    wheels = sorted(Path(item) for item in args.wheel)
    siblings = build_module.siblings_of(wheels, repo_root)
    for wheel in wheels:
        manifest = build_module.build(
            repo_root,
            build_dir,
            wheel,
            out_dir,
            license_data,
            discovery=discovery,
            siblings=siblings,
        )
        manifests.append(manifest)
        print(f"packaged evidence into {wheel.name}")

    # The merge a release uses: the set is stamped after its last wheel, and one
    # filename cannot stand for two wheels.
    merged = build_module.merge_records(
        [item for manifest in manifests for item in manifest["wheels"]]
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"manifest: {manifest_path}")

    for item in merged["wheels"]:
        for key in report_module.GAP_LISTS:
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
    build.add_argument(
        "--wheel",
        required=True,
        action="append",
        help="wheel to describe; rewritten in place, repeatable",
    )
    build.add_argument(
        "--out-dir", required=True, help="receives the sidecars and manifest.json"
    )
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
    check.add_argument(
        "--wheel", required=True, action="append", help="wheel to check; repeatable"
    )
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
    check_set.add_argument(
        "--manifest", required=True, help="merged manifest of the release"
    )
    check_set.add_argument(
        "--wheel-dir",
        required=True,
        help="the release's wheels; one the manifest does not list fails the set",
    )
    check_set.add_argument(
        "--evidence-dir", required=True, help="where the release's sidecars live"
    )
    check_set.set_defaults(handler=_cmd_check_set)

    merge = subparsers.add_parser(
        "merge-manifests", help="fold per-variant manifests into one"
    )
    merge.add_argument("--output", required=True, help="merged manifest to write")
    merge.add_argument("manifest", nargs="+", help="per-variant manifests")
    merge.set_defaults(handler=_cmd_merge_manifests)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (
        evidence_module.EvidenceError,
        build_module.BuildError,
        licensing.CorpusError,
        licensing.ExpressionReadError,
        discovery.ArchiveError,
        discovery.SupplierError,
        # Raised both for a build tree with no file API reply and for a repo
        # root git cannot answer for; its message is the instruction to the
        # operator, and a traceback buries it.
        discovery.FileApiError,
        wheelfile.WheelError,
        validate_module.ManifestError,
        # Malformed input reaches these through a document, a manifest or a
        # wheel's own metadata: a corrupt JSON body, a requirement that will not
        # parse, a `--wheel` that is not a zip, a manifest missing a key, a file
        # that is not there. Each has something worth printing; a traceback
        # buries it and exits 1 where the caller expects 2.
        ValueError,
        OSError,
        KeyError,
        zipfile.BadZipFile,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

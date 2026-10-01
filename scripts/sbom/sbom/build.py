# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a repaired wheel plus its build tree into a wheel that carries both."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from . import TOOL_NAME, TOOL_VERSION, stamped_now
from . import document as document_module
from . import evidence as evidence_module
from . import licensing
from . import inventory as inventory_module
from . import report as report_module
from . import wheelfile
from .discovery import project_supplier
from .inventory import Attribution
from .wheelfile import Entry

# Supplier is absent on purpose: it is read from the project's own packaging
# metadata at build time, so a fork or a rename cannot leave this file naming
# somebody else's organisation with nothing to catch it.
PROJECT = {
    "homepage": "https://github.com/NVIDIA/IsaacCapture",
    "license": "Apache-2.0",
    "document_namespace": "https://github.com/NVIDIA/IsaacCapture/spdx",
}


class BuildError(Exception):
    """The wheel cannot be published with the evidence available."""


def _entry(name: str, data: bytes) -> Entry:
    return Entry(
        name=name,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        sha1=hashlib.sha1(data).hexdigest(),  # noqa: S324 - SPDX 2.3 requires SHA1
    )


def _declared_path(dist_info: str, member: str) -> str:
    """What METADATA calls a packaged license file.

    PEP 639 states License-File relative to `.dist-info/licenses/`, which is
    what setuptools already does for the project's own texts; declaring the
    member path instead yields an entry that resolves to nothing.
    """
    return member.removeprefix(f"{dist_info}/licenses/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_license_data(build_dir: Path) -> Path:
    """Where deps/third_party materializes the SPDX license list for this build."""
    return build_dir / "_deps" / "license-list-data-src" / "json"


def build(
    repo_root: Path,
    build_dir: Path,
    wheel_path: Path,
    out_dir: Path,
    license_data: Path | None = None,
    system_resolver=None,
) -> dict:
    """Package evidence into `wheel_path` in place and write the sidecars."""
    licensing.load_corpus(license_data or default_license_data(build_dir))
    discovery = evidence_module.discover(repo_root, build_dir)
    wheel = wheelfile.scan(wheel_path, analyze_elf=True)
    sbom_name = wheelfile.sbom_member(wheel.dist_info, wheel_path.name)
    if any(item.name == sbom_name for item in wheel.entries):
        raise BuildError(
            f"{wheel_path.name} already carries {sbom_name}. Run this on a freshly "
            "repaired wheel; rewriting one twice would duplicate its members."
        )
    metadata_raw, _ = wheelfile.read_metadata(wheel_path, wheel.dist_info)

    resolver = inventory_module.Resolver(discovery, system_resolver)
    inventory = inventory_module.build(
        wheel, metadata_raw, resolver, discovery.staged_root
    )

    if inventory.unattributed:
        raise BuildError(
            "these wheel members could not be traced to anything this build "
            "produced or fetched:\n  "
            + "\n  ".join(inventory.unattributed)
            + "\nThey reached the wheel by a route the collector cannot see."
        )

    # A REUSE LICENSES/ pool is a real text even though it names no single
    # expression for the component, so it satisfies the packaging obligation.
    missing = [
        key
        for key, component in inventory.components_present.items()
        if not any(item.kind in ("grant", "pool") for item in component.evidence)
    ]
    if missing:
        raise BuildError(
            "no license text was found in the files of redistributed component(s): "
            + ", ".join(missing)
            + "\nThe component ships no LICENSE/COPYING file and states no grant in "
            "its README; obtain the terms before publishing it."
        )

    # Everything the wheel gains, computed before a byte is written so the
    # document can describe the wheel it is about to be embedded in.
    additions: dict[str, bytes] = {}
    packaged_paths: list[str] = []
    for key, component in inventory.components_present.items():
        for item in component.evidence:
            target = report_module.packaged_license_path(wheel.dist_info, key, item)
            if target in additions:
                continue
            additions[target] = item.text.encode("utf-8")
            packaged_paths.append(_declared_path(wheel.dist_info, target))

    notices_path = f"{wheel.dist_info}/licenses/THIRD-PARTY-NOTICES.md"
    additions[notices_path] = report_module.notices_markdown(
        inventory, wheel.dist_info, wheel_path.name
    ).encode("utf-8")
    packaged_paths.append(_declared_path(wheel.dist_info, notices_path))

    patched_metadata = wheelfile.add_license_files(metadata_raw, sorted(packaged_paths))
    metadata_declared = patched_metadata != metadata_raw
    replacements = {f"{wheel.dist_info}/METADATA": patched_metadata}

    # Neither file can state its own digest. RECORD hashes every member, this
    # document included, and the document is rendered before RECORD is
    # regenerated -- listing either is a fixed point no hash function has. SPDX
    # 2.3 requires a checksum on every file listed, so they are omitted rather
    # than listed without one; `check` reads both against the archive instead.
    excluded = [f"{wheel.dist_info}/RECORD", sbom_name]

    projected = [item for item in wheel.entries if item.name not in replacements] + [
        _entry(name, data) for name, data in {**replacements, **additions}.items()
    ]
    projected_wheel = replace(wheel, entries=tuple(projected))
    for name in [*additions, *replacements]:
        inventory.attributions[name] = Attribution(
            path=name,
            origin="metadata",
            detail="license evidence packaged by the SBOM collector"
            if name in additions
            else "distribution metadata, updated to declare the packaged license files",
        )
    # The document is added below, after it has been rendered; record it now so
    # the evidence accounts for every member of the finished wheel.
    inventory.attributions[sbom_name] = Attribution(
        path=sbom_name,
        origin="metadata",
        detail="the contents SBOM this collector embedded",
    )

    evidence_doc = evidence_module.document(discovery, wheel_path.name, inventory)
    evidence_doc["ci"] = environment_summary()
    evidence_doc["system_libraries"] = resolver.system_libraries

    spdx = document_module.build_document(
        {**PROJECT, "supplier": project_supplier(repo_root)},
        evidence_doc,
        projected_wheel,
        inventory,
        excluded,
    )
    spdx_bytes = json.dumps(spdx, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    additions[sbom_name] = spdx_bytes

    with tempfile.TemporaryDirectory(dir=str(wheel_path.parent)) as scratch:
        staged = Path(scratch) / wheel_path.name
        wheelfile.rewrite(
            wheel_path,
            staged,
            additions=additions,
            replacements=replacements,
            dist_info=wheel.dist_info,
        )
        shutil.move(str(staged), str(wheel_path))

    payload = report_module.report(inventory, evidence_doc, wheel_path.name)
    payload["packaged_license_files"] = sorted(packaged_paths)
    payload["metadata_license_files_declared"] = metadata_declared

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = wheel_path.name.removesuffix(".whl")
    sidecars = {
        "spdx": out_dir / f"{stem}.spdx.json",
        "license_report_json": out_dir / f"{stem}.licenses.json",
        "license_report_md": out_dir / f"{stem}.licenses.md",
        "build_evidence": out_dir / f"{stem}.build-evidence.json",
    }
    sidecars["spdx"].write_bytes(spdx_bytes)
    sidecars["license_report_json"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecars["license_report_md"].write_text(
        report_module.report_markdown(payload), encoding="utf-8"
    )
    sidecars["build_evidence"].write_text(
        json.dumps(evidence_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # The wheel digest is bound here, after the last mutation: a wheel that is
    # touched again no longer matches its own manifest entry.
    return {
        "schema": "isaaccapture-sbom-manifest/1",
        "generated_at": stamped_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "wheels": [
            {
                "filename": wheel_path.name,
                "sha256": sha256_file(wheel_path),
                "size": wheel_path.stat().st_size,
                "sbom_in_wheel": sbom_name,
                "components_without_license_evidence": payload[
                    "components_without_license_evidence"
                ],
                "components_with_unidentified_license": payload[
                    "components_with_unidentified_license"
                ],
                "sidecars": {
                    kind: {
                        "filename": path.name,
                        "sha256": sha256_file(path),
                        "size": path.stat().st_size,
                    }
                    for kind, path in sidecars.items()
                },
            }
        ],
    }


def merge_manifests(paths: list[Path]) -> dict:
    """Fold per-variant manifests into the one a release advertises."""
    return merge_records(
        [
            item
            for path in sorted(paths)
            for item in json.loads(path.read_text(encoding="utf-8"))["wheels"]
        ]
    )


def merge_records(wheels: list[dict]) -> dict:
    """The manifest for a set of wheels, stamped when the set was complete."""
    names = [item["filename"] for item in wheels]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise BuildError(f"the same wheel filename was published twice: {duplicates}")
    return {
        "schema": "isaaccapture-sbom-manifest/1",
        "generated_at": stamped_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "wheels": sorted(wheels, key=lambda item: item["filename"]),
    }


def environment_summary() -> dict:
    """Recorded so a rerun can tell a cold build from a cache-restored one."""
    return {
        "runner_os": os.environ.get("RUNNER_OS"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_ref": os.environ.get("GITHUB_REF"),
    }

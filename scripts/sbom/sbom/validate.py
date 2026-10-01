# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks that decide whether a wheel may be published.

`verify_wheel` is the consumer-side half: it needs the wheel and nothing else,
so the instructions published with a release are the same code that gates it.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from . import SPDX_VERSION, licensing
from . import document as document_module
from . import wheelfile

_REQUIRED_KEYS = (
    "spdxVersion",
    "dataLicense",
    "SPDXID",
    "name",
    "documentNamespace",
    "creationInfo",
    "packages",
    "files",
    "relationships",
)


def _embedded_sbom(
    archive: zipfile.ZipFile, dist_info: str, wheel_name: str
) -> tuple[str, dict]:
    name = wheelfile.sbom_member(dist_info, wheel_name)
    if name not in archive.namelist():
        found = [
            item
            for item in archive.namelist()
            if item.startswith(f"{dist_info}/sboms/")
        ]
        raise LookupError(
            f"expected {name}, found {found or 'nothing'} under {dist_info}/sboms/"
        )
    return name, json.loads(archive.read(name))


def verify_wheel(
    wheel_path: Path, scanned: wheelfile.WheelInfo | None = None
) -> list[str]:
    """Verify a wheel against the SBOM it carries. No repository required."""
    failures: list[str] = []
    wheel = scanned or wheelfile.scan(wheel_path)

    with zipfile.ZipFile(wheel_path) as archive:
        try:
            sbom_name, spdx = _embedded_sbom(archive, wheel.dist_info, wheel_path.name)
        except (LookupError, json.JSONDecodeError) as error:
            return [f"embedded SBOM: {error}"]

    for key in _REQUIRED_KEYS:
        if key not in spdx:
            failures.append(f"SPDX document is missing required field {key!r}")
    if spdx.get("spdxVersion") != SPDX_VERSION:
        failures.append(
            f"spdxVersion is {spdx.get('spdxVersion')!r}, expected {SPDX_VERSION!r}"
        )
    if failures:
        return failures

    identifiers = [spdx["SPDXID"]]
    identifiers += [item["SPDXID"] for item in spdx["packages"]]
    identifiers += [item["SPDXID"] for item in spdx["files"]]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        failures.append(f"duplicate SPDXIDs: {duplicates}")
    known = set(identifiers)
    for relationship in spdx["relationships"]:
        for side in ("spdxElementId", "relatedSpdxElement"):
            if relationship[side] not in known:
                failures.append(
                    f"relationship {relationship['relationshipType']} references unknown "
                    f"element {relationship[side]!r}"
                )

    wheel_package = next(
        (
            item
            for item in spdx["packages"]
            if item["SPDXID"] == "SPDXRef-Package-wheel"
        ),
        None,
    )
    if wheel_package is None:
        return failures + ["SPDXRef-Package-wheel is missing"]
    if wheel_package.get("packageFileName") != wheel_path.name:
        failures.append(
            f"packageFileName {wheel_package.get('packageFileName')!r} does not name this wheel"
        )
    if (
        f"{wheel_package.get('name')}-{wheel_package.get('versionInfo')}"
        != wheel.dist_info.removesuffix(".dist-info")
    ):
        failures.append(
            "wheel package name/version do not match the wheel's .dist-info"
        )

    excluded = {
        item.removeprefix("./")
        for item in wheel_package.get("packageVerificationCode", {}).get(
            "packageVerificationCodeExcludedFiles", []
        )
    }
    if sbom_name not in excluded:
        failures.append("the embedded SBOM must be listed among the excluded files")

    documented = {item["fileName"].removeprefix("./"): item for item in spdx["files"]}
    actual = {item.name: item for item in wheel.entries}

    for name in sorted(set(actual) - set(documented) - excluded):
        failures.append(f"wheel file not covered by the SBOM: {name}")
    for name in sorted(set(documented) - set(actual)):
        failures.append(f"SBOM documents a file the wheel does not contain: {name}")

    sha1_digests = []
    for name, record in sorted(documented.items()):
        entry = actual.get(name)
        if entry is None:
            continue
        checksums = {
            item["algorithm"]: item["checksumValue"] for item in record["checksums"]
        }
        if checksums.get("SHA256") != entry.sha256:
            failures.append(f"SHA256 mismatch for {name}")
        if checksums.get("SHA1") != entry.sha1:
            failures.append(f"SHA1 mismatch for {name}")
        sha1_digests.append(entry.sha1)

    expected_code = document_module.verification_code(sha1_digests)
    actual_code = wheel_package.get("packageVerificationCode", {}).get(
        "packageVerificationCodeValue"
    )
    if expected_code != actual_code:
        failures.append(
            f"packageVerificationCode is {actual_code!r} but the documented files hash to {expected_code!r}"
        )

    failures.extend(_check_record(wheel_path, wheel))
    failures.extend(_check_licenses(wheel, spdx))
    return failures


def _check_record(wheel_path: Path, wheel: wheelfile.WheelInfo) -> list[str]:
    """RECORD is what an installer trusts; a wrong entry here is a broken wheel."""
    failures: list[str] = []
    record = wheelfile.read_record(wheel_path, wheel.dist_info)
    record_name = f"{wheel.dist_info}/RECORD"
    for entry in wheel.entries:
        if entry.name == record_name:
            continue
        row = record.get(entry.name)
        if row is None:
            failures.append(f"RECORD does not list {entry.name}")
            continue
        # scan() already hashed every member; RECORD states the same digest in
        # base64, so re-reading the archive to recompute it is wasted work.
        if row[0] != wheelfile.record_hash_from_sha256(entry.sha256):
            failures.append(f"RECORD digest mismatch for {entry.name}")
        if row[1] != str(entry.size):
            failures.append(f"RECORD size mismatch for {entry.name}")
    for name in sorted(set(record) - {item.name for item in wheel.entries}):
        failures.append(f"RECORD lists a file the wheel does not contain: {name}")
    return failures


def _check_attribution(evidence_doc: dict, wheel: wheelfile.WheelInfo) -> list[str]:
    """Every wheel member must name where it came from, and every shipped
    component must have had a license text to read."""
    failures: list[str] = []
    attributed = {item["path"] for item in evidence_doc.get("attributions", [])}

    for entry in wheel.entries:
        if entry.name not in attributed:
            failures.append(
                f"build evidence does not explain where {entry.name} came from"
            )

    for key, component in evidence_doc.get("components", {}).items():
        if not component.get("in_this_wheel"):
            continue
        if not any(item["kind"] == "grant" for item in component.get("evidence", [])):
            failures.append(
                f"{key} is redistributed but no license text was found for it"
            )
    return failures


def _check_licenses(wheel: wheelfile.WheelInfo, spdx: dict) -> list[str]:
    failures: list[str] = []
    names = {item.name for item in wheel.entries}
    notices = f"{wheel.dist_info}/licenses/THIRD-PARTY-NOTICES.md"
    if notices not in names:
        failures.append(f"{notices} is not packaged")

    declared_refs = set()
    for item in spdx.get("files", []):
        declared_refs.update(
            licensing.license_refs(str(item.get("licenseConcluded", "")))
        )
    for package in spdx["packages"]:
        for field in ("licenseConcluded", "licenseDeclared"):
            for token in (
                str(package.get(field, "")).replace("(", " ").replace(")", " ").split()
            ):
                if token.startswith("LicenseRef-"):
                    declared_refs.add(token)
    extracted = {
        item["licenseId"] for item in spdx.get("hasExtractedLicensingInfos", [])
    }
    for missing in sorted(declared_refs - extracted):
        failures.append(
            f"{missing} is used but has no hasExtractedLicensingInfos entry"
        )
    for item in spdx.get("hasExtractedLicensingInfos", []):
        if not item.get("extractedText", "").strip():
            failures.append(f"{item['licenseId']} has an empty extractedText")
    return failures


def check(
    wheel: Path,
    build_evidence: Path | None = None,
    manifest: Path | None = None,
    evidence_dir: Path | None = None,
) -> list[str]:
    """Check a wheel, and as much of its published evidence as you can supply.

    One operation over widening scope rather than three commands: a consumer has
    only the wheel, a release build also has the evidence it just produced, and a
    release has the whole set. Each input adds checks; none replaces the others.
    """
    scanned = wheelfile.scan(wheel)
    failures = verify_wheel(wheel, scanned)

    if build_evidence is not None:
        evidence_doc = json.loads(build_evidence.read_text(encoding="utf-8"))
        failures.extend(_check_attribution(evidence_doc, scanned))

    if manifest is not None:
        failures.extend(_check_manifest(manifest, wheel, evidence_dir))

    return failures


def _check_manifest(
    manifest_path: Path, wheel: Path, evidence_dir: Path | None
) -> list[str]:
    """The manifest binds a wheel's digest and names the sidecars beside it."""
    failures: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        (item for item in manifest["wheels"] if item["filename"] == wheel.name), None
    )
    if record is None:
        return [f"{manifest_path.name} has no entry for {wheel.name}"]

    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if record["sha256"] != digest:
        failures.append(
            "the wheel was modified after its digest was bound: manifest says "
            f"{record['sha256']}, the file hashes to {digest}"
        )

    beside = evidence_dir if evidence_dir is not None else manifest_path.parent
    for kind, sidecar in record["sidecars"].items():
        path = beside / sidecar["filename"]
        if not path.is_file():
            failures.append(f"sidecar {kind} is missing: {sidecar['filename']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != sidecar["sha256"]:
            failures.append(f"sidecar {kind} does not match its recorded digest")

    published = beside / record["sidecars"]["spdx"]["filename"]
    if published.is_file():
        with zipfile.ZipFile(wheel) as archive:
            if archive.read(record["sbom_in_wheel"]) != published.read_bytes():
                failures.append(
                    "the published SBOM copy differs from the one embedded in the wheel"
                )
    return failures


def check_set(manifest_path: Path, wheel_dir: Path, evidence_dir: Path) -> list[str]:
    """Run :func:`check` over every wheel a merged manifest advertises."""
    failures: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["wheels"]:
        wheel = wheel_dir / record["filename"]
        if not wheel.is_file():
            failures.append(f"{record['filename']} is missing from {wheel_dir}")
            continue
        failures.extend(
            f"{record['filename']}: {item}"
            for item in check(wheel, manifest=manifest_path, evidence_dir=evidence_dir)
        )
    return failures

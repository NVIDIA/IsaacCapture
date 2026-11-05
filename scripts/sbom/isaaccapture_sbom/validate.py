# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify wheel contents and optional build/release evidence.

verify_wheel needs only the archive and is shared by CI and consumers.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from . import SPDX_VERSION, licensing, report
from .discovery import sha256_file
from . import document as document_module
from . import wheelfile


# Derived from the wheel alone, so every build of one wheel writes the same bytes.
CONTENT_SIDECARS = ("spdx", "license_report_json", "license_report_md")
_SIDECAR_KINDS = (*CONTENT_SIDECARS, "build_evidence")


class ManifestError(Exception):
    """A manifest lacks the structure required for verification."""


def _sidecar(evidence_dir: Path, record: dict, kind: str) -> Path | None:
    named = record.get("sidecars", {}).get(kind, {}).get("filename")
    if not named:
        return None
    path = evidence_dir / named
    return path if path.is_file() else None


def manifest_records(path: Path) -> list[dict]:
    """A manifest's wheel records; one listing none checks nothing, so it fails."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ManifestError(f"{path}: the manifest is not JSON ({error})") from error
    if not isinstance(manifest, dict):
        raise ManifestError(f"{path}: the manifest is not a JSON object")
    records = manifest.get("wheels")
    if not isinstance(records, list):
        raise ManifestError(f"{path}: the manifest has no 'wheels' list")
    if not records:
        raise ManifestError(f"{path}: the manifest lists no wheels")
    for item in records:
        if not isinstance(item, dict):
            raise ManifestError(f"{path}: a 'wheels' entry is not an object")
        if not isinstance(item.get("filename"), str) or not item["filename"]:
            raise ManifestError(f"{path}: a 'wheels' entry names no wheel")
        if not _is_bare_name(item["filename"]):
            raise ManifestError(
                f"{path}: {item['filename']!r} is not a bare wheel file name"
            )
    return records


def bound_names(records) -> list[str]:
    return [
        sidecar["filename"]
        for record in records
        for sidecar in record["sidecars"].values()
    ]


def sidecar_failures(beside: Path, record: dict) -> list[str]:
    """Return missing-file or digest failures for a checked record's sidecars."""
    failures = []
    for kind, sidecar in record["sidecars"].items():
        path = beside / sidecar["filename"]
        if not path.is_file():
            failures.append(f"sidecar {kind} is missing: {sidecar['filename']}")
        elif sha256_file(path) != sidecar["sha256"]:
            failures.append(f"sidecar {kind} does not match its recorded digest")
    return failures


def _is_bare_name(name: str) -> bool:
    """A file name joined onto a directory must stay inside that directory."""
    return name not in ("", ".", "..") and not any(sep in name for sep in "/\\:")


def _record_for(records: list[dict], filename: str) -> dict:
    """Find exactly one record for a wheel, or raise ManifestError."""
    seen = [item for item in records if item.get("filename") == filename]
    if len(seen) > 1:
        raise ManifestError(f"{filename} is advertised {len(seen)} times")
    return checked_record(seen[0]) if seen else {}


def checked_record(record: dict) -> dict:
    """A record holding every field the checks read, or ManifestError."""
    filename = record["filename"]
    for key in ("sha256", "size", "sbom_in_wheel", "sidecars"):
        if key not in record:
            raise ManifestError(f"{filename}: manifest record has no {key!r}")
    sidecars = record["sidecars"]
    if not isinstance(sidecars, dict) or not all(
        isinstance(item, dict)
        and isinstance(item.get("filename"), str)
        and isinstance(item.get("sha256"), str)
        for item in sidecars.values()
    ):
        raise ManifestError(
            f"{filename}: manifest record's sidecars must each name a file and its sha256"
        )
    for kind, item in sidecars.items():
        if not _is_bare_name(item["filename"]):
            raise ManifestError(
                f"{filename}: {kind} sidecar {item['filename']!r} is not a bare "
                "file name"
            )
    # Require every sidecar kind; omissions must not bypass evidence checks.
    missing = sorted(set(_SIDECAR_KINDS) - set(record["sidecars"]))
    if missing:
        raise ManifestError(
            f"{filename}: manifest record names no {', '.join(missing)} sidecar"
        )
    return record


_SHARED_ID = re.compile(re.escape(licensing.SHARED_REF) + "[0-9a-f]{12}")
_EVIDENCE_DIGEST = re.compile(
    r"\(sha256:(?P<digest>[0-9a-f]{64})[^)]*\) packaged at (?P<where>\S+)"
)

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
    wheel_path: Path,
    scanned: wheelfile.WheelInfo | None = None,
    carried: list[dict] | None = None,
) -> list[str]:
    """Verify an archive against its embedded SBOM without a repository.

    Append the parsed document to carried for reuse by evidence checks.
    """
    failures: list[str] = []
    wheel = scanned or wheelfile.scan(wheel_path)

    with zipfile.ZipFile(wheel_path) as archive:
        try:
            sbom_name, spdx = _embedded_sbom(archive, wheel.dist_info, wheel_path.name)
        except (LookupError, json.JSONDecodeError) as error:
            return [f"embedded SBOM: {error}"]
    if carried is not None:
        carried.append(spdx)

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
    duplicates = sorted(item for item, seen in Counter(identifiers).items() if seen > 1)
    if duplicates:
        failures.append(f"duplicate SPDXIDs: {duplicates}")
    known = set(identifiers)
    # Cross-document element IDs must name a declared DocumentRef.
    declared_documents = {
        item["externalDocumentId"] for item in spdx.get("externalDocumentRefs", [])
    }
    for relationship in spdx["relationships"]:
        for side in ("spdxElementId", "relatedSpdxElement"):
            element = relationship[side]
            prefix, _, _ = element.partition(":")
            if prefix.startswith("DocumentRef-"):
                if prefix not in declared_documents:
                    failures.append(
                        f"relationship {relationship['relationshipType']} references "
                        f"{element!r}, and no externalDocumentRef declares {prefix!r}"
                    )
                continue
            if element not in known:
                failures.append(
                    f"relationship {relationship['relationshipType']} references unknown "
                    f"element {element!r}"
                )
    for item in spdx.get("externalDocumentRefs", []):
        checksum = item.get("checksum") or {}
        if checksum.get("algorithm") != "SHA1" or not checksum.get("checksumValue"):
            failures.append(
                f"{item.get('externalDocumentId')!r} names a document with no SHA1 "
                "to check it by"
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
    # Do not trust the document to choose exclusions: arbitrary entries here
    # could hide added files from coverage checks.
    permitted = {f"{wheel.dist_info}/RECORD", sbom_name}
    if excluded != permitted:
        failures.append(
            "the excluded-files list must be exactly "
            f"{sorted(permitted)}, not {sorted(excluded)}"
        )
        excluded = permitted

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
    failures.extend(_check_license_texts(wheel, spdx))
    return failures


def _check_license_texts(wheel: wheelfile.WheelInfo, spdx: dict) -> list[str]:
    """Check that cited license paths contain the bytes the document describes.

    Per-file hashes alone can agree with a substituted or emptied license file.
    """
    failures: list[str] = []
    by_name = {entry.name: entry.sha256 for entry in wheel.entries}
    packaged = {
        digest
        for name, digest in by_name.items()
        if report.is_packaged_license(wheel.dist_info, name)
    }
    if not packaged:
        return failures

    for package in spdx.get("packages", []):
        for text in package.get("attributionTexts", []):
            match = _EVIDENCE_DIGEST.search(text)
            if not match:
                continue
            # Check the cited path too: another component can contain identical license bytes.
            where = match.group("where")
            if by_name.get(where) != match.group("digest"):
                failures.append(
                    f"{package['name']}: {where} does not hold the licence text "
                    f"this document records for it (sha256:{match.group('digest')})"
                )

    return failures


def _check_record(wheel_path: Path, wheel: wheelfile.WheelInfo) -> list[str]:
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
        # Reuse scan hashes; RECORD encodes the same digest in base64.
        if row[0] != wheelfile.record_hash_from_sha256(entry.sha256):
            failures.append(f"RECORD digest mismatch for {entry.name}")
        if row[1] != str(entry.size):
            failures.append(f"RECORD size mismatch for {entry.name}")
    for name in sorted(set(record) - {item.name for item in wheel.entries}):
        failures.append(f"RECORD lists a file the wheel does not contain: {name}")
    return failures


def _check_attribution(
    evidence_doc: dict, wheel: wheelfile.WheelInfo, spdx: dict | None
) -> list[str]:
    """Require attribution for every member and packaged terms for each component."""
    stated = evidence_doc.get("wheel")
    if stated and stated != wheel.path.name:
        return [f"build evidence describes {stated}, not {wheel.path.name}"]

    failures: list[str] = []
    attributed = {item["path"] for item in evidence_doc.get("attributions", [])}
    packaged = {
        entry.sha256
        for entry in wheel.entries
        if report.is_packaged_license(wheel.dist_info, entry.name)
    }
    recorded: set[str] = set()

    for entry in wheel.entries:
        if entry.name not in attributed:
            failures.append(
                f"build evidence does not explain where {entry.name} came from"
            )

    for key, component in evidence_doc.get("components", {}).items():
        if not component.get("in_this_wheel"):
            continue
        # Include NOTICE and other non-grant evidence cited by attributionTexts.
        recorded.update(item["sha256"] for item in component.get("evidence", []))
        texts = [
            item
            for item in component.get("evidence", [])
            if licensing.satisfies_obligation(item["kind"])
        ]
        if not texts:
            failures.append(
                f"{key} is redistributed but no license text was found for it"
            )
            continue
        # Existence at build time is insufficient; the recorded terms must still be packaged.
        if not any(item["sha256"] in packaged for item in texts):
            failures.append(
                f"{key} is redistributed but the license text recorded for it is "
                "not packaged in this wheel"
            )
    failures.extend(_document_agrees_with_evidence(wheel, recorded, evidence_doc, spdx))
    return failures


def _document_agrees_with_evidence(
    wheel: wheelfile.WheelInfo,
    recorded: set[str],
    evidence_doc: dict,
    spdx: dict | None,
) -> list[str]:
    """Compare license claims with separately recorded build evidence.

    A self-consistent rewrite of the SBOM and wheel can pass _check_license_texts;
    the evidence supplies independent digests and conclusions.
    """
    if spdx is None:
        return []

    failures = []
    for package in spdx.get("packages", []):
        for text in package.get("attributionTexts", []):
            match = _EVIDENCE_DIGEST.search(text)
            if match and match.group("digest") not in recorded:
                failures.append(
                    f"{package['name']}: the document names a licence text "
                    f"(sha256:{match.group('digest')}) the build never read"
                )

    # Compare expressions as well as text hashes; an unchanged EULA could otherwise
    # be relabeled Apache-2.0.
    components = evidence_doc.get("components", {})
    # Join by component ID: source trees, archives, and libraries can share display names.
    by_id = {
        document_module.component_package_id(key): (key, component)
        for key, component in components.items()
    }
    for package in spdx.get("packages", []):
        found = by_id.get(package["SPDXID"])
        if found is None:
            continue
        key, component = found
        for field, stated in (
            ("licenseConcluded", "license_concluded"),
            ("licenseDeclared", "license_declared"),
        ):
            expected = component.get(stated)
            actual = package.get(field)
            # Shared LicenseRefs are resolved before rendering, so require exact ID agreement.
            if not expected or actual == expected:
                continue
            failures.append(
                f"{key}: the document says {field} {actual!r}, "
                f"the build recorded {expected!r}"
            )

    # Bind each LicenseRef to the combined terms recorded for its owner, or to
    # a recorded shared text. Individual file hashes cannot verify a joined body.
    by_ref = {licensing.license_ref(key): key for key in components}
    stated = {
        key: component.get("verbatim_terms_sha256")
        for key, component in components.items()
    }
    any_stated = {digest for digest in stated.values() if digest}
    for extracted in spdx.get("hasExtractedLicensingInfos", []):
        ref, text = extracted["licenseId"], extracted.get("extractedText", "")
        # A shared ID must match its text hash, not just any component's terms.
        if _SHARED_ID.fullmatch(ref) and ref != licensing.shared_ref(text):
            failures.append(f"{ref}: the text it carries is not the one its id names")
            continue
        if not any_stated:
            # Legacy evidence without this digest cannot verify the combined text.
            continue
        owner = by_ref.get(ref)
        expected = {stated[owner]} if owner and stated.get(owner) else any_stated
        if hashlib.sha256(text.encode("utf-8")).hexdigest() not in expected:
            failures.append(
                f"{ref}: the text it carries is not the terms the build read"
                + (f" for {owner}" if owner else "")
            )
    return failures


def _refs_in(value) -> set[str]:
    """Read LicenseRefs from an expression or list, failing on malformed expressions."""
    if value is None:
        return set()
    items = value if isinstance(value, list) else [value]
    found: set[str] = set()
    for item in items:
        found.update(licensing.license_refs(str(item)))
    return found


def _check_licenses(wheel: wheelfile.WheelInfo, spdx: dict) -> list[str]:
    failures: list[str] = []
    names = {item.name for item in wheel.entries}
    notices = report.notices_path(wheel.dist_info)
    if notices not in names:
        failures.append(f"{notices} is not packaged")

    declared_refs = set()
    for item in spdx.get("files", []):
        for field in ("licenseConcluded", "licenseInfoInFiles"):
            declared_refs.update(_refs_in(item.get(field)))
    for package in spdx["packages"]:
        for field in ("licenseConcluded", "licenseDeclared", "licenseInfoFromFiles"):
            declared_refs.update(_refs_in(package.get(field)))
    file_ids = {item["SPDXID"] for item in spdx.get("files", [])}
    for package in spdx["packages"]:
        for held in package.get("hasFiles", []):
            if held not in file_ids:
                failures.append(f"{package['name']} claims {held}, which is not here")

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
    """Verify a wheel, adding checks for each supplied evidence or manifest input."""
    scanned = wheelfile.scan(wheel)
    carried: list[dict] = []
    failures = verify_wheel(wheel, scanned, carried)

    if build_evidence is not None:
        evidence_doc = json.loads(build_evidence.read_text(encoding="utf-8"))
        failures.extend(
            _check_attribution(evidence_doc, scanned, carried[0] if carried else None)
        )

    if manifest is not None:
        failures.extend(_check_manifest(manifest, wheel, evidence_dir))

    return failures


def _check_manifest(
    manifest_path: Path, wheel: Path, evidence_dir: Path | None
) -> list[str]:
    """The manifest binds a wheel's digest and names the sidecars beside it."""
    failures: list[str] = []
    record = _record_for(manifest_records(manifest_path), wheel.name)
    if not record:
        return [f"{manifest_path.name} has no entry for {wheel.name}"]

    digest = sha256_file(wheel)
    if record["sha256"] != digest:
        failures.append(
            "the wheel was modified after its digest was bound: manifest says "
            f"{record['sha256']}, the file hashes to {digest}"
        )

    beside = evidence_dir if evidence_dir is not None else manifest_path.parent
    failures.extend(sidecar_failures(beside, record))

    published = beside / record["sidecars"]["spdx"]["filename"]
    if published.is_file():
        with zipfile.ZipFile(wheel) as archive:
            embedded = record["sbom_in_wheel"]
            if embedded not in archive.namelist():
                failures.append(
                    f"the manifest names {embedded} as this wheel's SBOM, and the "
                    "wheel carries no such member"
                )
            elif archive.read(embedded) != published.read_bytes():
                failures.append(
                    "the published SBOM copy differs from the one embedded in the wheel"
                )
    return failures


def _cross_document_refs(
    spdx: dict, documents: dict[str, bytes], filename: str
) -> list[str]:
    """Resolve external references against this release, checking document hashes
    and referenced elements to reject stale sibling references.
    """
    failures: list[str] = []
    by_namespace = {
        json.loads(body).get("documentNamespace"): (name, body)
        for name, body in documents.items()
    }
    for ref in spdx.get("externalDocumentRefs", []):
        found = by_namespace.get(ref.get("spdxDocument"))
        if found is None:
            failures.append(
                f"{filename}: {ref.get('externalDocumentId')} names a document "
                f"this release does not publish ({ref.get('spdxDocument')})"
            )
            continue
        name, body = found
        stated = (ref.get("checksum") or {}).get("checksumValue")
        actual = hashlib.sha1(body).hexdigest()  # noqa: S324 - SPDX 2.3 says SHA1
        if stated != actual:
            failures.append(
                f"{filename}: {ref.get('externalDocumentId')} states sha1 {stated} "
                f"for {name}, which hashes to {actual}"
            )
    ids = {
        ref["externalDocumentId"]: json.loads(by_namespace[ref["spdxDocument"]][1])
        for ref in spdx.get("externalDocumentRefs", [])
        if ref.get("spdxDocument") in by_namespace
    }
    for relationship in spdx.get("relationships", []):
        for side in ("spdxElementId", "relatedSpdxElement"):
            prefix, _, element = str(relationship[side]).partition(":")
            target = ids.get(prefix)
            if target is None or not element:
                continue
            known = {target["SPDXID"]}
            known |= {item["SPDXID"] for item in target.get("packages", [])}
            known |= {item["SPDXID"] for item in target.get("files", [])}
            if element not in known:
                failures.append(
                    f"{filename}: {relationship[side]} names an element the "
                    "referenced document does not define"
                )
    return failures


def check_set(manifest_path: Path, wheel_dir: Path, evidence_dir: Path) -> list[str]:
    """Verify the release and reject wheels or evidence omitted from its manifest."""
    failures: list[str] = []
    records = manifest_records(manifest_path)
    for name, seen in sorted(Counter(item["filename"] for item in records).items()):
        if seen > 1:
            raise ManifestError(f"{name} is advertised {seen} times")
    by_name = {item["filename"]: checked_record(item) for item in records}

    # Every downloaded wheel is published, so unlisted wheels cannot be left unchecked.
    for stray in sorted({item.name for item in wheel_dir.glob("*.whl")} - set(by_name)):
        failures.append(
            f"{stray} is in {wheel_dir}, and {manifest_path.name} does not list it"
        )
    # Reject unbound evidence too; the wheel check above handles shared-directory wheels.
    expected = set(bound_names(by_name.values()))
    if manifest_path.parent.resolve() == evidence_dir.resolve():
        expected.add(manifest_path.name)
    if evidence_dir.resolve() == wheel_dir.resolve():
        expected |= {item.name for item in wheel_dir.glob("*.whl")}
    present = {item.name for item in evidence_dir.glob("*") if item.is_file()}
    for stray in sorted(present - expected):
        failures.append(
            f"{stray} is in {evidence_dir}, and {manifest_path.name} does not bind it"
        )

    published: dict[str, bytes] = {}
    for record in by_name.values():
        sidecar = _sidecar(evidence_dir, record, "spdx")
        if sidecar:
            published[sidecar.name] = sidecar.read_bytes()

    for name, body in sorted(published.items()):
        try:
            failures.extend(_cross_document_refs(json.loads(body), published, name))
        except json.JSONDecodeError as error:
            failures.append(f"{name}: not readable as a document ({error})")

    for record in by_name.values():
        wheel = wheel_dir / record["filename"]
        if not wheel.is_file():
            failures.append(f"{record['filename']} is missing from {wheel_dir}")
            continue
        failures.extend(
            f"{record['filename']}: {item}"
            for item in check(
                wheel,
                build_evidence=_sidecar(evidence_dir, record, "build_evidence"),
                manifest=manifest_path,
                evidence_dir=evidence_dir,
            )
        )
    return failures

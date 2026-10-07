# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks that decide whether a wheel may be published.

`verify_wheel` is the consumer-side half: it needs the wheel and nothing else,
so the instructions published with a release are the same code that gates it.
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


_SIDECAR_KINDS = ("spdx", "build_evidence", "license_report_json", "license_report_md")


class ManifestError(Exception):
    """A manifest is not shaped the way this tool writes them."""


def _sidecar(evidence_dir: Path, record: dict, kind: str) -> Path | None:
    """A named sidecar, or None so the missing-file failure is reported as one."""
    named = record.get("sidecars", {}).get(kind, {}).get("filename")
    if not named:
        return None
    path = evidence_dir / named
    return path if path.is_file() else None


def _record_for(manifest: dict, filename: str) -> dict:
    """One wheel's record, or a clear error rather than a KeyError traceback."""
    try:
        records = manifest["wheels"]
    except (KeyError, TypeError) as error:
        raise ManifestError("manifest has no 'wheels' list") from error
    seen = [item for item in records if item.get("filename") == filename]
    if len(seen) > 1:
        raise ManifestError(f"{filename} is advertised {len(seen)} times")
    if not seen:
        return {}
    record = seen[0]
    for key in ("sha256", "size", "sbom_in_wheel", "sidecars"):
        if key not in record:
            raise ManifestError(f"{filename}: manifest record has no {key!r}")
    # Every sidecar, not just the ones a record happens to name. An absent entry
    # is not an absent check: omitting `build_evidence` left `check-set` passing
    # a wheel the evidence would have failed.
    missing = sorted(set(_SIDECAR_KINDS) - set(record["sidecars"]))
    if missing:
        raise ManifestError(
            f"{filename}: manifest record names no {', '.join(missing)} sidecar"
        )
    return record


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
    """Verify a wheel against the SBOM it carries. No repository required.

    `carried` collects the parsed document for callers that go on to check it
    against evidence, so the wheel is opened and the document parsed once.
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
    # An element in another document is written `DocumentRef-x:SPDXRef-y`, and
    # the prefix has to be one this document declares -- otherwise it names a
    # document no reader can fetch or check.
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
    # Exactly these two, not "at least". The list lives in the document being
    # checked, so treating it as a set of files to skip lets anyone who can edit
    # the document exempt anything they add to the wheel from the coverage check
    # below -- which is the whole of what this command promises a consumer.
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
    """Every text the document describes must be a text the wheel carries.

    The document states a digest for each licence it read and carries the
    verbatim text of each LicenseRef. The wheel packages those same bytes under
    `licenses/`. Nothing compared the two, so a packaged licence could be
    swapped for another -- or emptied -- and every per-file digest still agreed,
    because those describe the replacement.
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
            # The path matters, not just the bytes existing somewhere: two
            # components can ship an identical text, so a swap inside one of them
            # survives a check that only asks whether the digest is present.
            where = match.group("where")
            if by_name.get(where) != match.group("digest"):
                failures.append(
                    f"{package['name']}: {where} does not hold the licence text "
                    f"this document records for it (sha256:{match.group('digest')})"
                )

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


def _check_attribution(
    evidence_doc: dict, wheel: wheelfile.WheelInfo, spdx: dict | None
) -> list[str]:
    """Every wheel member must name where it came from, and every shipped
    component must have had a license text to read."""
    stated = evidence_doc.get("wheel")
    if stated and stated != wheel.path.name:
        # Checking a wheel against another wheel's evidence compares two unrelated
        # member sets; the mismatch it produces describes the wrong problem.
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
        # Every kind: the document cites one attributionText per evidence item,
        # so comparing only the licence-granting ones called a NOTICE the build
        # plainly read a text it never did.
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
        # Found is not shipped. The build gate asks whether a text existed to
        # read; this asks whether the wheel in hand still carries it. Without
        # the second, every statement the document makes about a component --
        # that it is here, under these terms, with this text -- could be deleted
        # along with the text and nothing would notice.
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
    """The texts the document names must be the ones the build actually read.

    `_check_license_texts` reads the expected digest out of the document it is
    checking, so a rewrite consistent with itself is invisible to it. The build
    evidence was produced before the wheel was published and states the digests
    independently, so where it is supplied it is the thing to compare against.
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

    # A digest is the one thing about a component that cannot change its terms.
    # Comparing only that left every expression in the document rewritable --
    # a proprietary EULA could be published as Apache-2.0, with the evidence
    # beside it still saying otherwise and every gate reporting OK.
    components = evidence_doc.get("components", {})
    # By the id the document assigns, not by display name: a source tree, a
    # system library and an archive can all be called the same thing, and a name
    # join silently drops all but one of them.
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
            if not expected or actual == expected:
                continue
            # Both unnameable is agreement: components sharing one text share one
            # LicenseRef, so the document's id is not the evidence's. The text
            # behind it is checked below, which is what the id stands for.
            if "LicenseRef-" in str(actual) and "LicenseRef-" in expected:
                continue
            failures.append(
                f"{key}: the document says {field} {actual!r}, "
                f"the build recorded {expected!r}"
            )

    # And the text a LicenseRef stands for has to be the terms the build read
    # for some component. A component can state its licence across several
    # files, so the body is a join with no digest of its own -- the evidence
    # states that digest, which is what makes every entry checkable, shared ids
    # included. An id minted for one component is held to that component's.
    by_ref = {licensing.license_ref(key): key for key in components}
    stated = {
        key: component.get("verbatim_terms_sha256")
        for key, component in components.items()
    }
    any_stated = {digest for digest in stated.values() if digest}
    for extracted in spdx.get("hasExtractedLicensingInfos", []):
        if not any_stated:
            # Evidence written before this digest existed cannot answer.
            break
        digest = hashlib.sha256(
            extracted.get("extractedText", "").encode("utf-8")
        ).hexdigest()
        owner = by_ref.get(extracted["licenseId"])
        expected = {stated[owner]} if owner and stated.get(owner) else any_stated
        if digest not in expected:
            failures.append(
                f"{extracted['licenseId']}: the text it carries is not the terms "
                "the build read" + (f" for {owner}" if owner else "")
            )
    return failures


def _refs_in(value) -> set[str]:
    """LicenseRef identifiers in a field that may hold an expression or a list.

    An expression that cannot be read is reported as such rather than treated as
    naming nothing.
    """
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

    # Every field that can name one, through one parser.
    declared_refs = set()
    for item in spdx.get("files", []):
        for field in ("licenseConcluded", "licenseInfoInFiles"):
            declared_refs.update(_refs_in(item.get(field)))
    for package in spdx["packages"]:
        for field in ("licenseConcluded", "licenseDeclared", "licenseInfoFromFiles"):
            declared_refs.update(_refs_in(package.get(field)))
    # Every file id a package claims to hold has to be a file this document has.
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
    """Check a wheel, and as much of its published evidence as you can supply.

    One operation over widening scope rather than three commands: a consumer has
    only the wheel, a release build also has the evidence it just produced, and a
    release has the whole set. Each input adds checks; none replaces the others.
    """
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = _record_for(manifest, wheel.name)
    if not record:
        return [f"{manifest_path.name} has no entry for {wheel.name}"]

    digest = sha256_file(wheel)
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
        elif sha256_file(path) != sidecar["sha256"]:
            failures.append(f"sidecar {kind} does not match its recorded digest")

    published = beside / record["sidecars"]["spdx"]["filename"]
    if published.is_file():
        with zipfile.ZipFile(wheel) as archive:
            embedded = record["sbom_in_wheel"]
            # Reported, not raised: the member name comes out of the manifest,
            # so one the wheel does not hold is a disagreement between the two
            # and reads as such, rather than as a KeyError from the zip reader.
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
    """An external document reference has to resolve to a document in this set.

    Declaring one is not the check: the digest is what binds it, and a wheel
    rebuilt without its sibling would otherwise carry a stale one and pass.
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
    # And the elements those references point at have to be in that document.
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
    """Run :func:`check` over every wheel a merged manifest advertises."""
    failures: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # The whole release is here, which is the only place a reference from one
    # document to another can be resolved rather than merely declared.
    published: dict[str, bytes] = {}
    for entry in manifest.get("wheels") or []:
        record = _record_for(manifest, entry.get("filename") or "")
        sidecar = _sidecar(evidence_dir, record, "spdx") if record else None
        if sidecar and sidecar.is_file():
            published[sidecar.name] = sidecar.read_bytes()

    for name, body in sorted(published.items()):
        try:
            failures.extend(_cross_document_refs(json.loads(body), published, name))
        except json.JSONDecodeError as error:
            failures.append(f"{name}: not readable as a document ({error})")

    for entry in manifest.get("wheels") or []:
        named = entry.get("filename")
        if not named:
            raise ManifestError("a manifest record names no wheel")
        record = _record_for(manifest, named)
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

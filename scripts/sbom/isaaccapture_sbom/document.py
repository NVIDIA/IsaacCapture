# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the discovered inventory as an SPDX 2.3 JSON document."""

from __future__ import annotations

import hashlib
import re

from packageurl import PackageURL
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from . import SPDX_VERSION, TOOL_NAME, TOOL_VERSION, licensing, report, stamped_now
from .discovery import Component
from .inventory import Inventory
from .wheelfile import WheelInfo

NOASSERTION = "NOASSERTION"
_LINK_HOWS = {"compiled-source", "header-include", "prebuilt-library"}
# Shared content identifies candidates, not proven CONTAINS/COPY_OF relationships.
_AMBIGUOUS_HOWS = {"shared-content"}


WHEEL_PACKAGE_ID = "SPDXRef-Package-wheel"


def _spdx_id(prefix: str, value: str) -> str:
    return f"SPDXRef-{prefix}-{licensing.spdx_safe(value)}"


def component_package_id(key: str) -> str:
    """The id this document gives a component. The verifier joins on it."""
    return _spdx_id("Package", key)


def file_id(path: str, dist_info: str) -> str:
    """Derive a stable ID from the path, normalizing versioned .dist-info/.data roots.

    Include a path digest to distinguish names that sanitize alike.
    """
    stem = dist_info.removesuffix(".dist-info")
    hashed = path
    for directory, token in ((dist_info, "dist-info"), (f"{stem}.data", "data")):
        if path.startswith(f"{directory}/"):
            path = token + path.removeprefix(directory)
            # No member name holds a NUL, so a real top-level `data/` hashes apart.
            hashed = "\0" + path
            break
    return _spdx_id("File", f"{path}-{licensing.sha256_text(hashed)[:8]}")


def verification_code(sha1_digests: list[str]) -> str:
    """SPDX package verification code: SHA1 over the sorted file SHA1s."""
    return hashlib.sha1("".join(sorted(sha1_digests)).encode("ascii")).hexdigest()  # noqa: S324


def _scored(match) -> str:
    license_id, containment, coverage = match
    return f"{license_id} (containment {containment:.2f}, coverage {coverage:.2f})"


def _attribution_texts(component: Component, dist_info: str) -> list[str]:
    texts = [
        f"License evidence: {item.path} (sha256:{item.sha256}, {item.size} bytes, "
        f"{item.origin}, {item.kind}) packaged at "
        # Whitespace after the path: validate.py reads it up to the first one.
        f"{report.packaged_license_path(dist_info, component.key, item)}"
        + (
            " "
            + licensing.matched_prose(
                item.matches, name=_scored, join="; ", lead="identified as "
            )
            if item.matches
            else f" — the SPDX reference text for {item.identified}, an identifier "
            "REUSE declares for this component's files; not the component's own file"
            if item.origin == licensing.SPDX_REFERENCE
            else ""
        )
        + (f" — {report.NESTED_ROLE}" if item.kind == "nested" else "")
        for item in component.evidence
    ]
    if not licensing.grant_texts(component.evidence):
        texts.append(
            "No license grant was found in this component's own files; see the "
            "component-to-license report."
        )
    return texts


def _component_package(component: Component, roles: set[str], dist_info: str) -> dict:
    package = {
        "SPDXID": component_package_id(component.key),
        "name": component.name,
        "versionInfo": component.version,
        "supplier": component.supplier,
        "downloadLocation": component.download_location,
        "filesAnalyzed": False,
        "licenseConcluded": component.license_concluded,
        "licenseDeclared": component.license_declared,
        "copyrightText": component.copyright_text,
        **(
            {
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": component.purl,
                    }
                ]
            }
            if component.purl != NOASSERTION
            else {}
        ),
        "attributionTexts": _attribution_texts(component, dist_info),
        "sourceInfo": f"{component.source_info} Reaches this wheel as: {', '.join(sorted(roles))}.",
    }
    if component.homepage != NOASSERTION:
        package["homepage"] = component.homepage
    return package


def _extracted_entries(components: list[Component]) -> list[dict]:
    """One entry per distinct text, using the shared component LicenseRef."""
    entries: dict[str, dict] = {}
    for component in components:
        carried = licensing.carried_texts(component.evidence)
        if not carried:
            continue
        body = licensing.verbatim_terms(component.evidence)
        reason = licensing.why_unnamed(component.evidence)
        for ref in licensing.license_refs(
            component.license_concluded, component.license_declared
        ):
            if ref in entries and entries[ref]["extractedText"] != body:
                raise ValueError(
                    f"{ref} was minted for two components with different terms; "
                    "a LicenseRef identifies one text"
                )
            # A shared text must cite all components and evidence paths that supply it.
            covers = set(entries.get(ref, {}).get("_for", ())) | {component.name}
            cited = sorted(
                set(entries.get(ref, {}).get("_from", ()))
                | {item.path for item in carried}
            )
            entries[ref] = {
                "licenseId": ref,
                "name": "License terms shipped with " + ", ".join(sorted(covers)),
                "_for": covers,
                "_from": cited,
                "extractedText": body,
                "comment": (
                    reason[:1].upper()
                    + reason[1:]
                    + "; reproduced here verbatim from "
                    + ", ".join(cited)
                    + "."
                ),
            }
    for entry in entries.values():
        entry.pop("_for", None)
        entry.pop("_from", None)
    return sorted(entries.values(), key=lambda item: item["licenseId"])


_EXTRA_MARKER = re.compile(r"\bextra\s*==")


def _unanalyzed_package(
    package_id: str, name: str, comment: str, external_refs: list[dict] | None = None
) -> dict:
    """Describe an external package without asserting its contents."""
    package = {
        "SPDXID": package_id,
        "name": name,
        "versionInfo": NOASSERTION,
        "supplier": NOASSERTION,
        "downloadLocation": NOASSERTION,
        "filesAnalyzed": False,
        "licenseConcluded": NOASSERTION,
        "licenseDeclared": NOASSERTION,
        "copyrightText": NOASSERTION,
        "comment": comment,
    }
    if external_refs:
        package["externalRefs"] = external_refs
    return package


def _is_extra_gated(requirement: Requirement) -> bool:
    return requirement.marker is not None and bool(
        _EXTRA_MARKER.search(str(requirement.marker))
    )


def _by_version(specifier):
    """Specifier clauses in version order, or as given if one will not parse."""
    try:
        return sorted(specifier, key=lambda item: Version(item.version))
    except InvalidVersion:
        return list(specifier)


def _dependency_refs(
    name: str, requirement: Requirement, version: str | None = None
) -> list[dict]:
    """Represent requirements with an unversioned purl and a VERS constraint.

    A purl version must be exact; it cannot encode a requirement range.
    """
    refs = [
        {
            "referenceCategory": "PACKAGE-MANAGER",
            "referenceType": "purl",
            "referenceLocator": PackageURL(
                type="pypi", name=name, version=version
            ).to_string(),
        }
    ]
    # VERS requires version ordering; sorting operators puts <1.9 before >=1.6.
    constraints = [
        f"{item.operator}{item.version}" for item in _by_version(requirement.specifier)
    ]
    if constraints:
        refs.append(
            {
                "referenceCategory": "OTHER",
                "referenceType": "vers",
                "referenceLocator": "vers:pypi/" + "|".join(constraints),
            }
        )
    return refs


def _source_info(project: dict, evidence_doc: dict, build: dict, wheel) -> str:
    """Describe source and build facts applicable to this wheel.

    Universal wheels must not inherit the build matrix's native architecture.
    """
    source = evidence_doc.get("project", {})
    parts = []
    if project["homepage"] != NOASSERTION:
        parts.append(f"Built from {project['homepage']}")
    commit = source.get("commit")
    if commit:
        described = source.get("describe") or ""
        parts.append(
            f"at commit {commit}"
            + (" with uncommitted changes" if described.endswith("-dirty") else "")
        )
    if "none-any" not in wheel.path.name:
        parts.append(f"arch {build.get('arch')}")
        parts.append(f"CMAKE_BUILD_TYPE {build.get('config')}")
    return "; ".join(parts) + "." if parts else NOASSERTION


def _wheel_conclusion(declared: str, inventory: Inventory) -> str:
    """Conjoin the distribution's declaration and every redistributed component's terms.

    Any unresolved component makes the result NOASSERTION, which cannot be AND-joined.
    """
    present = inventory.components_present.values()
    if any(item.license_concluded == NOASSERTION for item in present):
        return NOASSERTION
    # Flatten AND operands for deduplication while preserving OR choices.
    operands: set[str] = set()
    for component in present:
        operands.update(licensing.operands_of(component.license_concluded))
    if declared != NOASSERTION:
        operands.update(licensing.operands_of(declared))
    return licensing.combine(sorted(operands)) if operands else declared


def build_document(
    project: dict,
    evidence_doc: dict,
    wheel: WheelInfo,
    inventory: Inventory,
    excluded_files: list[str],
    siblings: dict[str, dict] | None = None,
) -> dict:
    """Render SPDX, rejecting file-ID collisions.

    The caller validates the result. siblings maps canonical distribution names
    to metadata from other wheels in this run.
    """
    files: list[dict] = []
    file_ids: dict[str, str] = {}
    sha1_digests: list[str] = []

    stated_in_files: set[str] = set()
    # Only first-party file notices contribute to the wheel package's own copyright.
    stated_copyright: set[str] = set()
    paths_by_id: dict[str, str] = {}
    for entry in sorted(wheel.entries, key=lambda item: item.name):
        if entry.name in excluded_files:
            continue
        spdx_id = file_id(entry.name, wheel.dist_info)
        # Paths that sanitise alike and share a 32-bit digest prefix.
        if spdx_id in paths_by_id:
            raise ValueError(
                f"{wheel.path.name}: {paths_by_id[spdx_id]} and {entry.name} "
                f"would both be {spdx_id}"
            )
        paths_by_id[spdx_id] = entry.name
        file_ids[entry.name] = spdx_id
        sha1_digests.append(entry.sha1)

        attribution = inventory.attributions.get(entry.name)
        record = {
            "SPDXID": spdx_id,
            "fileName": f"./{entry.name}",
            "checksums": [
                {"algorithm": "SHA1", "checksumValue": entry.sha1},
                {"algorithm": "SHA256", "checksumValue": entry.sha256},
            ],
            "licenseConcluded": NOASSERTION,
            "copyrightText": (attribution.copyright_text if attribution else None)
            or NOASSERTION,
        }
        if attribution and attribution.spdx_tag:
            record["licenseConcluded"] = attribution.spdx_tag
            record["licenseInfoInFiles"] = [attribution.spdx_tag]
            stated_in_files.add(attribution.spdx_tag)
        else:
            # Missing file declarations must remain visible in the package roll-up.
            stated_in_files.add(NOASSERTION)
        if (
            attribution
            and attribution.copyright_text
            and not attribution.primary
            # Packaged terms may name a component or license author, not this distribution.
            and not report.is_packaged_license(wheel.dist_info, entry.name)
        ):
            stated_copyright.add(attribution.copyright_text)

        comment = []
        if attribution:
            comment.append(f"Origin: {attribution.origin}; {attribution.detail}")
        if entry.is_elf:
            comment.append(
                f"ELF {entry.machine}; SONAME {entry.soname or 'none'}; "
                f"DT_NEEDED: {', '.join(entry.needed) or 'none'}"
            )
        if comment:
            record["comment"] = " | ".join(comment)
        files.append(record)

    code = verification_code(sha1_digests)
    wheel_id = WHEEL_PACKAGE_ID
    build = evidence_doc.get("build", {})

    # The distribution declares its license; the conclusion also includes bundled terms.
    wheel_package = {
        "SPDXID": wheel_id,
        "name": wheel.distribution,
        "versionInfo": wheel.version,
        "packageFileName": wheel.path.name,
        "supplier": project["supplier"],
        "originator": project["supplier"],
        "downloadLocation": NOASSERTION,
        "filesAnalyzed": True,
        "homepage": project["homepage"],
        "licenseConcluded": _wheel_conclusion(project["license"], inventory),
        "licenseDeclared": project["license"],
        "licenseInfoFromFiles": sorted(stated_in_files) or [NOASSERTION],
        # Metadata-only wheels need the project notice as a fallback.
        "copyrightText": "\n".join(licensing.fold_notices(stated_copyright))
        or project.get("copyright", NOASSERTION),
        "packageVerificationCode": {
            "packageVerificationCodeValue": code,
            "packageVerificationCodeExcludedFiles": [
                f"./{name}" for name in sorted(excluded_files)
            ],
        },
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": PackageURL(
                    type="pypi", name=wheel.distribution, version=wheel.version
                ).to_string(),
            }
        ],
        "hasFiles": [file_ids[name] for name in sorted(file_ids)],
        "sourceInfo": _source_info(project, evidence_doc, build, wheel),
    }

    packages = [wheel_package]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relatedSpdxElement": wheel_id,
            "relationshipType": "DESCRIBES",
        }
    ]

    for key, component in inventory.components_present.items():
        roles = inventory.component_roles.get(key, set())
        package = _component_package(component, roles, wheel.dist_info)
        packages.append(package)
        package_id = package["SPDXID"]

        if roles & _LINK_HOWS:
            relationships.append(
                {
                    "spdxElementId": wheel_id,
                    "relatedSpdxElement": package_id,
                    "relationshipType": "STATIC_LINK",
                }
            )
        if roles - _LINK_HOWS - _AMBIGUOUS_HOWS:
            relationships.append(
                {
                    "spdxElementId": wheel_id,
                    "relatedSpdxElement": package_id,
                    "relationshipType": "CONTAINS",
                }
            )

        for name in inventory.files_of(key):
            if name not in file_ids:
                continue
            attribution = inventory.attributions[name]
            exact = attribution.exact
            relationships.append(
                {
                    "spdxElementId": file_ids[name],
                    "relatedSpdxElement": package_id,
                    "relationshipType": "COPY_OF" if exact else "GENERATED_FROM",
                }
            )

        # Ambiguous files have no primary owner; encode their candidate relationships too.
        for name in inventory.files_sharing(key):
            if name in file_ids:
                relationships.append(
                    {
                        "spdxElementId": file_ids[name],
                        "relatedSpdxElement": package_id,
                        "relationshipType": "OTHER",
                    }
                )

    for soname, consumers in inventory.external_runtime.items():
        package_id = _spdx_id("Package-external", soname)
        packages.append(
            _unanalyzed_package(
                package_id,
                soname,
                "Supplied by the installation environment, not redistributed in "
                f"this wheel. Required by: {', '.join(consumers)}.",
            )
        )
        relationships.append(
            {
                "spdxElementId": wheel_id,
                "relatedSpdxElement": package_id,
                "relationshipType": "DEPENDS_ON",
            }
        )

    # Group repeated requirements by package and version range; markers belong
    # to relationships, not separate package nodes.
    wanted: dict[tuple[str, str], list[Requirement]] = {}
    for requirement in inventory.requires_dist:
        parsed = Requirement(requirement)
        wanted.setdefault((parsed.name, str(parsed.specifier)), []).append(parsed)

    for (name, _), group in sorted(wanted.items()):
        package_id = _spdx_id(
            "Package-pypi",
            f"{name}-{hashlib.sha256(str(sorted(map(str, group))).encode()).hexdigest()[:8]}",
        )
        extras = sorted({extra for item in group for extra in item.extras})
        declared = (
            "Consumer requirement declared in wheel metadata: "
            # " | ", not "; ": a requirement string contains its own semicolon
            # before the marker.
            + " | ".join(sorted(str(item) for item in group))
            + "."
            + (f" Requested with: {', '.join(extras)}." if extras else "")
        )
        sibling = (siblings or {}).get(canonicalize_name(name))
        if sibling is None:
            element = package_id
            comment = (
                f"{declared} Resolved at install time; this document states no "
                "license for it."
            )
            packages.append(
                _unanalyzed_package(
                    package_id, name, comment, _dependency_refs(name, group[0])
                )
            )
        else:
            # Use sibling metadata, not document hashes: universal wheels must be identical
            # across matrix entries, and cyclic requirements cannot have cyclic hashes.
            element = package_id
            comment = (
                f"{declared} Built by this same build; its own document "
                "describes its contents."
            )
            packages.append(
                {
                    "SPDXID": package_id,
                    "name": name,
                    "versionInfo": sibling["version"],
                    "supplier": sibling["supplier"],
                    "downloadLocation": NOASSERTION,
                    "filesAnalyzed": False,
                    "licenseConcluded": NOASSERTION,
                    "licenseDeclared": sibling["license_declared"],
                    "copyrightText": NOASSERTION,
                    "externalRefs": _dependency_refs(
                        name, group[0], version=sibling["version"]
                    ),
                    "comment": comment,
                }
            )
        # Optional only if every line that asks for it is gated on an extra. One
        # unconditional line makes the dependency unconditional.
        if all(_is_extra_gated(item) for item in group):
            relationships.append(
                {
                    "spdxElementId": element,
                    "relatedSpdxElement": wheel_id,
                    "relationshipType": "OPTIONAL_DEPENDENCY_OF",
                    "comment": comment,
                }
            )
        else:
            relationships.append(
                {
                    "spdxElementId": wheel_id,
                    "relatedSpdxElement": element,
                    "relationshipType": "DEPENDS_ON",
                    "comment": comment,
                }
            )

    stamp = stamped_now()

    extracted = _extracted_entries(list(inventory.components_present.values()))

    return {
        "spdxVersion": SPDX_VERSION,
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": wheel.path.name.removesuffix(".whl"),
        "documentNamespace": f"{project['document_namespace']}/{wheel.path.name}/{code}",
        "creationInfo": {
            "created": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "creators": [
                f"Tool: {TOOL_NAME}-{TOOL_VERSION}",
                f"Organization: {project['supplier'].split(': ', 1)[-1]}",
            ],
            # Record the corpus actually used for identification, not a default version.
            "licenseListVersion": ".".join(
                str(evidence_doc["license_list_version"]).split(".")[:2]
            ),
            "comment": (
                "Contents inventory for one built wheel, discovered from the build "
                "that produced it. Build-time-only inputs, toolchains and "
                "source-distribution contents are out of scope here."
            ),
        },
        "packages": packages,
        "files": files,
        "relationships": relationships,
        "hasExtractedLicensingInfos": extracted,
    }

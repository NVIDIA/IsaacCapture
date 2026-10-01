# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the discovered inventory as an SPDX 2.3 JSON document."""

from __future__ import annotations

import hashlib
import re
import os
from datetime import datetime, timezone

from packageurl import PackageURL
from packaging.requirements import Requirement

from . import SPDX_VERSION, TOOL_NAME, TOOL_VERSION, licensing
from .discovery import Component
from .inventory import Inventory
from .wheelfile import WheelInfo

NOASSERTION = "NOASSERTION"
# How a component reaches the wheel decides the relationship that says so.
_LINK_HOWS = {"compiled-source", "header-include", "prebuilt-library"}


def _spdx_id(prefix: str, value: str) -> str:
    return f"SPDXRef-{prefix}-{licensing.spdx_safe(value)}"


def verification_code(sha1_digests: list[str]) -> str:
    """SPDX package verification code: SHA1 over the sorted file SHA1s."""
    return hashlib.sha1("".join(sorted(sha1_digests)).encode("ascii")).hexdigest()  # noqa: S324


def _attribution_texts(component: Component) -> list[str]:
    texts = [
        f"License evidence: {item.path} (sha256:{item.sha256}, {item.size} bytes, "
        f"{item.origin}, {item.kind})"
        + (
            " identified as "
            + "; ".join(
                f"{license_id} (containment {containment:.2f}, coverage {coverage:.2f})"
                for license_id, containment, coverage in item.matches
            )
            if item.matches
            else ""
        )
        for item in component.evidence
    ]
    if not any(item.kind == "grant" for item in component.evidence):
        texts.append(
            "No license grant was found in this component's own files; see the "
            "component-to-license report."
        )
    return texts


def _component_package(component: Component, roles: set[str]) -> dict:
    package = {
        "SPDXID": _spdx_id("Package", component.key),
        "name": component.name,
        "versionInfo": component.version,
        "supplier": component.supplier,
        "downloadLocation": component.download_location,
        "filesAnalyzed": False,
        "licenseConcluded": component.license_concluded,
        "licenseDeclared": component.license_declared,
        "copyrightText": NOASSERTION,
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": component.purl,
            }
        ],
        "attributionTexts": _attribution_texts(component),
        "sourceInfo": f"{component.source_info} Reaches this wheel as: {', '.join(sorted(roles))}.",
    }
    if component.homepage != NOASSERTION:
        package["homepage"] = component.homepage
    return package


def _file_license_refs(inventory: Inventory) -> set[str]:
    """LicenseRef- identifiers that arrive through a file's own REUSE tag."""
    refs: set[str] = set()
    for attribution in inventory.attributions.values():
        if attribution.spdx_tag:
            refs.update(licensing.license_refs(attribution.spdx_tag))
    return refs


def _extracted_licenses(components: list[Component]) -> list[dict]:
    """Texts for every LicenseRef the document uses, taken from the shipped file."""
    entries: dict[str, dict] = {}
    for component in components:
        refs = licensing.license_refs(
            component.license_concluded, component.license_declared
        )
        grants = [item for item in component.evidence if item.kind == "grant"]
        if not refs or not grants:
            continue
        for ref in refs:
            entries[ref] = {
                "licenseId": ref,
                "name": f"License terms shipped with {component.name}",
                "extractedText": "\n\n".join(item.text for item in grants),
                "comment": (
                    "Matched no text in the SPDX reference corpus; reproduced here "
                    "verbatim from " + ", ".join(item.path for item in grants) + "."
                ),
            }
    return [entries[key] for key in sorted(entries)]


_EXTRA_MARKER = re.compile(r"\bextra\s*==")


def _is_extra_gated(requirement: Requirement) -> bool:
    return requirement.marker is not None and bool(
        _EXTRA_MARKER.search(str(requirement.marker))
    )


def _dependency_refs(name: str, requirement: Requirement) -> list[dict]:
    """purl for the package, and the constraint as a range rather than prose.

    purl carries one exact version, which a requirement does not have. VERS (the
    purl project's range notation) does, so the specifier stays machine-readable
    instead of living only in a sentence.
    """
    refs = [
        {
            "referenceCategory": "PACKAGE-MANAGER",
            "referenceType": "purl",
            "referenceLocator": PackageURL(type="pypi", name=name).to_string(),
        }
    ]
    constraints = sorted(
        f"{item.operator}{item.version}" for item in requirement.specifier
    )
    if constraints:
        refs.append(
            {
                "referenceCategory": "OTHER",
                "referenceType": "vers",
                "referenceLocator": "vers:pypi/" + "|".join(constraints),
            }
        )
    return refs


def build_document(
    project: dict,
    evidence_doc: dict,
    wheel: WheelInfo,
    inventory: Inventory,
    excluded_files: list[str],
) -> dict:
    """Assemble the document. Callers validate it; this only renders."""
    files: list[dict] = []
    file_ids: dict[str, str] = {}
    sha1_digests: list[str] = []

    for index, entry in enumerate(sorted(wheel.entries, key=lambda item: item.name)):
        if entry.name in excluded_files:
            continue
        file_id = _spdx_id("File", f"{index}-{entry.name.rsplit('/', 1)[-1]}")
        file_ids[entry.name] = file_id
        sha1_digests.append(entry.sha1)

        attribution = inventory.attributions.get(entry.name)
        record = {
            "SPDXID": file_id,
            "fileName": f"./{entry.name}",
            "checksums": [
                {"algorithm": "SHA1", "checksumValue": entry.sha1},
                {"algorithm": "SHA256", "checksumValue": entry.sha256},
            ],
            "licenseConcluded": NOASSERTION,
            "copyrightText": NOASSERTION,
        }
        if attribution and attribution.spdx_tag:
            # A REUSE tag in the file is upstream stating that file's license.
            record["licenseConcluded"] = attribution.spdx_tag
            record["licenseInfoInFiles"] = [attribution.spdx_tag]

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
    wheel_id = "SPDXRef-Package-wheel"
    build = evidence_doc.get("build", {})

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
        "licenseConcluded": project["license"],
        "licenseDeclared": project["license"],
        "licenseInfoFromFiles": [NOASSERTION],
        "copyrightText": NOASSERTION,
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
        "sourceInfo": (
            f"Built from {project['homepage']} at commit "
            f"{evidence_doc.get('project', {}).get('commit') or NOASSERTION}; "
            f"arch {build.get('arch')}; CMAKE_BUILD_TYPE {build.get('config')}."
        ),
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
        package = _component_package(component, roles)
        packages.append(package)
        package_id = package["SPDXID"]

        # Linked in, copied in, or both; say whichever the evidence supports.
        if roles & _LINK_HOWS:
            relationships.append(
                {
                    "spdxElementId": wheel_id,
                    "relatedSpdxElement": package_id,
                    "relationshipType": "STATIC_LINK",
                }
            )
        if roles - _LINK_HOWS:
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
            exact = attribution.origin in {"copied", "archive-copy"}
            relationships.append(
                {
                    "spdxElementId": file_ids[name],
                    "relatedSpdxElement": package_id,
                    "relationshipType": "COPY_OF" if exact else "GENERATED_FROM",
                }
            )

    for soname, consumers in inventory.external_runtime.items():
        package_id = _spdx_id("Package-external", soname)
        packages.append(
            {
                "SPDXID": package_id,
                "name": soname,
                "versionInfo": NOASSERTION,
                "supplier": NOASSERTION,
                "downloadLocation": NOASSERTION,
                "filesAnalyzed": False,
                "licenseConcluded": NOASSERTION,
                "licenseDeclared": NOASSERTION,
                "copyrightText": NOASSERTION,
                "comment": (
                    "Supplied by the installation environment, not redistributed in "
                    f"this wheel. Required by: {', '.join(consumers)}."
                ),
            }
        )
        relationships.append(
            {
                "spdxElementId": wheel_id,
                "relatedSpdxElement": package_id,
                "relationshipType": "DEPENDS_ON",
            }
        )

    for requirement in inventory.requires_dist:
        parsed = Requirement(requirement)
        name = parsed.name
        # Not the requirement string: slugifying `websockets>=14.0` yields
        # `websockets-14.0`, which reads as a pinned version beside a
        # versionInfo this document deliberately leaves unasserted. The same
        # name recurs under different extras, so uniqueness comes from a digest
        # of the whole requirement; its text is in the comment below.
        digest = hashlib.sha256(requirement.encode("utf-8")).hexdigest()[:8]
        package_id = _spdx_id("Package-pypi", f"{name}-{digest}")
        packages.append(
            {
                "SPDXID": package_id,
                "name": name,
                "versionInfo": NOASSERTION,
                "supplier": NOASSERTION,
                "downloadLocation": NOASSERTION,
                "filesAnalyzed": False,
                "licenseConcluded": NOASSERTION,
                "licenseDeclared": NOASSERTION,
                "copyrightText": NOASSERTION,
                "externalRefs": _dependency_refs(name, parsed),
                "comment": (
                    f"Consumer requirement declared in wheel metadata: {requirement}. "
                    "Resolved at install time; this document states no license for it."
                ),
            }
        )
        # An `extra ==` marker means the requirement only applies when that extra
        # is requested, which SPDX has a relationship for. Its direction is the
        # reverse of DEPENDS_ON.
        if _is_extra_gated(parsed):
            relationships.append(
                {
                    "spdxElementId": package_id,
                    "relatedSpdxElement": wheel_id,
                    "relationshipType": "OPTIONAL_DEPENDENCY_OF",
                }
            )
        else:
            relationships.append(
                {
                    "spdxElementId": wheel_id,
                    "relatedSpdxElement": package_id,
                    "relationshipType": "DEPENDS_ON",
                }
            )

    # SOURCE_DATE_EPOCH makes the whole wheel reproducible, which is what lets a
    # rebuild be compared against a published one byte for byte.
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    stamp = (
        datetime.fromtimestamp(int(epoch), timezone.utc)
        if epoch and epoch.isdigit()
        else datetime.now(timezone.utc)
    )

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
            "licenseListVersion": ".".join(
                str(evidence_doc.get("license_list_version", "3.29")).split(".")[:2]
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
        "hasExtractedLicensingInfos": _extracted_licenses(
            list(inventory.components_present.values())
        ),
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the discovered inventory as an SPDX 2.3 JSON document."""

from __future__ import annotations

import hashlib
import re

from packageurl import PackageURL
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from . import SPDX_VERSION, TOOL_NAME, TOOL_VERSION, licensing, report, stamped_now
from .discovery import Component
from .inventory import Inventory
from .wheelfile import WheelInfo

NOASSERTION = "NOASSERTION"
# How a component reaches the wheel decides the relationship that says so.
_LINK_HOWS = {"compiled-source", "header-include", "prebuilt-library"}
# Bytes this component also holds, which is not evidence the wheel carries its
# file. It states a candidate, so it earns no CONTAINS and no COPY_OF.
_AMBIGUOUS_HOWS = {"shared-content"}


# The package every document describes: the wheel it is about.
WHEEL_PACKAGE_ID = "SPDXRef-Package-wheel"


def _spdx_id(prefix: str, value: str) -> str:
    return f"SPDXRef-{prefix}-{licensing.spdx_safe(value)}"


def component_package_id(key: str) -> str:
    """The id this document gives a component. The verifier joins on it."""
    return _spdx_id("Package", key)


def verification_code(sha1_digests: list[str]) -> str:
    """SPDX package verification code: SHA1 over the sorted file SHA1s."""
    return hashlib.sha1("".join(sorted(sha1_digests)).encode("ascii")).hexdigest()  # noqa: S324


def _attribution_texts(component: Component, dist_info: str) -> list[str]:
    texts = [
        f"License evidence: {item.path} (sha256:{item.sha256}, {item.size} bytes, "
        f"{item.origin}, {item.kind}) packaged at "
        f"{report.packaged_license_path(dist_info, component.key, item)}"
        + (
            " identified as "
            + "; ".join(
                f"{license_id} (containment {containment:.2f}, coverage {coverage:.2f})"
                for license_id, containment, coverage in item.matches
            )
            if item.matches
            # No text of its own to package: this is the SPDX reference text for
            # the identifier the component's files state about themselves, which
            # is a substitution and not a reading of the component.
            else f" the SPDX reference text for {item.identified}, substituted "
            "because this component ships no license file"
            if item.origin == licensing.SPDX_REFERENCE
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
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": component.purl,
            }
        ],
        "attributionTexts": _attribution_texts(component, dist_info),
        "sourceInfo": f"{component.source_info} Reaches this wheel as: {', '.join(sorted(roles))}.",
    }
    if component.homepage != NOASSERTION:
        package["homepage"] = component.homepage
    return package


def _extracted_entries(components: list[Component]) -> list[dict]:
    """One entry per distinct text, named by the id the components now share."""
    entries: dict[str, dict] = {}
    for component in components:
        refs = licensing.license_refs(
            component.license_concluded, component.license_declared
        )
        grants = licensing.obligation_texts(component.evidence)
        if not refs or not grants:
            continue
        body = licensing.verbatim_terms(component.evidence)
        for ref in refs:
            if ref in entries and entries[ref]["extractedText"] != body:
                raise ValueError(
                    f"{ref} was minted for two components with different terms; "
                    "a LicenseRef identifies one text"
                )
            shipped = ", ".join(item.path for item in grants)
            entries[ref] = {
                "licenseId": ref,
                "name": "License terms shipped with "
                + ", ".join(
                    sorted(
                        {entries[ref]["_for"], component.name}
                        if ref in entries
                        else {component.name}
                    )
                ),
                "_for": component.name,
                "extractedText": body,
                "comment": (
                    (
                        "Matched no text in the SPDX reference corpus"
                        if not any(item.matches for item in grants)
                        # Matched, but under an id the grammar shipped with this
                        # document cannot validate, so it cannot be named here.
                        else "Matched an identifier newer than the SPDX grammar "
                        "this document ships with"
                    )
                    + f"; reproduced here verbatim from {shipped}."
                ),
            }
    for entry in entries.values():
        entry.pop("_for", None)
    return sorted(entries.values(), key=lambda item: item["licenseId"])


_EXTRA_MARKER = re.compile(r"\bextra\s*==")


def _unanalyzed_package(
    package_id: str, name: str, comment: str, external_refs: list[dict] | None = None
) -> dict:
    """A package this document names but did not open: everything unasserted."""
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


def _dependency_refs(
    name: str, requirement: Requirement, version: str | None = None
) -> list[dict]:
    """purl for the package, and the constraint as a range rather than prose.

    purl carries one exact version, which a requirement does not have. VERS (the
    purl project's range notation) does, so the specifier stays machine-readable
    instead of living only in a sentence.
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


def _source_info(project: dict, evidence_doc: dict, build: dict, wheel) -> str:
    """Where this wheel came from, stating only what applies to it.

    One build produces several wheels, so its architecture and configuration are
    facts about the compilation -- which a pure-python wheel did not have. A
    placeholder rendered as a noun ("Built from NOASSERTION") is worse than an
    omitted clause.
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
            # A tree with uncommitted changes did not produce what that commit
            # produces, and a report nothing can reproduce has to say so.
            + (" with uncommitted changes" if described.endswith("-dirty") else "")
        )
    # `none-any` is the tag for a wheel with nothing compiled in it.
    if "none-any" not in wheel.path.name:
        parts.append(f"arch {build.get('arch')}")
        parts.append(f"CMAKE_BUILD_TYPE {build.get('config')}")
    return "; ".join(parts) + "." if parts else NOASSERTION


def _wheel_conclusion(declared: str, inventory: Inventory) -> str:
    """The declared licence, AND the terms of everything redistributed under its own.

    A component whose terms the build could not name travels as a LicenseRef,
    and those are exactly the ones a reader must not miss -- a proprietary SDK
    EULA among them. Components under a licence the declaration already covers
    add nothing and are left out.
    """
    others = sorted(
        {
            component.license_concluded
            for component in inventory.components_present.values()
            if "LicenseRef-" in component.license_concluded
        }
    )
    if not others:
        return declared
    if declared == NOASSERTION:
        return licensing.combine(others)
    return licensing.combine([declared, *others])


def build_document(
    project: dict,
    evidence_doc: dict,
    wheel: WheelInfo,
    inventory: Inventory,
    excluded_files: list[str],
    siblings: dict[str, dict] | None = None,
) -> dict:
    """Assemble the document. Callers validate it; this only renders.

    `siblings` are the other distributions of this same run, by canonical name.
    A requirement naming one is not an install-time unknown: the build made it,
    and its own document describes it.
    """
    files: list[dict] = []
    file_ids: dict[str, str] = {}
    external_documents: list[dict] = []
    sha1_digests: list[str] = []

    # What the files themselves state, for the package-level roll-up below.
    stated_in_files: set[str] = set()
    # Notices stated by files this distribution's own, for the package below. A
    # file attributed to a component carries that component's notice, which
    # belongs to that component's package and not to this one.
    stated_copyright: set[str] = set()
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
            # The notice sits directly above the identifier already parsed, so
            # declaring none would discard something read and in hand.
            "copyrightText": (attribution.copyright_text if attribution else None)
            or NOASSERTION,
        }
        if attribution and attribution.spdx_tag:
            # A REUSE tag in the file is upstream stating that file's license.
            record["licenseConcluded"] = attribution.spdx_tag
            record["licenseInfoInFiles"] = [attribution.spdx_tag]
            stated_in_files.add(attribution.spdx_tag)
        if (
            attribution
            and attribution.copyright_text
            and not attribution.primary
            # A packaged licence text states its component's notice, or its own
            # author's -- WTFPL names the person who wrote WTFPL. Neither is a
            # notice this distribution makes about itself.
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
        # Declared is what the distribution says of itself. Concluded is what
        # this document found in it, and a wheel carrying a component under
        # other terms is not wholly under the declared licence -- saying so is
        # the field a compliance tool reads for "what is this artifact".
        "licenseConcluded": _wheel_conclusion(project["license"], inventory),
        "licenseDeclared": project["license"],
        "licenseInfoFromFiles": sorted(stated_in_files) or [NOASSERTION],
        # The notices its own files carry. Declaring none beside a supplier, a
        # licence and files that each state one left the package saying less
        # about itself than anything in it.
        "copyrightText": "\n".join(licensing.fold_notices(stated_copyright))
        or NOASSERTION,
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

        # Linked in, copied in, or both; say whichever the evidence supports.
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

        # A file whose bytes several components hold names no single source, so
        # it has no primary and `files_of` does not see it. Stating the candidates
        # only in prose would leave it attached to nothing any tool reads.
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

    # One node per package and version range, not per requirement line. A
    # dependency recurs once per extra that wants it, and the repeats differ
    # only in a marker the relationship already carries.
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
        elif sibling.get("namespace"):
            # Built by this same run and already described, so the dependency
            # points straight at the package in that wheel's own document. A
            # local stub beside it would restate version, supplier and licence
            # from the same source.
            document_ref = f"DocumentRef-{licensing.spdx_safe(sibling['filename'])}"
            external_documents.append(
                {
                    "externalDocumentId": document_ref,
                    "spdxDocument": sibling["namespace"],
                    "checksum": {
                        "algorithm": "SHA1",
                        "checksumValue": sibling["sha1"],
                    },
                }
            )
            element = f"{document_ref}:{WHEEL_PACKAGE_ID}"
            comment = (
                f"{declared} Built by this same build as {sibling['name']} "
                f"{sibling['version']}; described by {sibling['filename']}."
            )
        else:
            # Built by this same run but not yet described: two wheels can
            # require each other, and a document citing another's digest cannot
            # be written before it. What the wheel states about itself needs no
            # document, so none of it is left unasserted.
            element = package_id
            comment = (
                f"{declared} Built by this same build; its own document "
                f"describes it, and is not referenced here because the two "
                f"wheels require each other."
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

    # SOURCE_DATE_EPOCH makes the whole wheel reproducible, which is what lets a
    # rebuild be compared against a published one byte for byte.
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
            # From the corpus this build identified against, never a default: a
            # stated version the document was not built against is a false claim
            # about every identification in it.
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
        **({"externalDocumentRefs": external_documents} if external_documents else {}),
    }

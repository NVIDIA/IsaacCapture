# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component-to-license report, and the notices packaged inside the wheel."""

from __future__ import annotations

import re

from . import licensing
from .inventory import Inventory

NOASSERTION = "NOASSERTION"

_HOW_PROSE = {
    "compiled-source": "its sources were compiled into a shipped binary",
    "header-include": "its headers were compiled into a shipped binary",
    "prebuilt-library": "a pre-built library of its was linked in",
    "copied-file": "files were copied in unchanged",
    "derived-file": "files were copied in and modified by the build",
    "extracted-file": "files were unpacked from its archive",
    "vendored-library": "the library was vendored in by auditwheel",
    "shared-content": (
        "a shipped file holds bytes this component also holds, and so do others"
    ),
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

# Distinguish vendored copyright holders from installable packages.
_KIND_PROSE = {
    "source-tree": "fetched source",
    "archive": "fetched archive",
    "system-library": "build machine",
    "vendored-source": "vendored source",
    "vcpkg-port": "vcpkg port",
}


_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# Shared gap categories for JSON, Markdown, manifests, and CI warnings.
GAP_LISTS = {
    "components_without_license_evidence": "No license text found for:",
    "components_with_unresolved_license": "No license concluded for:",
    "components_with_unidentified_license": "License text carried verbatim:",
}


def _short(version: str) -> str:
    """Abbreviate commit hashes only; truncating a package version changes its identity."""
    return version[:12] if _COMMIT.match(version) else version


def licenses_root(dist_info: str) -> str:
    """Where a wheel carries licence texts. PEP 639 names them relative to this."""
    return f"{dist_info}/licenses/"


def notices_path(dist_info: str) -> str:
    return f"{licenses_root(dist_info)}THIRD-PARTY-NOTICES.md"


def is_packaged_license(dist_info: str, member: str) -> bool:
    return member.startswith(licenses_root(dist_info))


def is_third_party_license(dist_info: str, member: str) -> bool:
    return member.startswith(f"{licenses_root(dist_info)}third-party/")


def declared_license_path(dist_info: str, member: str) -> str:
    """PEP 639 License-File paths are relative to .dist-info/licenses/."""
    return member.removeprefix(licenses_root(dist_info))


# Nested texts may repeat the root grant; describe their scope without implying a difference.
NESTED_ROLE = "terms stated for part of this component"


def packaged_license_path(dist_info: str, component_key: str, evidence) -> str:
    """Place terms under the component's full evidence path to avoid basename collisions."""
    folder = _SAFE_NAME.sub("_", component_key)
    relative = evidence.path.rsplit("!", 1)[-1].split("/", 1)[-1]
    # Preserve path separators when sanitizing to avoid flattening distinct paths.
    safe = "/".join(_SAFE_NAME.sub("_", part) for part in relative.split("/"))
    return f"{licenses_root(dist_info)}third-party/{folder}/{safe}"


def _roles(inventory: Inventory, key: str) -> str:
    hows = sorted(inventory.component_roles.get(key, set()))
    return "; ".join(_HOW_PROSE.get(how, how) for how in hows) or "unrecorded"


def notices_markdown(
    inventory: Inventory,
    dist_info: str,
    wheel_name: str,
    project: dict | None = None,
) -> str:
    """Render notices with the distribution's own license and copyright first.

    Reference templates do not name the holder of a metadata-only wheel.
    """
    lines = ["# Notices", ""]
    if project:
        supplier = project["supplier"].split(": ", 1)[-1]
        lines += [f"`{wheel_name}` is distributed by {supplier}."]
        if project.get("license", NOASSERTION) != NOASSERTION:
            lines.append(f"It is licensed under `{project['license']}`.")
        if project.get("copyright", NOASSERTION) != NOASSERTION:
            lines.append(f"{project['copyright']}")
        lines.append("")
    lines += ["## Third-party notices", ""]
    if not inventory.components_present:
        lines += [
            f"`{wheel_name}` redistributes no third-party component, so no "
            "license text is packaged here.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        f"Components redistributed in `{wheel_name}`, with the license text each",
        "obligation was read from. Texts are packaged under",
        f"`{licenses_root(dist_info)}third-party/`.",
        "",
    ]
    for key, component in inventory.components_present.items():
        lines.append(f"### {component.name}")
        lines.append("")
        lines.append(f"- License: `{component.license_concluded}`")
        if component.copyright_text != "NOASSERTION":
            # Reference texts omit component holders. Render their notices separately,
            # with explicit breaks so Markdown does not merge them into one paragraph.
            lines.append(
                "- Copyright: " + "<br>".join(component.copyright_text.splitlines())
            )
        if component.license_declared != component.license_concluded:
            lines.append(f"- Declared: `{component.license_declared}`")
        if component.homepage != "NOASSERTION":
            lines.append(f"- Homepage: {component.homepage}")
        if component.version != "NOASSERTION":
            lines.append(f"- Version: `{component.version}`")
        lines.append(f"- Reaches this wheel because {_roles(inventory, key)}")
        for item in component.evidence:
            packaged = packaged_license_path(dist_info, key, item)
            lines.append(
                f"- Text: `{packaged}`"
                + (
                    " — the SPDX reference text for this "
                    + (
                        "exception"
                        if licensing.listed_exception(item.identified or "")
                        else "licence"
                    )
                    + ", not this component's own file"
                    if item.origin == licensing.SPDX_REFERENCE
                    else f" — {NESTED_ROLE}"
                    if item.kind == "nested"
                    else ""
                )
            )
        if not component.evidence:
            lines.append("- No license text was found in this component's own files.")
        lines.append("")
    return "\n".join(lines)


def _carried_subject(carried: int, total: int) -> str:
    """How many of a component's grants travel verbatim, as a sentence subject."""
    if carried == total == 1:
        return "the shipped grant is"
    if carried == total:
        return f"its {total} shipped grants are"
    return f"{carried} of its {total} shipped grants {'is' if carried == 1 else 'are'}"


def _carried_reason(evidence) -> str:
    """Which of a component's grants travel verbatim, whether their ids are
    named too, and why they travel."""
    carried = licensing.carried_texts(evidence)
    named = len(licensing.carried_and_named(evidence))
    subject = _carried_subject(len(carried), len(licensing.grant_texts(evidence)))
    manner = (
        " rather than named"
        if not named
        else " as well as named"
        if named == len(carried)
        else f", {named} of them as well as named"
    )
    return f"{subject} carried verbatim{manner}: {licensing.why_unnamed(evidence)}"


def report(inventory: Inventory, evidence_doc: dict, wheel_name: str) -> dict:
    """Machine-readable component-to-license report, including what is missing."""
    components = []
    gaps: dict[str, list[dict]] = {key: [] for key in GAP_LISTS}

    for key, component in inventory.components_present.items():
        obligations = licensing.obligation_texts(component.evidence)
        components.append(
            {
                "component": key,
                "name": component.name,
                "kind": component.kind,
                "version": component.version,
                "how_it_reaches_the_wheel": sorted(
                    inventory.component_roles.get(key, set())
                ),
                "license_concluded": component.license_concluded,
                "license_declared": component.license_declared,
                "supplier": component.supplier,
                "homepage": component.homepage,
                "purl": component.purl,
                "source_info": component.source_info,
                "wheel_files": inventory.files_of(key),
                "evidence": [item.as_json() for item in component.evidence],
            }
        )
        if not obligations:
            gaps["components_without_license_evidence"].append(
                {
                    "component": key,
                    "reason": "no license grant found in the component's own files",
                }
            )
        elif component.license_concluded == NOASSERTION:
            # Packaged text can satisfy the gate without identifying a license.
            gaps["components_with_unresolved_license"].append(
                {
                    "component": key,
                    "reason": licensing.why_unresolved(component.evidence),
                    "evidence": [item.path for item in obligations],
                }
            )
        elif carried := licensing.carried_texts(component.evidence):
            # A partially matched grant still carries unresolved terms.
            gaps["components_with_unidentified_license"].append(
                {
                    "component": key,
                    "reason": _carried_reason(component.evidence),
                    "evidence": [item.path for item in carried],
                }
            )

    return {
        "schema": "isaaccapture-license-report/2",
        "wheel": wheel_name,
        "generated_from_commit": evidence_doc.get("project", {}).get("commit"),
        "source_tree_modified": str(
            evidence_doc.get("project", {}).get("describe") or ""
        ).endswith("-dirty"),
        "spdx_license_list_version": evidence_doc.get("license_list_version"),
        # Keep builder-local corpus paths in build evidence only.
        "license_data": {
            key: value
            for key, value in evidence_doc.get("license_data", {}).items()
            if key != "source"
        },
        "components": components,
        "externally_supplied_runtime": [
            {"soname": soname, "required_by": consumers}
            for soname, consumers in inventory.external_runtime.items()
        ],
        "consumer_requirements": inventory.requires_dist,
        **gaps,
    }


def report_markdown(payload: dict) -> str:
    """Human-readable twin of :func:`report`."""
    lines = [
        f"# Component-to-license report — `{payload['wheel']}`",
        "",
        f"Source commit: `{payload['generated_from_commit'] or 'unknown'}`"
        + (
            " (built from a tree with uncommitted changes)"
            if payload.get("source_tree_modified")
            else ""
        )
        + "  ",
        f"SPDX license list: `{payload['spdx_license_list_version'] or 'unknown'}` "
        f"(spdx/license-list-data `{payload.get('license_data', {}).get('commit', '?')[:12]}`)",
        "",
        "Every row was discovered from the build that produced this wheel. Nothing",
        "here is declared in a checked-in list.",
        "",
        "## Redistributed components",
        "",
        "| Component | Kind | Version | License | How it reaches the wheel | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload["components"]:
        # Summarize reference pools for readability; JSON and packaged notices retain all paths.
        shown = [record for record in item["evidence"] if record["kind"] != "pool"]
        pooled = [record for record in item["evidence"] if record["kind"] == "pool"]
        parts = [
            f"`{record['path']}`"
            + (f" → {record['identified']}" if record["identified"] else "")
            + (f" ({NESTED_ROLE})" if record["kind"] == "nested" else "")
            + (
                " (SPDX reference text, substituted)"
                if record["origin"] == licensing.SPDX_REFERENCE
                else ""
            )
            for record in shown
        ]
        if pooled:
            names = licensing.pool_names(record["path"] for record in pooled)
            parts.append(f"REUSE pool ({len(pooled)}): {names}")
        evidence = "<br>".join(parts) or "—"
        how = "; ".join(
            _HOW_PROSE.get(value, value) for value in item["how_it_reaches_the_wheel"]
        )
        version = item["version"]
        lines.append(
            f"| {item['name']} | {_KIND_PROSE.get(item['kind'], item['kind'])} | "
            f"`{_short(version)}` | `{item['license_concluded']}` | "
            f"{how or '—'} | {evidence} |"
        )

    lines += ["", "## Externally supplied at runtime", ""]
    if payload["externally_supplied_runtime"]:
        lines += [
            "Resolved from the installation environment; not redistributed here.",
            "",
        ]
        lines += [
            f"- `{item['soname']}` — required by {', '.join(item['required_by'])}"
            for item in payload["externally_supplied_runtime"]
        ]
    else:
        lines.append("None recorded.")

    lines += ["", "## License texts packaged in this wheel", ""]
    lines += [
        f"- `{item}`" for item in payload.get("packaged_license_files") or []
    ] or ["None."]
    if payload.get("packaged_license_files"):
        lines += [
            "",
            "Declared in METADATA as `License-File`: "
            + ("yes" if payload.get("metadata_license_files_declared") else "no"),
        ]

    lines += ["", "## Consumer requirements (wheel metadata)", ""]
    lines += [f"- `{item}`" for item in payload["consumer_requirements"]] or [
        "None declared."
    ]

    lines += ["", "## License evidence gaps", ""]
    listed = [key for key in GAP_LISTS if payload[key]]
    for key in listed:
        lines += [GAP_LISTS[key], ""]
        lines += [
            f"- `{item['component']}` — {item['reason']}" for item in payload[key]
        ]
        lines.append("")
    if not listed:
        lines.append(
            "None; every redistributed component ships an identified license."
            if payload["components"]
            else "This distribution redistributes no third-party component."
        )

    return "\n".join(lines) + "\n"

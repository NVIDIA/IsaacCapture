# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component-to-license report, and the notices packaged inside the wheel."""

from __future__ import annotations

import re

from . import licensing
from .inventory import Inventory

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


_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _short(version: str) -> str:
    """Abbreviate a commit; leave anything else whole.

    Truncating to a fixed width turned `1.1.0-2build1.1` into `1.1.0-2build`,
    which reads as a complete Debian version and is not one.
    """
    return version[:12] if _COMMIT.match(version) else version


def licenses_root(dist_info: str) -> str:
    """Where a wheel carries licence texts. PEP 639 names them relative to this."""
    return f"{dist_info}/licenses/"


def notices_path(dist_info: str) -> str:
    """The notice file that travels inside the wheel, by name."""
    return f"{licenses_root(dist_info)}THIRD-PARTY-NOTICES.md"


def is_packaged_license(dist_info: str, member: str) -> bool:
    return member.startswith(licenses_root(dist_info))


def is_third_party_license(dist_info: str, member: str) -> bool:
    """A text read out of a component, as against this distribution's own."""
    return member.startswith(f"{licenses_root(dist_info)}third-party/")


def declared_license_path(dist_info: str, member: str) -> str:
    """What METADATA calls a packaged licence file.

    PEP 639 states License-File relative to `.dist-info/licenses/`, which is
    what setuptools already does for the project's own texts; declaring the
    member path instead yields an entry that resolves to nothing.
    """
    return member.removeprefix(licenses_root(dist_info))


def packaged_license_path(dist_info: str, component_key: str, evidence) -> str:
    """Where one component's license text lands inside the wheel.

    Named after the evidence's whole path within its component, not its
    basename: a REUSE pool makes `LICENSE` and `LICENSES/MIT.txt` collide, and
    dropping one would leave the notices pointing at a text that is not there.
    """
    folder = _SAFE_NAME.sub("_", component_key)
    relative = evidence.path.rsplit("!", 1)[-1].split("/", 1)[-1]
    # Keep the separators: flattening them re-created the collision this whole
    # scheme exists to avoid, and the loser was dropped while the document went
    # on naming the path it was dropped from.
    safe = "/".join(_SAFE_NAME.sub("_", part) for part in relative.split("/"))
    return f"{licenses_root(dist_info)}third-party/{folder}/{safe}"


def _roles(inventory: Inventory, key: str) -> str:
    hows = sorted(inventory.component_roles.get(key, set()))
    return "; ".join(_HOW_PROSE.get(how, how) for how in hows) or "unrecorded"


def notices_markdown(inventory: Inventory, dist_info: str, wheel_name: str) -> str:
    """The notice file that travels inside the wheel."""
    lines = ["# Third-party notices", ""]
    if not inventory.components_present:
        # Saying where texts live when none were written sends a reader to a
        # directory the wheel does not contain.
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
        lines.append(f"## {component.name}")
        lines.append("")
        lines.append(f"- License: `{component.license_concluded}`")
        if component.copyright_text != "NOASSERTION":
            # A reference text names no holder, so for a component that ships no
            # licence file of its own this is the only notice in the wheel. One
            # per line, indented: a bare newline inside a list item is a lazy
            # continuation and renders several notices as one paragraph.
            notices = component.copyright_text.splitlines()
            lines.append(f"- Copyright: {notices[0]}")
            lines.extend(f"  {item}" for item in notices[1:])
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
                    " — the SPDX reference text for this licence, not this "
                    "component's own file, which it ships none of"
                    if item.origin == licensing.SPDX_REFERENCE
                    else ""
                )
            )
        if not component.evidence:
            lines.append("- No license text was found in this component's own files.")
        lines.append("")
    return "\n".join(lines)


def report(inventory: Inventory, evidence_doc: dict, wheel_name: str) -> dict:
    """Machine-readable component-to-license report, including what is missing."""
    components = []
    no_evidence = []
    unidentified = []

    for key, component in inventory.components_present.items():
        grants = licensing.obligation_texts(component.evidence)
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
        # Same test as the build gate and the verifier: a REUSE pool is a real
        # text. Requiring a grant here advertised "no licence evidence" for a
        # component the gates deliberately accepted on the pool they packaged.
        if not grants:
            no_evidence.append(
                {
                    "component": key,
                    "reason": "no license grant found in the component's own files",
                }
            )
        elif "LicenseRef-" in component.license_concluded:
            # Keyed on the conclusion, not on whether anything matched: a text
            # can match and still be unusable, when the id it matched is newer
            # than the expression grammar a consumer validates with. Either way
            # the terms are only readable as the text this wheel carries.
            unidentified.append(
                {
                    "component": key,
                    "reason": (
                        "the shipped grant is carried verbatim rather than named: "
                        + licensing.why_unnamed(grants).lower()
                    ),
                    "evidence": [item.path for item in grants],
                }
            )

    return {
        "schema": "isaaccapture-license-report/2",
        "wheel": wheel_name,
        "generated_from_commit": evidence_doc.get("project", {}).get("commit"),
        # The commit alone does not describe a tree that was modified, and this
        # report is what a reader uses to reproduce the build.
        "source_tree_modified": str(
            evidence_doc.get("project", {}).get("describe") or ""
        ).endswith("-dirty"),
        "spdx_license_list_version": evidence_doc.get("license_list_version"),
        "license_data": evidence_doc.get("license_data", {}),
        "components": components,
        "externally_supplied_runtime": [
            {"soname": soname, "required_by": consumers}
            for soname, consumers in inventory.external_runtime.items()
        ],
        "consumer_requirements": inventory.requires_dist,
        "exclusions": [
            {"component": key, "reason": reason}
            for key, reason in sorted(inventory.absent_components.items())
        ],
        "components_without_license_evidence": no_evidence,
        "components_with_unidentified_license": unidentified,
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
        "| Component | Version | License | How it reaches the wheel | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in payload["components"]:
        evidence = (
            "<br>".join(
                f"`{record['path']}`"
                + (f" → {record['identified']}" if record["identified"] else "")
                + (
                    " (SPDX reference text, substituted)"
                    if record["origin"] == licensing.SPDX_REFERENCE
                    else ""
                )
                for record in item["evidence"]
            )
            or "—"
        )
        how = "; ".join(
            _HOW_PROSE.get(value, value) for value in item["how_it_reaches_the_wheel"]
        )
        version = item["version"]
        lines.append(
            f"| {item['name']} | `{_short(version)}` | `{item['license_concluded']}` | "
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

    lines += ["", "## Fetched for this build but not in this wheel", ""]
    if payload["exclusions"]:
        lines += [
            f"- `{item['component']}` — {item['reason']}"
            for item in payload["exclusions"]
        ]
    else:
        lines.append("None.")

    lines += ["", "## License evidence gaps", ""]
    gaps = payload["components_without_license_evidence"]
    unidentified = payload["components_with_unidentified_license"]
    if gaps:
        lines += ["No license text found for:", ""]
        lines += [f"- `{item['component']}` — {item['reason']}" for item in gaps]
        lines.append("")
    if unidentified:
        lines += ["License text shipped but not identified:", ""]
        lines += [
            f"- `{item['component']}` — {item['reason']}" for item in unidentified
        ]
        lines.append("")
    if not gaps and not unidentified:
        lines.append(
            "None; every redistributed component ships an identified license."
            if payload["components"]
            else "This distribution redistributes no third-party component."
        )

    return "\n".join(lines) + "\n"

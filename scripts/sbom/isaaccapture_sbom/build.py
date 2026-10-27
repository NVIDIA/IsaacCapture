# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package a wheel's contents SBOM and license evidence from its build tree."""

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
from . import validate as validate_module
from . import wheelfile
from packaging.utils import canonicalize_name

from . import discovery as discovery_module
from .discovery import (
    PREBUILT,
    Reach,
    covers,
    project_copyright,
    project_supplier,
    reach_rank,
    sha256_file,
)
from .document import NOASSERTION
from .inventory import Attribution
from .wheelfile import Entry

# Only the document namespace is fixed; distribution identity comes from metadata.
PROJECT = {"document_namespace": "https://github.com/NVIDIA/IsaacCapture/spdx"}


def _project_facts(repo_root: Path, metadata: dict) -> dict:
    urls = metadata.get("project_urls") or {}
    homepage = next(
        (
            value
            for key, value in urls.items()
            if key.lower() in ("homepage", "home-page", "source", "repository")
        ),
        metadata.get("home_page") or NOASSERTION,
    )
    # A sibling distribution may have a different author than the root project.
    author = (metadata.get("author") or "").strip()
    return {
        **PROJECT,
        "supplier": f"Organization: {author}"
        if author
        else project_supplier(repo_root),
        # The legacy License field is free text, not an SPDX expression.
        "license": metadata.get("license_expression") or NOASSERTION,
        "homepage": homepage,
        # Metadata-only wheels have no source file carrying the project notice.
        "copyright": project_copyright(repo_root),
    }


class BuildError(Exception):
    """The wheel cannot be published with the evidence available."""


_REMEDIES = "\nSee 'What stops the Release build' in scripts/sbom/README.md."


def _entry(name: str, data: bytes) -> Entry:
    return Entry(
        name=name,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        sha1=hashlib.sha1(data).hexdigest(),  # noqa: S324 - SPDX 2.3 requires SHA1
    )


def default_license_data(build_dir: Path) -> Path:
    """Where deps/third_party materializes the SPDX license list for this build."""
    return build_dir / "_deps" / "license-list-data-src" / "json"


def _with_reached_nested(inventory, components: dict) -> dict:
    """Keep nested terms only for files or compiled code that reached this wheel."""
    # Deduplicate reach shared by several members, retaining the first member as evidence.
    reached: dict[str, dict[tuple[str, str], Reach]] = {}
    # A prebuilt archive conservatively retains all nested terms of its component.
    prebuilt: dict[str, str] = {}
    for member, attribution in inventory.attributions.items():
        for key, paths in attribution.read.items():
            into = reached.setdefault(key, {})
            for path, how in paths.items():
                known = into.get((path, how))
                if known is None or member < known.member:
                    into[(path, how)] = Reach(how, path, member)
        for key, roles in attribution.components.items():
            if PREBUILT in roles:
                prebuilt[key] = min(member, prebuilt.get(key, member))

    def strength(hit: Reach):
        return reach_rank(hit.how), hit.path, hit.member

    updated = {}
    for key, component in components.items():
        if not any(item.kind == "nested" for item in component.evidence):
            continue
        evidence = []
        for item in component.evidence:
            if item.kind != "nested":
                evidence.append(item)
                continue
            hits = [
                hit
                for hit in reached.get(key, {}).values()
                if covers(hit.path, hit.how, item.scope)
            ]
            if key in prebuilt:
                whole = component.root.name if component.root else key
                hits.append(Reach(PREBUILT, whole, prebuilt[key]))
            if hits:
                evidence.append(replace(item, reached_by=min(hits, key=strength)))
        updated[key] = replace(
            component,
            evidence=tuple(evidence),
            copyright_text=discovery_module.notice_from(evidence),
        )
    return updated


def _declared_without_text(declared: str, shipped: set[str]) -> list[str]:
    """Identifiers in the declared expression with no packaged text to match."""
    if declared == NOASSERTION:
        return []
    try:
        identifiers = licensing.identifiers_in(declared)
    except licensing.ExpressionReadError:
        # Malformed declarations must fail the missing-text gate.
        return [declared]
    return sorted(
        item
        for item in identifiers
        # Exceptions modify a license; this gate requires texts only for license operands.
        if not licensing.is_exception(item) and f"{item.lower()}.txt" not in shipped
    )


def _own_texts(shipped: set[str]) -> set[str]:
    """Return paths relative to .dist-info/licenses/, excluding third-party terms.

    A component's text must not discharge the distribution's own obligation: removing
    that component would also remove the distribution's license.
    """
    return {item for item in shipped if not item.startswith("third-party/")}


def _own_license_texts(declared: str, shipped: set[str]) -> dict[str, str]:
    """Supply reference texts missing from the distribution's declared licenses.

    Third-party terms must instead come from the components' own files.
    """
    if declared == NOASSERTION:
        return {}
    names = {Path(item).name.lower() for item in _own_texts(shipped)}
    wanted = {}
    for item in licensing.canonical_evidence("this-distribution", {declared}):
        identifier = item.identified
        if identifier and f"{identifier.lower()}.txt" not in names:
            wanted[identifier] = item.text
    return wanted


def siblings_of(wheels: list[Path], repo_root: Path) -> dict[str, dict]:
    """Read sibling wheel metadata before rendering, indexed by canonical name.

    Metadata supports cyclic requirements without cyclic document hashes.
    """
    known: dict[str, dict] = {}
    for wheel in wheels:
        _, parsed = wheelfile.read_metadata(wheel, wheelfile.dist_info_of(wheel))
        facts = _project_facts(repo_root, parsed)
        known[canonicalize_name(parsed["name"])] = {
            "version": parsed.get("version") or NOASSERTION,
            "supplier": facts["supplier"],
            "license_declared": facts["license"],
        }
    return known


def build(
    repo_root: Path,
    build_dir: Path,
    wheel_path: Path,
    out_dir: Path,
    license_data: Path | None = None,
    system_resolver=None,
    discovery: evidence_module.Discovery | None = None,
    siblings: dict[str, dict] | None = None,
) -> dict:
    """Package evidence into `wheel_path` in place and write the sidecars.

    `discovery` is the build tree read once. Reading it per wheel walks and
    hashes the whole tree, and decompresses every fetched archive, again.
    """
    licensing.load_corpus(license_data or default_license_data(build_dir))
    if discovery is None:
        discovery = evidence_module.discover(repo_root, build_dir)
    wheel = wheelfile.scan(wheel_path, analyze_elf=True)
    sbom_name = wheelfile.sbom_member(wheel.dist_info, wheel_path.name)
    if any(item.name == sbom_name for item in wheel.entries):
        raise BuildError(
            f"{wheel_path.name} already carries {sbom_name}. Run this on a freshly "
            "repaired wheel; rewriting one twice would duplicate its members."
        )
    metadata_raw, wheel_metadata = wheelfile.read_metadata(wheel_path, wheel.dist_info)

    resolver = inventory_module.Resolver(discovery, system_resolver)
    inventory = inventory_module.build(
        wheel, metadata_raw, resolver, discovery.staged_root
    )

    # Resolve nested terms and shared LicenseRefs before rendering any output,
    # so notices, reports, and evidence use the same terms and identifiers.
    settled = _with_reached_nested(inventory, inventory.components_present)
    present = {**inventory.components_present, **settled}
    renamed = licensing.shared_refs(present.values())
    for key, component in list(present.items()):
        present[key] = replace(
            component,
            license_concluded=licensing.rename_refs(
                component.license_concluded, renamed
            ),
            license_declared=licensing.rename_refs(component.license_declared, renamed),
        )
    inventory.components_present.update(present)
    resolver.components.update(present)

    if inventory.unattributed:
        raise BuildError(
            "these wheel members could not be traced to anything this build "
            "produced or fetched:\n  "
            + "\n  ".join(inventory.unattributed)
            + "\nThey reached the wheel by a route the collector cannot see; teach "
            "the collector that route rather than excluding the file." + _REMEDIES
        )

    if inventory.untraced:
        raise BuildError(
            "the link lines of wheel members name these, and nothing says whose "
            "code they hold:\n  "
            + "\n  ".join(inventory.untraced)
            + "\nLink only what the collector can trace -- a target of this build, a "
            "fetched dependency's tree, a system package or a known toolkit -- or "
            "teach it the route this input took. ExternalProject installs into the "
            "build tree by a route it does not know." + _REMEDIES
        )

    unlicensed = sorted(
        f"{path} ({key}): {reason}"
        for key, component in inventory.components_present.items()
        for path, reason in component.unlicensed
    )
    if unlicensed:
        raise BuildError(
            "the REUSE licence info of these checked-in third-party files fails "
            "the per-file gate:\n  "
            + "\n  ".join(unlicensed)
            + "\nRecord upstream's terms as REUSE info: a header tag, a .license "
            "sidecar or a REUSE.toml annotation. A stray tag is set aside instead."
            + _REMEDIES
        )

    missing = [
        key
        for key, component in inventory.components_present.items()
        if not licensing.obligation_texts(component.evidence)
    ]
    if missing:
        raise BuildError(
            "no license text was found in the files of redistributed component(s): "
            + ", ".join(missing)
            + "\nThe component ships no LICENSE/COPYING file and states no grant in "
            "its README; obtain the terms before publishing it." + _REMEDIES
        )

    # Compute the final member set before rendering its SBOM.
    additions: dict[str, bytes] = {}
    packaged_paths: list[str] = []
    for key, component in inventory.components_present.items():
        for item in component.evidence:
            target = report_module.packaged_license_path(wheel.dist_info, key, item)
            if target in additions:
                continue
            additions[target] = item.text.encode("utf-8")
            packaged_paths.append(
                report_module.declared_license_path(wheel.dist_info, target)
            )

    # Metadata-only wheels still need their own declared license texts.
    declared = _project_facts(repo_root, wheel_metadata)["license"]
    shipped = {
        report_module.declared_license_path(wheel.dist_info, item.name)
        for item in wheel.entries
        if report_module.is_packaged_license(wheel.dist_info, item.name)
    } | set(packaged_paths)
    missing_own = _own_license_texts(declared, shipped)
    for identifier, text in sorted(missing_own.items()):
        target = (
            f"{report_module.licenses_root(wheel.dist_info)}LICENSES/{identifier}.txt"
        )
        additions[target] = text.encode("utf-8")
        packaged_paths.append(
            report_module.declared_license_path(wheel.dist_info, target)
        )

    notices_path = report_module.notices_path(wheel.dist_info)
    additions[notices_path] = report_module.notices_markdown(
        inventory,
        wheel.dist_info,
        wheel_path.name,
        _project_facts(repo_root, wheel_metadata),
    ).encode("utf-8")
    packaged_paths.append(
        report_module.declared_license_path(wheel.dist_info, notices_path)
    )

    unmet = _declared_without_text(
        declared,
        {Path(item).name.lower() for item in _own_texts(set(packaged_paths) | shipped)},
    )
    if unmet:
        raise BuildError(
            f"{wheel_path.name} declares {declared} and packages no text for "
            + ", ".join(unmet)
            + ".\nA distribution has to ship the terms it declares, the same as "
            "anything it redistributes: add the text to license-files in "
            "pyproject.toml as <identifier>.txt, or drop the declaration." + _REMEDIES
        )

    patched_metadata = wheelfile.add_license_files(metadata_raw, sorted(packaged_paths))
    metadata_declared = patched_metadata != metadata_raw
    replacements = {f"{wheel.dist_info}/METADATA": patched_metadata}

    # Exclude the SBOM and RECORD to avoid circular digests. SPDX files require
    # checksums; the verifier checks these excluded members against the archive.
    excluded = [f"{wheel.dist_info}/RECORD", sbom_name]

    projected = [item for item in wheel.entries if item.name not in replacements] + [
        _entry(name, data) for name, data in {**replacements, **additions}.items()
    ]
    projected_wheel = replace(wheel, entries=tuple(projected))
    for name in [*additions, *replacements]:
        # Do not scrape our generated summaries as upstream copyright notices.
        packaged_text = name in additions and report_module.is_third_party_license(
            wheel.dist_info, name
        )
        body = additions.get(name, b"").decode("utf-8", "replace")
        inventory.attributions[name] = Attribution(
            path=name,
            origin="metadata",
            exact=True,
            detail="license evidence packaged by the SBOM collector"
            if name in additions
            else "distribution metadata, updated to declare the packaged license files",
            spdx_tag=licensing.normalized_tag(body) if packaged_text else None,
            # Retain all holders, including those in multi-component copyright files.
            copyright_text="\n".join(
                licensing.fold_notices(licensing.read_notices(body))
            )
            if packaged_text
            else None,
        )
    # Record the SBOM before rendering so the evidence covers the final member set.
    inventory.attributions[sbom_name] = Attribution(
        path=sbom_name,
        origin="metadata",
        detail="the contents SBOM this collector embedded",
    )

    evidence_doc = evidence_module.document(
        discovery, wheel_path.name, inventory, resolver.components
    )
    evidence_doc["ci"] = environment_summary()
    evidence_doc["system_libraries"] = resolver.system_libraries
    evidence_doc["static_inputs"] = resolver.static_inputs

    spdx = document_module.build_document(
        _project_facts(repo_root, wheel_metadata),
        evidence_doc,
        projected_wheel,
        inventory,
        excluded,
        siblings,
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

    # Bind the wheel digest only after the final mutation.
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
                **{key: payload[key] for key in report_module.GAP_LISTS},
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


def merge_manifests(paths: list[Path], evidence_out: Path | None = None) -> dict:
    """Merge variant manifests, optionally copying exactly their bound sidecars."""
    found = [
        (path.parent, item)
        for path in sorted(paths)
        for item in validate_module.manifest_records(path)
    ]
    merged = merge_records([item for _, item in found])
    if evidence_out is None:
        return merged

    # Copy from the selected record's directory, not another build of the same wheel.
    beside = {id(item): parent for parent, item in found}
    bound = [(beside[id(record)], record) for record in merged["wheels"]]
    # Validate the entire set before copying any files.
    failures = [
        f"{record['filename']}: {failure}"
        for parent, record in bound
        for failure in validate_module.sidecar_failures(parent, record)
    ]
    if failures:
        raise BuildError("; ".join(failures))
    evidence_out.mkdir(parents=True, exist_ok=True)
    for parent, record in bound:
        if parent.resolve() == evidence_out.resolve():
            continue
        for name in validate_module.bound_names([record]):
            shutil.copyfile(parent / name, evidence_out / name)
    return merged


def merge_records(wheels: list[dict]) -> dict:
    # Universal wheels repeat across matrix entries. Their bytes and wheel-derived
    # sidecars must agree; build evidence may differ, so bind the first entry's.
    identity = ("filename", "sha256", "size", "sbom_in_wheel")
    unique: dict[str, dict] = {}
    conflicting: set[str] = set()
    disagreeing: set[str] = set()
    for item in wheels:
        validate_module.checked_record(item)
        seen = unique.setdefault(item["filename"], item)
        if seen is item:
            continue
        if [seen.get(key) for key in identity] != [item.get(key) for key in identity]:
            conflicting.add(item["filename"])
        disagreeing.update(
            f"{item['filename']} ({kind})"
            for kind in validate_module.CONTENT_SIDECARS
            if seen["sidecars"][kind]["sha256"] != item["sidecars"][kind]["sha256"]
        )
    if conflicting:
        raise BuildError(
            "the same wheel filename was published with different contents: "
            + ", ".join(sorted(conflicting))
        )
    if disagreeing:
        raise BuildError(
            "builds of one wheel described it differently: "
            + ", ".join(sorted(disagreeing))
        )
    return {
        "schema": "isaaccapture-sbom-manifest/1",
        "generated_at": stamped_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "wheels": sorted(unique.values(), key=lambda item: item["filename"]),
    }


def environment_summary() -> dict:
    """Record CI/cache context for reproducibility."""
    return {
        "runner_os": os.environ.get("RUNNER_OS"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_ref": os.environ.get("GITHUB_REF"),
    }

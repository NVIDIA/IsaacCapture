# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a repaired wheel plus its build tree into a wheel that carries both."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from . import TOOL_NAME, TOOL_VERSION, stamped_now
from . import document as document_module
from . import evidence as evidence_module
from . import licensing
from . import inventory as inventory_module
from . import report as report_module
from . import wheelfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from . import discovery as discovery_module
from .discovery import project_copyright, project_supplier, sha256_file
from .document import NOASSERTION
from .inventory import Attribution
from .wheelfile import Entry

# Only the namespace this tool mints documents under. Supplier, licence and
# homepage are read from the wheel's own packaging metadata, so a fork, a rename
# or a second distribution built from the same tree cannot leave this file
# stating facts about somebody else's.
PROJECT = {"document_namespace": "https://github.com/NVIDIA/IsaacCapture/spdx"}


def _project_facts(repo_root: Path, metadata: dict) -> dict:
    """What this distribution says about itself, from its own metadata."""
    urls = metadata.get("project_urls") or {}
    homepage = next(
        (
            value
            for key, value in urls.items()
            if key.lower() in ("homepage", "home-page", "source", "repository")
        ),
        metadata.get("home_page") or NOASSERTION,
    )
    # The wheel's own `Author` first: a second distribution built from this tree
    # is a different project, and the checkout's `pyproject.toml` describes only
    # the one it is the root of.
    author = (metadata.get("author") or "").strip()
    return {
        **PROJECT,
        "supplier": f"Organization: {author}"
        if author
        else project_supplier(repo_root),
        # `license` is the legacy free-text field; only an expression is an
        # expression, and a wheel declaring neither declares nothing.
        "license": metadata.get("license_expression") or NOASSERTION,
        "homepage": homepage,
        # What the project states about itself, for a distribution whose own
        # files state nothing -- a metadata-only wheel has no file to carry it.
        "copyright": project_copyright(repo_root),
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


def default_license_data(build_dir: Path) -> Path:
    """Where deps/third_party materializes the SPDX license list for this build."""
    return build_dir / "_deps" / "license-list-data-src" / "json"


def _with_reached_nested(inventory, components: dict) -> dict:
    """Drop nested terms whose directory contributed nothing to this wheel.

    A nested LICENSE states a holder for part of a component -- the assets under
    it -- and dropping it names the wrong holder for those bytes. Keeping all of
    them names holders for code that never shipped.
    """
    reached: dict[str, set[str]] = {}
    for attribution in inventory.attributions.values():
        if attribution.source_path:
            for key in attribution.components:
                reached.setdefault(key, set()).add(attribution.source_path)

    updated = {}
    for key, component in components.items():
        nested = [item for item in component.evidence if item.kind == "nested"]
        if not nested:
            continue
        sources = reached.get(key, set())
        kept = [
            item
            for item in nested
            if any(
                source.startswith(item.path.rsplit("/", 1)[0] + "/")
                for source in sources
            )
        ]
        if len(kept) == len(nested):
            continue
        evidence = tuple(
            item for item in component.evidence if item.kind != "nested" or item in kept
        )
        updated[key] = replace(
            component,
            evidence=evidence,
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
        # An expression nothing can read names nothing that can be checked, and
        # the document carries it verbatim; say so rather than pass silently.
        return [declared]
    return sorted(
        item
        for item in identifiers
        # An exception modifies the licence it is attached to and has no text in
        # the list this build fetches, so demanding one blocks every
        # `X WITH <exception>` declaration with an instruction nothing can meet.
        if not licensing.is_exception(item) and f"{item.lower()}.txt" not in shipped
    )


def _own_texts(shipped: set[str]) -> set[str]:
    """The texts a wheel carries as its own, not as a third party's.

    These are named relative to `.dist-info/licenses/`, so a third party's sits
    under `third-party/`. Counting those let somebody else's copy discharge this
    distribution's obligation -- remove that component and the wheel silently
    loses its own terms.
    """
    return {item for item in shipped if not item.startswith("third-party/")}


def _own_license_texts(declared: str, shipped: set[str]) -> dict[str, str]:
    """Reference texts for the ids this distribution declares and does not ship.

    The identifiers come from the wheel's own `License-Expression`, so this adds
    the terms the distribution says apply to itself -- never a third party's,
    which are packaged from the files those components actually carry.
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
    """What each wheel of this run states about itself, by canonical name.

    Read before any of them is described, so a requirement naming one is never
    reported as an install-time unknown. Two wheels can require each other -- a
    renamed project and its transition wheel do -- and only one of them can
    carry a digest-bound reference to the other's document; both can state what
    the metadata says, because that needs no document.
    """
    known: dict[str, dict] = {}
    for wheel in wheels:
        _, parsed = wheelfile.read_metadata(wheel, wheelfile.dist_info_of(wheel))
        facts = _project_facts(repo_root, parsed)
        known[canonicalize_name(parsed["name"])] = {
            "name": parsed["name"],
            "version": parsed.get("version") or NOASSERTION,
            "supplier": facts["supplier"],
            "license_declared": facts["license"],
            "wheel": wheel.name,
        }
    return known


def in_dependency_order(wheels: list[Path]) -> list[Path]:
    """Wheels of one run, each after any sibling it requires.

    A document that references another cites its digest, so the referenced one
    has to exist first. Order is derived from the requirements rather than from
    the filenames, which only happen to sort the right way.
    """
    metadata = {}
    for wheel in wheels:
        _, parsed = wheelfile.read_metadata(wheel, wheelfile.dist_info_of(wheel))
        metadata[wheel] = (
            canonicalize_name(parsed["name"]),
            {
                canonicalize_name(Requirement(item).name)
                for item in parsed.get("requires_dist") or []
            },
        )

    ordered: list[Path] = []
    placed: set[str] = set()
    remaining = list(wheels)
    while remaining:
        ready = [
            wheel
            for wheel in remaining
            if not (
                {name for name, _ in metadata.values()}
                & metadata[wheel][1] - placed - {metadata[wheel][0]}
            )
        ]
        # A cycle between two wheels of one run cannot be ordered; keep the
        # caller's order rather than refusing to describe either.
        batch = ready or remaining
        for wheel in batch:
            ordered.append(wheel)
            placed.add(metadata[wheel][0])
        remaining = [item for item in remaining if item not in batch]
    return ordered


def _describes(spdx: dict, metadata_raw: bytes, sbom_filename: str) -> dict:
    """What another wheel's document needs to cite this one, SPDX 2.3 style."""
    metadata = wheelfile.parse_metadata(metadata_raw)
    body = json.dumps(spdx, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    wheel_package = next(
        item
        for item in spdx["packages"]
        if item["SPDXID"] == document_module.WHEEL_PACKAGE_ID
    )
    return {
        "name": metadata["name"],
        "version": metadata.get("version") or "NOASSERTION",
        "namespace": spdx["documentNamespace"],
        "sha1": hashlib.sha1(body).hexdigest(),  # noqa: S324
        "filename": sbom_filename,
        "license_declared": wheel_package["licenseDeclared"],
        "supplier": wheel_package["supplier"],
    }


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

    # Two settlements before anything renders, applied to both views of a
    # component so the document, the reports, the packaged notices and the
    # evidence sidecar all describe the same thing.
    #
    # Terms stated for one subdirectory apply only if that directory reached
    # this wheel: a test suite's licence is not this wheel's business, a
    # vendored asset set's is. And components carrying byte-identical terms
    # share one LicenseRef -- settled here because deciding it in the renderer
    # left the notices file naming an id the SBOM beside it did not define.
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
            + "\nThey reached the wheel by a route the collector cannot see."
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
            packaged_paths.append(
                report_module.declared_license_path(wheel.dist_info, target)
            )

    # The distribution's own declared licence is an obligation like any other.
    # Apache-2.0 asks that recipients get a copy, and the gate above covers only
    # what the build redistributes -- so a wheel could declare a licence whose
    # text it does not carry, and the transition wheel did.
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

    # Whatever could not be supplied from the corpus -- a LicenseRef, an id the
    # pinned list does not carry -- is a declaration with no terms behind it.
    unmet = _declared_without_text(
        declared,
        {Path(item).name.lower() for item in _own_texts(set(packaged_paths) | shipped)},
    )
    if unmet:
        raise BuildError(
            f"{wheel_path.name} declares {declared} and packages no text for "
            + ", ".join(unmet)
            + ".\nA distribution has to ship the terms it declares, the same as "
            "anything it redistributes; obtain the text before publishing it."
        )

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
        # Only the third-party texts: the notices file and the document below
        # are this collector's own output, and reading them back found the first
        # `- Copyright:` bullet of the summary it had just generated.
        packaged_text = name in additions and report_module.is_third_party_license(
            wheel.dist_info, name
        )
        body = additions.get(name, b"").decode("utf-8", "replace")
        # A licence text can state its own terms and carry a notice, and these
        # were the only members never read for either -- the collector packaged
        # them, so nothing scanned them the way it scans what it is describing.
        inventory.attributions[name] = Attribution(
            path=name,
            origin="metadata",
            exact=True,
            detail="license evidence packaged by the SBOM collector"
            if name in additions
            else "distribution metadata, updated to declare the packaged license files",
            spdx_tag=licensing.normalized_tag(body) if packaged_text else None,
            # Every holder the text names, as the package roll-up does: keeping
            # the first makes a per-file notice that does not discharge the
            # obligation -- libbsd's copyright file names forty-six.
            copyright_text="\n".join(
                licensing.fold_notices(licensing.read_notices(body))
            )
            if packaged_text
            else None,
        )
    # The document is added below, after it has been rendered; record it now so
    # the evidence accounts for every member of the finished wheel.
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
                # What a sibling wheel's document cites to reference this one.
                "describes": _describes(spdx, metadata_raw, sidecars["spdx"].name),
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
    duplicates = sorted(name for name, seen in Counter(names).items() if seen > 1)
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

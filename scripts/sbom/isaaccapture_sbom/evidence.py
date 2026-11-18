# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble every discovery index for one configured build tree."""

from __future__ import annotations

import hashlib
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from packageurl import PackageURL

from . import TOOL_NAME, TOOL_VERSION, licensing, stamped_now
from .discovery import (
    ARCHIVE_SUFFIXES,
    ArchiveIndex,
    component_of,
    Ownership,
    git,
    BuildGraph,
    Component,
    cmake_cache,
    FileIndex,
    discover_archives,
    discover_vendored,
    project_authors,
    discover_source_trees,
    discover_vcpkg,
    notice_from,
    tracked_files,
    version_in_name,
    EMPTY_SHA256,
)

# Record reproducibility settings without unrelated builder-local cache paths.
_CACHE_KEYS = re.compile(
    r"^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_C_COMPILER|CMAKE_SYSTEM_PROCESSOR"
    r"|CMAKE_CXX_COMPILER_VERSION|ISAAC_TELEOP_PYTHON_VERSION|BUILD_[A-Z0-9_]+"
    r"|ENABLE_[A-Z0-9_]+|VCPKG_[A-Z0-9_]+)$"
)


class EvidenceError(Exception):
    """The build tree does not carry the evidence a published wheel needs."""


@dataclass
class Discovery:
    """Everything one build tree can tell us, indexed for attribution."""

    repo_root: Path
    build_dir: Path
    config: str
    cache: dict[str, str]
    components: dict[str, Component]
    graph: BuildGraph
    archives: ArchiveIndex
    source_files: FileIndex
    repo_files: FileIndex
    build_files: FileIndex
    ownership: Ownership
    # build_files display prefix -> component whose tree the build staged there.
    staged_component_dirs: dict[str, str]

    @property
    def staged_root(self) -> Path | None:
        staged = self.build_dir / "python_package" / self.config
        return staged if staged.is_dir() else None


def _cmake_cache(build_dir: Path) -> dict[str, str]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise EvidenceError(
            f"{cache} not found; point --build-dir at a configured build tree"
        )
    return {
        name: value
        for name, value in cmake_cache(build_dir).items()
        if _CACHE_KEYS.match(name)
    }


def _fetched_from(archive: Path) -> str | None:
    """Read the download URL preserved beside an archive as a .source marker."""
    marker = archive.with_name(archive.name + ".source")
    if not marker.is_file():
        return None
    url = marker.read_text(encoding="utf-8", errors="replace").strip()
    return url.splitlines()[0].strip() if url else None


def _supplier_of(url: str | None) -> str:
    """Infer a supplier from the URL's penultimate host label when available."""
    host = urlparse(url).hostname if url else None
    if not host:
        return "NOASSERTION"
    labels = host.split(".")
    # For api.ngc.nvidia.com this yields nvidia; this is not a public-suffix parser.
    name = labels[-2] if len(labels) >= 2 else ""
    return f"Organization: {name}" if len(name) > 2 else "NOASSERTION"


def _archive_components(
    archives: ArchiveIndex, repo_root: Path
) -> dict[str, Component]:
    """Describe each indexed archive with its root license evidence."""
    components: dict[str, Component] = {}
    for display, record in archives.archives.items():
        name = Path(display).name
        for suffix in ARCHIVE_SUFFIXES:
            name = name.removesuffix(suffix)

        evidence = [
            licensing.evidence_for(
                display,
                "archive-member",
                f"{display}!{member.path}",
                data.decode("utf-8", "replace"),
                # Sidecars were set aside when the member was kept.
                licensing.classify(Path(member.path).name, lambda _: False) or "grant",
            )
            for member, data in record.get("license_members", [])
        ]
        # Use the full display path to distinguish archives with the same basename.
        concluded, declared = licensing.expression(evidence, display)

        source_url = _fetched_from(repo_root / display)
        held = (
            f"fetched from {source_url}" if source_url else "present in the source tree"
        )
        components[display] = Component(
            key=display,
            name=name,
            kind="archive",
            supplier=_supplier_of(source_url),
            homepage="NOASSERTION",
            purl=PackageURL(type="generic", name=name).to_string(),
            version=_archive_version(archives, display) or "NOASSERTION",
            download_location=source_url or "NOASSERTION",
            source_info=(
                f"Unpacked from {display} (sha256:{record['sha256']}, "
                f"{record['member_count']} files), {held}."
            ),
            license_concluded=concluded,
            license_declared=declared,
            copyright_text=notice_from(evidence),
            evidence=tuple(evidence),
        )
    return components


def _archive_version(archives: ArchiveIndex, display: str) -> str | None:
    """SDK tarballs state their version in a root VERSION member, or in the name."""
    stated = archives.archives.get(display, {}).get("version_text")
    if stated:
        return stated
    return version_in_name(Path(display).name)


def discover(repo_root: Path, build_dir: Path) -> Discovery:
    """Index a configured build tree."""
    cache = _cmake_cache(build_dir)
    deps_dir = build_dir / "_deps"
    if not deps_dir.is_dir():
        raise EvidenceError(
            f"{deps_dir} not found; configure the build before collecting evidence"
        )

    # Require the configuration that produced the graph and staged package tree.
    config = cache.get("CMAKE_BUILD_TYPE")
    if not config:
        raise EvidenceError(
            f"{build_dir}/CMakeCache.txt states no CMAKE_BUILD_TYPE; the build "
            "graph and the staged tree are both read per configuration"
        )
    graph = BuildGraph(build_dir, config)
    # Exclude other build trees too; their staged copies create false source matches.
    not_source = (
        build_dir,
        repo_root / "build-wheel",
        repo_root / "dist",
        *(
            path.parent
            for pattern in ("*/CMakeCache.txt", "*/*/CMakeCache.txt")
            for path in repo_root.glob(pattern)
        ),
    )

    tracked = tracked_files(repo_root)
    components = discover_source_trees(deps_dir)
    archives = discover_archives(repo_root, frozenset(tracked), skip=not_source)
    components.update(_archive_components(archives, repo_root))

    source_files = FileIndex()
    for root in sorted(deps_dir.glob("*-src")):
        if root.is_dir():
            source_files.add_tree(root, root.name)

    # Only tracked files are repository sources; install prefixes have no identifying marker.
    repo_files = FileIndex()
    repo_files.add_paths(repo_root, ".", tracked)

    # Index generated output, including _deps/*-build, separately from fetched sources.
    # Exclude staging so the wheel cannot explain itself. Preserve the actual build
    # directory name in evidence paths.
    build_files = FileIndex()
    build_files.add_tree(
        build_dir,
        build_dir.name,
        skip=(*sorted(deps_dir.glob("*-src")), build_dir / "python_package"),
    )

    # Infer staged-tree ownership from unchanged siblings after transformations.
    # Require a unique owner to avoid inventing a license obligation.
    staged_component_dirs: dict[str, str] = {}
    for root in sorted(deps_dir.iterdir()):
        if not root.is_dir() or root.name.endswith("-src"):
            continue
        display = f"{build_dir.name}/{root.relative_to(build_dir).as_posix()}"
        owners = set()
        for _, digest in build_files.under(display):
            # Empty files cannot identify an owner.
            if digest == EMPTY_SHA256:
                continue
            match = source_files.by_hash.get(digest)
            if match:
                owners.add(component_of(match))
        if len(owners) == 1:
            staged_component_dirs[display] = owners.pop()

    ownership = Ownership()
    for root in sorted(deps_dir.glob("*-src")):
        if root.is_dir():
            ownership.register_root(root, root.name.removesuffix("-src"))
    # Generated dependency headers belong to their component, not the root project.
    for root in sorted(deps_dir.glob("*-build")):
        if root.is_dir():
            ownership.register_root(root, root.name.removesuffix("-build"))

    ports, port_files = discover_vcpkg(
        Path(cache.get("VCPKG_INSTALLED_DIR") or build_dir / "vcpkg_installed"),
        cache.get("VCPKG_TARGET_TRIPLET"),
    )
    components.update(ports)
    for key, paths in port_files.items():
        for path in paths:
            ownership.register_file(path, key, by_directory=False)

    # Include tracked data and Python as well as native build inputs.
    vendored, owned_paths = discover_vendored(
        repo_root,
        graph.repo_inputs(repo_root, build_dir),
        project_authors(repo_root),
        tracked=tracked,
    )
    components.update(vendored)
    for key, paths in owned_paths.items():
        for path in paths:
            ownership.register_file(path, key)

    return Discovery(
        repo_root=repo_root,
        build_dir=build_dir,
        config=config,
        cache=cache,
        components=components,
        graph=graph,
        archives=archives,
        source_files=source_files,
        repo_files=repo_files,
        build_files=build_files,
        ownership=ownership,
        staged_component_dirs=staged_component_dirs,
    )


def document(
    discovery: Discovery, wheel_name: str, inventory, components: dict[str, Component]
) -> dict:
    """Render build evidence using this wheel's resolved components.

    Keep host libraries per wheel; Discovery is shared across the run.
    """
    repo_root = discovery.repo_root
    arch = {
        "x86_64": "amd64",
        "AMD64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(
        discovery.cache.get("CMAKE_SYSTEM_PROCESSOR", platform.machine()),
        platform.machine(),
    )

    artifacts = []
    for artifact in discovery.graph.artifacts.values():
        contributions = discovery.graph.contributions(artifact, discovery.ownership)
        # Resolved to components only where a wheel member links them; see
        # `static_inputs`.
        inputs = sorted({item.key for item in discovery.graph.static_inputs(artifact)})
        if not contributions and not inputs:
            continue
        try:
            output = artifact.output.relative_to(discovery.build_dir).as_posix()
        except ValueError:
            output = artifact.output.as_posix()
        artifacts.append(
            {
                "target": artifact.target,
                "output": output,
                "first_party": discovery.graph.first_party(
                    artifact, repo_root, discovery.ownership
                ),
                "contributions": {
                    key: sorted(how) for key, how in sorted(contributions.items())
                },
                **({"static_inputs": inputs} if inputs else {}),
            }
        )

    return {
        "schema": "isaaccapture-build-evidence/2",
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "collected_at": stamped_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wheel": wheel_name,
        "project": {
            "commit": git(repo_root, "rev-parse", "HEAD"),
            "describe": git(repo_root, "describe", "--tags", "--always", "--dirty"),
            "version_file": (repo_root / "VERSION").read_text(encoding="utf-8").strip()
            if (repo_root / "VERSION").is_file()
            else None,
        },
        "build": {
            "build_dir": str(discovery.build_dir),
            "staged_root": (
                str(discovery.staged_root) if discovery.staged_root else None
            ),
            "config": discovery.config,
            "arch": arch,
            "platform": platform.platform(),
            "cmake_cache": discovery.cache,
        },
        "license_data": licensing.corpus_provenance(),
        "license_list_version": licensing.corpus()[0],
        "components": {
            key: {
                "name": component.name,
                "kind": component.kind,
                "supplier": component.supplier,
                "homepage": component.homepage,
                "purl": component.purl,
                "version": component.version,
                "download_location": component.download_location,
                "source_info": component.source_info,
                "license_concluded": component.license_concluded,
                "license_declared": component.license_declared,
                "evidence": [item.as_json() for item in component.evidence],
                # Bind the combined terms as well as individual files so LicenseRef text is verifiable.
                "verbatim_terms_sha256": hashlib.sha256(
                    licensing.verbatim_terms(component.evidence).encode("utf-8")
                ).hexdigest(),
                "in_this_wheel": key in inventory.components_present,
            }
            for key, component in sorted(components.items())
        },
        "archives": {
            display: {
                key: value for key, value in record.items() if key != "license_members"
            }
            for display, record in sorted(discovery.archives.archives.items())
        },
        "link_artifacts": sorted(artifacts, key=lambda item: item["output"]),
        "attributions": [item.as_json() for item in inventory.attributions.values()],
    }

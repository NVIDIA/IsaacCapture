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
    FileIndex,
    discover_archives,
    discover_vendored,
    project_authors,
    discover_source_trees,
    _notice_from,
    tracked_files,
    EMPTY_SHA256,
)

# Suffixes whose files conventionally carry a REUSE header naming their holder.
_TAGGABLE_SOURCE = {".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".pyi", ".sh", ".cmake"}

# Cache entries that pin what a wheel contains. The rest of the cache is paths on
# the builder.
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
    entries: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        name = line.split(":", 1)[0]
        if _CACHE_KEYS.match(name):
            entries[name] = line.split("=", 1)[1]
    return entries


def _fetched_from(archive: Path) -> str | None:
    """The URL a fetch script recorded beside an archive it downloaded.

    Most archives here are downloaded at configure time rather than checked in,
    so the origin is only knowable if whatever fetched it wrote the URL down.
    """
    marker = archive.with_name(archive.name + ".source")
    if not marker.is_file():
        return None
    url = marker.read_text(encoding="utf-8", errors="replace").strip()
    return url.splitlines()[0].strip() if url else None


def _supplier_of(url: str | None) -> str:
    """Whoever served the archive, named by the host that did."""
    if not url:
        return "NOASSERTION"
    host = urlparse(url).hostname
    return f"Organization: {host}" if host else "NOASSERTION"


def _archive_components(
    archives: ArchiveIndex, repo_root: Path
) -> dict[str, Component]:
    """A tarball in the repo is a component: its members ship as they are."""
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
                licensing.classify(Path(member.path).name) or "grant",
            )
            for member, data in record.get("license_members", [])
        ]
        # `display`, not `name`: the name is the archive's basename, so two
        # archives in different directories would share a LicenseRef.
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
                f"{record['member_count']} members), {held}."
            ),
            license_concluded=concluded,
            license_declared=declared,
            copyright_text=_notice_from(evidence),
            evidence=tuple(evidence),
        )
    return components


def _archive_version(archives: ArchiveIndex, display: str) -> str | None:
    """SDK tarballs state their version in a root VERSION member, or in the name."""
    stated = archives.archives.get(display, {}).get("version_text")
    if stated:
        return stated
    match = re.search(
        r"(\d+\.\d+[\w.+-]*?)(?=-|\.tar|\.tgz|\.zip|$)", Path(display).name
    )
    return match.group(1) if match else None


def discover(repo_root: Path, build_dir: Path) -> Discovery:
    """Index a configured build tree. Nothing here is declared anywhere."""
    cache = _cmake_cache(build_dir)
    deps_dir = build_dir / "_deps"
    if not deps_dir.is_dir():
        raise EvidenceError(
            f"{deps_dir} not found; configure the build before collecting evidence"
        )

    # Guessing a configuration would point the build graph and the staged-tree
    # probe at a directory that need not exist, and every member they would have
    # explained then reads as having reached the wheel by an unknown route.
    config = cache.get("CMAKE_BUILD_TYPE")
    if not config:
        raise EvidenceError(
            f"{build_dir}/CMakeCache.txt states no CMAKE_BUILD_TYPE; the build "
            "graph and the staged tree are both read per configuration"
        )
    graph = BuildGraph(build_dir, config)
    # Everything that is a build tree rather than source, skipped the same way
    # everywhere it matters. The active one is given; any other is found by the
    # CMakeCache.txt that makes it one. A second build directory holds staged
    # copies of the repo, and walking those as source leaves a file's origin
    # ambiguous rather than wrong, which surfaces as an unexplained member.
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

    # Tracked files only: an install prefix carries no marker distinguishing it
    # from source, so a directory walk cannot, and a wheel member matching this
    # build's own output would be reported as a copy of repository source.
    repo_files = FileIndex()
    repo_files.add_paths(repo_root, ".", tracked)

    # _deps/*-src is the component-source domain, already indexed above; a wheel
    # member matching one is a copy of upstream, not something this build made.
    # Everything else under _deps is this build's own output -- FetchContent
    # build trees, and trees a dependency stages for packaging -- so it belongs
    # in build_files like any other generated file. The staged tree stays out:
    # indexing it would let the wheel explain itself.
    # Named after the directory that was read: `--build-dir` is free-form, and a
    # citation reading `build/...` resolves to the wrong tree, or to nothing, for
    # anyone whose build directory is not called that.
    build_files = FileIndex()
    build_files.add_tree(
        build_dir,
        build_dir.name,
        skip=(*sorted(deps_dir.glob("*-src")), build_dir / "python_package"),
    )

    # A dependency may stage a packaged copy of its own tree under _deps/<name>/.
    # Files the build transformed on the way in no longer hash to the component,
    # so infer the directory's owner from the siblings that still do; one clear
    # owner or nothing, since a guess here would invent a license obligation.
    staged_component_dirs: dict[str, str] = {}
    for root in sorted(deps_dir.iterdir()):
        if not root.is_dir() or root.name.endswith("-src"):
            continue
        display = f"{build_dir.name}/{root.relative_to(build_dir).as_posix()}"
        owners = set()
        # From the index rather than a second walk: `build_files` already hashed
        # every file under the build tree.
        for _, digest in build_files.under(display):
            # Zero bytes match every other empty file, so a CMake scaffolding
            # directory full of stamps would adopt whichever component happened
            # to ship an empty file. They are evidence of nothing here too.
            if digest == EMPTY_SHA256:
                continue
            match = source_files.by_hash.get(digest)
            if match:
                owners.add(component_of(match))
        if len(owners) == 1:
            staged_component_dirs[display] = owners.pop()

    # Third-party code checked in here rather than fetched: the candidates are
    # what the build actually compiles or includes, so nothing outside the build
    # is inspected.
    ownership = Ownership()
    for root in sorted(deps_dir.glob("*-src")):
        if root.is_dir():
            ownership.register_root(root, root.name.removesuffix("-src"))
    # A dependency generates sources into its own build tree -- glfw's Wayland
    # protocol headers, Catch2's generated config. Those sit under the build
    # directory and belong to no *-src checkout, so without this they count as
    # this project's own sources and the target reads as first-party.
    for root in sorted(deps_dir.glob("*-build")):
        if root.is_dir():
            ownership.register_root(root, root.name.removesuffix("-build"))

    # Every tracked file that can carry a REUSE header, plus whatever the build
    # compiled. Scanning only compiled sources missed a third party whose code
    # reaches the wheel as Python: nothing compiles it, so nothing looked at it,
    # and the component was reported as contributing no file.
    candidates: set[Path] = {
        repo_root / item
        for item in tracked
        if Path(item).suffix.lower() in _TAGGABLE_SOURCE
    }
    candidates.update(graph.repo_inputs(repo_root, build_dir))

    vendored, owned_paths = discover_vendored(
        repo_root, candidates, project_authors(repo_root)
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
    """The build-evidence sidecar: resolved facts, not the raw indexes.

    `components` is what this wheel resolved against, which is the discovered set
    plus the system libraries its own members turned out to need. One `Discovery`
    serves every wheel in a run, so it cannot hold those.
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
        if not contributions:
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
            # Named once: a member's origin reads "staged" rather than repeating
            # this prefix and the member's own path for every file staged here.
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
                # The digest of the terms verbatim, as the document carries
                # them. A component can state its licence across several files,
                # and the join has no digest of its own for the verifier to
                # compare against unless the build states one.
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
        "excluded_components": inventory.absent_components,
    }

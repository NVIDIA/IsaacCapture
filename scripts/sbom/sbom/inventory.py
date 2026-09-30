# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Account for every file in a wheel by where it came from.

Each member is traced to a link step, a fetched source tree, an archive in the
repo, the working tree, or the build's own generated output. A member that none
of those explain is an error: it means something reached the wheel by a route
this collector cannot see, and it stops publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import licensing
from .discovery import (
    Component,
    resolve_system_library,
    sha256_file,
    system_component,
)
from . import wheelfile
from .wheelfile import Entry, WheelInfo

if TYPE_CHECKING:  # pragma: no cover
    from .evidence import Discovery

FIRST_PARTY = "__first_party__"

# Text-ish members worth reading a REUSE tag out of.
_TAGGABLE = {
    ".py",
    ".pyi",
    ".txt",
    ".cfg",
    ".toml",
    ".yaml",
    ".yml",
    ".xml",
    ".json",
    ".md",
}


@dataclass
class Attribution:
    """One wheel member and the evidence that explains it."""

    path: str
    origin: str
    detail: str
    components: dict[str, set[str]] = field(default_factory=dict)
    primary: str | None = None
    spdx_tag: str | None = None

    def as_json(self) -> dict:
        return {
            "path": self.path,
            "origin": self.origin,
            "detail": self.detail,
            "primary": self.primary,
            "components": {
                key: sorted(how) for key, how in sorted(self.components.items())
            },
            "spdx_tag": self.spdx_tag,
        }


@dataclass
class Inventory:
    attributions: dict[str, Attribution]
    components_present: dict[str, Component]
    component_roles: dict[str, set[str]]
    external_runtime: dict[str, list[str]]
    requires_dist: list[str]
    unattributed: list[str] = field(default_factory=list)
    absent_components: dict[str, str] = field(default_factory=dict)

    def files_of(self, key: str) -> list[str]:
        return sorted(
            path for path, item in self.attributions.items() if item.primary == key
        )


class Resolver:
    """Probes a candidate's bytes and path against every discovered index."""

    def __init__(self, discovery: Discovery, system_resolver=None) -> None:
        self._system_resolver = system_resolver or resolve_system_library
        self.system_libraries: dict[str, dict] = {}
        self.repo_root = discovery.repo_root
        self.build_dir = discovery.build_dir
        self.graph = discovery.graph
        self.components = discovery.components
        self.archives = discovery.archives
        self.repo_files = discovery.repo_files
        self.build_files = discovery.build_files
        self.ownership = discovery.ownership
        self.source_files = discovery.source_files
        self._artifacts_by_hash: dict[str, list] = {}
        for artifact in self.graph.artifacts.values():
            if artifact.output.is_file():
                # build_files already hashed everything under the build tree.
                digest = self.build_files.digest_of(artifact.output) or sha256_file(
                    artifact.output
                )
                self._artifacts_by_hash.setdefault(digest, []).append(artifact)

    def _from_artifact(self, artifact, path: str, detail: str) -> Attribution:
        contributions = self.graph.contributions(artifact, self.ownership)
        first_party = self.graph.first_party(artifact, self.repo_root, self.ownership)
        primary = None
        if not first_party:
            # The component that supplied the compiled sources owns the binary;
            # the rest are linked into it.
            sources = [
                key for key, how in contributions.items() if "compiled-source" in how
            ]
            primary = min(
                sources, key=lambda key: (key != artifact.target, key), default=None
            )
        return Attribution(
            path=path,
            origin="built",
            detail=f"{detail}; linked by target {artifact.target}",
            components={key: set(how) for key, how in contributions.items()},
            primary=primary,
        )

    def resolve(
        self,
        wheel_path: str,
        digest: str,
        probe: Path | None,
        soname: str | None = None,
    ) -> Attribution | None:
        """Explain one member: by content, then by path, then by the machine."""
        candidate_digest = digest
        source = "wheel member"
        if probe is not None and probe.is_file():
            candidate_digest = sha256_file(probe)
            source = f"staged at {probe.relative_to(self.build_dir).as_posix()}"

        artifacts = self._artifacts_by_hash.get(candidate_digest)
        if artifacts:
            return self._from_artifact(
                artifacts[0], wheel_path, f"{source}; exact build output"
            )

        component_file = self.source_files.by_hash.get(candidate_digest)
        if component_file:
            key = component_file.split("/", 1)[0].removesuffix("-src")
            return Attribution(
                path=wheel_path,
                origin="copied",
                detail=f"{source}; byte-identical to {component_file}",
                components={key: {"copied-file"}},
                primary=key,
            )

        member = self.archives.by_hash.get(candidate_digest)
        if member:
            return Attribution(
                path=wheel_path,
                origin="archive-copy",
                detail=f"{source}; byte-identical to {member.path} in {member.container}",
                components={member.container: {"extracted-file"}},
                primary=member.container,
            )

        repo_file = self.repo_files.by_hash.get(candidate_digest)
        if repo_file:
            return Attribution(
                path=wheel_path, origin="repo-source", detail=f"{source}; {repo_file}"
            )

        build_file = self.build_files.by_hash.get(candidate_digest)
        if build_file:
            return Attribution(
                path=wheel_path, origin="generated", detail=f"{source}; {build_file}"
            )

        by_path = self._resolve_by_path(wheel_path, source)
        if by_path is not None:
            return by_path

        # Nothing this build produced or fetched explains these bytes. A shared
        # library reaching that point came from the machine -- whatever the
        # repair tool chose to call its directory.
        if soname:
            return self._from_build_host(wheel_path, soname)
        return None

    def _from_build_host(self, wheel_path: str, soname: str) -> Attribution | None:
        """Claim host origin only if the host actually has this library.

        Resolving the SONAME is the evidence for the claim. Without it there is
        nothing behind "the machine supplied this", and the member is better
        reported as unexplained than attributed to a library no one found.
        """
        origin = self._system_resolver(soname)
        if not origin.get("resolved_path"):
            return None
        self.system_libraries[soname] = origin
        component = system_component(soname, origin)
        self.components[component.key] = component

        detail = "no index explains these bytes; supplied by the build machine"
        if origin.get("package"):
            detail += f" as {origin['package']}"
        if origin.get("resolved_path"):
            detail += f" ({origin['resolved_path']})"
        return Attribution(
            path=wheel_path,
            origin="build-host-library",
            detail=detail,
            components={component.key: {"vendored-library"}},
            primary=component.key,
        )

    def _resolve_by_path(self, wheel_path: str, source: str) -> Attribution | None:
        """Post-processing changes bytes; the path still says where they came from.

        patchelf, auditwheel's RPATH rewrite and the MJCF mesh stripping all leave
        a file that no longer hashes to its origin, so a unique path match is the
        remaining evidence -- recorded as derived, never as an exact copy.
        """
        name = Path(wheel_path).name

        members = self.archives.by_name.get(name, [])
        containers = {member.container for member in members}
        if len(containers) == 1:
            member = members[0]
            return Attribution(
                path=wheel_path,
                origin="archive-derived",
                detail=(
                    f"{source}; modified copy of {member.path} in {member.container} "
                    f"(origin sha256:{member.sha256})"
                ),
                components={member.container: {"extracted-file"}},
                primary=member.container,
            )

        artifacts = self.graph.by_name(name)
        if len(artifacts) == 1:
            return self._from_artifact(
                artifacts[0], wheel_path, f"{source}; modified build output"
            )

        component_file = self.source_files.path_suffix_match(wheel_path)
        if component_file:
            key = component_file.split("/", 1)[0].removesuffix("-src")
            return Attribution(
                path=wheel_path,
                origin="derived",
                detail=f"{source}; derived from {component_file}",
                components={key: {"derived-file"}},
                primary=key,
            )

        repo_file = self.repo_files.path_suffix_match(wheel_path)
        if repo_file:
            return Attribution(
                path=wheel_path,
                origin="repo-source",
                detail=f"{source}; derived from {repo_file}",
            )

        return None


def build(
    wheel: WheelInfo,
    metadata_bytes: bytes,
    resolver: Resolver,
    staged_root: Path | None,
) -> Inventory:
    """Attribute every member, then fold the result into component presence."""
    attributions: dict[str, Attribution] = {}
    unattributed: list[str] = []
    provided_sonames = {entry.soname for entry in wheel.entries if entry.soname}

    for entry in wheel.entries:
        attribution = _attribute(entry, wheel, resolver, staged_root)
        if attribution is None:
            unattributed.append(entry.name)
            continue
        attribution.spdx_tag = _spdx_tag(entry)
        attributions[entry.name] = attribution

    roles: dict[str, set[str]] = {}
    for attribution in attributions.values():
        for key, how in attribution.components.items():
            roles.setdefault(key, set()).update(how)

    present = {
        key: resolver.components[key]
        for key in sorted(roles)
        if key in resolver.components
    }
    absent = {
        key: "fetched for this build; contributes to no file in this wheel"
        for key in sorted(resolver.components)
        if key not in roles
    }

    external: dict[str, list[str]] = {}
    for entry in wheel.entries:
        for soname in entry.needed:
            if soname not in provided_sonames:
                external.setdefault(soname, []).append(entry.name)

    metadata = wheelfile.parse_metadata(metadata_bytes)
    return Inventory(
        attributions=attributions,
        components_present=present,
        component_roles=roles,
        external_runtime={
            name: sorted(set(files)) for name, files in sorted(external.items())
        },
        requires_dist=sorted(metadata.get("requires_dist") or []),
        unattributed=sorted(unattributed),
        absent_components=absent,
    )


def _attribute(
    entry: Entry,
    wheel: WheelInfo,
    resolver: Resolver,
    staged_root: Path | None,
) -> Attribution | None:
    if entry.name.startswith(f"{wheel.dist_info}/"):
        return Attribution(
            path=entry.name,
            origin="metadata",
            detail="distribution metadata written by the build",
        )

    probe = staged_root / entry.name if staged_root is not None else None
    return resolver.resolve(entry.name, entry.sha256, probe, entry.soname)


def _spdx_tag(entry: Entry) -> str | None:
    """REUSE tags are upstream stating the license of that exact file."""
    if Path(entry.name).suffix.lower() not in _TAGGABLE:
        return None
    tag = licensing.read_spdx_tag(entry.head.decode("utf-8", "replace"))
    if tag is None:
        return None
    # Tags are written by hand: lowercase operators and stray text reach the
    # document as licenseConcluded, where an invalid expression fails the
    # published SBOM's own conformance check.
    normalized = licensing.normalized_expression(tag)
    return None if normalized == "NOASSERTION" else normalized

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
import re
from pathlib import Path
from typing import TYPE_CHECKING

from . import elf, licensing
from .discovery import (
    Component,
    EMPTY_SHA256,
    resolve_system_library,
    resolve_system_library_by_build_id,
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
    copyright_text: str | None = None
    # Whether the member is those bytes, as opposed to something made from them.
    # Kept apart from `origin`, which says where it came from: inferring one from
    # the other made the same fact answerable two ways depending on which index
    # happened to hold the bytes.
    exact: bool = False

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

    def files_sharing(self, key: str) -> list[str]:
        """Files naming this component as one candidate among several."""
        return sorted(
            path
            for path, item in self.attributions.items()
            if item.primary is None and "shared-content" in item.components.get(key, ())
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
        self.staged_component_dirs = discovery.staged_component_dirs
        self._artifacts_by_hash: dict[str, list] = {}
        self._artifacts_by_build_id: dict[str, list] = {}
        for artifact in self.graph.artifacts.values():
            if artifact.output.is_file():
                # build_files already hashed everything under the build tree.
                digest = self.build_files.digest_of(artifact.output) or sha256_file(
                    artifact.output
                )
                self._artifacts_by_hash.setdefault(digest, []).append(artifact)
                build_id = elf.read_build_id(artifact.output)
                if build_id:
                    self._artifacts_by_build_id.setdefault(build_id, []).append(
                        artifact
                    )

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
        build_id: str | None = None,
    ) -> Attribution | None:
        """Explain one member: by content, then by path, then by the machine."""
        candidate_digest = digest
        source = "wheel member"
        patched = False
        if probe is not None and probe.is_file():
            candidate_digest = sha256_file(probe)
            # The staged file is what the build put there; the member is what the
            # wheel ships. Tracing from the staged bytes finds the origin, but
            # the two can differ -- a repair tool rewrites members after staging
            # -- and saying the member is a copy of something it does not hash to
            # would describe a file the wheel does not contain.
            patched = candidate_digest != digest
            relative = probe.relative_to(self.build_dir).as_posix()
            # The staged path is the staging root plus the member's own path
            # unless the build renamed the file on the way in. Printing it
            # whenever it is derivable repeats the member path once per member
            # and says nothing; where it is not, it is the interesting part.
            source = (
                "staged"
                if relative.endswith(f"/{wheel_path}")
                else f"staged at {relative}"
            )

        found = self._trace(
            wheel_path, candidate_digest, digest, source, soname, build_id
        )
        if found is not None and patched:
            found.origin = {
                "copied": "derived",
                "archive-copy": "archive-derived",
            }.get(found.origin, found.origin)
            found.exact = False
            found.detail += (
                "; the member differs from that staged file, so it was rewritten "
                "after staging"
            )
        return found

    def _trace(
        self,
        wheel_path: str,
        candidate_digest: str,
        digest: str,
        source: str,
        soname: str | None,
        build_id: str | None,
    ) -> Attribution | None:
        """Find what explains these bytes, in order of how much it proves."""
        # Zero bytes are identical everywhere, so a content match proves nothing:
        # it would hand an empty marker file to whichever component happened to be
        # indexed first, and invent that component's license obligation with it.
        # The path is the only evidence such a file carries.
        if candidate_digest == EMPTY_SHA256:
            return self._resolve_by_path(wheel_path, source, build_id, candidate_digest)

        artifacts = self._artifacts_by_hash.get(candidate_digest)
        if artifacts:
            return self._from_artifact(
                artifacts[0], wheel_path, f"{source}; exact build output"
            )

        # Identical bytes are not evidence of origin when several components
        # hold them: a stock Apache-2.0 text is the same file in every project
        # that ships one. Say so, and let the path decide, rather than naming
        # whichever happened to be indexed first.
        shared = self._shared_content(candidate_digest)
        if shared is not None:
            by_path = self._resolve_by_path(
                wheel_path, source, build_id, candidate_digest
            )
            if by_path is not None:
                return by_path
            return shared(wheel_path, source)

        component_file = self.source_files.by_hash.get(candidate_digest)
        if component_file:
            key = component_file.split("/", 1)[0].removesuffix("-src")
            return Attribution(
                path=wheel_path,
                origin="copied",
                exact=True,
                detail=f"{source}; byte-identical to {component_file}",
                components={key: {"copied-file"}},
                primary=key,
            )

        member = self.archives.by_hash.get(candidate_digest)
        if member:
            return Attribution(
                path=wheel_path,
                origin="archive-copy",
                exact=True,
                detail=f"{source}; byte-identical to {member.path} in {member.container}",
                components={member.container: {"extracted-file"}},
                primary=member.container,
            )

        repo_file = self.repo_files.by_hash.get(candidate_digest)
        if repo_file:
            return self._from_repo_file(
                wheel_path, source, repo_file, f"{source}; ", True
            )

        build_file = self.build_files.by_hash.get(candidate_digest)
        if build_file:
            # An empty file hashes the same as every other empty file, so the
            # first indexed path wins and says nothing. Where the build tree
            # holds the same bytes under the wheel's own path, that is the
            # better answer.
            if Path(build_file).name != Path(wheel_path).name:
                by_path = self.build_files.path_suffix_match(wheel_path)
                if by_path:
                    build_file = by_path
            owner = self._staged_component_of(build_file)
            return Attribution(
                path=wheel_path,
                # Transforming a component's own file leaves a derivative work,
                # which carries that component's terms. Only output that owes
                # nothing to a third party is merely "generated".
                origin="derived" if owner else "generated",
                detail=f"{source}; {build_file}",
                components={owner: {"derived-file"}} if owner else {},
                primary=owner,
            )

        # Content no longer matches, so the bytes were patched on the way in.
        # The build-id is what the linker wrote and patchelf leaves alone, which
        # makes it evidence where a matching file name is only a coincidence.
        if build_id:
            by_build_id = self._resolve_by_build_id(wheel_path, source, build_id)
            if by_build_id is not None:
                return by_build_id

        by_path = self._resolve_by_path(wheel_path, source, build_id, candidate_digest)
        if by_path is not None:
            return by_path

        if build_id:
            from_host = self._from_build_host_by_build_id(wheel_path, build_id)
            if from_host is not None:
                return from_host

        # Nothing this build produced or fetched explains these bytes. A shared
        # library reaching that point came from the machine -- whatever the
        # repair tool chose to call its directory.
        if soname:
            return self._from_build_host(wheel_path, soname)
        return None

    def _from_repo_file(
        self, wheel_path: str, source: str, repo_file: str, lead: str, same: bool
    ) -> Attribution:
        """A file this repository holds -- and, where it is not ours, whose.

        Checked-in third-party code states its own copyright, which is how the
        vendored scan found it. Reporting the member as plain repository source
        would drop that holder, and the component would be reported as
        contributing nothing to a wheel that ships its file.
        """
        owner = self.ownership.owner_of(self.repo_root / repo_file.removeprefix("./"))
        return Attribution(
            path=wheel_path,
            origin="repo-source",
            exact=same,
            detail=f"{lead}{repo_file}",
            components={owner: {"copied-file" if same else "derived-file"}}
            if owner
            else {},
            primary=owner,
        )

    def _staged_component_of(self, build_file: str) -> str | None:
        """The component whose tree this build staged under _deps/<name>/."""
        for prefix, key in self.staged_component_dirs.items():
            if build_file == prefix or build_file.startswith(prefix + "/"):
                return key
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

    def _shared_content(self, digest: str):
        """How to describe content that more than one component also holds.

        Returns None when the digest names exactly one component, which is the
        case worth stating as a fact. Otherwise it returns a builder that names
        every candidate and claims none of them, keeping each component's
        licence obligation without asserting a copy that did not happen.
        """
        owners: dict[str, str] = {}
        for display in self.source_files.all_by_hash.get(digest, []):
            owners[display.split("/", 1)[0].removesuffix("-src")] = display
        for member in self.archives.all_by_hash.get(digest, []):
            owners[member.container] = f"{member.path} in {member.container}"
        # This repository holding the same bytes counts as a candidate too: a
        # module that is nothing but an SPDX header is identical in every project
        # that uses the same header, and calling ours a copy of a dependency's
        # would ship a first-party file as redistributed third-party content.
        ours = self.repo_files.all_by_hash.get(digest, [])
        if len(owners) + bool(ours) < 2:
            return None

        def build(wheel_path: str, source: str) -> Attribution:
            where = ", ".join(sorted([*owners.values(), *ours]))
            return Attribution(
                path=wheel_path,
                origin="shared-content",
                detail=(
                    f"{source}; these bytes are held by more than one component "
                    f"({where}); the build does not record which supplied this file"
                ),
                components={key: {"shared-content"} for key in owners},
            )

        return build

    def _resolve_by_build_id(
        self, wheel_path: str, source: str, build_id: str
    ) -> Attribution | None:
        """The pristine bytes this member was patched from, named by build-id."""
        # The same library can sit in two archives under two sets of terms, and
        # these two SDKs share five build-ids. Every neighbouring index refuses
        # to name one of several candidates; this one must too, or a patched
        # member takes whichever archive happened to be walked first.
        members = self.archives.all_by_build_id.get(build_id, [])
        containers = {item.container for item in members}
        member = members[0] if len(containers) == 1 else None
        if member:
            return Attribution(
                path=wheel_path,
                origin="archive-derived",
                detail=(
                    f"{source}; build-id {build_id} matches {member.path} in "
                    f"{member.container} (origin sha256:{member.sha256})"
                ),
                components={member.container: {"extracted-file"}},
                primary=member.container,
            )

        artifacts = self._artifacts_by_build_id.get(build_id)
        if artifacts:
            return self._from_artifact(
                artifacts[0], wheel_path, f"{source}; build-id {build_id}, patched"
            )
        return None

    def _from_build_host_by_build_id(
        self, wheel_path: str, build_id: str
    ) -> Attribution | None:
        """The machine's own copy, found by build-id after a SONAME rewrite."""
        origin = resolve_system_library_by_build_id(build_id)
        if not origin.get("resolved_path"):
            return None
        soname = origin["soname"]
        self.system_libraries[soname] = origin
        component = system_component(soname, origin)
        self.components[component.key] = component

        detail = (
            f"build-id {build_id} matches the build machine's {origin['resolved_path']}"
        )
        if origin.get("package"):
            detail += f", supplied by {origin['package']}"
        return Attribution(
            path=wheel_path,
            origin="build-host-library",
            detail=detail,
            components={component.key: {"vendored-library"}},
            primary=component.key,
        )

    def _resolve_by_path(
        self,
        wheel_path: str,
        source: str,
        build_id: str | None = None,
        digest: str | None = None,
    ) -> Attribution | None:
        """Post-processing changes bytes; the path still says where they came from.

        patchelf, auditwheel's RPATH rewrite and the MJCF mesh stripping all leave
        a file that no longer hashes to its origin, so a unique path match is the
        remaining evidence -- recorded as derived, never as an exact copy.

        A name match is only ever a guess, so a build-id that disagrees overrules
        it: two libraries called libcloudxr.so from different SDKs are not each
        other, and claiming they are would hand one the other's licence.
        """
        name = Path(wheel_path).name

        members = [
            member
            for member in self.archives.by_name.get(name, [])
            if not _build_ids_disagree(build_id, member.build_id)
        ]
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

        artifacts = [
            artifact
            for artifact in self.graph.by_name(name)
            if not _build_ids_disagree(build_id, elf.read_build_id(artifact.output))
        ]
        if len(artifacts) == 1:
            return self._from_artifact(
                artifacts[0], wheel_path, f"{source}; modified build output"
            )

        # This repository first. A member matching a file this project tracks is
        # this project's copy; a dependency whose tree happens to end in the same
        # two path segments is a coincidence -- our own BSL-1.0 pool text was
        # being credited to an SDK that ships one under the same relative path.
        # A third-party file checked in here still carries its component, because
        # the repository branch looks its holder up.
        repo_file = self.repo_files.path_suffix_match(wheel_path)
        if repo_file:
            same = self.repo_files.digest_of_display(repo_file) == digest
            lead = f"{source}; {'identical to' if same else 'derived from'} "
            return self._from_repo_file(wheel_path, source, repo_file, lead, same)

        component_file = self.source_files.path_suffix_match(wheel_path)
        if component_file:
            key = component_file.split("/", 1)[0].removesuffix("-src")
            origin_digest = self.source_files.digest_of_display(component_file)
            # Reaching this branch means the content matched more than one place,
            # not that anything transformed the file. Where the input's bytes are
            # these bytes it is a copy, and quoting an origin digest equal to the
            # member's own would say nothing.
            same = origin_digest == digest
            return Attribution(
                path=wheel_path,
                origin="copied" if same else "derived",
                exact=same,
                detail=(
                    f"{source}; "
                    + ("identical to " if same else "derived from ")
                    + component_file
                    + (
                        ""
                        if same or not origin_digest
                        else f" (origin sha256:{origin_digest})"
                    )
                ),
                components={key: {"copied-file" if same else "derived-file"}},
                primary=key,
            )

        build_file = self.build_files.path_suffix_match(wheel_path)
        if build_file:
            owner = self._staged_component_of(build_file)
            return Attribution(
                path=wheel_path,
                origin="derived" if owner else "generated",
                detail=f"{source}; {build_file}",
                components={owner: {"derived-file"}} if owner else {},
                primary=owner,
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
        attribution.copyright_text = _copyright_text(entry)
        attributions[entry.name] = attribution

    _resolve_shared_by_sibling(attributions)

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


def _build_ids_disagree(left: str | None, right: str | None) -> bool:
    """True only when both are known and differ -- that is evidence, not absence."""
    return bool(left and right and left != right)


def _resolve_shared_by_sibling(attributions: dict[str, Attribution]) -> None:
    """Let a staged directory settle what identical bytes cannot.

    `libcloudxr.so` differs between two SDKs and resolves by build-id, but the
    libraries beside it are byte-identical in both, and MuJoCo's licence text is
    the same file every Apache-2.0 project ships. In each case the directory the
    build staged the member into is already explained by exactly one of the
    candidates, which is a record of where it came from -- so saying the build
    kept none would be untrue, and naming every candidate overstates what the
    others contribute.
    """
    settled: dict[str, set[str]] = {}
    for name, item in attributions.items():
        if item.primary:
            settled.setdefault(name.rsplit("/", 1)[0], set()).add(item.primary)

    for name, item in attributions.items():
        if item.primary or item.origin != "shared-content":
            continue
        owners = settled.get(name.rsplit("/", 1)[0], set()) & set(item.components)
        if len(owners) != 1:
            continue
        key = owners.pop()
        item.origin = "copied"
        item.exact = True
        item.detail = (
            f"{item.detail.split(';')[0]}; these bytes are held by more than one "
            f"component, and the directory it is staged into is explained by {key}"
        )
        item.components = {key: {"copied-file"}}
        item.primary = key


# What a packaging backend writes, and SBOM documents other tools leave here.
# `licenses/` is deliberately absent: the project's own texts are copies of files
# this repository tracks, so they trace like any other copy, and the texts this
# collector adds are attributed as it adds them. Calling the directory metadata
# asserted an origin for files that have a real one.
_DIST_INFO_WRITTEN = re.compile(
    r"^(METADATA|WHEEL|RECORD|INSTALLER|REQUESTED|entry_points\.txt|top_level\.txt"
    r"|direct_url\.json|zip-safe|namespace_packages\.txt)$|^sboms/"
)


def _attribute(
    entry: Entry,
    wheel: WheelInfo,
    resolver: Resolver,
    staged_root: Path | None,
) -> Attribution | None:
    if entry.name.startswith(f"{wheel.dist_info}/"):
        relative = entry.name[len(wheel.dist_info) + 1 :]
        # Only what a packaging backend and this collector actually write there.
        # Treating the whole directory as metadata asserted an origin for
        # anything placed in it, which is the one route around the gate that
        # every other member has to pass.
        if _DIST_INFO_WRITTEN.match(relative):
            return Attribution(
                path=entry.name,
                origin="metadata",
                exact=True,
                detail="distribution metadata written by the build",
            )

    probe = staged_root / entry.name if staged_root is not None else None
    return resolver.resolve(
        entry.name, entry.sha256, probe, entry.soname, entry.build_id
    )


def _copyright_text(entry: Entry) -> str | None:
    """The notice a file states about itself, beside the identifier."""
    if Path(entry.name).suffix.lower() not in _TAGGABLE:
        return None
    return licensing.read_copyright(entry.head.decode("utf-8", "replace"))


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
    if normalized == "NOASSERTION":
        return None
    # A LicenseRef is only meaningful alongside its text, and a file tag carries
    # none: naming one here produced a document that failed its own check for an
    # undefined reference, on a wheel the build had already rewritten and would
    # refuse to process again.
    if licensing.license_refs(normalized):
        return None
    return normalized

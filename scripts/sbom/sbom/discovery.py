# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find out what went into a build by reading what the build left behind.

There is no dependency list to keep in step with the code. A component exists
because a source tree was fetched into the build, and it reaches a wheel because
the link graph or a content hash says so.

Four indexes do the work:

* source trees   ``_deps/*-src`` checkouts -> identity, from git
* build graph    CMake's file API codemodel -> what links and compiles into what
* archives       members of tarballs in the repo, by content hash
* files          repo and build-tree files, by content hash
"""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from packageurl import PackageURL

from . import elf, licensing

# Directories that never hold an input worth indexing.
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tar", ".zip")
_MAX_ARCHIVE_BYTES = 1 << 31

_GITHUB_REMOTE = re.compile(
    r"github\.com[:/](?P<org>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)
_GITCLONE_URL = re.compile(r'clone\b[^\n]*?"(?P<url>[a-z]+[^"\s]*://[^"]+|git@[^"]+)"')
_GITCLONE_TAG = re.compile(r'(?<![-\w])checkout\s+"(?P<tag>[^"]+)"')
_INCLUDE_FLAG = re.compile(r"-(?:I|isystem)\s*(\S+)")
_DEPS_SOURCE = re.compile(r"_deps/(?P<name>[A-Za-z0-9_.+-]+)-src(?:/|$)")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(root: Path):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in _SKIP_DIRS:
                    stack.append(entry)
            elif entry.is_file():
                yield entry


# ==============================================================================
# Source trees
# ==============================================================================


@dataclass(frozen=True)
class Component:
    """Something a wheel can contain, identified from where it came from."""

    key: str
    name: str
    kind: str  # "source-tree" | "archive" | "system-library"
    supplier: str
    homepage: str
    purl: str
    version: str
    download_location: str
    source_info: str
    root: Path | None = None
    license_concluded: str = "NOASSERTION"
    license_declared: str = "NOASSERTION"
    evidence: tuple[licensing.LicenseEvidence, ...] = ()


def git(repo: Path, *args: str) -> str | None:
    """Run a read-only git query, or return None if git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (
        result.stdout.strip()
        if result.returncode == 0 and result.stdout.strip()
        else None
    )


def _identity(
    remote: str | None, name: str, commit: str | None
) -> tuple[str, str, str, str]:
    """Supplier, homepage, purl and download location, read off the remote URL."""
    if remote:
        match = _GITHUB_REMOTE.search(remote)
        if match:
            org, repo = match.group("org"), match.group("repo")
            purl = PackageURL(
                type="github", namespace=org, name=repo, version=commit
            ).to_string()
            download = f"git+https://github.com/{org}/{repo}.git" + (
                f"@{commit}" if commit else ""
            )
            return (
                f"Organization: {org}",
                f"https://github.com/{org}/{repo}",
                purl,
                download,
            )
        purl = f"pkg:generic/{name}" + (f"@{commit}" if commit else "")
        return (
            "NOASSERTION",
            remote,
            purl,
            (f"git+{remote}@{commit}" if commit else remote),
        )
    return "NOASSERTION", "NOASSERTION", f"pkg:generic/{name}", "NOASSERTION"


def _declared_constraint(deps_dir: Path, name: str) -> dict[str, str | None]:
    clone = (
        deps_dir
        / f"{name}-subbuild"
        / f"{name}-populate-prefix"
        / "tmp"
        / f"{name}-populate-gitclone.cmake"
    )
    if not clone.is_file():
        return {"url": None, "tag": None}
    text = clone.read_text(encoding="utf-8", errors="replace")
    url = _GITCLONE_URL.search(text)
    tag = _GITCLONE_TAG.search(text)
    return {
        "url": url.group("url") if url else None,
        "tag": tag.group("tag") if tag else None,
    }


def discover_source_trees(deps_dir: Path) -> dict[str, Component]:
    """Every FetchContent checkout under the build's ``_deps``."""
    components: dict[str, Component] = {}
    for root in sorted(deps_dir.glob("*-src")):
        if not root.is_dir():
            continue
        key = root.name.removesuffix("-src")
        declared = _declared_constraint(deps_dir, key)
        commit = git(root, "rev-parse", "HEAD")
        remote = git(root, "config", "--get", "remote.origin.url") or declared["url"]
        describe = git(root, "describe", "--tags", "--always")

        supplier, homepage, purl, download = _identity(remote, key, commit)
        evidence = licensing.discover_in_tree(key, root, root.name)
        concluded, declared_expression = licensing.expression(evidence, key)

        details = [f"Fetched into {root.name}."]
        if declared["tag"]:
            details.append(f"Declared constraint: {declared['tag']}.")
        if describe:
            details.append(f"git describe: {describe}.")

        name_match = _GITHUB_REMOTE.search(remote) if remote else None
        components[key] = Component(
            key=key,
            name=name_match.group("repo") if name_match else key,
            kind="source-tree",
            supplier=supplier,
            homepage=homepage,
            purl=purl,
            version=commit or declared["tag"] or "NOASSERTION",
            download_location=download,
            source_info=" ".join(details),
            root=root,
            license_concluded=concluded,
            license_declared=declared_expression,
            evidence=tuple(evidence),
        )
    return components


# ==============================================================================
# Build graph
# ==============================================================================


@dataclass(frozen=True)
class Artifact:
    """One file a target produces, and the target that produced it."""

    output: Path
    target: str
    target_id: str


@dataclass(frozen=True)
class _Node:
    """A target as the CMake file API describes it."""

    name: str
    kind: str
    sources: tuple[Path, ...]
    includes: tuple[Path, ...]
    dependencies: tuple[str, ...]


def _owner(path: Path, ownership: Ownership | None) -> str | None:
    """The component a compiled source belongs to, if any."""
    match = _DEPS_SOURCE.search(path.as_posix())
    if match:
        return match.group("name")
    return ownership.owner_of(path) if ownership is not None else None


class FileApiError(Exception):
    """The build tree carries no CMake file API reply to read."""


class BuildGraph:
    """The link and compile graph, read from CMake's own file API.

    The top-level ``CMakeLists.txt`` asks for the codemodel with
    ``cmake_file_api()``, so any configure of this project leaves one behind,
    under every generator. Scraping ``link.txt`` instead would only work for the
    Makefile generators -- Ninja writes no such file.
    """

    def __init__(self, build_dir: Path, config: str | None = None) -> None:
        self.build_dir = build_dir
        self.artifacts: dict[Path, Artifact] = {}
        self._nodes: dict[str, _Node] = {}
        self._by_name: dict[str, list[Artifact]] = {}
        self._load(config)

    def _load(self, config: str | None) -> None:
        reply = self.build_dir / ".cmake" / "api" / "v1" / "reply"
        models = sorted(reply.glob("codemodel-v2-*.json")) if reply.is_dir() else []
        if not models:
            raise FileApiError(
                f"no CMake file API codemodel under {reply}. Configure the project "
                "with CMake 3.27 or newer, which is where cmake_file_api() asks for it."
            )

        codemodel = json.loads(models[-1].read_text(encoding="utf-8"))
        source_root = Path(codemodel["paths"]["source"])
        build_root = Path(codemodel["paths"]["build"])

        configurations = codemodel["configurations"]
        chosen = next(
            (item for item in configurations if item["name"] == (config or "")),
            configurations[0],
        )

        def resolve(raw: str, root: Path) -> Path:
            path = Path(raw)
            return path if path.is_absolute() else (root / path)

        for entry in chosen["targets"]:
            target = json.loads((reply / entry["jsonFile"]).read_text(encoding="utf-8"))
            includes = {
                resolve(item["path"], source_root).resolve()
                for group in target.get("compileGroups", [])
                for item in group.get("includes", [])
            }
            self._nodes[entry["id"]] = _Node(
                name=target["name"],
                kind=target["type"],
                sources=tuple(
                    resolve(item["path"], source_root).resolve()
                    for item in target.get("sources", [])
                ),
                includes=tuple(sorted(includes)),
                dependencies=tuple(
                    item["id"] for item in target.get("dependencies", [])
                ),
            )
            for item in target.get("artifacts", []):
                output = resolve(item["path"], build_root).resolve()
                artifact = Artifact(
                    output=output, target=target["name"], target_id=entry["id"]
                )
                self.artifacts[output] = artifact
                self._by_name.setdefault(output.name, []).append(artifact)

    def by_name(self, name: str) -> list[Artifact]:
        return self._by_name.get(name, [])

    def _closure(self, target_id: str) -> list[_Node]:
        """The target and everything it depends on, transitively."""
        seen: set[str] = set()
        stack = [target_id]
        found: list[_Node] = []
        while stack:
            current = stack.pop()
            if current in seen or current not in self._nodes:
                continue
            seen.add(current)
            node = self._nodes[current]
            found.append(node)
            stack.extend(node.dependencies)
        return found

    def contributions(
        self, artifact: Artifact, ownership: Ownership | None = None
    ) -> dict[str, set[str]]:
        """Which components contributed to an artifact, and in what way.

        Walks the target's dependency closure. Compiled sources name a component
        outright; include paths are the only trace a header-only dependency
        leaves anywhere.
        """
        found: dict[str, set[str]] = {}

        def note(key: str, how: str) -> None:
            found.setdefault(key, set()).add(how)

        for node in self._closure(artifact.target_id):
            for source in node.sources:
                owner = _owner(source, ownership)
                if owner:
                    note(owner, "compiled-source")
            for include in node.includes:
                match = _DEPS_SOURCE.search(include.as_posix())
                if match:
                    note(match.group("name"), "header-include")
                elif ownership is not None:
                    # Which file in the directory was used is unknowable, so
                    # every component with files there is credited, the same
                    # way a fetched include directory is.
                    for owner in ownership.owners_under(include):
                        note(owner, "header-include")
        return found

    def first_party(
        self, artifact: Artifact, repo_root: Path, ownership: Ownership | None = None
    ) -> bool:
        """True when this build compiled the artifact from the project's own sources."""
        for node in self._closure(artifact.target_id):
            if node.name != artifact.target:
                continue
            for source in node.sources:
                if _owner(source, ownership):
                    continue
                try:
                    source.relative_to(repo_root)
                except ValueError:
                    continue
                return True
        return False


# ==============================================================================
# Ownership
# ==============================================================================


class Ownership:
    """Which component a path belongs to.

    One lookup for every kind of third-party code, so recognising a new kind is
    registering a root rather than adding a branch wherever paths are matched.
    """

    def __init__(self) -> None:
        self._files: dict[Path, str] = {}
        self._roots: list[tuple[Path, str]] = []

    def register_root(self, root: Path, key: str) -> None:
        self._roots.append((root.resolve(), key))

    def register_file(self, path: Path, key: str) -> None:
        self._files[path.resolve()] = key

    def owner_of(self, path: Path) -> str | None:
        resolved = path.resolve()
        owner = self._files.get(resolved)
        if owner is not None:
            return owner
        # Longest root wins, so a vendored subdirectory beats its parent.
        best: tuple[int, str] | None = None
        for root, key in self._roots:
            text = str(root)
            if str(resolved) == text or str(resolved).startswith(text + "/"):
                if best is None or len(text) > best[0]:
                    best = (len(text), key)
        return best[1] if best else None

    def owners_under(self, directory: Path) -> set[str]:
        """Components with files inside a directory, for include-path evidence."""
        prefix = str(directory.resolve())
        found = {
            key
            for path, key in self._files.items()
            if str(path) == prefix or str(path).startswith(prefix + "/")
        }
        found.update(
            key
            for root, key in self._roots
            if str(root) == prefix or str(root).startswith(prefix + "/")
        )
        return found


# ==============================================================================
# Third-party sources vendored in this repository
# ==============================================================================
# Not everything third-party arrives through FetchContent. A file checked in
# here that someone else wrote says so in its REUSE header, which is the same
# thing the project states about its own files -- so the copyright holder is
# what separates them, not the directory it happens to sit in.

# REUSE's tag where a file has one, and a plain copyright line where it does
# not -- upstream files usually carry the latter.
_COPYRIGHT_TAG = re.compile(
    r"SPDX-FileCopyrightText:\s*(?P<holder>[^\n\r]+)"
    r"|^[^\w\n]*Copyright\b(?P<plain>[^\n\r]+)",
    re.MULTILINE,
)
_YEARS = re.compile(r"(copyright|\(c\)|©|\d{4}(\s*-\s*\d{4})?|,|\.)", re.IGNORECASE)


def project_authors(repo_root: Path) -> set[str]:
    """Who the project says it is, read from its own packaging metadata."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return set()
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    return {
        str(author["name"]).lower()
        for author in data.get("project", {}).get("authors", [])
        if author.get("name")
    }


def _holder_of(path: Path) -> tuple[str, str | None] | None:
    """The copyright holder and licence a file states about itself."""
    try:
        head = path.open("rb").read(4096).decode("utf-8", "replace")
    except OSError:
        return None
    match = _COPYRIGHT_TAG.search(head)
    if not match:
        return None
    holder = _YEARS.sub(" ", match.group("holder") or match.group("plain") or "")
    holder = " ".join(holder.split()).removesuffix(" All rights reserved").strip()
    return (holder, licensing.read_spdx_tag(head)) if holder else None


def discover_vendored(
    repo_root: Path, candidates: set[Path], authors: set[str]
) -> tuple[dict[str, Component], dict[str, list[Path]]]:
    """Group checked-in files whose copyright holder is not the project's."""
    grouped: dict[str, list[Path]] = {}
    licences: dict[str, set[str]] = {}
    names: dict[str, str] = {}

    for path in sorted(candidates):
        stated = _holder_of(path)
        if stated is None:
            continue
        holder, expression = stated
        if any(author in holder.lower() for author in authors):
            continue
        key = f"vendored:{licensing.spdx_safe(holder).lower()}"
        grouped.setdefault(key, []).append(path)
        names[key] = holder
        if expression:
            licences.setdefault(key, set()).add(expression)

    components: dict[str, Component] = {}
    for key, paths in grouped.items():
        evidence = licensing.pool_evidence(repo_root, key, licences.get(key, set()))
        concluded, declared = licensing.expression(evidence, key)
        if licences.get(key):
            declared = licensing.combine(sorted(licences[key]))
        relative = ", ".join(
            sorted(str(item.relative_to(repo_root)) for item in paths)[:4]
        )
        components[key] = Component(
            key=key,
            name=names[key],
            kind="vendored-source",
            supplier=f"Organization: {names[key]}",
            homepage="NOASSERTION",
            purl=PackageURL(
                type="generic", name=licensing.spdx_safe(names[key]).lower()
            ).to_string(),
            version="NOASSERTION",
            download_location="NOASSERTION",
            source_info=(
                f"Checked into this repository rather than fetched; states its own "
                f"copyright. Files: {relative}."
            ),
            license_concluded=concluded,
            license_declared=declared,
            evidence=tuple(evidence),
        )
    return components, grouped


# ==============================================================================
# The build host
# ==============================================================================
# The last discovery source: a shared library that no index explains was not
# produced or fetched by this build, so ask the machine that supplied it.

_SEARCH_ROOTS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib",
    "/lib/x86_64-linux-gnu",
    "/lib/aarch64-linux-gnu",
    "/usr/local/lib",
)


def _host_library_paths() -> list[Path]:
    """Every shared library the loader can see, plus the CUDA trees it cannot."""
    paths: set[Path] = set()
    if shutil.which("ldconfig"):
        try:
            result = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            for line in result.stdout.splitlines():
                _, _, target = line.partition("=>")
                target = target.strip()
                if target:
                    paths.add(Path(target))
        except (OSError, subprocess.SubprocessError):
            pass
    roots = [Path(item) for item in _SEARCH_ROOTS if Path(item).is_dir()]
    roots += [Path(item) for item in sorted(glob.glob("/usr/local/cuda*/lib64"))]
    for root in roots:
        paths.update(item for item in root.glob("*.so*") if item.is_file())
    return sorted(paths)


@functools.lru_cache(maxsize=1)
def _host_build_id_index() -> dict[str, str]:
    """Host libraries by GNU build-id.

    A repair tool rewrites SONAME on the way into the wheel, so the name is no
    longer a way back to the machine's copy. The build-id is, and it is the same
    note the linker wrote. Built once; reading it from every library the loader
    knows costs about a second.
    """
    index: dict[str, str] = {}
    for path in _host_library_paths():
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not elf.is_elf(data[:4]):
            continue
        try:
            build_id = elf.read_dynamic(data).build_id
        except Exception:  # noqa: BLE001 - a malformed host library is not ours to fix
            continue
        if not build_id:
            continue
        # A regular file beats a symlink to it: dpkg-query knows the real path.
        current = index.get(build_id)
        if current is None or (
            path.is_symlink() is False and Path(current).is_symlink()
        ):
            index[build_id] = str(path)
    return index


def resolve_system_library_by_build_id(build_id: str) -> dict:
    """Trace a vendored library back by build-id, whatever it was renamed to."""
    origin: dict = {
        "soname": None,
        "build_id": build_id,
        "resolved_path": None,
        "package": None,
        "copyright": None,
    }
    found = _host_build_id_index().get(build_id)
    if found is None:
        return origin
    library = Path(found).resolve()
    origin["soname"] = library.name
    origin["resolved_path"] = str(library)
    return _identify_host_package(library, origin)


def resolve_system_library(soname: str) -> dict:
    """Trace an auditwheel-vendored library back to the package that supplied it."""
    origin: dict = {
        "soname": soname,
        "resolved_path": None,
        "package": None,
        "copyright": None,
    }
    roots = [Path(item) for item in _SEARCH_ROOTS if Path(item).is_dir()]
    roots += [Path(item) for item in sorted(glob.glob("/usr/local/cuda*/lib64"))]

    library = next((root / soname for root in roots if (root / soname).exists()), None)
    if library is None:
        return origin
    library = library.resolve()
    origin["resolved_path"] = str(library)
    return _identify_host_package(library, origin)


def _identify_host_package(library: Path, origin: dict) -> dict:
    """Name the distro package that owns a resolved host library."""
    if shutil.which("dpkg-query") is None:
        return origin
    try:
        result = subprocess.run(
            ["dpkg-query", "-S", str(library)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return origin
    if result.returncode != 0 or ":" not in result.stdout:
        return origin

    package = result.stdout.split(":", 1)[0].strip()
    origin["package"] = package
    copyright_file = Path("/usr/share/doc") / package / "copyright"
    if copyright_file.is_file():
        origin["copyright"] = str(copyright_file)
    return origin


def _license_near(library: Path) -> Path | None:
    """A toolchain that installs outside the package manager still ships its terms.

    CUDA is the case that matters: the runtime lands in /usr/local/cuda*/lib64 with
    EULA.txt a level or two up, and dpkg knows nothing about it.
    """
    current = library.parent
    for _ in range(3):
        for entry in sorted(current.iterdir()) if current.is_dir() else []:
            if entry.is_file() and licensing._classify(entry.name):  # noqa: SLF001
                return entry
            if entry.is_file() and entry.name.upper().startswith("EULA"):
                return entry
        if current.parent == current:
            break
        current = current.parent
    return None


def system_component(soname: str, origin: dict) -> Component:
    """A pre-built library the build machine supplied, named by what it came from."""
    evidence: list[licensing.LicenseEvidence] = []
    key = f"system:{soname}"
    license_path = origin.get("copyright")
    if not license_path and origin.get("resolved_path"):
        found = _license_near(Path(origin["resolved_path"]))
        license_path = str(found) if found else None
        origin["license_file"] = license_path
    if license_path and Path(license_path).is_file():
        text = Path(license_path).read_text(encoding="utf-8", errors="replace")
        evidence.append(
            licensing._evidence(key, "build-host", license_path, text, "grant")  # noqa: SLF001
        )
    concluded, declared = licensing.expression(evidence, soname)

    package = origin.get("package")
    details = [
        f"Vendored into the wheel by auditwheel from {origin.get('resolved_path') or soname}."
    ]
    if package:
        details.append(f"Supplied by system package {package}.")
    return Component(
        key=key,
        name=soname,
        kind="system-library",
        supplier=f"Organization: {package}" if package else "NOASSERTION",
        homepage="NOASSERTION",
        purl=PackageURL(type="generic", name=soname).to_string(),
        version="NOASSERTION",
        download_location="NOASSERTION",
        source_info=" ".join(details),
        license_concluded=concluded,
        license_declared=declared,
        evidence=tuple(evidence),
    )


# ==============================================================================
# Content indexes
# ==============================================================================


@dataclass(frozen=True)
class Member:
    """A file inside an archive, or a file on disk."""

    container: str
    path: str
    sha256: str
    size: int


class ArchiveIndex:
    """Members of every archive in the repo, so extracted bytes can be traced back."""

    def __init__(self) -> None:
        self.by_hash: dict[str, Member] = {}
        self.by_name: dict[str, list[Member]] = {}
        # Survives the RPATH/SONAME rewrite a repair tool applies on the way in,
        # so a patched wheel member still points at the archive it came from.
        self.by_build_id: dict[str, Member] = {}
        self.archives: dict[str, dict] = {}

    def add_archive(self, path: Path, display: str) -> None:
        if path.stat().st_size > _MAX_ARCHIVE_BYTES:
            return
        members = self._read_members(path, display)
        if members is None:
            return
        self.archives[display] = {
            "path": display,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "member_count": len(members),
        }
        for member, data in members:
            self.by_hash.setdefault(member.sha256, member)
            self.by_name.setdefault(Path(member.path).name, []).append(member)
            if elf.is_elf(data[:4]):
                try:
                    build_id = elf.read_dynamic(data).build_id
                except Exception:  # noqa: BLE001 - a malformed member is not ours to fix
                    build_id = None
                if build_id:
                    self.by_build_id.setdefault(build_id, member)
            self._maybe_license(display, member, data)

    def _maybe_license(self, display: str, member: Member, data: bytes) -> None:
        name = Path(member.path).name
        if Path(member.path).parent.as_posix() not in {".", ""}:
            return  # only the archive root states the bundle's own terms
        if name.upper() == "VERSION" and len(data) < 256:
            self.archives[display]["version_text"] = data.decode(
                "utf-8", "replace"
            ).strip()
            return
        if not licensing._classify(name):  # noqa: SLF001 - same package
            return
        self.archives[display].setdefault("license_members", []).append((member, data))

    @staticmethod
    def _read_members(path: Path, display: str) -> list[tuple[Member, bytes]] | None:
        """Hash every member; keep the bytes only of the few worth keeping.

        An SDK tarball expands to far more than it compresses to, so holding
        every member would size peak memory by the largest archive rather than
        by what is retained.
        """
        collected: list[tuple[Member, bytes]] = []

        def take(name: str, data: bytes) -> None:
            member = Member(display, name, sha256_bytes(data), len(data))
            keep = b"" if Path(name).parent.as_posix() not in {".", ""} else data
            collected.append((member, keep))

        try:
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if not info.is_dir():
                            take(info.filename, archive.read(info.filename))
            else:
                with tarfile.open(path) as archive:
                    for info in archive:
                        if not info.isfile():
                            continue
                        handle = archive.extractfile(info)
                        if handle is not None:
                            take(info.name.removeprefix("./"), handle.read())
        except (OSError, tarfile.TarError, zipfile.BadZipFile):
            return None
        return collected


def discover_archives(repo_root: Path, skip: tuple[Path, ...] = ()) -> ArchiveIndex:
    """Index archives held in the repository, not ones a build unpacked."""
    index = ArchiveIndex()
    skipped = tuple(item.resolve() for item in skip)
    for path in _walk(repo_root):
        if any(str(path).startswith(str(item)) for item in skipped):
            continue
        name = path.name.lower()
        if not name.endswith(ARCHIVE_SUFFIXES):
            continue
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        index.add_archive(path, relative)
    return index


class FileIndex:
    """Files on disk by content hash, for tracing a copied wheel member back."""

    def __init__(self) -> None:
        self.by_hash: dict[str, str] = {}
        self.by_suffix: dict[str, list[str]] = {}
        self._digests: dict[str, str] = {}

    def digest_of(self, path: Path) -> str | None:
        """The digest this index already computed for a path, if it indexed it."""
        return self._digests.get(str(path.resolve()))

    def add_tree(
        self, root: Path, display_root: str, skip: tuple[Path, ...] = ()
    ) -> None:
        skipped = tuple(item.resolve() for item in skip)
        for path in _walk(root):
            if any(str(path).startswith(str(item)) for item in skipped):
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                continue
            display = f"{display_root}/{relative}"
            try:
                digest = sha256_file(path)
            except OSError:
                continue
            self._digests[str(path.resolve())] = digest
            self.by_hash.setdefault(digest, display)
            # Two trailing segments is enough to disambiguate an __init__.py.
            self.by_suffix.setdefault(
                "/".join(relative.rsplit("/", 2)[-2:]), []
            ).append(display)

    def path_suffix_match(self, wheel_path: str) -> str | None:
        key = "/".join(wheel_path.rsplit("/", 2)[-2:])
        matches = self.by_suffix.get(key, [])
        return matches[0] if len(matches) == 1 else None

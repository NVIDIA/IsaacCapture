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
import os
import platform
import re
import shlex
import subprocess
import tomllib
import tarfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
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
# How much of a library nested in an archive is kept, to read its build-id from.
# An SDK ships its libraries under `lib/`, not at the archive root, so without
# this the only members with a build-id are the ones nothing links against.
_ELF_HEAD_BYTES = 1 << 16

_GITHUB_REMOTE = re.compile(
    r"github\.com[:/](?P<org>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)
_GITCLONE_URL = re.compile(r'clone\b[^\n]*?"(?P<url>[a-z]+[^"\s]*://[^"]+|git@[^"]+)"')
_GITCLONE_TAG = re.compile(r'(?<![-\w])checkout\s+"(?P<tag>[^"]+)"')
_INCLUDE_FLAG = re.compile(r"-(?:I|isystem)\s*(\S+)")
_DEPS_SOURCE = re.compile(r"_deps/(?P<name>[A-Za-z0-9_.+-]+)-src(?:/|$)")


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def component_of(display: str) -> str:
    """The component a `FileIndex` display path belongs to.

    `add_tree` renders `<tree-name>/<relative>` and a fetched checkout's tree is
    `<key>-src`, so this is the inverse of how the index was populated. Inverting
    it at the call site made the convention an undeclared contract.
    """
    return display.split("/", 1)[0].removesuffix("-src")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(root: Path):
    """Every regular file under `root`, never following a symlink."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in _SKIP_DIRS:
                    stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)


# ==============================================================================
# Source trees
# ==============================================================================


@dataclass(frozen=True)
class Component:
    """Something a wheel can contain, identified from where it came from."""

    key: str
    name: str
    # "source-tree" | "archive" | "system-library" | "vendored-source" | "vcpkg-port"
    kind: str
    supplier: str
    homepage: str
    purl: str
    version: str
    download_location: str
    source_info: str
    root: Path | None = None
    license_concluded: str = "NOASSERTION"
    license_declared: str = "NOASSERTION"
    # The component's own copyright notice, where the build can read one. Naming
    # a holder and then declaring none would contradict the same document.
    copyright_text: str = "NOASSERTION"
    evidence: tuple[licensing.LicenseEvidence, ...] = ()
    # (path, reason) for checked-in files whose REUSE info fails the per-file
    # gate. Pooling a holder's licences would otherwise lend one file the terms
    # of another, or void them all.
    unlicensed: tuple[tuple[str, str], ...] = ()


def repo_head(directory: Path) -> str | None:
    """The commit of the checkout rooted at `directory`, if it is one.

    `git -C <dir> rev-parse HEAD` answers for whichever repository encloses the
    directory, so a dependency whose checkout carries no .git of its own would
    otherwise be stamped with this project's commit.
    """
    top = git(directory, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != directory.resolve():
        return None
    return git(directory, "rev-parse", "HEAD")


def git(repo: Path, *args: str) -> str | None:
    """Run a read-only git query, or return None if git cannot answer."""
    return _ask("git", "-C", str(repo), *args).strip() or None


def notice_from(evidence) -> str:
    """The copyright a component's own licence text states, where it states one.

    Permissive licences generally require the notice to travel with the terms,
    and for MIT and the BSDs it is a line of the same file -- shipped in the
    wheel but, until now, named nowhere a reader could act on.
    """
    # Grants only, though a pool counts as licence evidence everywhere else: a
    # pool is a directory of reference texts, and the notice in one is the
    # licence's own author -- OpenXR ships WTFPL, so reading its pool credited
    # OpenXR to the person who wrote WTFPL.
    notices = [
        found
        for item in evidence
        if item.kind in ("grant", "nested")
        for found in licensing.read_notices(item.text)
        if not licensing.notice_is_the_licence_authors(
            found, [license_id for license_id, _, _ in item.matches]
        )
    ]
    return "\n".join(licensing.fold_notices(notices)) if notices else "NOASSERTION"


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
        commit = repo_head(root)
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
            copyright_text=notice_from(evidence),
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
    # What its link line names. Only linkable targets have one; a static
    # library's own requirements land on its consumers' lines.
    linked: tuple[Path, ...] = ()  # outputs of targets in this graph
    inputs: tuple[LinkInput, ...] = ()  # everything else that can carry code


@dataclass(frozen=True)
class LinkInput:
    """An archive or object a link line names, or a name it could not resolve."""

    named: str  # as the link line spells it
    path: Path | None
    kind: str  # "archive" | "object" | "unresolved"
    reason: str = ""
    members: int | None = None  # an archive's object files

    @property
    def key(self) -> str:
        return str(self.path) if self.path else self.named


@dataclass
class _Line:
    """One target's link line while it is read."""

    link_dir: Path
    directories: list[Path]
    sysroot: str = ""
    linked: list[Path] = field(default_factory=list)
    inputs: list[LinkInput] = field(default_factory=list)


@dataclass(frozen=True)
class _Search:
    """Where `-l` looks after the `-L` directories, and the sysroot `=` names.

    The linker's own directories are read under `sysroot`, which a target's
    CMAKE_SYSROOT replaces.
    """

    compiler_dirs: list[Path]
    compiler: str | None
    sysroot: str

    @property
    def directories(self) -> list[Path]:
        linker = (
            _linker_search_dirs(self.compiler, self.sysroot) if self.compiler else []
        )
        return self.compiler_dirs + linker


# Driver and linker options whose value is the next word. Skipping the value
# keeps a version script or an rpath from being read as an input.
_DRIVER_TAKES_VALUE = frozenset(
    """
    -o -x -T -u -e -z -I -B -isystem -include -imacros -MF -MT -MQ
    -Xassembler -Xpreprocessor --param
    """.split()
)
_LINKER_TAKES_VALUE = frozenset(
    """
    -rpath --rpath -rpath-link --rpath-link --version-script -soname --soname -h
    -z -T --script -Map --Map --dynamic-list -e --entry -u --undefined --defsym
    -o --output --sysroot -y --trace-symbol --wrap -m -R --just-symbols
    -dynamic-linker --dynamic-linker -I -F -f --filter --auxiliary --exclude-libs
    -plugin --plugin -plugin-opt --plugin-opt --section-start -Ttext -Tdata -Tbss
    --retain-symbols-file -Y -G --gpsize -a -A --architecture -b --format -assert
    -init -fini -c --mri-script -P --depaudit --audit --require-defined
    --export-dynamic-symbol --export-dynamic-symbol-list --dependency-file
    --sort-section --section-ordering-file
    """.split()
)
_STATIC_MODE = frozenset({"-Bstatic", "-dn", "-non_shared", "-static"})
_DYNAMIC_MODE = frozenset({"-Bdynamic", "-dy", "-call_shared"})
# GNU ld script commands that name inputs; AS_NEEDED nests inside them.
_SCRIPT_INPUTS = re.compile(r"\b(?:INPUT|GROUP)\s*\(")
_SCRIPT_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SCRIPT_DEPTH = 8
_NOT_FOUND = (
    "not found; if it is the value of a linker option, that option is one the "
    "collector does not know takes a value"
)
_UNKNOWN_KIND = "not an archive, an object, a shared library or a linker script"
_UNSEARCHED = (
    "found in no -L directory, compiler link directory or linker search directory"
)
_NO_COMPILER = (
    "found in no -L directory, and the build records no compiler to ask for its "
    "search directories; reconfigure so the file API writes its toolchains reply"
)


def _linker_words(fragments: list[dict]) -> tuple[list[str], list[str]]:
    """What a link line hands the linker, in order, and fragments that do not parse.

    Every role is read, because CMake files a path given as a link option
    under `flags`. Driver options go with their values; `-Wl,` lists and
    `-Xlinker` words are unpacked, which is how `--whole-archive` wrappers
    arrive. The driver's `-static` governs the whole line wherever it sits.
    """
    stream: list[str] = []
    unparsed: list[str] = []
    static = False
    for fragment in fragments:
        try:
            words = iter(shlex.split(fragment["fragment"]))
        except ValueError:
            unparsed.append(fragment["fragment"])
            continue
        if fragment.get("role") == "libraryPath":
            stream.extend(f"-L{word.removeprefix('-L')}" for word in words)
            continue
        for word in words:
            if word.startswith("-Wl,"):
                stream.extend(word[4:].split(","))
            elif word == "-Xlinker":
                stream.append(next(words, ""))
            elif word in _DRIVER_TAKES_VALUE:
                next(words, None)
            elif word == "-static":
                static = True
            elif word.startswith(("-l", "-L")) or not word.startswith("-"):
                stream.append(word)
    return (["-Bstatic"] if static else []) + _joined(stream), unparsed


def _joined(stream: list[str]) -> list[str]:
    """Each search option as one word: `-l x`, `--library x` and `--library=x`
    all become `-lx`, and the same for `-L`."""
    words: list[str] = []
    pending = iter(word for word in stream if word)
    for word in pending:
        option, equals, value = word.partition("=")
        if word in ("-L", "--library-path", "-l", "--library"):
            option, value = word, next(pending, "")
        elif not (equals and option in ("--library-path", "--library")):
            words.append(word)
            continue
        words.append(("-L" if option in ("-L", "--library-path") else "-l") + value)
    return words


def _search_library(name: str, directories: list[Path], static: bool) -> Path | None:
    """The file ld takes for `-l<name>`: `.so` before `.a` in each directory."""
    if name.startswith(":"):
        candidates = [name[1:]]
    else:
        candidates = ([] if static else [f"lib{name}.so"]) + [f"lib{name}.a"]
    for directory in directories:
        for candidate in candidates:
            if (directory / candidate).is_file():
                return directory / candidate
    return None


def _sysrooted(text: str, sysroot: str, base: Path) -> Path:
    """A search directory as ld reads it: a leading `=` stands for the sysroot."""
    if text.startswith("="):
        return Path(sysroot + text[1:]).resolve()
    return (base / text).resolve()


def _inspect(path: Path) -> tuple[str | None, int | None, list[str] | None]:
    """(kind, archive members, script entries), judged by content.

    The kind is "archive", "object", "shared", "script", "missing", or None for
    a file that is none of them.
    """
    if not path.is_file():
        return "missing", None, None
    members = archive_members(path)
    if members is not None:
        return "archive", members, None
    kind = {"ET_REL": "object", "ET_DYN": "shared"}.get(elf.file_type(path) or "")
    if kind:
        return kind, None, None
    entries = _script_inputs(path)
    return ("script" if entries is not None else None), None, entries


def _script_inputs(path: Path) -> list[str] | None:
    """The files and `-l` names a GNU ld script's INPUT and GROUP commands name.

    None if the file is not such a script.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(1 << 16)
    except OSError:
        return None
    if b"\0" in head:
        return None
    body = _SCRIPT_COMMENT.sub(" ", head.decode("utf-8", "replace"))
    commands = list(_SCRIPT_INPUTS.finditer(body))
    if not commands:
        return None
    found: list[str] = []
    for match in commands:
        depth, end = 1, match.end()
        while end < len(body) and depth:
            depth += {"(": 1, ")": -1}.get(body[end], 0)
            end += 1
        found += [
            item
            for item in re.split(r"[\s,()]+", body[match.end() : end - 1])
            if item and item != "AS_NEEDED"
        ]
    return found


def owned_input(item: LinkInput, ownership: Ownership | None) -> str | None:
    """The component a linked archive or object belongs to by the ownership
    index -- a fetched tree, a vcpkg port, vendored material -- if it holds code."""
    if ownership is None or item.path is None or item.kind == "unresolved":
        return None
    if item.members == 0:
        return None
    return ownership.owner_of(item.path)


def _owner(path: Path, ownership: Ownership | None) -> str | None:
    """The component a compiled source belongs to, if any."""
    match = _DEPS_SOURCE.search(path.as_posix())
    if match:
        return match.group("name")
    return ownership.owner_of(path) if ownership is not None else None


class SupplierError(Exception):
    """The project does not name exactly one author to record as supplier."""


class ArchiveError(Exception):
    """An archive the build fetched cannot be read."""


class FileApiError(Exception):
    """The build tree carries no CMake file API reply to read."""


def _within(repo_root: Path, build_dir: Path):
    """Predicate: under the checkout, outside its build tree.

    Both roots are resolved by the caller; `--build-dir` is free-form and a
    relative one compared against absolute paths matches nothing.
    """

    def check(path: Path) -> bool:
        resolved = path if path.is_absolute() else path.resolve()
        return resolved.is_relative_to(repo_root) and not resolved.is_relative_to(
            build_dir
        )

    return check


def _reply_objects(reply: Path) -> tuple[dict[str, Path], str]:
    """The current reply's objects by kind, reached through its index, and the generator.

    Every other reply filename is documented as unspecified and not to be
    interpreted, and CMake only *attempts* to delete what it did not just
    write -- so globbing for a codemodel can find one from an earlier configure,
    and sorting the matches orders content hashes. The index alone carries an
    order: "the one with the largest name in lexicographic order is the current
    index file".
    """
    indexes = sorted(reply.glob("index-*.json")) if reply.is_dir() else []
    if not indexes:
        return {}, ""
    index = json.loads(indexes[-1].read_text(encoding="utf-8"))
    generator = index.get("cmake", {}).get("generator", {}).get("name", "")
    wanted = {"codemodel": 2, "toolchains": 1}
    objects: dict[str, Path] = {}
    for item in index.get("objects", []):
        version = item.get("version") or {}
        if item.get("kind") in wanted and wanted[item["kind"]] == version.get("major"):
            # Index references are relative to the reply directory.
            found = reply / item["jsonFile"]
            if found.is_file():
                objects[item["kind"]] = found
    return objects, generator


_CACHE_COMPILER = re.compile(r"^CMAKE_(?P<language>[A-Z]+)_COMPILER$")
_SEARCH_DIR = re.compile(r'SEARCH_DIR\("(?P<path>[^"]+)"\)')


def cmake_cache(build_dir: Path) -> dict[str, str]:
    """Every entry of a build tree's CMakeCache.txt, by name; empty if it has none."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    entries: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        entries[line.split(":", 1)[0]] = line.split("=", 1)[1]
    return entries


def _search_directories(toolchains: Path | None, build_dir: Path) -> dict[str, _Search]:
    """Where `-l` looks after the `-L` directories, by link language.

    First the directories the compiler hands the linker: CMake's toolchains
    reply, or the compiler asked directly for a language the reply does not
    cover -- CMake records them only for languages the top-level project
    enables. Then the linker's own built-in SEARCH_DIRs.
    """
    compilers = {
        match["language"]: value
        for name, value in cmake_cache(build_dir).items()
        if (match := _CACHE_COMPILER.match(name))
    }
    recorded: dict[str, list[str] | None] = {}
    if toolchains is not None:
        payload = json.loads(toolchains.read_text(encoding="utf-8"))
        for item in payload.get("toolchains", []):
            compiler = item.get("compiler", {})
            recorded[item["language"]] = compiler.get("implicit", {}).get(
                "linkDirectories"
            )
            if compiler.get("path"):
                compilers[item["language"]] = compiler["path"]

    found: dict[str, _Search] = {}
    for language in sorted(set(compilers) | set(recorded)):
        compiler = compilers.get(language)
        listed = recorded.get(language)
        if listed is not None:
            directories = [Path(item) for item in listed]
        elif compiler:
            directories = _compiler_search_dirs(compiler)
        else:
            directories = None
        if directories is None:
            continue
        sysroot = _ask(compiler, "-print-sysroot").strip() if compiler else ""
        found[language] = _Search(directories, compiler, sysroot)
    return found


def _ask(*command: str) -> str:
    """A command's standard output, or empty where it cannot run or fails."""
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


@functools.lru_cache(maxsize=None)
def _compiler_search_dirs(compiler: str) -> list[Path] | None:
    """The `libraries:` list of `<compiler> -print-search-dirs`, GCC's and Clang's."""
    for line in _ask(compiler, "-print-search-dirs").splitlines():
        if line.startswith("libraries:"):
            listed = line.split(":", 1)[1].strip().removeprefix("=")
            return [
                Path(item).resolve()
                for item in listed.split(":")
                if Path(item).is_dir()
            ]
    return None


@functools.lru_cache(maxsize=None)
def _linker_search_dirs(compiler: str, sysroot: str) -> list[Path]:
    """The SEARCH_DIRs of the linker the compiler runs, from `ld --verbose`."""
    linker = _ask(compiler, "-print-prog-name=ld").strip() or "ld"
    found = []
    for match in _SEARCH_DIR.finditer(_ask(linker, "--verbose")):
        directory = _sysrooted(match["path"], sysroot, Path("/"))
        if directory.is_dir():
            found.append(directory)
    return found


class BuildGraph:
    """The link and compile graph, read from CMake's own file API.

    The top-level ``CMakeLists.txt`` asks for the codemodel with
    ``cmake_file_api()`` where that command exists, so a configure by CMake 3.27
    or newer leaves one behind, under every generator. Scraping ``link.txt``
    instead would only work for the Makefile generators -- Ninja writes no such
    file.
    """

    def __init__(self, build_dir: Path, config: str | None = None) -> None:
        self.build_dir = build_dir
        self.artifacts: dict[Path, Artifact] = {}
        self._nodes: dict[str, _Node] = {}
        self._by_name: dict[str, list[Artifact]] = {}
        self._closures: dict[str, tuple[_Node, ...]] = {}
        self._static: dict[str, dict[LinkInput, set[str]]] = {}
        self._inspected: dict[
            Path, tuple[str | None, int | None, list[str] | None]
        ] = {}
        self._load(config)

    def _load(self, config: str | None) -> None:
        reply = self.build_dir / ".cmake" / "api" / "v1" / "reply"
        objects, generator = _reply_objects(reply)
        if "codemodel" not in objects:
            raise FileApiError(
                f"no CMake file API codemodel under {reply}. Configure the project "
                "with CMake 3.27 or newer, which is where cmake_file_api() asks for it."
            )
        model_file = objects["codemodel"]
        search = _search_directories(objects.get("toolchains"), self.build_dir)
        # A relative link-line path is relative to where the link runs: the
        # target's own build directory under the Makefile generators, the top
        # of the build tree under Ninja.
        per_directory = "Makefiles" in generator or "WMake" in generator

        codemodel = json.loads(model_file.read_text(encoding="utf-8"))
        source_root = Path(codemodel["paths"]["source"])
        build_root = Path(codemodel["paths"]["build"])

        configurations = codemodel["configurations"]
        chosen = next(
            (item for item in configurations if item["name"] == (config or "")), None
        )
        if chosen is None:
            # The caller names the configuration from CMakeCache.txt, and a
            # reply that does not describe it is a reply for another build --
            # taking any other credits a wheel to a graph that did not build it.
            # Unnamed is the single-configuration tree the tests construct.
            if config:
                raise FileApiError(
                    f"codemodel under {reply} describes no {config!r} configuration, "
                    f"only {[item['name'] for item in configurations]}"
                )
            chosen = configurations[0]

        def resolve(raw: str, root: Path) -> Path:
            path = Path(raw)
            return path if path.is_absolute() else (root / path)

        lines: dict[str, tuple[list[str], list[str], Path, _Search | None]] = {}
        for entry in chosen["targets"]:
            # Relative to the codemodel, which is what the format says, even
            # though CMake writes it into the reply directory alongside these.
            target = json.loads(
                (model_file.parent / entry["jsonFile"]).read_text(encoding="utf-8")
            )
            includes = {
                resolve(item["path"], source_root).resolve()
                for group in target.get("compileGroups", [])
                for item in group.get("includes", [])
            }
            link_dir = (
                resolve(target.get("paths", {}).get("build", "."), build_root)
                if per_directory
                else build_root
            )
            link = target.get("link", {})
            words, unparsed = _linker_words(link.get("commandFragments", []))
            found = search.get(link.get("language", ""))
            # CMAKE_SYSROOT reaches the linker as --sysroot, which the compiler's
            # own -print-sysroot does not report.
            sysroot = link.get("sysroot", {}).get("path")
            if found is not None and sysroot:
                found = replace(found, sysroot=sysroot)
            lines[entry["id"]] = (words, unparsed, link_dir, found)
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

        # Read once every target's outputs are known: an output of this graph
        # is that target's contribution, whatever its content.
        for target_id, (words, unparsed, link_dir, found) in lines.items():
            line = self._read_line(words, link_dir, found)
            line.inputs += [
                LinkInput(item, None, "unresolved", "does not parse as shell words")
                for item in unparsed
            ]
            self._nodes[target_id] = replace(
                self._nodes[target_id],
                linked=tuple(dict.fromkeys(line.linked)),
                inputs=tuple(dict.fromkeys(line.inputs)),
            )

    def _read_line(
        self, words: list[str], link_dir: Path, search: _Search | None
    ) -> _Line:
        """What a linker word stream names.

        ld applies every `-L` to every `-l`, then the compiler's directories,
        then its own built-in ones -- `search`, or None where the build records
        no compiler. A `-l` name that ld would satisfy with a shared library
        adds nothing, since DT_NEEDED records it.
        """
        sysroot = search.sysroot if search else ""
        line = _Line(
            link_dir,
            [
                _sysrooted(word[2:], sysroot, link_dir)
                for word in words
                if word.startswith("-L")
            ]
            + (search.directories if search else []),
            sysroot,
        )
        static = False
        saved: list[bool] = []
        pending = iter(words)
        for word in pending:
            if word in _STATIC_MODE:
                static = True
            elif word in _DYNAMIC_MODE:
                static = False
            elif word == "--push-state":
                saved.append(static)
            elif word == "--pop-state":
                static = saved.pop() if saved else False
            elif word.startswith("-l"):
                hit = _search_library(word[2:], line.directories, static)
                if hit is None:
                    reason = _UNSEARCHED if search else _NO_COMPILER
                    line.inputs.append(LinkInput(word, None, "unresolved", reason))
                else:
                    self._classify(word, hit, line, static)
            elif word.startswith("-"):
                if word in _LINKER_TAKES_VALUE:
                    next(pending, None)
            else:
                self._classify(word, link_dir / word, line, static)
        return line

    def _classify(
        self, spelling: str, path: Path, line: _Line, static: bool, depth: int = 0
    ) -> None:
        """Add what one path on a link line brings in, a linker script followed
        under the `-Bstatic` state in force where the line names it.

        A shared library adds nothing, since DT_NEEDED records it.
        """
        # ld looks beside the name it was given, not beside a symlink's target.
        beside = path.parent
        path = path.resolve()
        if path in self.artifacts:
            line.linked.append(path)
            return
        if path not in self._inspected:
            self._inspected[path] = _inspect(path)
        kind, members, entries = self._inspected[path]
        if kind in ("archive", "object"):
            line.inputs.append(LinkInput(spelling, path, kind, members=members))
        elif kind == "missing":
            line.inputs.append(LinkInput(spelling, path, "unresolved", _NOT_FOUND))
        elif kind is None:
            line.inputs.append(LinkInput(spelling, path, "unresolved", _UNKNOWN_KIND))
        elif kind == "script" and depth > _SCRIPT_DEPTH:
            line.inputs.append(
                LinkInput(
                    spelling, path, "unresolved", "linker scripts nest too deeply here"
                )
            )
        elif kind == "script":
            # An absolute name in a script inside the sysroot, or one opening
            # with `=`, is read under the sysroot.
            in_sysroot = bool(line.sysroot) and path.is_relative_to(
                Path(line.sysroot).resolve()
            )
            for entry in entries or []:
                if entry.startswith("-l"):
                    hit = _search_library(entry[2:], line.directories, static)
                else:
                    if entry.startswith("=") or (in_sysroot and entry.startswith("/")):
                        entry = line.sysroot + entry.removeprefix("=")
                    # Beside the script, then where the link runs, then the
                    # search path; an absolute name is itself in each.
                    hit = _search_library(
                        ":" + entry, [beside, line.link_dir, *line.directories], static
                    )
                if hit is None:
                    named = f"{spelling} ({path.name}: {entry})"
                    line.inputs.append(LinkInput(named, None, "unresolved", _NOT_FOUND))
                else:
                    self._classify(spelling, hit, line, static, depth + 1)

    def by_name(self, name: str) -> list[Artifact]:
        return self._by_name.get(name, [])

    def repo_inputs(self, repo_root: Path, build_dir: Path) -> set[Path]:
        """Files this build compiled or included that the checkout itself holds.

        The build tree is not among them, for sources as much as for include
        directories: `_deps` is the component-source domain and is attributed
        from its own checkouts, generated headers state no upstream holder, and
        vcpkg installs its ports under the build directory -- whose headers
        carry their upstream contributors' copyright lines, which the vendored
        scan reads as this repository having vendored each of those people.
        """
        inside = _within(repo_root.resolve(), build_dir.resolve())
        found: set[Path] = set()
        for node in self._nodes.values():
            found.update(item for item in node.sources if inside(item))
            for include in node.includes:
                if inside(include) and include.is_dir():
                    found.update(
                        item
                        for item in include.rglob("*")
                        if item.is_file() and inside(item)
                    )
        return found

    def _reach(self, target_id: str, prune=None) -> tuple[_Node, ...]:
        """The target and everything it depends on or links, transitively.

        A link line naming another target's output brings that target, whether
        or not it is also a declared dependency. A node `prune` accepts, other
        than the target itself, is left out with everything reached only
        through it.
        """
        seen: set[str] = set()
        stack = [target_id]
        found: list[_Node] = []
        while stack:
            current = stack.pop()
            if current in seen or current not in self._nodes:
                continue
            seen.add(current)
            node = self._nodes[current]
            if prune is not None and current != target_id and prune(node):
                continue
            found.append(node)
            stack.extend(node.dependencies)
            stack.extend(self.artifacts[item].target_id for item in node.linked)
        return tuple(found)

    def _closure(self, target_id: str) -> tuple[_Node, ...]:
        """`_reach`, memoised per instance, not with `lru_cache`: a cache on the
        class would hold every graph ever built alive for the life of the process."""
        if target_id not in self._closures:
            self._closures[target_id] = self._reach(target_id)
        return self._closures[target_id]

    def static_inputs(self, artifact: Artifact) -> dict[LinkInput, set[str]]:
        """What the closure links that no target here produced.

        Maps each to the targets whose link line names it. Another executable
        or module is a build-order dependency -- a code generator, say -- and
        nothing links it or what it links.
        """
        if artifact.target_id not in self._static:
            found: dict[LinkInput, set[str]] = {}
            for node in self._reach(
                artifact.target_id,
                lambda node: node.kind in ("EXECUTABLE", "MODULE_LIBRARY"),
            ):
                for item in node.inputs:
                    found.setdefault(item, set()).add(node.name)
            self._static[artifact.target_id] = found
        return self._static[artifact.target_id]

    def contributions(
        self, artifact: Artifact, ownership: Ownership | None = None
    ) -> dict[str, set[str]]:
        """Which components contributed to an artifact, and in what way.

        Walks the target's dependency closure. Compiled sources name a component
        outright; include paths are the only trace a header-only dependency
        leaves anywhere; a linked archive or object names whoever holds it.
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
        for item in self.static_inputs(artifact):
            owner = owned_input(item, ownership)
            if owner:
                note(owner, "prebuilt-library")
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
        # path -> (owner, whether an include directory credits it outright)
        self._files: dict[Path, tuple[str, bool]] = {}
        self._roots: list[tuple[Path, str]] = []
        # directory -> (owners credited outright, owners credited if alone)
        self._index: dict[Path, tuple[set[str], set[str]]] | None = None

    def register_root(self, root: Path, key: str) -> None:
        self._roots.append((root.resolve(), key))

    def register_file(self, path: Path, key: str, *, by_directory: bool = True) -> None:
        """Record a file's owner.

        An include directory credits every owner of a file under it; one whose
        files are registered with `by_directory=False` only where it holds every
        such file there.
        """
        self._files[path.resolve()] = (key, by_directory)
        self._index = None

    def owner_of(self, path: Path) -> str | None:
        resolved = path.resolve()
        entry = self._files.get(resolved)
        if entry is not None:
            return entry[0]
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
        resolved = directory.resolve()
        outright, alone = self._by_directory().get(resolved, (set(), set()))
        found = set(outright)
        prefix = str(resolved)
        found.update(
            key
            for root, key in self._roots
            if str(root) == prefix or str(root).startswith(prefix + "/")
        )
        if len(alone) == 1:
            found |= alone
        return found

    def _by_directory(self) -> dict[Path, tuple[set[str], set[str]]]:
        """Each registered file's owner under the file and every directory above it."""
        if self._index is None:
            self._index = {}
            for path, (key, outright) in self._files.items():
                for directory in (path, *path.parents):
                    entry = self._index.setdefault(directory, (set(), set()))
                    entry[0 if outright else 1].add(key)
        return self._index


# ==============================================================================
# Third-party sources vendored in this repository
# ==============================================================================
# Not everything third-party arrives through FetchContent. A file checked in
# here that someone else wrote says so in its REUSE info, which is the same
# thing the project states about its own files -- so the copyright holder is
# what separates them, not the directory it happens to sit in.

# REUSE's tag where a file has one, and a plain copyright line where it does
# not -- upstream files usually carry the latter.
_COPYRIGHT_TAG = re.compile(
    r"SPDX-FileCopyrightText:\s*(?P<holder>[^\n\r]+)"
    r"|^[^\w\n]*Copyright\b(?P<plain>[^\n\r]+)",
    re.MULTILINE,
)
# Comment markup around a copyright line: `//`, `#`, ` * `, `<!-- ... -->`, `/* ... */`.
_COMMENT_MARKUP = re.compile(r"^(?:<!--|[/*#\s])+|\s*(?:-->|\*/)\s*$")
_YEARS = re.compile(r"(copyright|\(c\)|©|\d{4}(\s*-\s*\d{4})?|,|\.)", re.IGNORECASE)


def project_authors(repo_root: Path) -> set[str]:
    """Who the project says it is, read from its own packaging metadata.

    As written, not folded: the same names answer "is this holder ours" and
    "who supplied this distribution", and only the first wants them lowercased.
    """
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return set()
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    return {
        str(author["name"])
        for author in data.get("project", {}).get("authors", [])
        if author.get("name")
    }


def project_supplier(repo_root: Path) -> str:
    """Who supplied this distribution, as its own packaging metadata states it.

    SPDX takes one supplier. Choosing among several, or publishing a release
    that names none, is a question for whoever is publishing it -- so it is
    asked here rather than answered quietly.
    """
    authors = sorted(project_authors(repo_root))
    if len(authors) != 1:
        raise SupplierError(
            f"{repo_root}/pyproject.toml names {len(authors)} project authors "
            f"({', '.join(authors) or 'none'}); SPDX records one supplier for "
            "the distribution"
        )
    return f"Organization: {authors[0]}"


def project_copyright(repo_root: Path) -> str:
    """The notice this project states about itself, from its packaging metadata.

    A distribution that ships files of its own is covered by their notices. One
    that ships only metadata states its copyright nowhere a reader can reach --
    the reference text its declared licence packages names no holder -- and the
    REUSE header on the file the distribution is built from is where the project
    does state it.
    """
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return "NOASSERTION"
    try:
        head = pyproject.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return "NOASSERTION"
    return licensing.read_copyright(head) or "NOASSERTION"


@dataclass(frozen=True)
class _Stated:
    """Who holds a file, and the notices that say so."""

    holder: str
    notices: tuple[str, ...]


def _holder_in(text: str) -> str:
    """The holder a copyright statement names, without years or boilerplate."""
    holder = " ".join(_YEARS.sub(" ", _COMMENT_MARKUP.sub("", text)).split())
    return holder.removesuffix(" All rights reserved").strip()


def _notice_line(text: str) -> str:
    # The tag introduces the notice; it is not part of it.
    notice = _COMMENT_MARKUP.sub("", " ".join(text.split()))
    return re.sub(r"^SPDX-(?:File|Snippet)CopyrightText:\s*", "", notice)


def _stated_by(path: Path) -> _Stated | None:
    """Holder and notice, from the first copyright line in a text file's head."""
    from reuse.extract import HEURISTICS_CHUNK_SIZE, detect_encoding

    try:
        with path.open("rb") as handle:
            raw = handle.read(4096)
    except OSError:
        return None
    # REUSE's own text test: a PNG's `Copyright` text chunk is not a header.
    encoding = detect_encoding(raw[:HEURISTICS_CHUNK_SIZE])
    if encoding is None:
        return None
    match = _COPYRIGHT_TAG.search(raw.decode(encoding, "replace"))
    if not match:
        return None
    holder = _holder_in(match.group("holder") or match.group("plain") or "")
    return _Stated(holder, (_notice_line(match.group(0)),)) if holder else None


def _header_applies(project, used: Path, infos) -> bool:
    """Whether REUSE applies the file's own text beside its declared info.

    It does where it took info from the header, and under `aggregate` whatever
    it parsed -- unless a `.license` sidecar exists or any `override` applies,
    the conditions `Project.reuse_info_of` tests before reading the file.
    """
    from reuse.copyright import SourceType
    from reuse.global_licensing import PrecedenceType

    if any(info.source_type is SourceType.FILE_HEADER for info in infos):
        return True
    if project.global_licensing is None or Path(f"{used}.license").exists():
        return False
    found = project.global_licensing.reuse_info_of(project.relative_from_root(used))
    return PrecedenceType.AGGREGATE in found and PrecedenceType.OVERRIDE not in found


def _held_by(project, used: Path, infos, authors: set[str]) -> _Stated | None:
    """Who other than the project holds a file, as a `_Stated`, if anyone does.

    A sidecar or REUSE.toml notice naming someone else makes the file theirs,
    under the first such holder by name. The file's first copyright line counts
    as well unless REUSE applied declared notices that all name a holder in the
    file's place: a sidecar, `override`, or `closest` where REUSE found nothing
    in the file. Then the file is ours if those all name the project.
    """
    from reuse.copyright import SourceType

    def ours(holder: str) -> bool:
        return any(author.lower() in holder.lower() for author in authors)

    # (holder, notice, whether it names the project), once per declared notice.
    declared = [
        (
            _holder_in(notice.name),
            _notice_line(notice.to_string(original=True)),
            ours(notice.name),
        )
        for info in infos
        if info.source_type is not SourceType.FILE_HEADER
        for notice in info.copyright_notices
    ]
    set_aside = (
        bool(declared)
        and all(holder for holder, _, _ in declared)
        and not _header_applies(project, used, infos)
    )
    stated = None if set_aside else _stated_by(used)
    scraped = stated if stated is not None and not ours(stated.holder) else None
    foreign = [(holder, line) for holder, line, mine in declared if holder and not mine]
    if not foreign:
        return scraped
    holder = min(foreign, key=lambda item: (item[0].casefold(), item))[0]
    lines = [line for _, line in foreign] + list(scraped.notices if scraped else ())
    return _Stated(holder, tuple(lines))


def _covered(project, path: Path, *, built: bool) -> bool:
    """Whether REUSE covers a file.

    For a file the build read, only what REUSE excludes by name counts: a link
    is judged by its target, and a submodule, a meson subproject or a
    VCS-ignored file it compiled still ships.
    """
    from reuse.covered_files import is_path_ignored

    if not built and path.is_symlink():
        return False
    real = path.resolve()
    if not real.is_relative_to(project.root) or not real.is_file():
        return False
    # With no VCS strategy REUSE checks neither ignore rules nor submodules.
    return not any(
        is_path_ignored(
            item,
            include_submodules=project.include_submodules,
            include_meson_subprojects=built or project.include_meson_subprojects,
            vcs_strategy=None if built else project.vcs_strategy,
        )
        for item in (real, *real.parents)
        if item != project.root and item.is_relative_to(project.root)
    )


def _info_for(project, own: Path, target: Path):
    """REUSE info for a file's holder and for its licence, and the holder's path.

    A link's own declarations are its sidecar and the annotations on its path.
    Where they give a field, REUSE's view of the link decides it, as for a file
    in its place; otherwise REUSE's view of the target, which applies the
    target's precedence to the bytes they share.
    """
    from reuse.copyright import SourceType

    infos = project.reuse_info_of(own)
    if target == own:
        return infos, infos, own
    mine = [info for info in infos if info.source_type is not SourceType.FILE_HEADER]
    theirs = project.reuse_info_of(target)
    licensed = infos if any(info.spdx_expressions for info in mine) else theirs
    if any(info.copyright_notices for info in mine):
        return infos, licensed, own
    return theirs, licensed, target


def discover_vendored(
    repo_root: Path,
    candidates: set[Path],
    authors: set[str],
    *,
    tracked: Iterable[str],
) -> tuple[dict[str, Component], dict[str, list[Path]]]:
    """Group checked-in files whose copyright holder is not the project's.

    Reads the build inputs in `candidates` and the `tracked` files REUSE covers
    (`_covered`); `tracked` is required, as without it tracked data and Python
    go unread. Licences are REUSE's; holders are `_held_by`'s.
    """
    # Here, not at module level: only discovery needs reuse, and reuse needs libmagic.
    from reuse.project import Project

    project = Project.from_directory(repo_root.resolve())
    built = {path for path in candidates if _covered(project, path, built=True)}
    candidates = built | {
        path
        for path in (repo_root / item for item in tracked)
        if _covered(project, path, built=False)
    }
    grouped: dict[str, list[Path]] = {}
    licences: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    notices: dict[str, set[str]] = {}
    unlicensed: dict[str, list[tuple[str, str]]] = {}

    for path in sorted(candidates):
        under_root = path.relative_to(repo_root)
        own = project.root / under_root
        infos, licensed, used = _info_for(project, own, path.resolve())
        stated = _held_by(project, used, infos, authors)
        if stated is None:
            continue
        reuse_declared = [
            expression for info in licensed for expression in info.spdx_expressions
        ]
        key = f"vendored:{licensing.spdx_safe(stated.holder).lower()}"
        grouped.setdefault(key, []).append(path)
        names[key] = stated.holder
        notices.setdefault(key, set()).update(stated.notices)
        reason = licensing.why_unlicensed(reuse_declared)
        if reason:
            unlicensed.setdefault(key, []).append((str(under_root), reason))
        else:
            licences.setdefault(key, set()).update(map(str, reuse_declared))

    components: dict[str, Component] = {}
    for key, paths in grouped.items():
        evidence = licensing.canonical_evidence(key, licences.get(key, set()))
        concluded, declared = licensing.expression(evidence, key)
        if licences.get(key):
            # The declared expressions, structure and all. Rebuilding this
            # from the reference texts the identifiers name turns `A OR B` into
            # `A AND B` -- a choice restated as both obligations at once.
            declared = licensing.combine(sorted(licences[key]))
            # An id the pinned list knows and the grammar does not travels as a
            # LicenseRef carrying the texts, as `licensing.expression` does.
            if declared != "NOASSERTION" and not licensing.grammar_names(
                licensing.identifiers_in(declared)
            ):
                evidence += licensing.exception_evidence(key, declared)
                declared = licensing.license_ref(key)
            concluded = declared
        # What the build read first, so a shortened list names what ships; say
        # when it is shortened, or it reads as the whole set.
        named = [
            name
            for _, name in sorted(
                (item not in built, str(item.relative_to(repo_root))) for item in paths
            )
        ]
        relative = ", ".join(named[:4]) + (
            f", and {len(named) - 4} more" if len(named) > 4 else ""
        )
        components[key] = Component(
            key=key,
            # Not the bare holder: an organisation in a `name` field reads as a
            # product, and "Collabora Ltd" is not a package anyone can obtain.
            # The holder is the only identity this material has, so it is named
            # for what it is and who holds it; `sourceInfo` lists the files.
            name=f"Vendored source ({names[key]})",
            kind="vendored-source",
            supplier=f"Organization: {names[key]}",
            homepage="NOASSERTION",
            # No purl: this is third-party material identified by the copyright
            # holder its REUSE info or header names; there is no package to
            # locate. Minting `pkg:generic/<holder>` named a package that does
            # not exist, from an organisation rather than from anything shipped.
            purl="NOASSERTION",
            version="NOASSERTION",
            download_location="NOASSERTION",
            source_info=(
                f"Checked into this repository rather than fetched; its REUSE info "
                f"or header names its copyright holder. Files: {relative}."
            ),
            license_concluded=concluded,
            license_declared=declared,
            copyright_text="\n".join(
                licensing.fold_notices(notices.get(key, {names[key]}))
            ),
            evidence=tuple(evidence),
            unlicensed=tuple(unlicensed.get(key, ())),
        )
    return components, grouped


# ==============================================================================
# vcpkg
# ==============================================================================
# vcpkg installs into the build tree, so what it builds is neither a target of
# this build nor a fetched checkout. Its install records name the port that
# put each file there, as dpkg's do on the build host.


def discover_vcpkg(
    installed: Path, target_triplet: str | None = None
) -> tuple[dict[str, Component], dict[str, list[Path]]]:
    """Every port vcpkg installed under `installed`, and the files each owns.

    A port installed for the host as well is described from its install for
    `target_triplet` -- what the build links -- where there is one.
    """
    info = installed / "vcpkg" / "info"
    components: dict[str, Component] = {}
    owned: dict[str, list[Path]] = {}
    for listing in sorted(info.glob("*.list")) if info.is_dir() else []:
        # <port>_<version>_<triplet>.list; port names carry no underscore.
        port, _, rest = listing.stem.partition("_")
        version, _, triplet = rest.rpartition("_")
        key = f"vcpkg:{port}"
        # A port installed for both the host and the target triplet has a list
        # for each, and both installs are its files.
        owned.setdefault(key, []).extend(
            installed / line
            for line in listing.read_text(encoding="utf-8").splitlines()
            if line and not line.endswith("/")
        )
        if key not in components or triplet == target_triplet:
            components[key] = _vcpkg_component(key, port, version, triplet, installed)
    return components, owned


def _vcpkg_component(
    key: str, port: str, version: str, triplet: str, installed: Path
) -> Component:
    """A port, identified by the upstream source its own SBOM says it built.

    vcpkg writes `share/<port>/vcpkg.spdx.json` with the recipe and the
    downloaded sources, and copies upstream's licence to `share/<port>/copyright`.
    """
    share = installed / triplet / "share" / port
    try:
        sbom = json.loads((share / "vcpkg.spdx.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sbom = {}
    packages = {item.get("SPDXID"): item for item in sbom.get("packages", [])}
    recipe = packages.get("SPDXRef-port", {})
    sources = [
        item["downloadLocation"]
        for spdx_id, item in packages.items()
        if str(spdx_id).startswith("SPDXRef-resource-") and item.get("downloadLocation")
    ]
    port_version = recipe.get("versionInfo") or version
    # `#<n>` is the recipe's revision, not anything upstream released.
    version = port_version.partition("#")[0]
    # One source names the upstream; several (or none) name no single one.
    upstream = sources[0] if len(sources) == 1 else None
    location = (upstream or "").removeprefix("git+")
    remote, _, ref = (
        location.rpartition("@")
        if "@" in location.rsplit("/", 1)[-1]
        else (location, "", "")
    )
    if remote and _GITHUB_REMOTE.search(remote):
        supplier, homepage, purl, download = _identity(remote, port, ref or None)
    else:
        supplier, homepage, download = "NOASSERTION", "NOASSERTION", upstream
        purl = PackageURL(type="generic", name=port, version=version).to_string()
    homepage = recipe.get("homepage") or homepage

    evidence = []
    licence = share / "copyright"
    if licence.is_file():
        evidence.append(
            licensing.evidence_for(
                key,
                "vcpkg-port",
                licence.relative_to(installed.parent).as_posix(),
                licence.read_text(encoding="utf-8", errors="replace"),
                "grant",
            )
        )
    concluded, declared = licensing.expression(evidence, key)
    details = [
        f"Built and installed by vcpkg port {port} {port_version} for {triplet}."
    ]
    if upstream:
        details.append(f"Source: {upstream}.")
    if recipe.get("downloadLocation"):
        details.append(f"Port recipe: {recipe['downloadLocation']}.")
    return Component(
        key=key,
        name=port,
        kind="vcpkg-port",
        supplier=supplier,
        homepage=homepage,
        purl=purl,
        version=version,
        download_location=download or "NOASSERTION",
        source_info=" ".join(details),
        license_concluded=concluded,
        license_declared=declared,
        copyright_text=notice_from(evidence),
        evidence=tuple(evidence),
    )


# ==============================================================================
# The build host
# ==============================================================================
# The last discovery source: a shared library that no index explains, or a
# static archive linked from outside the tree, was not produced or fetched by
# this build, so ask the machine that supplied it.

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
    for line in _ask("ldconfig", "-p").splitlines():
        _, _, target = line.partition("=>")
        if target.strip():
            paths.add(Path(target.strip()))
    for root in _library_roots():
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
        build_id = elf.read_build_id(path)
        if not build_id:
            continue
        # A regular file beats a symlink to it: dpkg-query knows the real path.
        current = index.get(build_id)
        if current is None or (
            path.is_symlink() is False and Path(current).is_symlink()
        ):
            index[build_id] = str(path)
    return index


def _host_origin(library: Path | None, **known) -> dict:
    """What the build machine says about a file: where it resolves, and the
    package that owns it with that package's terms."""
    origin: dict = {**known, "resolved_path": None, "package": None, "copyright": None}
    if library is None:
        return origin
    library = library.resolve()
    origin["resolved_path"] = str(library)
    return _identify_host_package(library, origin)


def resolve_system_library_by_build_id(build_id: str) -> dict:
    """Trace a vendored library back by build-id, whatever it was renamed to."""
    found = _host_build_id_index().get(build_id)
    library = Path(found).resolve() if found else None
    return _host_origin(
        library, soname=library.name if library else None, build_id=build_id
    )


def resolve_system_library(soname: str) -> dict:
    """Trace an auditwheel-vendored library back to the package that supplied it."""
    library = next(
        (root / soname for root in _library_roots() if (root / soname).exists()), None
    )
    return _host_origin(library, soname=soname)


def resolve_static_archive(archive: Path) -> dict:
    """Trace an archive or object the build linked to the package or toolkit it came from.

    A shared library is explained once its SONAME resolves on the machine: that
    is how the repair tool found the file it vendored. A link line already
    names a static input's path, so finding it proves nothing; only a package
    or a toolkit that states its terms explains one.
    """
    if not archive.is_file():
        return _host_origin(None, toolkit_root=None)
    root = _toolkit_root(archive)
    return _host_origin(archive, toolkit_root=str(root) if root else None)


def archive_members(path: Path) -> int | None:
    """How many object files a `!<arch>` archive holds; None if it is not one.

    glibc 2.34 folded librt, libpthread and libdl into libc and left each as an
    archive with no members, so naming one on a link line links nothing.
    """
    try:
        with path.open("rb") as handle:
            magic = handle.read(8)
            if magic not in (b"!<arch>\n", b"!<thin>\n"):
                return None
            count = 0
            while header := handle.read(60):
                if len(header) < 60 or header[58:60] != b"`\n":
                    return None
                name = header[:16].rstrip()
                size = int(header[48:58])
                skip = size + size % 2
                # BSD writes a long name at the head of the member's data.
                if name.startswith(b"#1/"):
                    length = int(name[3:])
                    name = handle.read(length).rstrip(b"\0")
                    skip -= length
                # The symbol table and the long-name table are not members.
                index = name in (b"/", b"//", b"/SYM64/") or name.startswith(
                    b"__.SYMDEF"
                )
                count += not index
                # A thin archive stores only its index; members stay outside it.
                if index or magic == b"!<arch>\n":
                    handle.seek(skip, os.SEEK_CUR)
            return count
    except (OSError, ValueError):
        return None


def _identify_host_package(library: Path, origin: dict) -> dict:
    """Name the distro package that owns a resolved host library."""
    owned = _ask("dpkg-query", "-S", str(library))
    if ":" not in owned:
        return origin
    package = owned.split(":", 1)[0].strip()
    origin["package"] = package
    origin["supplier"] = _package_supplier(package)
    origin["version"] = _package_field(package, "Version")
    origin["distro"] = _os_release_id()
    copyright_file = Path("/usr/share/doc") / package / "copyright"
    if copyright_file.is_file():
        origin["copyright"] = str(copyright_file)
    return origin


def _package_field(package: str, field: str) -> str | None:
    """One field of a distro package's metadata, as dpkg records it."""
    return _ask("dpkg-query", "-W", f"-f=${{{field}}}", package).strip() or None


def _os_release_id() -> str | None:
    """The distribution this package manager belongs to, for the purl type."""
    try:
        return platform.freedesktop_os_release().get("ID") or None
    except OSError:
        return None


def _package_supplier(package: str) -> str | None:
    """Who ships a distro package, in the form SPDX wants.

    The package name is not a supplier: `libbsd0` names the thing, not whoever
    provided it. dpkg records a maintainer, which is the answer.
    """
    maintainer = _package_field(package, "Maintainer")
    if not maintainer:
        return None
    # "Name <email>" is Debian's form; SPDX wants "Organization: Name (email)".
    match = re.match(r"^(?P<name>.+?)\s*<(?P<email>[^>]+)>$", maintainer)
    if match:
        return f"Organization: {match['name']} ({match['email']})"
    return f"Organization: {maintainer}"


def _library_roots() -> list[Path]:
    """Where a pre-built library the build machine supplied can be found."""
    roots = [Path(item) for item in _SEARCH_ROOTS if Path(item).is_dir()]
    return roots + [root / "lib64" for root in _toolkit_roots()]


@functools.lru_cache(maxsize=None)
def _toolkit_roots() -> list[Path]:
    """Toolkits that may install outside the package manager: CUDA's."""
    return sorted(
        {
            Path(item).resolve()
            for item in glob.glob("/usr/local/cuda*")
            if Path(item).is_dir()
        }
    )


def _toolkit_root(path: Path) -> Path | None:
    resolved = path.resolve()
    return next(
        (root for root in _toolkit_roots() if resolved.is_relative_to(root)), None
    )


def _license_near(library: Path) -> Path | None:
    """A toolchain that installs outside the package manager still ships its terms.

    CUDA is the case that matters: its libraries resolve into
    targets/<arch>/lib with EULA.txt at the toolkit root, and dpkg need know
    nothing about it. Under a toolkit root the search runs up to that root.
    """
    root = _toolkit_root(library)
    current = library.resolve().parent
    levels = 3 if root is None else len(current.relative_to(root).parts) + 1
    for _ in range(levels):
        for entry in sorted(current.iterdir()) if current.is_dir() else []:
            if entry.is_file() and licensing.classify(entry.name):
                return entry
        if current.parent == current:
            break
        current = current.parent
    return None


def _system_notice(evidence: list[licensing.LicenseEvidence]) -> str:
    """The holders a machine-readable Debian copyright file names, or NOASSERTION.

    Any other text -- an EULA, a free-form copyright file -- quotes the notices
    of whatever it bundles, and scraping those names other people's code as the
    holders of this library. The packaged text still carries all of it.
    """
    for item in evidence:
        fields = licensing.dep5_copyright(item.text)
        if fields:
            return "\n".join(licensing.fold_notices(fields))
    return "NOASSERTION"


def system_component(
    soname: str, origin: dict, route: str | None = None, key: str | None = None
) -> Component:
    """A pre-built library the build machine supplied, named by what it came from.

    `route` says how it reached the wheel; the default is auditwheel's. `key`
    defaults to `system:<soname>`.
    """
    evidence: list[licensing.LicenseEvidence] = []
    key = key or f"system:{soname}"
    license_path = origin.get("copyright")
    if not license_path and origin.get("resolved_path"):
        found = _license_near(Path(origin["resolved_path"]))
        license_path = str(found) if found else None
        origin["license_file"] = license_path
    if license_path and Path(license_path).is_file():
        text = Path(license_path).read_text(encoding="utf-8", errors="replace")
        evidence.append(
            licensing.evidence_for(key, "build-host", license_path, text, "grant")
        )
    # `key`, not `soname`: the key is what identifies this component everywhere
    # else, and two components reduced to the same label would mint one LicenseRef
    # for two different texts.
    concluded, declared = licensing.expression(evidence, key)

    package = origin.get("package")
    details = [
        route
        or f"Vendored into the wheel by auditwheel from {origin.get('resolved_path') or soname}."
    ]
    if package:
        details.append(f"Supplied by system package {package}.")
    return Component(
        key=key,
        name=soname,
        kind="system-library",
        supplier=origin.get("supplier") or "NOASSERTION",
        copyright_text=_system_notice(evidence),
        homepage="NOASSERTION",
        # A distro package the machine can name resolves; pkg:generic does not.
        purl=(
            PackageURL(
                type="deb",
                namespace=origin.get("distro"),
                name=origin["package"],
                version=origin.get("version"),
            ).to_string()
            if origin.get("package")
            else PackageURL(type="generic", name=soname).to_string()
        ),
        version=origin.get("version") or "NOASSERTION",
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
    build_id: str | None = None


class ArchiveIndex:
    """Members of every archive in the repo, so extracted bytes can be traced back."""

    def __init__(self) -> None:
        self.by_hash: dict[str, Member] = {}
        self.all_by_hash: dict[str, list[Member]] = {}
        self.by_name: dict[str, list[Member]] = {}
        # Survives the RPATH/SONAME rewrite a repair tool applies on the way in,
        # so a patched wheel member still points at the archive it came from.
        self.all_by_build_id: dict[str, list[Member]] = {}
        self.archives: dict[str, dict] = {}

    def add_archive(self, path: Path, display: str) -> None:
        size = path.stat().st_size
        if size > _MAX_ARCHIVE_BYTES:
            return
        members = self._read_members(path, display)
        self.archives[display] = {
            "path": display,
            "sha256": sha256_file(path),
            "size": size,
            "member_count": len(members),
        }
        for member, data in members:
            # `data` is the whole member only at the archive root; elsewhere it
            # is the head kept for exactly this, so ask only for the build-id.
            build_id = elf.build_id_of(data)
            if build_id:
                member = replace(member, build_id=build_id)
                self.all_by_build_id.setdefault(build_id, []).append(member)
            self.by_hash.setdefault(member.sha256, member)
            self.all_by_hash.setdefault(member.sha256, []).append(member)
            self.by_name.setdefault(Path(member.path).name, []).append(member)
            self._maybe_license(display, member, data)

    def _maybe_license(self, display: str, member: Member, data: bytes) -> None:
        name = Path(member.path).name
        if Path(member.path).parent.as_posix() not in {".", ""}:
            return  # only the archive root states the bundle's own terms
        if name.upper() == "VERSION" and len(data) < 256:
            # First line only. These files carry more than one, and a newline in
            # a version breaks every tag-value and table rendering downstream.
            text = data.decode("utf-8", "replace").strip().splitlines()
            self.archives[display]["version_text"] = text[0].strip() if text else ""
            return
        if not licensing.classify(name):
            return
        self.archives[display].setdefault("license_members", []).append((member, data))

    @staticmethod
    def _read_members(path: Path, display: str) -> list[tuple[Member, bytes]]:
        """Hash every member; keep the bytes only of the few worth keeping.

        An SDK tarball expands to far more than it compresses to, so holding
        every member would size peak memory by the largest archive rather than
        by what is retained.
        """
        collected: list[tuple[Member, bytes]] = []

        def take(name: str, handle) -> None:
            # A root member's bytes are kept whole, for its licence text; every
            # other member streams through the digest, and keeps a head only if
            # it is a library, for its build-id. Reading every member whole
            # sizes peak memory by the largest file in the archive.
            at_root = Path(name).parent.as_posix() in {".", ""}
            digest = hashlib.sha256()
            size = 0
            kept = bytearray()
            head = b""
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                if at_root:
                    kept += chunk
                elif not size and elf.is_elf(chunk):
                    head = chunk[:_ELF_HEAD_BYTES]
                size += len(chunk)
            collected.append(
                (
                    Member(display, name, digest.hexdigest(), size),
                    bytes(kept) if at_root else head,
                )
            )

        try:
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if not info.is_dir():
                            with archive.open(info) as handle:
                                take(info.filename, handle)
            else:
                with tarfile.open(path) as archive:
                    for info in archive:
                        if not info.isfile():
                            continue
                        handle = archive.extractfile(info)
                        if handle is not None:
                            take(info.name.removeprefix("./"), handle)
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
            # Only archives this build deliberately fetched are indexed, so one
            # that will not open is a broken download, not a stray file. Saying
            # so beats reporting every file it should have explained as a member
            # that reached the wheel by an unknown route.
            raise ArchiveError(f"{display} cannot be read: {error}") from error
        return collected


def tracked_files(repo_root: Path) -> list[str]:
    """What this repository actually holds, as git records it.

    A working tree also holds whatever was built in it. An install prefix has no
    marker saying so, so walking the directory cannot tell the two apart, and a
    wheel member matching a *build output* would be reported as a copy of
    repository source. Git knows the difference.
    """
    listing = git(repo_root, "ls-files", "-z", "--cached")
    if listing is None:
        raise FileApiError(
            f"{repo_root} is not a git checkout this build can query; the SBOM "
            "distinguishes repository source from build output by what git tracks"
        )
    return [item for item in listing.split("\0") if item]


def discover_archives(
    repo_root: Path, held: frozenset[str], skip: tuple[Path, ...] = ()
) -> ArchiveIndex:
    """Index archives held in the repository, not ones a build unpacked.

    `held` is what the repository tracks. It has no default: an empty set
    silently indexes nothing, and a caller that forgot it would see every
    archive-derived wheel member become unexplainable rather than wrong.
    """
    index = ArchiveIndex()
    skipped = tuple(item.resolve() for item in skip)
    for path in _walk(repo_root):
        # By path segment, not by string prefix: `dist` prefixes `distribution`
        # and a build directory named `build` prefixes `buildtools`, and an
        # archive skipped that way goes on to explain nothing.
        if any(path.is_relative_to(item) for item in skipped):
            continue
        name = path.name.lower()
        if not name.endswith(ARCHIVE_SUFFIXES):
            continue
        relative = path.relative_to(repo_root).as_posix()
        # Held here, or fetched by this build and recorded as such. An archive
        # anywhere else is output -- staged into an install prefix, say -- and
        # letting it explain a wheel member is the build explaining itself.
        if relative not in held and not path.with_name(path.name + ".source").is_file():
            continue
        index.add_archive(path, relative)
    return index


class FileIndex:
    """Files on disk by content hash, for tracing a copied wheel member back."""

    def __init__(self) -> None:
        self.by_hash: dict[str, str] = {}
        # Every path sharing a digest, because one is not evidence of origin when
        # several files hold the same bytes -- a stock licence text, or a module
        # that is nothing but an SPDX header.
        self.all_by_hash: dict[str, list[str]] = {}
        self.by_suffix: dict[str, list[str]] = {}
        self._digests: dict[str, str] = {}
        self._by_display: dict[str, str] = {}

    def digest_of(self, path: Path) -> str | None:
        """The digest this index already computed for a path, if it indexed it."""
        return self._digests.get(str(path.resolve()))

    def add_tree(
        self, root: Path, display_root: str, skip: tuple[Path, ...] = ()
    ) -> None:
        # Compare skip entries as paths relative to `root`: `--build-dir` may be
        # relative, and `_deps/foo-srcx` string-prefixes `_deps/foo-src`.
        inside = []
        for item in skip:
            try:
                inside.append(item.resolve().relative_to(root.resolve()))
            except ValueError:
                continue
        self._add(
            root,
            display_root,
            (
                path
                for path in _walk(root)
                if not any(
                    path.relative_to(root).is_relative_to(item) for item in inside
                )
            ),
        )

    def add_paths(self, root: Path, display_root: str, paths) -> None:
        """Index an explicit list rather than whatever the tree happens to hold."""
        self._add(root, display_root, (root / item for item in paths))

    def _add(self, root: Path, display_root: str, paths) -> None:
        # One resolve for the tree, not one per file: `_walk` never crosses a
        # symlink, so the resolved root plus the relative path is the real path.
        resolved_root = root.resolve()
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            display = f"{display_root}/{relative}"
            try:
                digest = sha256_file(path)
            except OSError:
                continue
            self._digests[str(resolved_root / relative)] = digest
            self.by_hash.setdefault(digest, display)
            self.all_by_hash.setdefault(digest, []).append(display)
            self._by_display[display] = digest
            # Two trailing segments is enough to disambiguate an __init__.py.
            self.by_suffix.setdefault(
                "/".join(relative.rsplit("/", 2)[-2:]), []
            ).append(display)

    def digest_of_display(self, display: str) -> str | None:
        """The digest this index recorded for one of its own display paths."""
        return self._by_display.get(display)

    def under(self, display_prefix: str):
        """Every (display, digest) this index holds beneath a display path."""
        prefix = display_prefix.rstrip("/") + "/"
        return (
            (display, digest)
            for display, digest in self._by_display.items()
            if display.startswith(prefix)
        )

    def path_suffix_match(self, wheel_path: str) -> str | None:
        key = "/".join(wheel_path.rsplit("/", 2)[-2:])
        matches = self.by_suffix.get(key, [])
        return matches[0] if len(matches) == 1 else None

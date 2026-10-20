# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a link line names is a contribution, wherever it came from."""

from __future__ import annotations

from pathlib import Path

import pytest
import synth
from packageurl import PackageURL
from sbom import build as build_module
from sbom import discovery
from sbom import document as document_module
from sbom import evidence as evidence_module

EXTENSION = "_ext.cpython-312-x86_64-linux-gnu.so"
ONE_OBJECT = synth.ar_archive({"runtime.o": b"\x7fELF object"})
# An agreement that quotes the notices of what it bundles, as the CUDA EULA does.
EULA = (
    synth.PROPRIETARY_LICENSE
    + "\nThird-party notices\n\nCopyright (c) 2011 Upstream Compression Ltd.\n"
)


def _no_package(monkeypatch) -> None:
    """The build machine's package manager owns nothing outside its own trees."""
    monkeypatch.setattr(
        discovery, "_identify_host_package", lambda library, origin: origin
    )


def _packaged(monkeypatch, package, copyright, **fields) -> None:
    """dpkg names `package` as the owner of whatever is asked about.

    `package` and `copyright` may be callables of the library's path.
    """

    def owned(library, origin):
        def value(item):
            return item(library) if callable(item) else item

        return {
            **origin,
            **fields,
            "package": value(package),
            "copyright": str(value(copyright)),
        }

    monkeypatch.setattr(discovery, "_identify_host_package", owned)


def _row(collected: dict, key: str) -> dict:
    """The licence report's row for one component."""
    return next(
        item for item in collected["licenses"]["components"] if item["component"] == key
    )


def _library_dir(root, *names) -> None:
    """A directory holding each `lib<name>` as both a shared library and an archive."""
    for name in names:
        synth.write(root / f"lib{name}.so", synth.elf_shared_object(f"lib{name}.so"))
        synth.write(root / f"lib{name}.a", ONE_OBJECT)


def _inputs(workspace, *fragments, **link) -> dict[str, discovery.LinkInput]:
    synth.link_line(workspace.build, "ext", fragments, **link)
    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)
    linked = graph.static_inputs(graph.by_name(EXTENSION)[0])
    return {item.named: item for item in linked}


def _package(row: dict, spdx: dict) -> dict:
    package_id = document_module.component_package_id(row["component"])
    return next(item for item in spdx["packages"] if item["SPDXID"] == package_id)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (synth.ar_archive({}), 0),
        (ONE_OBJECT, 1),
        (synth.ar_archive({"a.o": b"a", "b.o": b"bb"}), 2),
        # A GNU symbol table is an index, not a member.
        (b"!<arch>\n" + synth.ar_header(b"/", 4) + b"\0\0\0\0" + ONE_OBJECT[8:], 1),
        # BSD names a long member, the symbol table among them, inside its data.
        (
            b"!<arch>\n"
            + synth.ar_header(b"#1/20", 24)
            + b"__.SYMDEF SORTED\0\0\0\0"
            + b"\0\0\0\0"
            + ONE_OBJECT[8:],
            1,
        ),
        # A thin archive stores no member data, only headers.
        (
            b"!<thin>\n"
            + synth.ar_header(b"a.o/", 100)
            + synth.ar_header(b"b.o/", 200),
            2,
        ),
        (b"/* GNU ld script */\nGROUP ( libm.so.6 )\n", None),
    ],
)
def test_archive_members_are_counted_from_the_ar_headers(tmp_path, data, expected):
    path = tmp_path / "lib.a"
    path.write_bytes(data)

    assert discovery.archive_members(path) == expected


@pytest.mark.parametrize(
    ("generator", "directory", "fragment"),
    [
        (None, None, "lib/libdelta.a"),
        ("Ninja", "src/ext", "lib/libdelta.a"),
        ("Unix Makefiles", "src/ext", "../../lib/libdelta.a"),
    ],
)
def test_an_archive_a_target_here_built_maps_to_that_target(
    workspace, generator, directory, fragment
):
    """Not a declared dependency of ext; its link line is the only evidence."""
    if generator:
        synth.set_generator(workspace.build, generator)
    synth.link_line(workspace.build, "ext", (fragment,), directory=directory)

    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)
    extension = graph.by_name(EXTENSION)[0]

    assert "compiled-source" in graph.contributions(extension)["delta"]
    assert graph.static_inputs(extension) == {}


def test_an_archive_through_a_static_library_is_on_the_consumers_link_line(
    workspace, tmp_path
):
    """CMake gives a static library no link line: what it requires is flattened
    onto each consumer's, beside the static library's own archive."""
    runtime = tmp_path / "host" / "libruntime.a"
    synth.write(runtime, ONE_OBJECT)
    synth.link_line(
        workspace.build,
        "ext",
        ("lib/libdelta.a", str(runtime)),
        dependencies=("delta",),
    )

    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)
    extension = graph.by_name(EXTENSION)[0]

    assert "delta" in graph.contributions(extension)
    assert graph.static_inputs(extension) == {
        discovery.LinkInput(str(runtime), runtime, "archive", members=1): {"ext"}
    }


def test_an_archive_a_dependency_links_is_followed_like_its_sources(
    workspace, tmp_path
):
    """ext -> delta (static) -> gamma (shared), whose link line names it."""
    runtime = tmp_path / "host" / "libruntime.a"
    synth.write(runtime, ONE_OBJECT)
    synth.link_line(workspace.build, "ext", (), dependencies=("delta",))
    synth.link_line(workspace.build, "delta", (), dependencies=("gamma",))
    synth.link_line(workspace.build, "gamma", (str(runtime),))

    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)
    extension = graph.by_name(EXTENSION)[0]

    assert list(graph.static_inputs(extension).values()) == [{"gamma"}]


def test_static_code_is_found_in_every_form_a_link_line_takes(workspace, tmp_path):
    host = tmp_path / "host dir"
    synth.write(host / "libspaced.a", ONE_OBJECT)
    synth.write(host / "libwhole.a", ONE_OBJECT)
    synth.write(host / "libflagged.a", ONE_OBJECT)
    synth.write(host / "libversioned.a.1", ONE_OBJECT)
    synth.write(host / "prebuilt.o", synth.elf_object())
    synth.write(host / "libshared.so", synth.elf_shared_object("libshared.so"))
    synth.write(host / "exports.map", "{ global: *; };\n")

    inputs = _inputs(
        workspace,
        f'"{host}/libspaced.a"',
        f'"-Wl,--whole-archive,{host}/libwhole.a,--no-whole-archive"',
        (f'"{host}/libflagged.a"', "flags"),
        f'"{host}/libversioned.a.1"',
        f'"{host}/prebuilt.o"',
        f'"{host}/libshared.so"',
        (f'"-Wl,--version-script,{host}/exports.map"', "flags"),
        ("-fuse-ld=gold", "flags"),
    )

    assert {name: item.kind for name, item in inputs.items()} == {
        f"{host}/libspaced.a": "archive",
        f"{host}/libwhole.a": "archive",
        f"{host}/libflagged.a": "archive",
        f"{host}/libversioned.a.1": "archive",
        f"{host}/prebuilt.o": "object",
    }


def test_a_library_name_resolves_the_way_ld_resolves_it(workspace, tmp_path):
    searched = tmp_path / "searched"
    implicit = tmp_path / "implicit"
    _library_dir(searched, "both", "forced")
    synth.write(searched / "libstaticonly.a", ONE_OBJECT)
    synth.write(implicit / "libcompiler.a", ONE_OBJECT)
    synth.set_toolchains(workspace.build, {"CXX": [str(implicit)]})

    inputs = _inputs(
        workspace,
        (f"-L{searched}", "libraryPath"),
        "-lboth",
        "-lstaticonly",
        "-lcompiler",
        "-l:libboth.a",
        "-lmissing",
        "-l:libmissing.a",
        "-Wl,-Bstatic",
        "-lforced",
    )

    # -lboth takes the shared library, which DT_NEEDED records; after -Bstatic,
    # -lforced takes the archive.
    assert {name: (item.kind, item.path) for name, item in inputs.items()} == {
        "-lforced": ("archive", searched / "libforced.a"),
        "-lstaticonly": ("archive", searched / "libstaticonly.a"),
        "-lcompiler": ("archive", implicit / "libcompiler.a"),
        "-l:libboth.a": ("archive", searched / "libboth.a"),
        "-lmissing": ("unresolved", None),
        "-l:libmissing.a": ("unresolved", None),
    }


def test_every_spelling_of_a_search_resolves_alike(workspace, tmp_path):
    joined, separated = tmp_path / "joined", tmp_path / "separated"
    _library_dir(joined, "eqsign", "spaced", "linkerspaced")
    _library_dir(separated, "fromsep")
    synth.write(joined / "libarchive.a.1", ONE_OBJECT)
    synth.write(joined / "libshared.so.1", synth.elf_shared_object("libshared.so.1"))
    synth.write(tmp_path / "xl" / "libxl.a", ONE_OBJECT)

    inputs = _inputs(
        workspace,
        f"-Wl,--library-path={joined}",
        f"-Wl,-L,{separated}",
        "-Wl,-Bstatic",
        "-Wl,--library=eqsign",
        "-l spaced",
        "-Wl,-l,linkerspaced",
        "-lfromsep",
        "-Wl,-Bdynamic",
        "-l:libarchive.a.1",
        "-l:libshared.so.1",
        f"-Xlinker {tmp_path}/xl/libxl.a",
    )

    assert {name: item.path for name, item in inputs.items()} == {
        "-leqsign": joined / "libeqsign.a",
        "-lspaced": joined / "libspaced.a",
        "-llinkerspaced": joined / "liblinkerspaced.a",
        "-lfromsep": separated / "libfromsep.a",
        # Named outright, so judged by content: an archive without the suffix
        # is still one, and a shared library is left to DT_NEEDED.
        "-l:libarchive.a.1": joined / "libarchive.a.1",
        f"{tmp_path}/xl/libxl.a": tmp_path / "xl" / "libxl.a",
    }


@pytest.mark.parametrize(
    ("fragments", "expected"),
    [
        # -Bdynamic resets; push-state saves and pop-state restores.
        (
            (
                "-Wl,-Bstatic",
                "-Wl,-Bdynamic",
                "-lreset",
                "-Wl,--push-state,-Bstatic,-lpushed,--pop-state",
                "-lafter",
            ),
            {"-lpushed": "libpushed.a"},
        ),
        # Pushed while static: the pop restores static mode, not the default.
        (
            (
                "-Wl,-Bstatic",
                "-Wl,--push-state,-Bdynamic,-lpushdyn,--pop-state",
                "-lpopped",
            ),
            {"-lpopped": "libpopped.a"},
        ),
        # The driver's -static covers the whole line, wherever it is written.
        (("-lafter", ("-static", "flags")), {"-lafter": "libafter.a"}),
    ],
)
def test_static_mode_follows_ld_and_the_driver(
    workspace, tmp_path, fragments, expected
):
    directory = tmp_path / "libs"
    _library_dir(directory, "reset", "pushed", "after", "pushdyn", "popped")

    inputs = _inputs(workspace, f"-L{directory}", *fragments)

    assert {name: item.path for name, item in inputs.items()} == {
        name: directory / file for name, file in expected.items()
    }


def test_an_option_value_is_not_an_input(workspace, tmp_path):
    synth.write(tmp_path / "libvalue.a", ONE_OBJECT)

    inputs = _inputs(
        workspace,
        (f"-o {tmp_path}/libvalue.a", "flags"),
        f"-Wl,--dependency-file,{tmp_path}/deps.d",
        f"-Wl,--require-defined,{tmp_path}/libvalue.a",
        (f"-Wl,-rpath,{tmp_path}", "flags"),
    )

    assert inputs == {}


def test_an_unknown_path_names_itself_and_the_option_it_may_belong_to(workspace):
    (item,) = _inputs(workspace, "-Wl,--some-new-option,mystery-value").values()

    assert item.kind == "unresolved"
    assert item.named == "mystery-value"
    assert "linker option" in item.reason


def test_the_compiler_and_linker_are_asked_where_the_reply_is_silent(
    workspace, tmp_path
):
    """C enabled only below the top level gets no implicit directories from
    CMake, so the compiler is asked, and then its linker's SEARCH_DIRs."""
    from_compiler = tmp_path / "compiler"
    from_linker = tmp_path / "sysroot" / "stub-only" / "lib"
    synth.write(from_compiler / "libfromcompiler.a", ONE_OBJECT)
    synth.write(from_linker / "libfromlinker.a", ONE_OBJECT)
    compiler = synth.stub_compiler(
        tmp_path / "bin", from_compiler, tmp_path / "sysroot", "/stub-only/lib"
    )
    synth.set_toolchains(workspace.build, {"CXX": None}, {"CXX": str(compiler)})

    inputs = _inputs(workspace, "-lfromcompiler", "-lfromlinker")

    assert {name: item.path for name, item in inputs.items()} == {
        "-lfromcompiler": from_compiler / "libfromcompiler.a",
        "-lfromlinker": from_linker / "libfromlinker.a",
    }


def test_a_search_directory_under_the_sysroot_is_read_there(workspace, tmp_path):
    """`-L=/dir` names `<sysroot>/dir`, not a directory called `=`."""
    sysroot = tmp_path / "sysroot"
    synth.write(sysroot / "sub" / "libinroot.a", ONE_OBJECT)
    synth.write(tmp_path / "compiler" / "libinroot.a", ONE_OBJECT)
    compiler = synth.stub_compiler(
        tmp_path / "bin", tmp_path / "compiler", sysroot, "/stub-only/lib"
    )
    synth.set_toolchains(workspace.build, {"CXX": None}, {"CXX": str(compiler)})

    (item,) = _inputs(workspace, "-L=/sub", "-linroot").values()

    assert item.path == sysroot / "sub" / "libinroot.a"


def test_a_script_in_the_sysroot_names_files_in_the_sysroot(workspace, tmp_path):
    """libc.so in a sysroot names `/usr/lib/...`, which ld reads under it."""
    sysroot = tmp_path / "sysroot"
    synth.write(sysroot / "lib" / "libroot_nonshared.a", ONE_OBJECT)
    synth.write(sysroot / "lib" / "libprefixed.a", ONE_OBJECT)
    synth.write(
        sysroot / "lib" / "libroot.so",
        "/* GNU ld script */\nGROUP ( /lib/libroot_nonshared.a =/lib/libprefixed.a )\n",
    )
    compiler = synth.stub_compiler(
        tmp_path / "bin", tmp_path / "compiler", sysroot, "/stub-only/lib"
    )
    synth.set_toolchains(workspace.build, {"CXX": None}, {"CXX": str(compiler)})

    synth.link_line(workspace.build, "ext", (f"-L{sysroot}/lib", "-lroot"))
    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)
    inputs = graph.static_inputs(graph.by_name(EXTENSION)[0])

    assert sorted(item.path for item in inputs) == [
        sysroot / "lib" / "libprefixed.a",
        sysroot / "lib" / "libroot_nonshared.a",
    ]


def test_cmake_sysroot_overrides_the_compilers_own(workspace, tmp_path):
    """CMAKE_SYSROOT reaches the linker as --sysroot, for `-L=` and for ld's
    `=` search directories alike."""
    compiler_root, cmake_root = tmp_path / "compiler-root", tmp_path / "cmake-root"
    for root in (compiler_root, cmake_root):
        synth.write(root / "sub" / "libfoo.a", ONE_OBJECT)
        synth.write(root / "stub-only" / "lib" / "libfromld.a", ONE_OBJECT)
    compiler = synth.stub_compiler(
        tmp_path / "bin", tmp_path / "compiler", compiler_root, "/stub-only/lib"
    )
    synth.set_toolchains(workspace.build, {"CXX": None}, {"CXX": str(compiler)})

    inputs = _inputs(workspace, "-L=/sub", "-lfoo", "-lfromld", sysroot=cmake_root)

    assert {name: item.path for name, item in inputs.items()} == {
        "-lfoo": cmake_root / "sub" / "libfoo.a",
        "-lfromld": cmake_root / "stub-only" / "lib" / "libfromld.a",
    }


@pytest.mark.parametrize(
    ("fragments", "expected"),
    [
        (("-Wl,-Bstatic", "-l:libgrp.so"), ["libz.a"]),
        (("-Wl,-Bstatic", "{d}/libgrp.so"), ["libz.a"]),
        (("-Wl,-Bstatic", "-l:libouter.so"), ["libz.a"]),
        (("-lgrp",), []),
    ],
)
def test_static_mode_applies_inside_a_linker_script(
    workspace, tmp_path, fragments, expected
):
    _library_dir(tmp_path, "z")
    synth.write(tmp_path / "libgrp.so", "/* GNU ld script */\nGROUP ( -lz )\n")
    synth.write(tmp_path / "libouter.so", "/* GNU ld script */\nINPUT ( libgrp.so )\n")

    inputs = _inputs(
        workspace, f"-L{tmp_path}", *(item.format(d=tmp_path) for item in fragments)
    )

    assert [item.path.name for item in inputs.values()] == expected


def test_no_compiler_on_record_says_to_reconfigure(workspace):
    cache = workspace.build / "CMakeCache.txt"
    cache.write_text(
        "".join(
            line
            for line in cache.read_text().splitlines(keepends=True)
            if "_COMPILER:" not in line
        )
    )

    (item,) = _inputs(workspace, "-lanything").values()

    assert "reconfigure" in item.reason


def test_a_linker_script_is_followed_to_what_it_names(workspace, tmp_path):
    """libc.so names libc_nonshared.a beside libc.so.6; glibc's libm.a is a
    script naming libm-<version>.a."""
    directory = tmp_path / "libs"
    synth.write(directory / "libscripted.so.6", synth.elf_shared_object("x.so.6"))
    synth.write(directory / "libscripted_nonshared.a", ONE_OBJECT)
    synth.write(directory / "libinner.a", ONE_OBJECT)
    synth.write(
        directory / "libscripted.so",
        "/* GNU ld script */\nOUTPUT_FORMAT(elf64-x86-64)\n"
        f"GROUP ( {directory}/libscripted.so.6 {directory}/libscripted_nonshared.a"
        f"  AS_NEEDED ( {directory}/libscripted.so.6 ) )\nINPUT(-linner)\n",
    )
    synth.write(directory / "libm-9.99.a", ONE_OBJECT)
    synth.write(directory / "libmath.a", "/* GNU ld script */\nGROUP ( libm-9.99.a )\n")
    # ld looks beside the script first, then where the link runs, then -L.
    elsewhere = tmp_path / "elsewhere"
    synth.write(directory / "libbeside.a", ONE_OBJECT)
    synth.write(elsewhere / "libbeside.a", ONE_OBJECT)
    synth.write(workspace.build / "libbeside.a", ONE_OBJECT)
    synth.write(workspace.build / "libcwd.a", ONE_OBJECT)
    synth.write(
        elsewhere / "libnear.so",
        "/* GNU ld script */\nGROUP ( libbeside.a libcwd.a )\n",
    )

    synth.link_line(
        workspace.build,
        "ext",
        (f"-L{directory}", "-lscripted", "-l:libmath.a", f"{elsewhere}/libnear.so"),
    )
    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)
    inputs = graph.static_inputs(graph.by_name(EXTENSION)[0])

    assert sorted((item.named, item.path) for item in inputs) == sorted(
        [
            (f"{elsewhere}/libnear.so", elsewhere / "libbeside.a"),
            (f"{elsewhere}/libnear.so", (workspace.build / "libcwd.a").resolve()),
            ("-l:libmath.a", directory / "libm-9.99.a"),
            ("-lscripted", directory / "libinner.a"),
            ("-lscripted", directory / "libscripted_nonshared.a"),
        ]
    )


def test_a_script_naming_itself_says_so(workspace, tmp_path):
    synth.write(tmp_path / "libself.so", "/* GNU ld script */\nGROUP ( libself.so )\n")

    inputs = _inputs(workspace, f"-L{tmp_path}", "-lself")

    assert [item.reason for item in inputs.values()] == [
        "linker scripts nest too deeply here"
    ]


@pytest.mark.parametrize("kind", ["EXECUTABLE", "MODULE_LIBRARY"])
def test_a_build_order_dependency_links_nothing_into_the_member(workspace, kind):
    """A code generator ext waits for is not linked into it, so neither its line
    nor a shared library reached only through it is."""
    synth.set_type(workspace.build, "unit_tests", kind)
    synth.link_line(workspace.build, "gamma", ("-l:libviagamma.a",))
    synth.link_line(
        workspace.build, "unit_tests", ("-l:libnothing.a",), dependencies=("gamma",)
    )
    synth.link_line(workspace.build, "ext", (), dependencies=("unit_tests",))

    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)

    assert graph.static_inputs(graph.by_name(EXTENSION)[0]) == {}
    assert {
        item.named for item in graph.static_inputs(graph.by_name("unit_tests")[0])
    } == {"-l:libnothing.a", "-l:libviagamma.a"}


def test_a_fragment_that_does_not_parse_is_unresolved_not_dropped(workspace):
    inputs = _inputs(workspace, '"/opt/lib/libunbalanced.a')

    (item,) = inputs.values()
    assert item.kind == "unresolved"
    assert "does not parse" in item.reason


def _link(workspace, *fragments) -> None:
    synth.link_line(workspace.build, "ext", fragments)


def test_a_packaged_archive_is_inventoried_through_its_package(
    workspace, tmp_path, monkeypatch
):
    archive = tmp_path / "host" / "usr" / "lib" / "libsquash.a"
    synth.write(archive, ONE_OBJECT)
    copyright_file = tmp_path / "host" / "copyright"
    synth.write(copyright_file, synth.spdx_text("Zlib"))
    _packaged(
        monkeypatch,
        "libsquash-dev",
        copyright_file,
        supplier="Organization: Example Distro (devel@example.com)",
        version="1.0-1",
        apt_origin="Ubuntu",
        apt_origin_from="installed version",
    )
    _link(workspace, str(archive))

    collected = synth.collect(workspace)

    row = _row(collected, "system:libsquash.a")
    assert row["license_concluded"] == "Zlib"
    assert row["how_it_reaches_the_wheel"] == ["prebuilt-library"]
    assert row["purl"] == "pkg:deb/ubuntu/libsquash-dev@1.0-1"
    assert row["source_info"].startswith(f"Statically linked from {archive}.")
    evidence = collected["build-evidence"]
    record = evidence["static_inputs"][str(archive)]
    assert record["package"] == "libsquash-dev"
    assert record["apt_origin"] == "Ubuntu"
    assert record["apt_origin_from"] == "installed version"
    assert record["linked_by"] == ["ext"]
    linked = next(
        item for item in evidence["link_artifacts"] if item["target"] == "ext"
    )
    assert linked["static_inputs"] == [str(archive)]


# `apt-cache policy` as apt prints it. NVIDIA's CUDA repository is a flat one,
# `deb <uri>/ /`, listed with an empty distribution.
UBUNTU_SOURCE = "http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages"
NVIDIA_SOURCE = (
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
    "  Packages"
)
SECURITY_SOURCE = "http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages"
FLAT_SOURCE = "https://repo.example.com/apt ./ Packages"
# A repository with no Release file states no Origin.
BARE_SOURCE = "file:/srv/debs ./ Packages"
# libnvidia-container's Origin is a URL.
URL_ORIGIN = "https://nvidia.github.io/libnvidia-container/stable"
URL_SOURCE = f"{URL_ORIGIN}/deb/amd64  Packages"
# A mirror spelling Ubuntu's Origin in lower case.
MIRROR_SOURCE = "http://mirror.example.org/ubuntu jammy-updates/main amd64 Packages"
HOST_SOURCE = "https://packages.example.com/apt stable/main amd64 Packages"
APT_SOURCES = f"""Package files:
 100 /var/lib/dpkg/status
     release a=now
 600 {NVIDIA_SOURCE}
     release o=NVIDIA,l=NVIDIA CUDA,c=
     origin developer.download.nvidia.com
 500 {UBUNTU_SOURCE}
     release v=22.04,o=Ubuntu,a=jammy-updates,n=jammy,l=Ubuntu,c=main,b=amd64
     origin archive.ubuntu.com
 500 {FLAT_SOURCE}
     release o=Example,c=
     origin repo.example.com
 500 {SECURITY_SOURCE}
     release v=22.04,o=Ubuntu,a=jammy-security,n=jammy,l=Ubuntu,c=main,b=amd64
     origin security.ubuntu.com
 500 {BARE_SOURCE}
 500 {URL_SOURCE}
     release o={URL_ORIGIN},l=NVIDIA CORPORATION <cudatools@nvidia.com>,c=
     origin nvidia.github.io
 500 {MIRROR_SOURCE}
     release o=ubuntu,a=jammy-updates,c=main,b=amd64
     origin mirror.example.org
 500 {HOST_SOURCE}
     release o=packages.example.com,a=stable,c=main,b=amd64
     origin packages.example.com
Pinned packages:
"""


def _version_table(*versions: tuple) -> str:
    """`(version, installed, sources[, note])` per version, as apt lists them."""
    lines = ["libsquash-dev:", "  Installed: 1.0-1", "  Version table:"]
    for version, installed, sources, *note in versions:
        lines.append(
            " ".join([f" {'***' if installed else '   '} {version} 500", *note])
        )
        lines += [f"        500 {source}" for source in sources]
    return "\n".join(lines) + "\n"


STATUS = "/var/lib/dpkg/status"
LIBRARY = "/usr/lib/libsquash.a"


def _host_package(
    monkeypatch,
    policy,
    owned=f"libsquash-dev:amd64: {LIBRARY}\n",
    qualified="libsquash-dev:amd64",
    sources=None,
):
    """What `_identify_host_package` makes of dpkg answering `owned`, and what it
    asked. dpkg-query -W and apt answer only for `qualified`."""
    answers = {
        ("dpkg-query", "-S", LIBRARY): owned,
        ("dpkg-query", "-W", "-f=${Version}", qualified): "1.0-1",
        ("apt-cache", "policy"): APT_SOURCES if sources is None else sources,
        ("apt-cache", "policy", qualified): policy,
    }
    asked = []

    def ask(*command, env=None):
        asked.append(command)
        return answers.get(command, "")

    monkeypatch.setattr(discovery, "_ask", ask)
    return discovery._identify_host_package(Path(LIBRARY), {}), asked


@pytest.mark.parametrize(
    ("policy", "namespace", "decided"),
    [
        pytest.param(
            _version_table(
                ("1.0-1", True, [UBUNTU_SOURCE, STATUS]),
                ("0.9-1", False, [FLAT_SOURCE]),
            ),
            "ubuntu",
            "installed version",
            id="installed-version-decides-not-another",
        ),
        pytest.param(
            _version_table(("1.0-1", True, [NVIDIA_SOURCE, STATUS])),
            "nvidia",
            "installed version",
            id="nvidia",
        ),
        pytest.param(
            _version_table(("1.0-1", True, [FLAT_SOURCE])),
            "example",
            "installed version",
            id="flat",
        ),
        pytest.param(
            _version_table(("1.0-1", True, [UBUNTU_SOURCE, FLAT_SOURCE, STATUS])),
            None,
            None,
            id="disagreeing",
        ),
        pytest.param(
            _version_table(
                ("1.0-1", True, [BARE_SOURCE, STATUS]),
                ("0.9-1", False, [UBUNTU_SOURCE]),
            ),
            None,
            None,
            id="no-origin",
        ),
        # As for libc6 in an ubuntu:22.04 image.
        pytest.param(
            _version_table(
                ("1.1-1", False, [UBUNTU_SOURCE, SECURITY_SOURCE]),
                ("1.0-1", True, [STATUS]),
                ("0.9-1", False, [UBUNTU_SOURCE]),
            ),
            "ubuntu",
            "other versions",
            id="installed-version-unlisted-other-versions-decide",
        ),
        pytest.param(
            _version_table(
                ("1.1-1", False, [UBUNTU_SOURCE]),
                ("1.0-1", True, [STATUS]),
                ("0.9-1", False, [FLAT_SOURCE]),
            ),
            None,
            None,
            id="installed-version-unlisted-others-disagree",
        ),
        pytest.param(
            _version_table(("1.0-1", True, [STATUS])), None, None, id="local-deb"
        ),
        pytest.param("", None, None, id="apt-knows-nothing"),
        # A phased update carries a note after its priority, translated.
        pytest.param(
            _version_table(
                ("1.0-1", True, [UBUNTU_SOURCE, STATUS], "(phased 20%)"),
                ("0.9-1", False, [FLAT_SOURCE]),
            ),
            "ubuntu",
            "installed version",
            id="phased-installed",
        ),
        pytest.param(
            _version_table(
                ("1.0-1", True, [NVIDIA_SOURCE, STATUS]),
                ("0.9-1", False, [UBUNTU_SOURCE], "(phased 20%)"),
            ),
            "nvidia",
            "installed version",
            id="phased-after-installed",
        ),
        pytest.param(
            _version_table(
                ("1.0-1", True, [NVIDIA_SOURCE, STATUS]),
                ("0.9-1", False, [UBUNTU_SOURCE], "(gestaffelt 20%)"),
            ),
            "nvidia",
            "installed version",
            id="phased-german",
        ),
        pytest.param(
            _version_table(("1.0-1", True, [UBUNTU_SOURCE, MIRROR_SOURCE, STATUS])),
            "ubuntu",
            "installed version",
            id="origins-agree-ignoring-case",
        ),
        pytest.param(
            _version_table(("1.0-1", True, [URL_SOURCE, STATUS])),
            None,
            "installed version",
            id="url-origin-is-no-namespace",
        ),
        pytest.param(
            _version_table(("1.0-1", True, [HOST_SOURCE, STATUS])),
            None,
            "installed version",
            id="host-origin-is-no-namespace",
        ),
    ],
)
def test_a_packages_namespace_is_the_archive_that_publishes_it(
    monkeypatch, policy, namespace, decided
):
    origin, _ = _host_package(monkeypatch, policy)

    purl = discovery.system_component("libsquash.a", origin).purl

    # pkg:deb requires its vendor namespace; without one, nothing is claimed.
    expected = f"pkg:deb/{namespace}/" if namespace else "pkg:generic/"
    assert purl == f"{expected}libsquash-dev@1.0-1"
    parsed = PackageURL.from_string(purl)
    assert (parsed.type, parsed.namespace, parsed.name, parsed.version) == (
        "deb" if namespace else "generic",
        namespace,
        "libsquash-dev",
        "1.0-1",
    )
    assert origin["apt_origin_from"] == decided


@pytest.mark.parametrize(
    ("owned", "qualified"),
    [
        pytest.param(
            f"libsquash-dev, libsquash-extra:amd64: {LIBRARY}\n",
            "libsquash-dev",
            id="several-owners-the-first-as-printed",
        ),
        pytest.param(
            f"libsquash-dev:i386: {LIBRARY}\n",
            "libsquash-dev:i386",
            id="foreign-architecture",
        ),
    ],
)
def test_dpkg_and_apt_are_asked_about_the_owner_dpkg_names(
    monkeypatch, owned, qualified
):
    origin, asked = _host_package(
        monkeypatch,
        _version_table(("1.0-1", True, [UBUNTU_SOURCE])),
        owned=owned,
        qualified=qualified,
    )

    assert origin["package"] == "libsquash-dev"
    assert origin["version"] == "1.0-1"
    assert origin["apt_origin"] == "Ubuntu"
    assert {
        ("dpkg-query", "-W", "-f=${Version}", qualified),
        ("dpkg-query", "-W", "-f=${Maintainer}", qualified),
        ("apt-cache", "policy", qualified),
    } <= set(asked)


@pytest.mark.parametrize(
    "owned",
    [
        pytest.param(
            f"diversion by squash-gl from: {LIBRARY}\n"
            f"diversion by squash-gl to: {LIBRARY}.distrib\n",
            id="diverted-unowned",
        ),
        pytest.param(
            f"diversion by squash-gl from: {LIBRARY}\n"
            f"diversion by squash-gl to: {LIBRARY}.distrib\n"
            f"libsquash-dev:amd64: {LIBRARY}\n",
            id="diverted-the-file-may-not-be-the-owners",
        ),
        pytest.param(
            f"diversion by squash-gl to: {LIBRARY}\n",
            id="only-a-diversion-names-the-path",
        ),
    ],
)
def test_a_diversion_line_names_no_owner(monkeypatch, owned):
    origin, _ = _host_package(monkeypatch, "", owned=owned)

    assert origin.get("package") is None
    purl = discovery.system_component("libsquash.a", origin).purl
    assert purl == "pkg:generic/libsquash.a"


def test_apt_sources_are_read_whatever_the_language(monkeypatch):
    """Headings are translated; a pinned package may be named `release`."""
    sources = (
        APT_SOURCES.replace("Package files:", "Fichiers du paquet :").replace(
            "Pinned packages:", "Paquets épinglés :"
        )
        + "     release -> 2.0-1 avec la priorité 990\n"
        + "     123 -> 1.0 avec la priorité 1001\n"
    )
    _host_package(monkeypatch, "", sources=sources)

    releases = discovery._apt_releases()

    assert set(releases) == {
        STATUS,
        NVIDIA_SOURCE,
        UBUNTU_SOURCE,
        FLAT_SOURCE,
        SECURITY_SOURCE,
        BARE_SOURCE,
        URL_SOURCE,
        MIRROR_SOURCE,
        HOST_SOURCE,
    }
    assert releases[BARE_SOURCE] == {}
    assert releases[NVIDIA_SOURCE]["o"] == "NVIDIA"
    assert releases[URL_SOURCE]["o"] == URL_ORIGIN


# A machine-readable copyright file whose licence paragraph quotes a notice.
DEP5 = (
    "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
    "Upstream-Name: squash\n\n"
    "Files: *\nCopyright: 2001-2020 Squash Authors\n"
    " 2019 Example Corp\nLicense: Squash\n\n"
    "License: Squash\n Permission is granted to use this software freely.\n"
    " Copyright (c) 1995 Someone Quoted In The Text\n"
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Not machine-readable: the notices it quotes belong to what it bundles.
        (EULA, "NOASSERTION"),
        (DEP5, "2001-2020 Squash Authors\n2019 Example Corp"),
        # Free-form, with a `Copyright:` heading at the start of a line.
        (
            "This package was debianized by someone.\n\n"
            "Copyright: 2001 Quoted Upstream\n",
            "NOASSERTION",
        ),
    ],
)
def test_a_system_component_names_only_the_holders_its_package_declares(
    workspace, tmp_path, monkeypatch, text, expected
):
    archive = tmp_path / "host" / "libsquash.a"
    synth.write(archive, ONE_OBJECT)
    copyright_file = tmp_path / "host" / "copyright"
    synth.write(copyright_file, text)
    _packaged(monkeypatch, "libsquash-dev", copyright_file)
    _link(workspace, str(archive))

    collected = synth.collect(workspace)

    row = _row(collected, "system:libsquash.a")
    assert _package(row, collected["spdx"])["copyrightText"] == expected


def test_a_cuda_archive_is_a_component_under_the_toolkits_terms(
    workspace, tmp_path, monkeypatch
):
    toolkit = tmp_path / "cuda-99.9"
    archive = toolkit / "targets" / "x86_64-linux" / "lib" / "libcudart_static.a"
    synth.write(archive, ONE_OBJECT)
    synth.write(toolkit / "EULA.txt", EULA)
    _no_package(monkeypatch)
    monkeypatch.setattr(discovery, "_toolkit_roots", lambda: [toolkit.resolve()])
    _link(workspace, str(archive))

    collected = synth.collect(workspace)

    key = "system:libcudart_static.a"
    row = _row(collected, key)
    assert row["how_it_reaches_the_wheel"] == ["prebuilt-library"]
    assert [item["path"] for item in row["evidence"]] == [
        str((toolkit / "EULA.txt").resolve())
    ]
    assert row["license_concluded"].startswith("LicenseRef-")
    assert _package(row, collected["spdx"])["copyrightText"] == "NOASSERTION"

    member = next(
        item
        for item in collected["build-evidence"]["attributions"]
        if item["path"].endswith(EXTENSION)
    )
    assert member["components"][key] == ["prebuilt-library"]
    assert {
        "spdxElementId": document_module.WHEEL_PACKAGE_ID,
        "relatedSpdxElement": document_module.component_package_id(key),
        "relationshipType": "STATIC_LINK",
    } in collected["spdx"]["relationships"]


def test_two_archives_sharing_a_name_are_two_components(
    workspace, tmp_path, monkeypatch
):
    first = tmp_path / "one" / "libsame.a"
    second = tmp_path / "two" / "libsame.a"
    for archive in (first, second):
        synth.write(archive, ONE_OBJECT)
        synth.write(archive.parent / "copyright", synth.spdx_text("Zlib"))
    _packaged(
        monkeypatch,
        lambda library: f"libsame-{library.parent.name}",
        lambda library: library.parent / "copyright",
    )
    _link(workspace, str(first), str(second))

    collected = synth.collect(workspace)

    records = collected["build-evidence"]["static_inputs"]
    keys = {records[str(first)]["component"], records[str(second)]["component"]}
    assert len(keys) == 2
    assert keys <= {item["component"] for item in collected["licenses"]["components"]}


def test_an_archive_with_no_members_is_recorded_not_inventoried(
    workspace, tmp_path, monkeypatch
):
    archive = tmp_path / "host" / "librt.a"
    synth.write(archive, synth.ar_archive({}))
    _no_package(monkeypatch)
    _link(workspace, str(archive))

    collected = synth.collect(workspace)

    record = collected["build-evidence"]["static_inputs"][str(archive)]
    assert record["members"] == 0
    assert record["component"] is None
    assert "no members" in record["outcome"]
    assert not any(
        item["component"].endswith("librt.a")
        for item in collected["licenses"]["components"]
    )


def _tracked_archive(workspace, monkeypatch, holder=None):
    """A tracked archive linked into ext, held by `holder` through REUSE.toml."""
    relative = "prebuilt/libtracked.a"
    synth.write(workspace.root / relative, ONE_OBJECT)
    if holder:
        synth.write(
            workspace.root / "REUSE.toml",
            synth.reuse_toml(synth.annotation(relative, holder)),
        )
    synth.track(workspace.root)
    _no_package(monkeypatch)
    _link(workspace, str(workspace.root / relative))
    return (workspace.root / relative).resolve()


def test_a_tracked_archive_is_credited_to_the_holder_reuse_names(
    workspace, monkeypatch
):
    _tracked_archive(workspace, monkeypatch, "Upstream Archive Co")

    collected = synth.collect(workspace)

    row = _row(collected, "vendored:upstream-archive-co")
    assert "prebuilt-library" in row["how_it_reaches_the_wheel"]


def test_a_tracked_archive_naming_no_other_holder_is_the_projects(
    workspace, monkeypatch
):
    """Every tracked file has REUSE info; one naming nobody else is the project's."""
    archive = _tracked_archive(workspace, monkeypatch)

    collected = synth.collect(workspace)

    record = collected["build-evidence"]["static_inputs"][str(archive)]
    assert record["component"] is None
    assert "held by this project" in record["outcome"]


@pytest.mark.parametrize("where", ["host", "build", "unresolved"])
def test_an_input_nothing_explains_stops_the_build(
    workspace, tmp_path, monkeypatch, where
):
    _no_package(monkeypatch)
    monkeypatch.setattr(discovery, "_toolkit_roots", list)
    if where == "unresolved":
        fragment = "-l:libmystery.a"
    else:
        root = workspace.build / "prebuilt" if where == "build" else tmp_path / "opt"
        synth.write(root / "libmystery.a", ONE_OBJECT)
        fragment = str((root / "libmystery.a").resolve())
    _link(workspace, fragment)

    with pytest.raises(
        build_module.BuildError, match=r"libmystery\.a \(linked by ext\)"
    ):
        synth.collect(workspace)


FMT_LICENCE_HEAD = (
    "Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors\n\n"
)


def _fmt_port(installed) -> None:
    synth.vcpkg_port(
        installed,
        "fmt",
        "11.0.2#1",
        {
            "lib/libfmt.a": ONE_OBJECT,
            "debug/lib/libfmtd.a": ONE_OBJECT,
            "include/fmt/core.h": b"#pragma once\n",
        },
        FMT_LICENCE_HEAD + synth.spdx_text("MIT"),
        "fmtlib/fmt",
    )


@pytest.mark.parametrize("outside_build", [False, True])
def test_a_vcpkg_archive_is_credited_to_the_port_that_installed_it(
    workspace, tmp_path, monkeypatch, outside_build
):
    """spdlog built against vcpkg's fmt puts libfmt.a on every consumer's line."""
    installed = (
        tmp_path / "elsewhere" / "vcpkg_installed"
        if outside_build
        else workspace.build / "vcpkg_installed"
    )
    if outside_build:
        cache = workspace.build / "CMakeCache.txt"
        cache.write_text(cache.read_text() + f"VCPKG_INSTALLED_DIR:PATH={installed}\n")
    _fmt_port(installed)
    _no_package(monkeypatch)
    _link(workspace, str(installed / "x64-linux" / "lib" / "libfmt.a"))

    collected = synth.collect(workspace)

    row = _row(collected, "vcpkg:fmt")
    assert row["kind"] == "vcpkg-port"
    assert row["version"] == "11.0.2"
    assert row["license_concluded"] == "MIT"
    assert row["purl"] == "pkg:github/fmtlib/fmt@11.0.2"
    assert row["supplier"] == "Organization: fmtlib"
    assert "prebuilt-library" in row["how_it_reaches_the_wheel"]
    assert "vcpkg port fmt 11.0.2#1 for x64-linux" in row["source_info"]
    package = _package(row, collected["spdx"])
    assert "Victor Zverovich" in package["copyrightText"]
    assert {
        "spdxElementId": document_module.WHEEL_PACKAGE_ID,
        "relatedSpdxElement": package["SPDXID"],
        "relationshipType": "STATIC_LINK",
    } in collected["spdx"]["relationships"]


@pytest.mark.parametrize("target", ["x64-linux", "arm64-linux"])
def test_a_port_installed_for_two_triplets_owns_both_installs(workspace, target):
    """A port the build needs as a host tool and as a library has a list per
    triplet. Both installs are its files; the component describes the target's."""
    installed = workspace.build / "vcpkg_installed"
    triplets = ("arm64-linux", "x64-linux")
    for triplet in triplets:
        synth.vcpkg_port(
            installed,
            "fmt",
            "11.0.2#1",
            {"lib/libfmt.a": ONE_OBJECT},
            synth.spdx_text("MIT"),
            "fmtlib/fmt",
            triplet=triplet,
        )

    components, owned = discovery.discover_vcpkg(installed, target)

    assert {installed / triplet / "lib" / "libfmt.a" for triplet in triplets} <= set(
        owned["vcpkg:fmt"]
    )
    assert components["vcpkg:fmt"].source_info.startswith(
        f"Built and installed by vcpkg port fmt 11.0.2#1 for {target}."
    )


def test_the_cache_names_the_triplet_a_port_is_described_from(workspace):
    installed = workspace.build / "vcpkg_installed"
    for triplet in ("arm64-linux", "x64-linux"):
        synth.vcpkg_port(
            installed,
            "fmt",
            "11.0.2#1",
            {"lib/libfmt.a": ONE_OBJECT},
            synth.spdx_text("MIT"),
            "fmtlib/fmt",
            triplet=triplet,
        )
    cache = workspace.build / "CMakeCache.txt"
    cache.write_text(cache.read_text() + "VCPKG_TARGET_TRIPLET:STRING=x64-linux\n")

    space = evidence_module.discover(workspace.root, workspace.build)

    assert "for x64-linux." in space.components["vcpkg:fmt"].source_info


def test_a_file_no_port_lists_is_still_unexplained(workspace, monkeypatch):
    installed = workspace.build / "vcpkg_installed"
    _fmt_port(installed)
    stray = installed / "x64-linux" / "lib" / "libstray.a"
    synth.write(stray, ONE_OBJECT)
    _no_package(monkeypatch)
    _link(workspace, str(stray))

    with pytest.raises(build_module.BuildError, match=r"libstray\.a \(linked by ext\)"):
        synth.collect(workspace)


def test_an_unlinked_port_is_excluded_as_installed_by_vcpkg(workspace):
    synth.vcpkg_port(
        workspace.build / "vcpkg_installed",
        "zlib",
        "1.3.1",
        {"lib/libz.a": ONE_OBJECT},
        synth.spdx_text("Zlib"),
        "madler/zlib",
    )

    collected = synth.collect(workspace)

    exclusion = next(
        item
        for item in collected["licenses"]["exclusions"]
        if item["component"] == "vcpkg:zlib"
    )
    assert exclusion["reason"].startswith("installed by vcpkg for this build")


def _eigen_port(installed) -> None:
    synth.vcpkg_port(
        installed,
        "eigen3",
        "3.4.0#5",
        {"include/eigen3/Eigen/Core": b"#pragma once\n"},
        synth.spdx_text("MIT"),
        ("libeigen/eigen", "libeigen/eigen-patches"),
    )


def test_an_include_directory_one_port_owns_credits_that_port(workspace):
    """The fixture's extension also searches the shared include root."""
    installed = workspace.build / "vcpkg_installed"
    _fmt_port(installed)
    _eigen_port(installed)
    synth.add_include(
        workspace.build, "ext", installed / "x64-linux" / "include" / "eigen3"
    )

    space = evidence_module.discover(workspace.root, workspace.build)
    graph = space.graph
    contributions = graph.contributions(graph.by_name(EXTENSION)[0], space.ownership)

    assert contributions["vcpkg:eigen3"] == {"header-include"}
    assert "vcpkg:fmt" not in contributions


def test_a_port_with_several_sources_names_no_upstream(workspace):
    _eigen_port(workspace.build / "vcpkg_installed")

    collected = synth.collect(workspace)

    row = _row(collected, "vcpkg:eigen3")
    assert row["version"] == "3.4.0"
    assert row["purl"] == "pkg:generic/eigen3@3.4.0"
    assert row["supplier"] == "NOASSERTION"
    assert "vcpkg port eigen3 3.4.0#5" in row["source_info"]


def test_a_symlinked_script_is_read_beside_the_link(workspace, tmp_path):
    """ld resolves a script's relative names beside the name it was given."""
    target, links = tmp_path / "target", tmp_path / "links"
    synth.write(target / "libreal.so", "/* GNU ld script */\nGROUP ( libmine.a )\n")
    synth.write(target / "libmine.a", ONE_OBJECT)
    synth.write(links / "libmine.a", ONE_OBJECT)
    (links / "libsym.so").symlink_to(target / "libreal.so")

    (item,) = _inputs(workspace, f"-L{links}", "-lsym").values()

    assert item.path == links / "libmine.a"


def test_ownership_sees_a_file_registered_after_a_lookup(tmp_path):
    ownership = discovery.Ownership()
    include = tmp_path / "include"
    ownership.register_file(include / "a" / "a.h", "port-a", by_directory=False)
    assert ownership.owners_under(include) == {"port-a"}

    ownership.register_file(include / "b" / "b.h", "port-b", by_directory=False)
    ownership.register_file(include / "c.h", "fetched")

    # Two listed owners under one directory credit neither; each alone does.
    assert ownership.owners_under(include) == {"fetched"}
    assert ownership.owners_under(include / "b") == {"port-b"}


def test_an_empty_archive_credits_no_one_however_it_is_held(workspace):
    installed = workspace.build / "vcpkg_installed"
    synth.vcpkg_port(
        installed,
        "stubs",
        "1.0",
        {"lib/libstub.a": synth.ar_archive({})},
        synth.spdx_text("MIT"),
        "example/stubs",
    )
    _link(workspace, str(installed / "x64-linux" / "lib" / "libstub.a"))

    space = evidence_module.discover(workspace.root, workspace.build)
    contributions = space.graph.contributions(
        space.graph.by_name(EXTENSION)[0], space.ownership
    )

    assert "vcpkg:stubs" not in contributions

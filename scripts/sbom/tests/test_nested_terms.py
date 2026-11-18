# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope nested license terms to the files contributed to the wheel."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
import synth
from isaaccapture_sbom import build as build_module
from isaaccapture_sbom import discovery
from isaaccapture_sbom import evidence as evidence_module
from isaaccapture_sbom import inventory as inventory_module
from isaaccapture_sbom import licensing
from isaaccapture_sbom import report

BUNDLED = "Copyright (c) 2019 Bundled Authors\n\n" + "Permission is hereby granted."
EXTENSION = "isaaccapture/_ext.cpython-312-x86_64-linux-gnu.so"


@pytest.fixture
def alpha(workspace):
    return workspace.build / "_deps" / "alpha-src"


@pytest.fixture
def beta(workspace):
    return workspace.build / "_deps" / "beta-src"


def _deps(workspace):
    return workspace.build / "_deps"


def _compiled(workspace):
    """The sources the fixture's `ext` target compiles, as a depfile names them."""
    return (
        (workspace.root / "src/cpp/module.cpp").resolve(),
        (_deps(workspace) / "alpha-src/alpha.cpp").resolve(),
    )


def _depfiles(workspace, read=(), extra=()) -> None:
    """A depfile per object `ext` compiles; the first also read `read`."""
    first, *rest = (*_compiled(workspace), *(item.resolve() for item in extra))
    synth.depfile(workspace.build, "ext", first, tuple(read))
    for source in rest:
        synth.depfile(workspace.build, "ext", source)


def _kept(workspace, built=None) -> dict[str, dict | None]:
    """Each nested text the build found, with what kept it, or None if dropped."""
    built = built or synth.collect(workspace)
    found = {
        item["path"]: item.get("reached_by")
        for component in built["build-evidence"]["components"].values()
        for item in component["evidence"]
        if item["kind"] == "nested"
    }
    for root in sorted(_deps(workspace).glob("*-src")):
        for item in licensing.discover_in_tree(root.name, root, root.name):
            if item.kind == "nested":
                found.setdefault(item.path, None)
    return found


def _package(built, name):
    return next(item for item in built["spdx"]["packages"] if item["name"] == name)


def test_a_compiled_in_library_keeps_its_terms_and_holder(workspace, alpha):
    synth.write(alpha / "third/bundled.cpp", "// bundled\n")
    synth.write(alpha / "third/LICENSE", BUNDLED)
    synth.write(alpha / "test/LICENSE", "Copyright (c) 2018 Test Authors\n")
    synth.add_source(workspace.build, "ext", alpha / "third/bundled.cpp")
    _depfiles(workspace, extra=(alpha / "third/bundled.cpp",))

    built = synth.collect(workspace)
    kept = _kept(workspace, built)

    assert kept["alpha-src/third/LICENSE"]["how"] == "compiled-source"
    assert kept["alpha-src/third/LICENSE"]["path"] == "alpha-src/third/bundled.cpp"
    assert kept["alpha-src/test/LICENSE"] is None
    copyright_text = _package(built, "alpha")["copyrightText"]
    assert "Bundled Authors" in copyright_text
    assert "Test Authors" not in copyright_text


def test_bundled_gpl_text_names_no_steward_and_reads_as_alternatives(workspace, alpha):
    # As upstreams ship it: `(C)` and `http://`, where the list has `©` and `https://`.
    gpl = (
        synth.spdx_text("GPL-3.0-only")
        .replace("©", "(C)")
        .replace("https://fsf.org/", "http://fsf.org/")
    )
    synth.write(alpha / "third/bundled.cpp", "// bundled\n")
    synth.write(alpha / "third/LICENSE-gpl3.txt", gpl)
    synth.add_source(workspace.build, "ext", alpha / "third/bundled.cpp")
    _depfiles(workspace, extra=(alpha / "third/bundled.cpp",))

    built = synth.collect(workspace)

    alternatives = "matches the reference text of GPL-3.0-only or GPL-3.0-or-later"
    package = _package(built, "alpha")
    assert "Free Software Foundation" not in package["copyrightText"]
    assert package["licenseConcluded"] == "MIT"
    texts = [item for item in package["attributionTexts"] if "LICENSE-gpl3" in item]
    assert "matches the reference text of GPL-3.0-only (" in texts[0]
    assert texts[0].endswith(report.NESTED_ROLE)
    row = next(line for line in built["md"].splitlines() if "| alpha |" in line)
    assert f"→ {alternatives} ({report.NESTED_ROLE})" in row
    assert "GPL-3.0-only AND" not in built["md"] + json.dumps(built["spdx"])
    assert f"LICENSE-gpl3.txt` — {report.NESTED_ROLE}" in built["notices"]


def test_a_header_the_depfile_says_was_read_keeps_its_terms(workspace, beta):
    synth.write(beta / "include/beta/fmt/format.h", "#pragma once\n")
    synth.write(beta / "include/beta/fmt/LICENSE", BUNDLED)
    synth.write(beta / "include/beta/unused/LICENSE", BUNDLED)
    synth.write(beta / "docs/LICENSE", BUNDLED)
    _depfiles(workspace, read=(beta / "include/beta/fmt/format.h",))

    kept = _kept(workspace)

    assert kept["beta-src/include/beta/fmt/LICENSE"] == {
        "how": "header-read",
        "path": "beta-src/include/beta/fmt/format.h",
        "member": EXTENSION,
    }
    assert kept["beta-src/include/beta/unused/LICENSE"] is None
    assert kept["beta-src/docs/LICENSE"] is None


def test_a_depfile_overrules_the_include_directory(workspace, beta):
    synth.write(beta / "include/COPYING", BUNDLED)
    _depfiles(workspace)

    assert _kept(workspace)["beta-src/include/COPYING"] is None


def _no_record(workspace) -> None:
    """No dependency output at all."""


def _one_record_short(workspace) -> None:
    """module.cpp includes alpha.cpp, which has no object record of its own."""
    module, alpha = _compiled(workspace)
    synth.depfile(workspace.build, "ext", module, (alpha,))


def _placeholder(workspace) -> None:
    """The compiler_depend.make CMake writes before the first build."""
    synth.compiler_depend(workspace.build, "ext", {})


@pytest.mark.parametrize("records", [_no_record, _one_record_short, _placeholder])
def test_without_a_record_per_object_include_directories_stand_in(
    workspace, beta, records
):
    synth.write(beta / "include/COPYING", BUNDLED)
    synth.write(beta / "include/beta/fmt/LICENSE", BUNDLED)
    synth.write(beta / "docs/LICENSE", BUNDLED)
    records(workspace)

    kept = _kept(workspace)

    assert kept["beta-src/include/COPYING"]["how"] == "include-dir"
    assert kept["beta-src/include/COPYING"]["path"] == "beta-src/include"
    assert kept["beta-src/include/beta/fmt/LICENSE"]["how"] == "include-dir"
    assert kept["beta-src/docs/LICENSE"] is None


def test_a_stale_depfile_is_ignored_and_a_precompiled_header_counts(workspace, beta):
    for name in ("stale", "pch", "unity"):
        synth.write(beta / f"{name}/{name}.h", "#pragma once\n")
        synth.write(beta / f"{name}/LICENSE", BUNDLED)
    _depfiles(workspace)
    # A source the target no longer compiles, and a header CMake generated into
    # the objects directory to precompile.
    synth.depfile(
        workspace.build,
        "ext",
        (workspace.root / "src/cpp/removed.cpp").resolve(),
        ((beta / "stale/stale.h").resolve(),),
    )
    synth.depfile(
        workspace.build,
        "ext",
        (workspace.build / "CMakeFiles/ext.dir/cmake_pch.hxx").resolve(),
        ((beta / "pch/pch.h").resolve(),),
    )
    # A unity source left from a build the codemodel no longer describes.
    synth.depfile(
        workspace.build,
        "ext",
        (workspace.build / "CMakeFiles/ext.dir/Unity/unity_0_cxx.cxx").resolve(),
        ((beta / "unity/unity.h").resolve(),),
    )

    kept = _kept(workspace)

    assert kept["beta-src/stale/LICENSE"] is None
    assert kept["beta-src/unity/LICENSE"] is None
    assert kept["beta-src/pch/LICENSE"]["how"] == "header-read"


def test_an_include_directory_holding_the_whole_tree_keeps_its_terms(workspace, beta):
    synth.write(beta / "vendor/LICENSE", BUNDLED)
    # `-I _deps` credits every fetched tree, so each needs its own terms.
    synth.write(_deps(workspace) / "license-list-data-src/LICENSE", BUNDLED)
    synth.add_include(workspace.build, "ext", workspace.build / "_deps")

    assert _kept(workspace)["beta-src/vendor/LICENSE"] == {
        "how": "include-dir",
        "path": "beta-src",
        "member": EXTENSION,
    }


def test_without_dependency_output_listed_sources_stand_in(workspace, beta):
    synth.write(beta / "src/impl.h", "#pragma once\n")
    synth.write(beta / "src/LICENSE", BUNDLED)
    synth.add_source(workspace.build, "ext", beta / "src/impl.h")

    assert _kept(workspace)["beta-src/src/LICENSE"]["how"] == "listed-source"


def _depfiles_in_sub(workspace, monkeypatch) -> None:
    """Depfiles name paths from where make ran the compiler."""
    module, alpha = _compiled(workspace)
    header = "../_deps/beta-src/include/beta/fmt/format.h"
    synth.depfile(workspace.build, "ext", module, (header,), directory="sub")
    synth.depfile(workspace.build, "ext", alpha, directory="sub")


def _compiler_depend_in_sub(workspace, monkeypatch) -> None:
    """compiler_depend.make names paths inside the build from its top."""
    module, _ = _compiled(workspace)
    objects = "sub/CMakeFiles/ext.dir"
    header = "_deps/beta-src/include/beta/fmt/format.h"
    synth.compiler_depend(
        workspace.build,
        "ext",
        {
            f"{objects}/module.cpp.o": (str(module), header),
            f"{objects}/alpha.cpp.o": ("_deps/alpha-src/alpha.cpp",),
        },
        directory="sub",
    )


def _ninja_log_in_sub(workspace, monkeypatch) -> None:
    """Ninja's deps log names paths from the top of the build."""
    module, _ = _compiled(workspace)
    objects = "sub/CMakeFiles/ext.dir"
    _ninja(
        workspace,
        monkeypatch,
        f"{objects}/module.cpp.o: #deps 2, deps mtime 1 (VALID)\n"
        f"    {module}\n    _deps/beta-src/include/beta/fmt/format.h\n\n"
        f"{objects}/alpha.cpp.o: #deps 1, deps mtime 1 (VALID)\n"
        "    _deps/alpha-src/alpha.cpp\n\n",
    )


@pytest.mark.parametrize(
    "records", [_depfiles_in_sub, _compiler_depend_in_sub, _ninja_log_in_sub]
)
def test_relative_paths_in_dependency_output_resolve_as_written(
    workspace, beta, records, monkeypatch
):
    synth.write(beta / "include/beta/fmt/LICENSE", BUNDLED)
    synth.set_directory(workspace.build, "ext", "sub")
    records(workspace, monkeypatch)

    assert _kept(workspace)["beta-src/include/beta/fmt/LICENSE"]["how"] == "header-read"


def test_a_linked_archive_keeps_every_nested_text_of_its_component(workspace, beta):
    synth.write(beta / "lib/libbeta.a", synth.ar_archive({"b.o": synth.elf_object()}))
    synth.write(beta / "vendor/LICENSE", BUNDLED)
    synth.link_line(workspace.build, "ext", (str(beta / "lib/libbeta.a"),))
    _depfiles(workspace)

    assert _kept(workspace)["beta-src/vendor/LICENSE"] == {
        "how": "prebuilt-library",
        "path": "beta-src",
        "member": EXTENSION,
    }


def test_an_archive_owned_outside_the_component_trees_keeps_them_too(
    workspace, beta, tmp_path
):
    prebuilt = tmp_path / "prebuilt"
    synth.write(prebuilt / "libbeta.a", synth.ar_archive({"b.o": synth.elf_object()}))
    synth.write(beta / "vendor/LICENSE", BUNDLED)
    synth.link_line(workspace.build, "ext", (str(prebuilt / "libbeta.a"),))
    _depfiles(workspace)
    space = evidence_module.discover(workspace.root, workspace.build)
    space.ownership.register_root(prebuilt, "beta")

    evidence = synth.collect(workspace, discovery=space)["build-evidence"]

    nested = next(
        item
        for item in evidence["components"]["beta"]["evidence"]
        if item["path"] == "beta-src/vendor/LICENSE"
    )
    assert nested["reached_by"]["how"] == "prebuilt-library"


def test_a_header_read_from_the_build_tree_is_its_source_file(workspace, beta):
    built = _deps(workspace) / "beta-build"
    synth.write(beta / "include/beta/fmt/format.h", "// fmt\n")
    synth.write(beta / "include/beta/fmt/LICENSE", BUNDLED)
    synth.write(beta / "gen/LICENSE", BUNDLED)
    # A copy staged elsewhere, and a header generated where its tree would hold it.
    synth.write(built / "staged/format.h", "// fmt\n")
    synth.write(built / "gen/config.h", "#define GENERATED 1\n")
    _depfiles(workspace, read=(built / "staged/format.h", built / "gen/config.h"))

    kept = _kept(workspace)

    assert kept["beta-src/include/beta/fmt/LICENSE"]["path"] == (
        "beta-src/include/beta/fmt/format.h"
    )
    assert kept["beta-src/gen/LICENSE"]["path"] == "beta-src/gen/config.h"


def test_a_component_path_outside_its_trees_is_not_reach(workspace, alpha, tmp_path):
    synth.write(alpha / "fmt/LICENSE", BUNDLED)
    elsewhere = tmp_path / "other" / "_deps" / "alpha-src" / "fmt" / "format.h"
    _depfiles(workspace, read=(elsewhere,))

    assert _kept(workspace)["alpha-src/fmt/LICENSE"] is None


def test_a_build_tree_file_keeps_its_place_and_names_only_its_own_sources(
    workspace, alpha, beta, tmp_path
):
    deps = _deps(workspace)
    synth.write(beta / "include/beta/fmt/format.h", "// fmt\n")
    synth.write(alpha / "other/other.h", "// other\n")
    synth.write(deps / "beta-build/gen/format.h", "// fmt\n")
    synth.write(deps / "beta-build/gen/other.h", "// other\n")
    synth.write(beta / "a/dup.h", "// dup\n")
    synth.write(beta / "b/dup.h", "// dup\n")
    # Empty, like many files of any tree: the bytes say nothing about origin.
    synth.write(beta / "empty/blank.h", "")
    synth.write(deps / "beta-build/gen/blank.h", "")
    # A source tree reached through a link has its `-build` sibling beside the link.
    real = tmp_path / "elsewhere" / "linked"
    real.mkdir(parents=True)
    (deps / "linked-src").symlink_to(real)
    synth.write(deps / "linked-build/gen/config.h", "#define X 1\n")
    space = evidence_module.discover(workspace.root, workspace.build)
    resolver = inventory_module.Resolver(space)

    def shown(key, path):
        return sorted(resolver._displayed({key: {path.resolve(): "header-read"}})[key])

    assert shown("beta", deps / "beta-build/gen/format.h") == [
        "beta-src/gen/format.h",
        "beta-src/include/beta/fmt/format.h",
    ]
    assert shown("beta", deps / "beta-build/gen/other.h") == ["beta-src/gen/other.h"]
    assert shown("beta", deps / "beta-build/gen/blank.h") == ["beta-src/gen/blank.h"]
    assert shown("beta", beta / "a/dup.h") == ["beta-src/a/dup.h"]
    assert shown("linked", deps / "linked-build/gen/config.h") == [
        "linked-src/gen/config.h"
    ]


def test_reach_is_attributed_with_the_ownership_asked_about(workspace):
    config = _deps(workspace) / "beta-build/gen/config.h"
    synth.write(config, "#define X 1\n")
    _depfiles(workspace, read=(config,))
    space = evidence_module.discover(workspace.root, workspace.build)
    artifact = space.graph.by_name(EXTENSION.rsplit("/", 1)[-1])[0]

    # Only the ownership index knows a `-build` tree's owner.
    assert "beta" not in space.graph.reached(artifact)
    assert "beta" in space.graph.reached(artifact, space.ownership)


def test_a_build_tool_or_custom_target_is_no_reach(workspace, beta):
    synth.write(beta / "codegen/LICENSE", BUNDLED)
    synth.write(beta / "scripts/LICENSE", BUNDLED)
    synth.add_target(
        workspace.build,
        "beta_codegen",
        "EXECUTABLE",
        (str(beta / "codegen/main.cpp"),),
        "ext",
    )
    synth.add_target(
        workspace.build,
        "beta_generate",
        "UTILITY",
        (str(beta / "scripts/generate.py"),),
        "ext",
    )
    _depfiles(workspace)

    kept = _kept(workspace)

    assert kept["beta-src/codegen/LICENSE"] is None
    assert kept["beta-src/scripts/LICENSE"] is None


def test_a_reuse_sidecar_covers_only_its_file(workspace, alpha):
    synth.write(alpha / "third/bundled.cpp", "// bundled\n")
    synth.write(alpha / "third/data.txt", "data\n")
    synth.write(alpha / "third/data.txt.license", "SPDX-FileCopyrightText: 2020 A\n")
    synth.write(alpha / "third/used.h", "#pragma once\n")
    synth.write(alpha / "third/used.h.license", "SPDX-FileCopyrightText: 2020 B\n")
    synth.add_source(workspace.build, "ext", alpha / "third/bundled.cpp")
    _depfiles(
        workspace, read=(alpha / "third/used.h",), extra=(alpha / "third/bundled.cpp",)
    )

    kept = _kept(workspace)

    assert kept["alpha-src/third/data.txt.license"] is None
    assert kept["alpha-src/third/used.h.license"] == {
        "how": "header-read",
        "path": "alpha-src/third/used.h",
        "member": EXTENSION,
    }


def test_copied_and_derived_members_keep_the_terms_beside_them(workspace):
    vendorpy = _deps(workspace) / "vendorpy-src"
    synth.write(vendorpy / "pkg/LICENSE", BUNDLED)
    synth.write(vendorpy / "pkg/assets/NOTICE", BUNDLED)
    synth.write(vendorpy / "docs/LICENSE", BUNDLED)

    kept = _kept(workspace)

    assert kept["vendorpy-src/pkg/LICENSE"] == {
        "how": "copied-file",
        "path": "vendorpy-src/pkg/mod.py",
        "member": "vendorpy/mod.py",
    }
    assert kept["vendorpy-src/pkg/assets/NOTICE"]["how"] == "derived-file"
    assert kept["vendorpy-src/docs/LICENSE"] is None


def _ninja(workspace, monkeypatch, log: str) -> None:
    synth.set_generator(workspace.build, "Ninja")
    asked = discovery._ask
    monkeypatch.setattr(
        discovery,
        "_ask",
        lambda *command, **options: (
            log if command[0] == "ninja" else asked(*command, **options)
        ),
    )


def test_ninja_deps_log_is_read_for_what_a_target_read(
    workspace, monkeypatch, caplog, beta
):
    synth.write(beta / "include/beta/fmt/LICENSE", BUNDLED)
    header = (beta / "include/beta/fmt/format.h").resolve()
    _ninja(
        workspace,
        monkeypatch,
        "".join(
            f"CMakeFiles/ext.dir/{source.name}.o: #deps 2, deps mtime 1 (VALID)\n"
            f"    {source}\n    {header}\n\n"
            for source in _compiled(workspace)
        )
        # An output whose depfile listed nothing, as a precompiled header's can.
        + "CMakeFiles/ext.dir/cmake_pch.hxx.gch: #deps 0, deps mtime 1 (VALID)\n\n",
    )

    with caplog.at_level(logging.WARNING, logger="isaaccapture.sbom"):
        kept = _kept(workspace)

    assert kept["beta-src/include/beta/fmt/LICENSE"]["how"] == "header-read"
    assert not [item for item in caplog.records if "ninja -t deps" in item.message]


def test_ninja_records_count_only_for_their_own_objects_directory(
    workspace, monkeypatch, beta
):
    synth.write(beta / "other/LICENSE", BUNDLED)
    other = (beta / "other/other.h").resolve()
    module, _ = _compiled(workspace)
    _ninja(
        workspace,
        monkeypatch,
        "".join(
            f"CMakeFiles/ext.dir/{source.name}.o: #deps 1, deps mtime 1 (VALID)\n"
            f"    {source}\n\n"
            for source in _compiled(workspace)
        )
        # Another target compiling the same source with other headers.
        + "sub/CMakeFiles/other.dir/module.cpp.o: #deps 2, deps mtime 1 (VALID)\n"
        f"    {module}\n    {other}\n\n",
    )

    assert _kept(workspace)["beta-src/other/LICENSE"] is None


def test_without_ninja_the_fallback_is_announced_once(
    workspace, monkeypatch, caplog, beta
):
    synth.write(beta / "include/COPYING", BUNDLED)
    _ninja(workspace, monkeypatch, "")

    with caplog.at_level(logging.WARNING, logger="isaaccapture.sbom"):
        kept = _kept(workspace)

    assert kept["beta-src/include/COPYING"]["how"] == "include-dir"
    assert (
        len([item for item in caplog.records if "ninja -t deps" in item.message]) == 1
    )


def test_depfile_rules_are_read_as_make_reads_them():
    text = (
        "# CMAKE generated file: DO NOT EDIT!\n"
        "obj.o: /src/a\\ b.cpp \\\n /inc/x$$y.h \\\n  /inc/z.h\n"
        "\n/inc/z.h:\n"
    )

    assert discovery._make_rules(text) == [["/src/a b.cpp", "/inc/x$y.h", "/inc/z.h"]]


def test_registering_an_owner_forgets_what_was_asked_before(tmp_path):
    ownership = discovery.Ownership()
    source = tmp_path / "tree" / "a.h"
    vendored = tmp_path / "vendor" / "b.h"
    assert ownership.owner_of(source) is None
    ownership.register_root(tmp_path / "tree", "tree")
    assert ownership.owner_of(source) == "tree"

    assert ownership.owner_of(vendored) is None
    ownership.register_file(vendored, "vendored:b")
    assert ownership.owner_of(vendored) == "vendored:b"


def test_a_steward_line_is_recognised_however_its_copy_spells_it():
    notice = "Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>"

    assert licensing.notice_is_the_licence_authors(notice, ["GPL-3.0-only"])


def test_the_evidence_names_the_first_member_that_reached_a_text(workspace, beta):
    synth.write(beta / "vendor/LICENSE", BUNDLED)
    component = evidence_module.discover(workspace.root, workspace.build).components[
        "beta"
    ]
    read = {"beta": {"beta-src/vendor/v.h": "header-read"}}
    inventory = SimpleNamespace(
        attributions={
            member: inventory_module.Attribution(
                path=member, origin="built", detail="", read=read
            )
            for member in ("z/late.so", "a/early.so")
        }
    )

    settled = build_module._with_reached_nested(inventory, {"beta": component})

    kept = next(
        item
        for item in settled["beta"].evidence
        if item.path == "beta-src/vendor/LICENSE"
    )
    assert kept.reached_by.member == "a/early.so"
